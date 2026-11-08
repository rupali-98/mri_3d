"""HTTP API for the hippocampus model (FastAPI): volumes and masks for a crop or a whole-brain MNI scan.

Serves a model package from hippo.package_model (default deploy/model/; the fold models' class
probabilities are averaged). Research use only, not for diagnosis: every response says so.

Endpoints (interactive documentation at /docs):
    GET  /health           liveness: is the model loaded
    GET  /model            name, version (hash of the weights), preprocessing, cross-validation results
    POST /segment/crop     one T1 crop around one hippocampus (NIfTI, 1 mm, at most 128 voxels per side)
    POST /segment/mni      a whole-brain T1 already registered to MNI space (NIfTI, 1 mm): both hippocampi
                           are cut out with the same atlas boxes hippo.external uses, and segmented

Each result gives the anterior, posterior and whole volume (ml), quality checks (pieces, distance to
the crop edge, missing part; for crops also the training volume range) and, with include_mask=true, the
mask as a base64 gzipped NIfTI on the crop's grid (with its affine, so it lines up with the scan).
Inputs the model was not made for are refused with HTTP 422: not a 3D NIfTI, voxel size not about 1 mm,
a crop larger than 128 voxels, or an "MNI" scan that doesn't contain the atlas boxes. The API cannot
check that a scan is really in MNI space, so /segment/mni always says so. Uploads are read from a
temporary file that is deleted during the request; nothing is stored.

Settings (environment variables):
    HIPPO_MODEL_DIR   model package folder (default deploy/model)
    HIPPO_API_KEY     if set, every request except /health needs the header X-API-Key with this value
    HIPPO_MAX_MB      largest upload accepted, in MB (default 30: Cloud Run takes at most 32 MiB per HTTP/1 request)

Usage:
    PYTHONPATH=src uvicorn hippo.api:app --port 8000          # then open http://127.0.0.1:8000/docs
    curl -F file=@data/Task04_Hippocampus/imagesTr/hippocampus_135.nii.gz http://127.0.0.1:8000/segment/crop
"""
from __future__ import annotations

import base64
import gzip
import json
import logging
import os
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import yaml
from fastapi import Depends, FastAPI, File, Header, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import RedirectResponse
from monai.transforms import Compose

from . import CLASS_NAMES
from .data import base_transforms
from .external import largest_piece, world_box
from .model import build_model
from .predict import edge_margin, pieces, to_image_grid

MAX_CROP = 128          # longest side (voxels) of a crop; the training crops are 24-59
VOXEL_MM = (0.8, 1.25)  # accepted voxel size: the model was trained on 1 mm voxels only
NOT_MNI_CHECKED = ("The scan is assumed to be registered to MNI space (as MNI-HISUB25); this is not checked. "
                   "MNI-space volumes are scaled to the template brain, not the subject's own head size.")
log = logging.getLogger("uvicorn.error")  # uvicorn's logger, so the lines reach the container logs


class Model:
    """The packaged fold models, loaded once; segment() averages their probabilities."""

    def __init__(self, folder: Path):
        self.folder = Path(folder)
        self.meta = json.loads((self.folder / "metadata.json").read_text())
        self.nets = []
        for member in self.meta["members"]:
            cfg = yaml.safe_load((self.folder / member / "config.yaml").read_text())
            net = build_model(cfg)
            net.load_state_dict(torch.load(self.folder / member / "best.pt", map_location="cpu", weights_only=True))
            self.nets.append(net.eval())
        self.transform = Compose(base_transforms(self.meta["normalize"], keys=("image",)))

    @torch.no_grad()
    def segment(self, path: str, image: nib.Nifti1Image) -> np.ndarray:
        """The mask (0, 1 anterior, 2 posterior) on the image's own grid."""
        loaded = self.transform({"image": path})["image"]
        x = loaded.as_tensor()[None]
        prob = torch.stack([torch.softmax(net(x), 1) for net in self.nets]).mean(0)
        return to_image_grid(prob.argmax(1)[0].numpy().astype(np.uint8), loaded.affine, image)


def check_image(image: nib.Nifti1Image, crop: bool) -> list[str]:
    """Why the model shouldn't be run on this image ([] if it can be)."""
    if len(image.shape) != 3:
        return [f"expected a 3D image, got shape {list(image.shape)}"]
    problems = []
    zooms = image.header.get_zooms()[:3]
    if not all(VOXEL_MM[0] <= z <= VOXEL_MM[1] for z in zooms):
        problems.append(f"voxel size {[round(float(z), 2) for z in zooms]} mm; the model needs about 1 mm "
                        f"({VOXEL_MM[0]}-{VOXEL_MM[1]} mm)")
    if crop and max(image.shape) > MAX_CROP:
        problems.append(f"{list(image.shape)} voxels is too large for a hippocampus crop (at most {MAX_CROP} per "
                        "side); send whole-brain MNI scans to /segment/mni")
    return problems


def describe(mask: np.ndarray, image: nib.Nifti1Image, limits: dict | None, include_mask: bool) -> dict:
    """Volumes, quality checks and (optionally) the mask of one segmented hippocampus."""
    spacing = np.array(image.header.get_zooms()[:3], dtype=float)
    voxel_ml = float(np.prod(spacing)) / 1000
    vols = {name: round(float((mask == cls).sum()) * voxel_ml, 3) for cls, name in CLASS_NAMES.items()}
    vols["whole"] = round(float((mask > 0).sum()) * voxel_ml, 3)
    n, margin = pieces(mask), edge_margin(mask, spacing)
    flags = []
    if not all(vols[name] for name in CLASS_NAMES.values()):
        flags.append("part missing")
    if n > 1:
        flags.append(f"{n} pieces (islands)")
    if limits and not margin >= limits["margin_min"]:  # also true for an empty mask (nan)
        flags.append("closer to the crop edge than any training label")
    if limits and not limits["vol_min"] <= vols["whole"] <= limits["vol_max"]:
        flags.append(f"volume outside the training labels' range ({limits['vol_min']}-{limits['vol_max']} ml)")
    out = {"volumes_ml": vols, "qc": {"pieces": n, "edge_margin_mm": None if margin != margin else round(margin, 1),
                                      "flags": flags, "needs_review": bool(flags)}}
    if include_mask:
        data = nib.Nifti1Image(mask.astype(np.uint8), image.affine).to_bytes()
        out["mask_nifti_gz_base64"] = base64.b64encode(gzip.compress(data)).decode()
    return out


async def read_nifti(upload: UploadFile, max_mb: float) -> tuple[str, nib.Nifti1Image]:
    """Save the upload to a temporary file (the caller deletes it) and open it as NIfTI, or refuse it."""
    data = await upload.read()
    if len(data) > max_mb * 1e6:
        raise HTTPException(413, f"file is {len(data) / 1e6:.1f} MB; the limit is {max_mb} MB")
    suffix = ".nii.gz" if data[:2] == b"\x1f\x8b" else ".nii"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(data)
    try:
        image = nib.load(f.name)
        image.get_fdata(dtype=np.float32)  # fail here, not in the model, if the file is broken
    except Exception:
        os.unlink(f.name)
        raise HTTPException(422, "not a readable NIfTI file (.nii or .nii.gz)") from None
    return f.name, image


def segment_box(model: Model, image: nib.Nifti1Image, data: np.ndarray, box: dict, side: str) -> tuple:
    """Cut one atlas box out of a whole-brain MNI scan and segment it; returns (mask, crop image)."""
    region = world_box(image, box["lo_mm"], box["hi_mm"])
    if [s.stop - s.start for s in region] != box["size_vox"]:
        raise HTTPException(422, f"the {side} atlas box lies partly outside the image: is it a whole-brain scan "
                                 "in MNI space?")
    affine = image.affine.copy()
    affine[:3, 3] = nib.affines.apply_affine(image.affine, [s.start for s in region])
    crop = nib.Nifti1Image(data[region], affine)
    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as f:
        path = f.name
    try:
        nib.save(crop, path)
        return model.segment(path, crop), crop
    finally:
        os.unlink(path)


def create_app(model_dir: str | None = None, api_key: str | None = None, max_mb: float | None = None) -> FastAPI:
    """The FastAPI app; the arguments override the environment variables (the tests use them)."""
    model_dir = Path(model_dir or os.environ.get("HIPPO_MODEL_DIR", "deploy/model"))
    api_key = api_key if api_key is not None else os.environ.get("HIPPO_API_KEY")
    max_mb = max_mb or float(os.environ.get("HIPPO_MAX_MB", 30))
    state = {}

    @asynccontextmanager
    async def lifespan(app):
        state["model"] = Model(model_dir)  # once, not per request
        log.info("loaded %s version %s", state["model"].meta["name"], state["model"].meta["version"])
        yield

    app = FastAPI(title="Hippocampus segmentation API", lifespan=lifespan,
                  description="3D U-Net for anterior/posterior hippocampus segmentation and volumetry on T1 MRI "
                              "(GET /model says which trained models it serves). **Research use only, not for "
                              "diagnosis.**")

    def authorise(x_api_key: str | None = Header(default=None)):
        if api_key and x_api_key != api_key:
            raise HTTPException(401, "missing or wrong X-API-Key")

    def header(model: Model) -> dict:
        return {"model": {"name": model.meta["name"], "version": model.meta["version"]},
                "intended_use": model.meta["intended_use"]}

    @app.get("/", include_in_schema=False)
    def root():
        return RedirectResponse("/docs")

    @app.get("/health")
    def health():
        return {"status": "ok" if "model" in state else "loading"}

    @app.get("/model", dependencies=[Depends(authorise)])
    def model_info():
        meta = state["model"].meta
        keys = ("name", "version", "experiment", "created", "members", "combination", "normalize", "labels",
                "input", "cross_validation", "intended_use")
        return {k: meta[k] for k in keys} | {"endpoints": {"crop": True, "mni": "mni_boxes" in meta}}

    @app.post("/segment/crop", dependencies=[Depends(authorise)])
    async def segment_crop(file: UploadFile = File(..., description="T1 crop around one hippocampus, NIfTI"),
                           include_mask: bool = Query(False, description="Return the mask as base64 NIfTI")):
        model, started, request = state["model"], time.time(), uuid.uuid4().hex[:8]
        path, image = await read_nifti(file, max_mb)
        try:
            if problems := check_image(image, crop=True):
                raise HTTPException(422, problems)
            # the model is CPU work: run it in a worker thread so the server keeps answering (e.g. /health)
            mask = await run_in_threadpool(model.segment, path, image)
        finally:
            os.unlink(path)
        result = describe(mask, image, model.meta["training_label_range"], include_mask)
        log.info("request %s crop %s %.2fs flags=%s", request, list(image.shape), time.time() - started,
                 result["qc"]["flags"])
        return header(model) | {"request_id": request, "input": {"shape": list(image.shape)},
                                "post_processing": "none", "hippocampi": [{"side": "unknown (crop)", **result}],
                                "seconds": round(time.time() - started, 2)}

    @app.post("/segment/mni", dependencies=[Depends(authorise)])
    async def segment_mni(file: UploadFile = File(..., description="Whole-brain T1 in MNI space, NIfTI"),
                          include_mask: bool = Query(False, description="Return each mask as base64 NIfTI"),
                          largest: bool = Query(False, description="Keep only each mask's largest piece "
                                                                   "(post-processing; off in the main results)")):
        model, started, request = state["model"], time.time(), uuid.uuid4().hex[:8]
        boxes = model.meta.get("mni_boxes")
        if not boxes:
            raise HTTPException(501, "this model package has no MNI boxes (package it with --src)")
        path, image = await read_nifti(file, max_mb)
        try:
            if problems := check_image(image, crop=False):
                raise HTTPException(422, problems)
            data = np.asarray(image.dataobj, dtype=np.float32)
        finally:
            os.unlink(path)
        hippocampi = []
        for side, box in boxes.items():
            mask, crop = await run_in_threadpool(segment_box, model, image, data, box, side)
            # quality flags on the plain mask; the training volume range is MSD's, not MNI space, so not used
            plain = describe(mask, crop, None, include_mask and not largest)
            if largest:
                kept, _ = largest_piece(mask)
                voxel_ml = float(np.prod(crop.header.get_zooms()[:3])) / 1000
                removed_ml = float((mask > 0).sum() - (kept > 0).sum()) * voxel_ml
                result = {**describe(kept, crop, None, include_mask), "qc_before_post_processing": plain["qc"],
                          "removed_ml": round(removed_ml, 3)}
            else:
                result = plain
            hippocampi.append({"side": "left" if side == "L" else "right", **result})
        log.info("request %s mni %s %.2fs", request, list(image.shape), time.time() - started)
        return header(model) | {"request_id": request, "input": {"shape": list(image.shape)},
                                "post_processing": "largest piece" if largest else "none",
                                "warnings": [NOT_MNI_CHECKED], "hippocampi": hippocampi,
                                "seconds": round(time.time() - started, 2)}

    return app


# `uvicorn hippo.api:app` serves this; the tests build their own with create_app(...)
app = create_app()
