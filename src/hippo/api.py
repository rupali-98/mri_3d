"""HTTP API for the hippocampus model (FastAPI): volumes and masks for a crop or a whole-brain MNI scan.

Serves a model package from hippo.package_model (default deploy/model/; the fold models' class
probabilities are averaged). Research use only, not for diagnosis: every response says so.

Endpoints (interactive documentation at /docs):
    GET  /health           liveness: are the models loaded
    GET  /models           the models this service holds (name to pass as ?model=, version, which is the default)
    GET  /model            name, version (hash of the weights), preprocessing, cross-validation results
    POST /segment/crop     one T1 crop around one hippocampus (NIfTI, 1 mm, at most 128 voxels per side)
    POST /segment/mni      a whole-brain T1 already registered to MNI space (NIfTI, 1 mm): both hippocampi
                           are cut out with the same atlas boxes hippo.external uses, and segmented

Each result gives the anterior, posterior and whole volume (ml), quality checks (pieces, distance to
the crop edge, missing part; for crops also the training volume range) and, with include_mask=true, the
mask as a base64 gzipped NIfTI on the crop's grid (with its affine, so it lines up with the scan).
largest=true (both endpoints, off by default) keeps only each mask's largest connected piece; the result
then also gives the plain mask's checks (qc_before_post_processing) and the volume removed (removed_ml).
The reported MSD results never use it.

Several models: the package at HIPPO_MODEL_DIR is the default; packages in its versions/<name>/ folders
(written by the cloud build, one per MLflow alias) are offered too. Every segment endpoint takes
?model=<name> (default: the default model), and every response names the model that answered. The name
of the default is HIPPO_DEFAULT_MODEL, else the file default_model in HIPPO_MODEL_DIR, else "default".
A versions/ entry with the same weights as the default shares its loaded copy.
Inputs the model was not made for are refused with HTTP 422: not a 3D NIfTI, voxel size not about 1 mm,
a crop larger than 128 voxels, or an "MNI" scan that doesn't contain the atlas boxes. The API cannot
check that a scan is really in MNI space, so /segment/mni always says so. Uploads are read from a
temporary file that is deleted during the request; nothing is stored.

Settings (environment variables):
    HIPPO_MODEL_DIR   model package folder (default deploy/model), optionally with versions/<name>/ packages
    HIPPO_DEFAULT_MODEL  name of the default model (see above)
    HIPPO_API_KEY     if set, every request except /health needs the header X-API-Key with this value
    HIPPO_API_KEY_FILE  the same key read from a file (Docker Compose secret /run/secrets/api_key), if HIPPO_API_KEY is unset
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


def result_for(mask: np.ndarray, image: nib.Nifti1Image, limits: dict | None, include_mask: bool,
               largest: bool) -> dict:
    """describe(), optionally after keeping only the largest piece (post-processing, off by default).

    With largest, the quality checks of the plain mask are kept too (qc_before_post_processing), so the
    islands that were removed still show, with the volume they had (removed_ml).
    """
    plain = describe(mask, image, limits, include_mask and not largest)
    if not largest:
        return plain
    kept, _ = largest_piece(mask)
    voxel_ml = float(np.prod(image.header.get_zooms()[:3])) / 1000
    removed_ml = float((mask > 0).sum() - (kept > 0).sum()) * voxel_ml
    return {**describe(kept, image, limits, include_mask), "qc_before_post_processing": plain["qc"],
            "removed_ml": round(removed_ml, 3)}


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


def load_models(folder: Path) -> tuple[dict[str, Model], str]:
    """The packages in a model folder by name, and the default's name (see the module docstring)."""
    folder = Path(folder)
    models = {sub.parent.name: Model(sub.parent) for sub in sorted((folder / "versions").glob("*/metadata.json"))}
    named = os.environ.get("HIPPO_DEFAULT_MODEL")
    if not named and (folder / "default_model").exists():
        named = (folder / "default_model").read_text().strip()
    if models and named and named not in models:  # with versions/, the default must be one of them
        raise SystemExit(f"No model package called {named!r} in {folder}/versions (found: {sorted(models)})")
    if (folder / "metadata.json").exists():
        default = named or "default"
        version = json.loads((folder / "metadata.json").read_text())["version"]
        if default not in models or models[default].meta["version"] != version:  # else: one copy for both
            models[default] = Model(folder)
    else:
        default = named or next(iter(models), None)
    if default not in models:
        raise SystemExit(f"No model package called {default!r} in {folder} (found: {sorted(models) or 'none'})")
    return models, default


def env_api_key() -> str | None:
    """The API key from HIPPO_API_KEY, else from the file HIPPO_API_KEY_FILE names, else None."""
    if os.environ.get("HIPPO_API_KEY"):
        return os.environ["HIPPO_API_KEY"]
    path = os.environ.get("HIPPO_API_KEY_FILE")
    if not path:
        return None
    # A file keeps the key out of `docker inspect` and process listings, where environment variables show
    return Path(path).read_text().strip() or None


def create_app(model_dir: str | None = None, api_key: str | None = None, max_mb: float | None = None) -> FastAPI:
    """The FastAPI app; the arguments override the environment variables (the tests use them)."""
    model_dir = Path(model_dir or os.environ.get("HIPPO_MODEL_DIR", "deploy/model"))
    api_key = api_key if api_key is not None else env_api_key()
    max_mb = max_mb or float(os.environ.get("HIPPO_MAX_MB", 30))
    state = {}

    @asynccontextmanager
    async def lifespan(app):
        state["models"], state["default"] = load_models(model_dir)  # once, not per request
        for key, m in state["models"].items():
            log.info("loaded %s: %s version %s%s", key, m.meta["name"], m.meta["version"],
                     " (default)" if key == state["default"] else "")
        yield

    app = FastAPI(title="Hippocampus segmentation API", lifespan=lifespan,
                  description="3D U-Net for anterior/posterior hippocampus segmentation and volumetry on T1 MRI "
                              "(GET /models lists the models it serves; choose one with ?model=). **Research use "
                              "only, not for diagnosis.**")

    def authorise(x_api_key: str | None = Header(default=None)):
        if api_key and x_api_key != api_key:
            raise HTTPException(401, "missing or wrong X-API-Key")

    def pick(name: str | None) -> tuple[str, Model]:
        """The requested model (the default without a name), or 404 listing the available ones."""
        key = name or state["default"]
        if key not in state["models"]:
            raise HTTPException(404, f"no model called {key!r}; available: {sorted(state['models'])} (GET /models)")
        return key, state["models"][key]

    def header(key: str, model: Model) -> dict:
        return {"model": {"key": key, "name": model.meta["name"], "version": model.meta["version"]},
                "intended_use": model.meta["intended_use"]}

    choose = Query(None, description="Which model to use (a name from GET /models); default: the default model")

    @app.get("/", include_in_schema=False)
    def root():
        return RedirectResponse("/docs")

    @app.get("/health")
    def health():
        return {"status": "ok" if "models" in state else "loading"}

    @app.get("/models", dependencies=[Depends(authorise)])
    def models_list():
        return {"default": state["default"],
                "models": [{"key": key, "name": m.meta["name"], "version": m.meta["version"],
                            "combination": m.meta["combination"], "experiment": m.meta.get("experiment"),
                            "default": key == state["default"], "mni": "mni_boxes" in m.meta}
                           for key, m in state["models"].items()]}

    @app.get("/model", dependencies=[Depends(authorise)])
    def model_info(model: str | None = choose):
        key, chosen = pick(model)
        meta = chosen.meta
        keys = ("name", "version", "experiment", "created", "members", "combination", "normalize", "labels",
                "input", "cross_validation", "intended_use")
        return {"key": key} | {k: meta[k] for k in keys} | {"endpoints": {"crop": True, "mni": "mni_boxes" in meta}}

    @app.post("/segment/crop", dependencies=[Depends(authorise)])
    async def segment_crop(file: UploadFile = File(..., description="T1 crop around one hippocampus, NIfTI"),
                           include_mask: bool = Query(False, description="Return the mask as base64 NIfTI"),
                           largest: bool = Query(False, description="Keep only the mask's largest piece "
                                                                    "(post-processing; off by default and in the "
                                                                    "reported MSD results)"),
                           model: str | None = choose):
        (key, chosen), started, request = pick(model), time.time(), uuid.uuid4().hex[:8]
        path, image = await read_nifti(file, max_mb)
        try:
            if problems := check_image(image, crop=True):
                raise HTTPException(422, problems)
            # the model is CPU work: run it in a worker thread so the server keeps answering (e.g. /health)
            mask = await run_in_threadpool(chosen.segment, path, image)
        finally:
            os.unlink(path)
        result = result_for(mask, image, chosen.meta["training_label_range"], include_mask, largest)
        log.info("request %s crop %s model %s %.2fs flags=%s", request, list(image.shape), key, time.time() - started,
                 result["qc"]["flags"])
        return header(key, chosen) | {"request_id": request, "input": {"shape": list(image.shape)},
                                "post_processing": "largest piece" if largest else "none",
                                "hippocampi": [{"side": "unknown (crop)", **result}],
                                "seconds": round(time.time() - started, 2)}

    @app.post("/segment/mni", dependencies=[Depends(authorise)])
    async def segment_mni(file: UploadFile = File(..., description="Whole-brain T1 in MNI space, NIfTI"),
                          include_mask: bool = Query(False, description="Return each mask as base64 NIfTI"),
                          largest: bool = Query(False, description="Keep only each mask's largest piece "
                                                                   "(post-processing; off in the main results)"),
                          model: str | None = choose):
        (key, chosen), started, request = pick(model), time.time(), uuid.uuid4().hex[:8]
        boxes = chosen.meta.get("mni_boxes")
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
            mask, crop = await run_in_threadpool(segment_box, chosen, image, data, box, side)
            # the training volume range is MSD's, not MNI space, so it isn't checked here
            result = result_for(mask, crop, None, include_mask, largest)
            hippocampi.append({"side": "left" if side == "L" else "right", **result})
        log.info("request %s mni %s model %s %.2fs", request, list(image.shape), key, time.time() - started)
        return header(key, chosen) | {"request_id": request, "input": {"shape": list(image.shape)},
                                "post_processing": "largest piece" if largest else "none",
                                "warnings": [NOT_MNI_CHECKED], "hippocampi": hippocampi,
                                "seconds": round(time.time() - started, 2)}

    return app


# `uvicorn hippo.api:app` serves this; the tests build their own with create_app(...)
app = create_app()
