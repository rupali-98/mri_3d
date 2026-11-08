"""Predict a set of MSD crops without scoring them: masks to open in other tools, and label-free checks.

MSD's imagesTs has no public labels, so no Dice or NSD can be computed on it. This saves each mask
on the image's own voxel grid (same shape and affine as the input file, not padded), so tools that
need a matching grid can load it, and writes label-free checks per case. The checks compare each mask
with the run's training labels; they can show that a prediction looks wrong, not that it is right.
No largest-piece post-processing: this is in-distribution data.

--images test does the same for the labelled test split, as a reference: a second tool's agreement
with the model on imagesTs only means something next to its agreement with the true labels there.

Outputs:
    <run>/predictions_<images>/<case>.nii.gz   uint8 mask: 1 = anterior, 2 = posterior
    <run>/predictions_<images>.csv             case, vol_anterior_ml, vol_posterior_ml, vol_total_ml, pieces,
                                               edge_margin_mm, partner_in, flags
partner_in: the set holding file 2k-1/2k's possible other side (not documented by MSD, see make_splits).

Usage:
    python -m hippo.predict --run runs/baseline
    python -m hippo.predict --run runs/baseline --images test
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import yaml
from monai.transforms import Compose
from scipy import ndimage

from . import defaults
from .data import base_transforms
from .model import build_model

IMAGE_SETS = ("imagesTs", "test")


def to_image_grid(mask: np.ndarray, mask_affine: np.ndarray, image: nib.Nifti1Image) -> np.ndarray:
    """Cut the padding off a mask made on the RAS, padded grid; MSD files are already RAS, so only padding differs."""
    start = np.rint(-(np.linalg.inv(image.affine) @ np.asarray(mask_affine, dtype=np.float64))[:3, 3]).astype(int)
    return mask[tuple(slice(s, s + n) for s, n in zip(start, image.shape[:3]))]


def edge_margin(mask: np.ndarray, spacing) -> float:
    """Smallest distance (mm) between the mask and the crop edge, over all six sides."""
    found = np.argwhere(mask > 0)
    if not len(found):
        return float("nan")
    return float(min(np.min(found.min(0) * spacing), np.min((np.array(mask.shape) - 1 - found.max(0)) * spacing)))


def pieces(mask: np.ndarray) -> int:
    """Connected pieces of the whole mask (26-neighbour, as external.largest_piece)."""
    return ndimage.label(mask > 0, structure=np.ones((3, 3, 3)))[1]


def training_range(cases: list[dict]) -> dict:
    """What the training labels look like: the limits a prediction is checked against."""
    volumes, margins = [], []
    for c in cases:
        label = nib.load(c["label"])
        spacing = np.array(label.header.get_zooms()[:3])
        mask = np.asarray(label.dataobj)
        volumes.append(float((mask > 0).sum() * np.prod(spacing)) / 1000)
        margins.append(edge_margin(mask, spacing))
    return {"vol_min": min(volumes), "vol_max": max(volumes), "margin_min": min(margins)}


def case_id(path: str) -> int:
    return int(re.search(r"(\d+)\.nii", Path(path).name).group(1))


def partner_in(case: int, splits: dict) -> str:
    """The set holding the file that may be this case's other side (2k-1 pairs with 2k)."""
    partner = case + 1 if case % 2 else case - 1
    for name, cases in splits.items():
        if partner in {case_id(c["image"]) for c in cases}:
            return name
    return "imagesTs"


def flags(row: dict, limits: dict) -> str:
    """Why a prediction looks unusual, compared with the training labels ("" if it doesn't)."""
    found = []
    if not row["vol_anterior_ml"] or not row["vol_posterior_ml"]:
        found.append("part missing")
    if not limits["vol_min"] <= row["vol_total_ml"] <= limits["vol_max"]:
        found.append("volume outside training range")
    if row["pieces"] > 1:
        found.append("several pieces")
    if not row["edge_margin_mm"] >= limits["margin_min"]:  # also true for an empty mask (nan)
        found.append("closer to the edge than any training label")
    return "; ".join(found)


@torch.no_grad()
def main(run_dir: str, images: str = "imagesTs", data_dir: str | None = None) -> list[dict]:
    if images not in IMAGE_SETS:
        raise SystemExit(f"--images must be one of {IMAGE_SETS}, not {images!r}")
    run = Path(run_dir)
    cfg = yaml.safe_load((run / "config.yaml").read_text())
    split_file = Path(data_dir) / defaults.SPLIT_FILENAME if data_dir else Path(cfg["data"]["split_file"])
    splits = json.loads(split_file.read_text())
    if images == "test":
        files = [c["image"] for c in splits["test"]]
    else:  # skips macOS "._" metadata files, which share the name but are not images
        folder = split_file.parent / "Task04_Hippocampus" / "imagesTs"
        files = sorted(str(p) for p in folder.glob("*.nii.gz") if not p.name.startswith("._"))
    if not files:
        raise SystemExit(f"No images found for {images}")

    device = defaults.pick_device()
    model = build_model(cfg).to(device)
    model.load_state_dict(torch.load(run / "best.pt", map_location=device, weights_only=True))
    model.eval()
    transform = Compose(base_transforms(cfg["data"].get("normalize", "zscore"), keys=("image",)))
    limits = training_range(splits["train"])

    out = run / f"predictions_{images}"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for file in files:
        loaded = transform({"image": file})["image"]
        pred = model(loaded.as_tensor()[None].to(device)).argmax(1)[0].cpu().numpy().astype(np.uint8)
        image = nib.load(file)
        mask = to_image_grid(pred, loaded.affine, image)
        spacing = np.array(image.header.get_zooms()[:3])
        nib.save(nib.Nifti1Image(mask, image.affine), out / Path(file).name)
        voxel_ml = float(np.prod(spacing)) / 1000
        row = {"case": Path(file).name.split(".nii")[0],
               "vol_anterior_ml": round(float((mask == 1).sum()) * voxel_ml, 3),
               "vol_posterior_ml": round(float((mask == 2).sum()) * voxel_ml, 3),
               "vol_total_ml": round(float((mask > 0).sum()) * voxel_ml, 3),
               "pieces": pieces(mask),
               "edge_margin_mm": edge_margin(mask, spacing),
               "partner_in": partner_in(case_id(file), splits)}
        rows.append({**row, "flags": flags(row, limits)})

    with open(run / f"predictions_{images}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    totals = [r["vol_total_ml"] for r in rows]
    print(f"{len(rows)} masks in {out}\nTotal volume median {np.median(totals):.2f} ml "
          f"(training labels {limits['vol_min']:.2f}-{limits['vol_max']:.2f} ml); "
          f"{sum(r['pieces'] == 1 for r in rows)} one piece; {sum(bool(r['flags']) for r in rows)} flagged")
    return rows


if __name__ == "__main__":
    p = defaults.make_parser("Predict MSD crops without scoring: masks and label-free checks.")
    p.add_argument("--run", default=defaults.RUN_DIR)
    p.add_argument("--images", default="imagesTs", choices=IMAGE_SETS)
    # Without --data-dir / HIPPO_DATA_DIR / hippo.conf, use the location saved in the run.
    p.set_defaults(data_dir=None)
    a = p.parse_args()
    main(a.run, a.images, a.data_dir)
