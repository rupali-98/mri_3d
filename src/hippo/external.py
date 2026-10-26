"""Evaluate a trained model on an external dataset of whole-brain scans in MNI space.

The external data (folder of subjects s01, s02, ...) has whole-brain T1 scans and
hippocampal subfield labels (1, 2, 3; separate left and right files), all on the
1 mm MNI grid of the SPAM_* subfield probability maps. The model was trained on
MSD crops with anterior/posterior labels, so:

  - each hippocampus is cut out as an MSD-sized box centred on the SPAM atlas
    (never on the subject's own label, so the ground truth does not place the crop);
  - only the whole hippocampus can be compared: prediction > 0 vs label > 0.

Crops are saved as ordinary NIfTI files (data/external/imagesTr, labelsTr), so
hippo.visualize --image ... --label ... works on them. Volumes are in MNI space,
i.e. scaled to the template brain, not the subject's own head size.

Outputs in <run>/external/:
    metrics_external.csv   one row per hemisphere: Dice, NSD (whole hippocampus), volumes
    summary_external.json  mean, median, std

Usage:
    python -m hippo.external --src data/mri_dataset --run runs/baseline
    python -m hippo.external --src data/mri_dataset --run runs/baseline --subjects s03
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import yaml
from monai.data import DataLoader, Dataset
from monai.metrics import compute_dice, compute_surface_dice
from monai.networks.utils import one_hot

from . import defaults
from .data import eval_transforms
from .model import build_model
from .train import spacing_from_affine

OUT_DIR = "data/external"
# MSD crops are 30-43 x 40-59 x 24-47 voxels at ~1 mm. Slightly larger here so that, despite
# subject-to-subject variation around the atlas centre, every hippocampus keeps a margin to the
# edge as in MSD (40x56x48 cut some off). Divisible by 8 for the U-Net.
CROP_MM = (48, 56, 56)
SUBFIELDS = ["CA1-3", "CA4-DG", "SUB"]  # SPAM_<subfield>_<side>_MNI.nii.gz


def atlas_centre(src: Path, side: str) -> tuple[np.ndarray, nib.Nifti1Image]:
    """Voxel centre of the hippocampus on one side: probability-weighted centroid of the SPAM maps."""
    maps = [nib.load(src / f"SPAM_{s}_{side}_MNI.nii.gz") for s in SUBFIELDS]
    prob = sum(np.asarray(m.dataobj, dtype=np.float32) for m in maps)
    centre = (np.argwhere(prob > 0) * prob[prob > 0][:, None]).sum(0) / prob[prob > 0].sum()
    return centre, maps[0]


def crop_box(centre: np.ndarray, shape: tuple, size: tuple) -> tuple[slice, ...]:
    """A box of `size` voxels around `centre`, shifted (not shrunk) to stay inside the volume."""
    starts = [int(np.clip(round(c - s / 2), 0, n - s)) for c, s, n in zip(centre, size, shape)]
    return tuple(slice(a, a + s) for a, s in zip(starts, size))


def save_crop(img: nib.Nifti1Image, box: tuple[slice, ...], data: np.ndarray, path: Path) -> None:
    """Save `data` (already cropped) with the affine moved to the box corner, so it stays in place."""
    affine = img.affine.copy()
    affine[:3, 3] = nib.affines.apply_affine(img.affine, [b.start for b in box])
    nib.save(nib.Nifti1Image(data, affine), path)


def prepare(src: str, out: str = OUT_DIR, subjects: list[str] | None = None, image: str = "t1w_standard") -> list[dict]:
    """Crop every subject's left and right hippocampus; returns the case list (image, label, coverage)."""
    src, out = Path(src), Path(out)
    for sub in ("imagesTr", "labelsTr"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    found = sorted(p.name for p in src.glob("s[0-9]*") if p.is_dir())
    missing = sorted(set(subjects or []) - set(found))
    if missing:
        raise SystemExit(f"Subjects not found in {src}: {missing}")
    names = subjects or found

    cases = []
    for side in ("L", "R"):
        centre, ref = atlas_centre(src, side)
        for s in names:
            img = nib.load(src / s / f"{s}_{image}_defaced_MNI.nii.gz")
            lbl = nib.load(src / s / f"{s}_hippolabels_{image}_{side}_MNI.nii.gz")
            if img.shape != ref.shape or not np.allclose(img.affine, ref.affine, atol=1e-3):
                raise SystemExit(f"{s}: {image} is not on the SPAM atlas grid; crops would be misplaced")
            box = crop_box(centre, img.shape, CROP_MM)
            y = np.rint(np.asarray(lbl.dataobj)).astype(np.uint8)
            name = f"{s}_{side}"
            save_crop(img, box, np.asarray(img.dataobj, dtype=np.float32)[box], out / "imagesTr" / f"{name}.nii.gz")
            save_crop(lbl, box, y[box], out / "labelsTr" / f"{name}.nii.gz")
            # share of the hippocampus inside the crop; below 1 means the box cut part of it off
            coverage = float((y[box] > 0).sum() / max((y > 0).sum(), 1))
            cases.append({"case": name, "image": str(out / "imagesTr" / f"{name}.nii.gz"),
                          "label": str(out / "labelsTr" / f"{name}.nii.gz"), "coverage": round(coverage, 4)})
    cases.sort(key=lambda c: c["case"])
    (out / "cases.json").write_text(json.dumps(cases, indent=2))
    print(f"Cropped {len(cases)} hippocampi into {out}; lowest label coverage "
          f"{min(c['coverage'] for c in cases):.3f} ({min(cases, key=lambda c: c['coverage'])['case']})")
    return cases


def whole_metrics(pred: torch.Tensor, label: torch.Tensor, spacing_mm, nsd_tolerance_mm: float) -> dict:
    """Dice, NSD and volumes of the whole hippocampus (any label > 0) for one case."""
    p, y = one_hot((pred > 0).long(), num_classes=2, dim=1), one_hot((label > 0).long(), num_classes=2, dim=1)
    voxel_ml = float(np.prod(spacing_mm)) / 1000
    return {
        "dice": float(compute_dice(p, y, include_background=False, ignore_empty=False)[0, 0]),
        "nsd": float(compute_surface_dice(p, y, class_thresholds=[nsd_tolerance_mm], include_background=False,
                                          spacing=list(map(float, spacing_mm)))[0, 0]),
        "vol_pred_ml": float((pred > 0).sum()) * voxel_ml,
        "vol_true_ml": float((label > 0).sum()) * voxel_ml,
    }


@torch.no_grad()
def evaluate(run_dir: str, cases: list[dict]) -> dict:
    run = Path(run_dir)
    cfg = yaml.safe_load((run / "config.yaml").read_text())
    device = defaults.pick_device()
    model = build_model(cfg).to(device)
    model.load_state_dict(torch.load(run / "best.pt", map_location=device, weights_only=True))
    model.eval()

    transforms = eval_transforms(cfg["data"].get("normalize", "zscore"))  # the run's own preprocessing
    loader = DataLoader(Dataset([{"image": c["image"], "label": c["label"]} for c in cases], transforms), batch_size=1)
    rows = []
    for case, batch in zip(cases, loader):
        pred = model(batch["image"].to(device)).argmax(dim=1, keepdim=True).cpu()
        m = whole_metrics(pred, batch["label"], spacing_from_affine(batch["image"].affine),
                          cfg["eval"]["nsd_tolerance_mm"])
        rows.append({"case": case["case"], **m, "label_coverage": case["coverage"]})

    out = run / "external"
    out.mkdir(exist_ok=True)
    with open(out / "metrics_external.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary = {"n_cases": len(rows), "metric": "whole hippocampus (prediction > 0 vs any subfield label)"}
    for k in ("dice", "nsd"):
        vals = np.array([r[k] for r in rows])
        summary[k] = {"mean": float(vals.mean()), "median": float(np.median(vals)), "std": float(vals.std())}
    (out / "summary_external.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    p = defaults.make_parser("Evaluate a trained model on external whole-brain MNI data.")
    p.add_argument("--src", required=True, help="External dataset folder (s01/, s02/, ..., SPAM_*_MNI.nii.gz)")
    p.add_argument("--run", default=defaults.RUN_DIR, help="Run folder with best.pt + config.yaml")
    p.add_argument("--subjects", nargs="+", help="Only these subjects, e.g. s03 (default: all)")
    p.add_argument("--out", default=OUT_DIR, help="Folder for the cropped cases")
    a = p.parse_args()
    evaluate(a.run, prepare(a.src, a.out, a.subjects))
