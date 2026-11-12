"""Error analysis: sort a run's worst cases on each test set into error types.

For each test set, the run's worst cases (lowest whole-hippocampus Dice, plain prediction, no
post-processing) are taken, and every wrong voxel of a case is put in one group:
  - islands: extra voxels in pieces apart from the prediction's largest piece (false-positive islands);
  - under-segmentation: label voxels the prediction misses, split into boundary (within 1 mm of the
    prediction, the outline stops short) and chunks (further away, a whole region left out);
  - split: hippocampus in both, but the wrong part (anterior/posterior); only where the label has parts;
  - over-segmentation: extra voxels on the main piece itself (outline too wide).
A case's bucket is the group with the most wrong voxels, except that a case with whole Dice below 0.5
(or an empty prediction) is a complete failure. Each case also lists every group holding at least 25% of
its error, as secondary buckets. Over-segmentation is not one of the four planned buckets, but it is kept
so that no error is hidden.

Test sets: the external crops of data/external/cases.json (MNI-HISUB25, atlas box, whole hippocampus only:
its labels have no anterior/posterior split, so the split bucket can't occur there), and the MSD test split
as the in-distribution reference, where the split bucket can be measured.

Outputs in <run>/error_analysis/:
    worst_cases.csv   one row per case: test set, case, Dice, error volume per group (ml), bucket, secondary
    summary.md        cases per bucket for each test set

Usage:
    python -m hippo.errors --run runs/baseline
    python -m hippo.errors --run runs/baseline --worst 10
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.ndimage import distance_transform_edt

from . import defaults
from .external import largest_piece, predictions
from .metrics import whole_metrics
from .train import spacing_from_affine

FAILURE_DICE = 0.5
SECONDARY_SHARE = 0.25
BUCKETS = {"islands": "false-positive islands", "under": "under-segmentation", "split": "anterior/posterior split",
           "over": "over-segmentation of the main piece", "failure": "complete failure (domain shift)"}
EXTERNAL_CASES = "data/external/cases.json"


def error_groups(pred: np.ndarray, label: np.ndarray, spacing, parts: bool) -> dict:
    """Wrong voxels of one case by group, in ml; parts=True when the label has anterior/posterior."""
    voxel_ml = float(np.prod(spacing)) / 1000
    main = largest_piece(pred)[0] > 0
    extra, missed = (pred > 0) & (label == 0), (label > 0) & (pred == 0)
    if (pred > 0).any():
        near = distance_transform_edt(pred == 0, sampling=spacing) <= 1.0
    else:
        near = np.zeros(pred.shape, bool)  # nothing predicted: every missed voxel is a missed chunk
    split = (pred > 0) & (label > 0) & (pred != label) if parts else np.zeros(pred.shape, bool)
    return {"islands_ml": (extra & ~main).sum() * voxel_ml, "over_ml": (extra & main).sum() * voxel_ml,
            "under_boundary_ml": (missed & near).sum() * voxel_ml, "under_chunks_ml": (missed & ~near).sum() * voxel_ml,
            "split_ml": split.sum() * voxel_ml}


def bucket(groups: dict, dice: float, empty: bool) -> tuple[str, list[str]]:
    """The case's main bucket and its secondary buckets (groups with at least 25% of its error)."""
    sizes = {"islands": groups["islands_ml"], "under": groups["under_boundary_ml"] + groups["under_chunks_ml"],
             "split": groups["split_ml"], "over": groups["over_ml"]}
    total = sum(sizes.values())
    secondary = [BUCKETS[k] for k, v in sizes.items() if total and v / total >= SECONDARY_SHARE]
    if empty or dice < FAILURE_DICE:
        return BUCKETS["failure"], secondary
    return BUCKETS[max(sizes, key=sizes.get)], secondary


def analyse(run: str, cases: list[dict], test_set: str, parts: bool) -> list[dict]:
    """Every case of a test set with its Dice, error groups and bucket."""
    cfg = yaml.safe_load((Path(run) / "config.yaml").read_text())
    rows = []
    with torch.no_grad():
        for case, pred, batch in predictions(run, cases):
            spacing = spacing_from_affine(batch["image"].affine)
            p, y = pred[0, 0].numpy().astype(int), batch["label"][0, 0].numpy().astype(int)
            dice = whole_metrics(pred, batch["label"], spacing, cfg["eval"]["nsd_tolerance_mm"])["dice"]
            groups = error_groups(p, y, spacing, parts)
            main, secondary = bucket(groups, dice, not (p > 0).any())
            rows.append({"test set": test_set, "case": case["case"], "dice_whole": round(dice, 4),
                         **{k: round(float(v), 3) for k, v in groups.items()},
                         "bucket": main, "secondary": "; ".join(secondary)})
    return rows


def summary(rows: list[dict], worst: int) -> str:
    """Cases per bucket for each test set, as Markdown."""
    lines = [f"# Error analysis: worst {worst} cases per test set", ""]
    for test_set in dict.fromkeys(r["test set"] for r in rows):
        mine = [r for r in rows if r["test set"] == test_set]
        lines += [f"## {test_set}", "", "| Bucket | Main bucket | Also ≥ 25% of the error |", "|---|---:|---:|"]
        for name in BUCKETS.values():
            main = sum(r["bucket"] == name for r in mine)
            also = sum(name in r["secondary"].split("; ") for r in mine)
            lines.append(f"| {name} | {main} | {also} |")
        lines += ["", "Cases (Dice, then error in ml: islands / over / under at boundary / under, chunks / split):", ""]
        lines += [f"- {r['case']}: Dice {r['dice_whole']:.3f}; {r['islands_ml']:.2f} / {r['over_ml']:.2f} / "
                  f"{r['under_boundary_ml']:.2f} / {r['under_chunks_ml']:.2f} / {r['split_ml']:.2f} -> {r['bucket']}"
                  for r in mine]
        lines.append("")
    lines.append("Plain predictions, no post-processing. Research use only, not for diagnosis.")
    return "\n".join(lines) + "\n"


def main(run: str, worst: int = 10, external_cases: str = EXTERNAL_CASES) -> list[dict]:
    cfg = yaml.safe_load((Path(run) / "config.yaml").read_text())
    sets = []
    if Path(external_cases).exists():
        sets.append(("MNI-HISUB25 (atlas box)", json.loads(Path(external_cases).read_text()), False))
    else:
        print(f"{external_cases} not found (run python -m hippo.external first); external set skipped")
    msd = json.loads(Path(cfg["data"]["split_file"]).read_text())["test"]
    sets.append(("MSD test", [{**c, "case": Path(c["image"]).name.split(".nii")[0]} for c in msd], True))
    rows = []
    for name, cases, parts in sets:
        scored = analyse(run, cases, name, parts)
        rows += sorted(scored, key=lambda r: r["dice_whole"])[:worst]
    out = Path(run) / "error_analysis"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "worst_cases.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    text = summary(rows, worst)
    (out / "summary.md").write_text(text)
    print(text)
    print(f"Saved {out / 'worst_cases.csv'} and summary.md")
    return rows


if __name__ == "__main__":
    p = defaults.make_parser("Sort a run's worst cases into error types.")
    p.add_argument("--run", default=defaults.RUN_DIR, help="Run folder with best.pt + config.yaml")
    p.add_argument("--worst", type=int, default=10, help="How many of the worst cases per test set")
    p.add_argument("--external-cases", default=EXTERNAL_CASES, help="cases.json of the external crops")
    a = p.parse_args()
    main(a.run, a.worst, a.external_cases)
