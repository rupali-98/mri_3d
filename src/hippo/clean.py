"""Clean the labelled dataset, using the exploration report.

Raw data is never changed. For each labelled case:
  - FAIL cases (from hippo.explore) and cases in clean.exclude are left out;
  - label pieces smaller than clean.min_piece_voxels, other than the main piece
    of each part, become background (26-neighbour connectivity); larger extra
    pieces are kept and flagged.
Changed labels are written to data/clean/labelsTr/; all other cases keep their
original label file.

Outputs in data/clean/:
    cases.csv          the cleaned dataset index (paths relative to the data folder),
                       one row per labelled case: included or not, and why
    cleaning_log.csv   every change: case, part, pieces and voxels removed, volume before/after
    summary.json       counts

Usage (run python -m hippo.explore first):
    python -m hippo.clean
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import yaml
from scipy import ndimage

from . import CLASS_NAMES, defaults
from .explore import CONNECTIVITY, OUT_DIR as EXPLORE_DIR, list_cases

OUT_DIR = "data/clean"


def remove_small_pieces(label: np.ndarray, min_voxels: int) -> tuple[np.ndarray, list[dict], list[dict]]:
    """Set small extra pieces of each part to background.

    Returns the cleaned label, what was removed (per part) and the extra pieces
    that were kept because they are not small. The largest piece of a part is
    never removed, whatever its size.
    """
    cleaned = label.copy()
    removed, kept = [], []
    for value, part in CLASS_NAMES.items():
        pieces, n = ndimage.label(label == value, structure=CONNECTIVITY)
        if n <= 1:
            continue
        sizes = np.bincount(pieces.ravel())[1:]  # voxels per piece, piece ids 1..n
        main = int(sizes.argmax()) + 1
        small = [i for i in range(1, n + 1) if i != main and sizes[i - 1] < min_voxels]
        large = [int(sizes[i - 1]) for i in range(1, n + 1) if i != main and sizes[i - 1] >= min_voxels]
        if small:
            cleaned[np.isin(pieces, small)] = 0
            removed.append({"part": part, "pieces": len(small), "voxels": int(sum(sizes[i - 1] for i in small))})
        if large:
            kept.append({"part": part, "sizes": large})
    return cleaned, removed, kept


def main(data_dir: str = defaults.DATA_DIR, explore_dir: str = EXPLORE_DIR, out: str = OUT_DIR,
         config: str = defaults.CONFIG) -> dict:
    rules = yaml.safe_load(Path(config).read_text())["clean"]
    report = Path(explore_dir) / "cases.csv"
    if not report.exists():
        raise SystemExit(f"{report} not found: run python -m hippo.explore first")
    with open(report) as f:
        explored = {r["case"]: r for r in csv.DictReader(f)}

    data = Path(data_dir)
    out_dir = Path(out)
    labels_dir = out_dir / "labelsTr"
    labels_dir.mkdir(parents=True, exist_ok=True)
    for old in labels_dir.glob("*.nii.gz"):  # outputs of an earlier run; rebuilt below
        old.unlink()

    exclude = set(rules["exclude"])
    index, log = [], []
    for c in list_cases(data / "Task04_Hippocampus"):
        if c["label"] is None:
            continue  # unlabelled cases can't be trained or evaluated on
        name = c["case"]
        info = explored.get(name)
        row = {"case": name, "image": c["image"].relative_to(data).as_posix(),
               "label": c["label"].relative_to(data).as_posix(), "included": True, "reason": "",
               "changes": "", "intensity_group": info.get("intensity_group", "") if info else ""}
        if info is None:
            row.update(included=False, reason="not in exploration report (rerun hippo.explore)")
        elif info["status"] == "FAIL":
            row.update(included=False, reason=f"explore: {info['issues']}")
        elif name in exclude:
            row.update(included=False, reason="listed in clean.exclude")
        else:
            original = nib.load(c["label"])
            y = np.rint(np.asarray(original.dataobj)).astype(np.uint8)
            cleaned, removed, kept = remove_small_pieces(y, rules["min_piece_voxels"])
            voxel_ml = float(np.prod(original.header.get_zooms()[:3])) / 1000
            for r in removed:
                value = next(v for v, p in CLASS_NAMES.items() if p == r["part"])
                log.append({"case": name, **r,
                            "volume_before_ml": round(float((y == value).sum()) * voxel_ml, 3),
                            "volume_after_ml": round(float((cleaned == value).sum()) * voxel_ml, 3)})
            if removed:
                path = labels_dir / c["label"].name
                nib.save(nib.Nifti1Image(cleaned, original.affine, original.header), path)
                row["label"] = path.relative_to(data).as_posix() if path.is_relative_to(data) else str(path)
                row["changes"] = "; ".join(f"removed {r['voxels']} voxels in {r['pieces']} {r['part']} piece(s)"
                                           for r in removed)
            if kept:
                row["changes"] = "; ".join(filter(None, [row["changes"]] + [
                    f"kept extra {k['part']} piece(s) of {k['sizes']} voxels (>= {rules['min_piece_voxels']}): review"
                    for k in kept]))
        index.append(row)

    with open(out_dir / "cases.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(index[0]))
        w.writeheader()
        w.writerows(index)
    with open(out_dir / "cleaning_log.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["case", "part", "pieces", "voxels", "volume_before_ml", "volume_after_ml"])
        w.writeheader()
        w.writerows(log)
    summary = {
        "n_labelled": len(index),
        "included": sum(r["included"] for r in index),
        "excluded": {r["case"]: r["reason"] for r in index if not r["included"]},
        "labels_changed": len({e["case"] for e in log}),
        "voxels_removed": int(sum(e["voxels"] for e in log)),
        "rules": rules,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"{summary['included']} of {summary['n_labelled']} labelled cases included; "
          f"{summary['labels_changed']} labels cleaned ({summary['voxels_removed']} voxels removed)")
    for case, reason in summary["excluded"].items():
        print(f"  excluded {case}: {reason}")
    for r in index:
        if r["changes"]:
            print(f"  {r['case']}: {r['changes']}")
    print(f"Cleaned dataset index: {out_dir / 'cases.csv'}")
    return summary


if __name__ == "__main__":
    p = defaults.make_parser("Clean the labelled dataset using the exploration report.")
    p.add_argument("--config", default=defaults.CONFIG, help="Experiment settings (YAML) with a clean: section")
    p.add_argument("--explore-dir", default=EXPLORE_DIR, help="Folder with the hippo.explore report")
    p.add_argument("--out", default=OUT_DIR, help="Output folder")
    a = p.parse_args()
    main(a.data_dir, a.explore_dir, a.out, a.config)
