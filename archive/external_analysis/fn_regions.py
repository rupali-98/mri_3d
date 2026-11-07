"""Where the 3D U-Net misses external hippocampus: false negatives by subfield and position.

A false negative is a voxel the MNI-HISUB25 label calls hippocampus (label > 0) and the
plain prediction does not (pred == 0), on the normal atlas box (the main result). Each one is
described by:
  - subfield: the label value (1 = CA1-3, 2 = subiculum, 3 = CA4-DG; names inferred from the atlas);
  - along the length (y, back -> front), on that hippocampus's own label: tail = back third,
    body = middle third, head = front third;
  - across (x): inner (towards the midline) or outer half, from the label's centre in that coronal
    slice (RAS: inner = larger x on the left, smaller x on the right);
  - distance to the prediction: rim (within 1 mm of a predicted voxel, an outline disagreement)
    or away (> 1 mm, a missed chunk).

Outputs: runs/analysis/external/fn_regions.csv (for runs/baseline; fn_regions_<run>.csv for another --run) (one row per hippocampus and subfield/position
group) and a printed summary.

Usage (from the project root):
    PYTHONPATH=src python archive/external_analysis/fn_regions.py
    PYTHONPATH=src python archive/external_analysis/fn_regions.py --run runs/random_trim
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

sys.path.insert(0, "src")
from hippo import external  # noqa: E402

RUN = "runs/baseline"
OUT_DIR = Path("runs/analysis/external")


def out_path(run: str) -> Path:
    """The baseline keeps the original file name, so earlier results are still found."""
    return OUT_DIR / ("fn_regions.csv" if run == RUN else f"fn_regions_{Path(run).name}.csv")
SUBFIELDS = {1: "CA1-3", 2: "subiculum", 3: "CA4-DG"}
LENGTH = ("tail", "body", "head")


def describe(label: np.ndarray, pred: np.ndarray, left: bool) -> list[dict]:
    """One dict per false-negative voxel group: subfield, length third, side, rim/away -> voxel count."""
    fn = (label > 0) & (pred == 0)
    xs, ys, zs = np.nonzero(fn)
    y_all = np.nonzero((label > 0).any((0, 2)))[0]
    third = np.minimum(((ys - y_all[0]) / (y_all[-1] - y_all[0] + 1) * 3).astype(int), 2)
    # the label's x-centre in each coronal slice (y), so "inner" follows the curve of the hippocampus
    centre = {y: np.nonzero(label[:, y] > 0)[0].mean() for y in np.unique(ys)}
    inner = np.array([(x > centre[y]) if left else (x < centre[y]) for x, y in zip(xs, ys)])
    near = distance_transform_edt(pred == 0)[fn] <= 1.0  # 1 mm voxels
    groups = {}
    for sub, t, i, n in zip(label[fn], third, inner, near):
        key = (SUBFIELDS[int(sub)], LENGTH[t], "inner" if i else "outer", "rim" if n else "away")
        groups[key] = groups.get(key, 0) + 1
    return [dict(subfield=k[0], length=k[1], side=k[2], distance=k[3], voxels=v) for k, v in groups.items()]


def main(run: str = RUN):
    OUT = out_path(run)
    cases = json.load(open("data/external/cases.json"))
    rows, totals = [], {}
    with torch.no_grad():
        for case, pred, batch in external.predictions(run, cases):
            label = batch["label"][0, 0].numpy().astype(int)
            p = pred[0, 0].numpy()
            totals[case["case"]] = {s: int((label == v).sum()) for v, s in SUBFIELDS.items()}
            rows += [dict(case=case["case"], **r) for r in describe(label, p, case["case"].endswith("_L"))]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["case", "subfield", "length", "side", "distance", "voxels"])
        w.writeheader()
        w.writerows(rows)

    fn_total = sum(r["voxels"] for r in rows)
    label_total = sum(sum(t.values()) for t in totals.values())
    print(f"{len(totals)} hippocampi; false negatives {fn_total} voxels ({fn_total / 1000:.1f} ml) "
          f"= {fn_total / label_total:.1%} of all labelled voxels\n")

    def table(by: str, order):
        print(f"by {by}:")
        for v in order:
            n = sum(r["voxels"] for r in rows if r[by] == v)
            extra = ""
            if by == "subfield":
                size = sum(t[v] for t in totals.values())
                extra = f"   ({n / size:.1%} of that subfield missed)"
            print(f"  {v:10} {n:7d}  {n / fn_total:6.1%}{extra}")
        print()

    table("subfield", SUBFIELDS.values())
    table("length", LENGTH)
    table("side", ("inner", "outer"))
    table("distance", ("rim", "away"))
    print("subfield x length x side (share of all false negatives):")
    print(f"  {'':10}" + "".join(f"{t + ' ' + s:>13}" for t in LENGTH for s in ("inner", "outer")))
    for sub in SUBFIELDS.values():
        cells = [sum(r["voxels"] for r in rows if (r["subfield"], r["length"], r["side"]) == (sub, t, s))
                 for t in LENGTH for s in ("inner", "outer")]
        print(f"  {sub:10}" + "".join(f"{c / fn_total:13.1%}" for c in cells))
    # how consistent is it: share of each hippocampus's false negatives that are inner
    per_case = {}
    for r in rows:
        c = per_case.setdefault(r["case"], [0, 0])
        c[0] += r["voxels"] * (r["side"] == "inner")
        c[1] += r["voxels"]
    shares = np.array([a / b for a, b in per_case.values()])
    print(f"\ninner share per hippocampus: median {np.median(shares):.1%}, range {shares.min():.1%}-{shares.max():.1%}, "
          f"{(shares > 0.5).sum()}/{len(shares)} with more than half inner")
    print(f"Saved {OUT}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run", default=RUN, help="Run folder with best.pt + config.yaml")
    main(p.parse_args().run)
