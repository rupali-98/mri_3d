"""Where the 3D U-Net over-predicts on external scans: false positives by position and tissue.

A false positive is a voxel the plain prediction calls hippocampus (pred > 0) and the MNI-HISUB25
label does not (label == 0), on the normal atlas box (the main result). Each one is described by:
  - piece: main (the prediction's largest connected piece) or island (a separate small piece);
  - along the length (y), against that hippocampus's own label: behind (behind the label's back
    end), tail / body / head (thirds of the label's length), in front (beyond the label's front end);
  - across (x): inner (towards the midline) or outer, and vertically (z): above or below,
    both from the label's centre in that coronal slice (its overall centre where the slice has no label);
  - distance to the label: rim (within 1 mm of a labelled voxel) or away (> 1 mm);
  - brightness: the z-scored T1 value, compared with the labelled hippocampus of the same scan
    (darker = fluid-like, similar = grey matter, brighter = white matter);
  - predicted part: anterior or posterior.

Outputs: runs/analysis/external/fp_regions.csv (for runs/baseline; fp_regions_<run>.csv for another --run) (one row per hippocampus and group) and a printed summary.

Usage (from the project root):
    PYTHONPATH=src python archive/external_analysis/fp_regions.py
    PYTHONPATH=src python archive/external_analysis/fp_regions.py --run runs/random_trim
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
    return OUT_DIR / ("fp_regions.csv" if run == RUN else f"fp_regions_{Path(run).name}.csv")
LENGTH = ("behind", "tail", "body", "head", "in front")
FIELDS = ["case", "piece", "length", "side", "height", "distance", "tissue", "part", "voxels"]


def describe(label: np.ndarray, pred: np.ndarray, image: np.ndarray, left: bool) -> list[dict]:
    """One dict per false-positive voxel group -> voxel count."""
    fp = (pred > 0) & (label == 0)
    main = external.largest_piece(pred)[0] > 0
    xs, ys, zs = np.nonzero(fp)
    lab = label > 0
    y_lab = np.nonzero(lab.any((0, 2)))[0]
    rel = (ys - y_lab[0]) / (y_lab[-1] - y_lab[0] + 1)
    length = np.where(ys < y_lab[0], 0, np.where(ys > y_lab[-1], 4, 1 + np.minimum((rel * 3).astype(int), 2)))
    all_x, _, all_z = np.nonzero(lab)
    cx, cz = {}, {}
    for y in np.unique(ys):
        sx, sz = np.nonzero(lab[:, y])
        cx[y], cz[y] = (sx.mean(), sz.mean()) if len(sx) else (all_x.mean(), all_z.mean())
    inner = np.array([(x > cx[y]) if left else (x < cx[y]) for x, y in zip(xs, ys)])
    above = np.array([z > cz[y] for z, y in zip(zs, ys)])
    near = distance_transform_edt(~lab)[fp] <= 1.0  # 1 mm voxels
    lo, hi = np.percentile(image[lab], [10, 90])     # the labelled hippocampus's own brightness range
    value = image[fp]
    tissue = np.where(value < lo, "darker", np.where(value > hi, "brighter", "similar"))
    groups = {}
    for m, ln, i, a, n, t, pt in zip(main[fp], length, inner, above, near, tissue, pred[fp]):
        key = ("main" if m else "island", LENGTH[ln], "inner" if i else "outer", "above" if a else "below",
               "rim" if n else "away", t, "anterior" if pt == 1 else "posterior")
        groups[key] = groups.get(key, 0) + 1
    return [dict(zip(FIELDS[1:-1], k), voxels=v) for k, v in groups.items()]


def main(run: str = RUN):
    OUT = out_path(run)
    cases = json.load(open("data/external/cases.json"))
    rows, predicted = [], 0
    with torch.no_grad():
        for case, pred, batch in external.predictions(run, cases):
            label = batch["label"][0, 0].numpy().astype(int)
            p = pred[0, 0].numpy()
            predicted += int((p > 0).sum())
            rows += [dict(case=case["case"], **r) for r in
                     describe(label, p, batch["image"][0, 0].numpy(), case["case"].endswith("_L"))]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    total = sum(r["voxels"] for r in rows)
    print(f"{len(cases)} hippocampi; false positives {total} voxels ({total / 1000:.1f} ml) "
          f"= {total / predicted:.1%} of all predicted voxels\n")
    share = lambda **kw: sum(r["voxels"] for r in rows if all(r[k] == v for k, v in kw.items())) / total
    for field, order in [("piece", ("main", "island")), ("length", LENGTH), ("side", ("inner", "outer")),
                         ("height", ("above", "below")), ("distance", ("rim", "away")),
                         ("tissue", ("darker", "similar", "brighter")), ("part", ("anterior", "posterior"))]:
        print(f"by {field}: " + ", ".join(f"{v} {share(**{field: v}):.1%}" for v in order))
    print("\nlength x height (share of all false positives):")
    for ln in LENGTH:
        print(f"  {ln:9} above {share(length=ln, height='above'):6.1%}   below {share(length=ln, height='below'):6.1%}")
    print("\nislands only: " + ", ".join(f"{h} {share(piece='island', height=h) / share(piece='island'):.1%}"
                                         for h in ("above", "below")) +
          "; " + ", ".join(f"{t} {share(piece='island', tissue=t) / share(piece='island'):.1%}"
                           for t in ("darker", "similar", "brighter")))
    print("in front of the label: " + ", ".join(
        f"{t} {share(length='in front', tissue=t) / max(share(length='in front'), 1e-9):.1%}"
        for t in ("darker", "similar", "brighter")))
    per_case = {}
    for r in rows:
        c = per_case.setdefault(r["case"], [0, 0])
        c[0] += r["voxels"] * (r["height"] == "below")
        c[1] += r["voxels"]
    below = np.array([a / b for a, b in per_case.values()])
    print(f"\nbelow share per hippocampus: median {np.median(below):.1%}, {(below > 0.5).sum()}/{len(below)} more than half below")
    print(f"Saved {OUT}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run", default=RUN, help="Run folder with best.pt + config.yaml")
    main(p.parse_args().run)
