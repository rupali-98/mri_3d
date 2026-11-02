"""Test whether the model depends on where the hippocampus sits inside the crop.

ARCHIVED (2026-10-28): was hippo.shift. The external analysis uses the plain atlas box with
no shift (docs/external_validation.md); kept here for the planned full-image training branch.

The model was trained on MSD's own crops; external crops place the hippocampus
differently. This builds shifted copies of the MSD test cases and scores a run on them:

  - every crop is padded with zeros on all sides (by the largest shift), so moving
    it never pushes the hippocampus off the edge: only its position changes;
  - its content is moved by each shift (default ±5 and ±10 mm) along one axis
    (x = left-right, y = posterior-anterior, z = inferior-superior), the same as
    moving the crop window the other way;
  - the padded but unshifted copy ("none") is the control, so a change from it is
    due to position alone. Each affine keeps the anatomy at its world position,
    so images, labels and predictions still line up in the viewers.

Each copy is scored with hippo.evaluate (the run's own preprocessing).

With --external, the same shifts are applied to the external dataset by moving the
SPAM-atlas box on the whole-brain scans (hippo.external). There the crop keeps real
tissue all round, nothing is padded, and the control is the usual external crop.
A shifted box can cut off part of a hippocampus; min_coverage reports it.
--largest-piece (with --external) keeps only each prediction's largest connected
piece before scoring (hippo.external --largest-piece), in its own output folder.
--axes limits the shifts to some axes, e.g. z. --conditions names exact conditions
instead, and a name joined with "_" moves along several axes at once: x-4_z-10 is
4 mm along -x and 10 mm along -z together. The control "none" is always included.

Outputs:
    data/shifted/<condition>/imagesTr, labelsTr, splits.json   condition: none, x+5, x-5, ..., z-10
    <run>/shifted/<condition>/metrics_test.csv, summary_test.json
    <run>/shifted/summary_shifted.csv    one row per condition: mean Dice and NSD (mean of
                                         anterior/posterior per case), change vs the control,
                                         cases worse, Wilcoxon p, lowest label coverage
  with --external:
    data/shifted_external/<condition>/imagesTr, labelsTr, cases.json
    <run>/shifted_external/<condition>/metrics_external.csv, summary_external.json
    <run>/shifted_external/summary_shifted.csv   as above, whole hippocampus
  with --external --largest-piece: the same in <run>/shifted_external_largest/
  condition names: none, x+5, ..., z-10, or combined such as x-4_z-10

Usage (from the project root):
    PYTHONPATH=src python archive/shift.py --run runs/baseline
    PYTHONPATH=src python archive/shift.py --run runs/baseline --shifts 5 10
    PYTHONPATH=src python archive/shift.py --external data/mri_dataset --run runs/baseline
    PYTHONPATH=src python archive/shift.py --external data/mri_dataset --run runs/baseline --axes z --largest-piece
    PYTHONPATH=src python archive/shift.py --external data/mri_dataset --run runs/baseline --conditions z-5 z-10 x-4_z-10
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.stats import wilcoxon

from hippo import defaults, evaluate, external

OUT_DIR = "data/shifted"
EXTERNAL_OUT = "data/shifted_external"
AXES = "xyz"
CONTROL = "none"


def parse(name: str) -> list[tuple[int, float]]:
    """The moves (axis, mm) of a condition name: "none" -> [], "z-10" -> [(2, -10)], "x-4_z-10" -> both."""
    if name == CONTROL:
        return []
    moves = []
    for part in name.split("_"):
        if len(part) < 3 or part[0] not in AXES or part[1] not in "+-":
            raise ValueError(f"not a shift condition: {name!r} (expected e.g. z-10 or x-4_z-10)")
        moves.append((AXES.index(part[0]), float(part[1:])))
    if len({a for a, _ in moves}) < len(moves):
        raise ValueError(f"{name!r} moves along the same axis twice")
    return moves


def conditions(shifts_mm: list[float], axes: str = AXES, names: list[str] | None = None) -> list[tuple[str, list]]:
    """(name, moves): the control first, then the named conditions, or each of `axes` and shift, + and -."""
    if names is None:
        names = [f"{a}{sign}{s:g}" for a in AXES if a in axes for s in shifts_mm for sign in "+-"]
    return [(CONTROL, [])] + [(n, parse(n)) for n in names if n != CONTROL]


def shift_volume(img: nib.Nifti1Image, axis: int | None, shift_vox: int, pad: int,
                 more: list[tuple[int, int]] = ()) -> nib.Nifti1Image:
    """`img` (RAS) padded with `pad` zero voxels per side and its content moved `shift_vox` voxels along `axis`
    (and along any further (axis, voxels) in `more`)."""
    data = np.asarray(img.dataobj)
    offset = np.full(3, pad)
    for a, v in ([(axis, shift_vox)] if axis is not None else []) + list(more):
        offset[a] += v
    canvas = np.zeros(tuple(n + 2 * pad for n in data.shape[:3]), data.dtype)
    canvas[tuple(slice(o, o + n) for o, n in zip(offset, data.shape[:3]))] = data
    # canvas voxel (offset + i) shows original voxel i, so it gets that voxel's world position
    affine = img.affine @ nib.affines.from_matvec(np.eye(3), -offset)
    return nib.Nifti1Image(canvas, affine)  # no header: the old one's qform would contradict the new affine


def build(data_dir: str, out: str = OUT_DIR, shifts_mm: list[float] = (5, 10), axes: str = AXES,
          names: list[str] | None = None) -> dict[str, Path]:
    """Write a shifted copy of every MSD test case for each condition; returns {condition: folder}."""
    cases = json.loads((Path(data_dir) / defaults.SPLIT_FILENAME).read_text())["test"]
    folders = {}
    todo = conditions(list(shifts_mm), axes, names)
    largest = max([abs(mm) for _, moves in todo for _, mm in moves] or [0])
    for name, moves in todo:
        folder = Path(out) / name
        written = []
        for c in cases:
            image, label = (nib.as_closest_canonical(nib.load(c[k])) for k in ("image", "label"))
            spacing = image.header.get_zooms()[:3]
            pad = int(np.ceil(largest / min(spacing)))
            vox = [(a, int(round(mm / spacing[a]))) for a, mm in moves]
            paths = {}
            for kind, img in (("image", image), ("label", label)):
                path = folder / ("imagesTr" if kind == "image" else "labelsTr") / Path(c[kind]).name
                path.parent.mkdir(parents=True, exist_ok=True)
                nib.save(shift_volume(img, None, 0, pad, vox), path)
                paths[kind] = str(path)
            written.append(paths)
        (folder / defaults.SPLIT_FILENAME).write_text(json.dumps({"test": written}, indent=2))
        folders[name] = folder
    print(f"Wrote {len(cases)} test cases x {len(folders)} conditions to {out}")
    return folders


def case_scores(path: Path) -> dict[str, tuple[float, float, float]]:
    """{case: (Dice, NSD, label coverage)} from an MSD (mean of anterior and posterior) or external
    (whole hippocampus) metrics CSV; unscored (blank) cases left out. Coverage is 1 for MSD."""
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if rows and "dice" in rows[0]:  # hippo.external
        return {r["case"]: (float(r["dice"]), float(r["nsd"]), float(r["label_coverage"])) for r in rows}
    mean = lambda r, m: (float(r[f"{m}_anterior"]) + float(r[f"{m}_posterior"])) / 2
    return {r["case"]: (mean(r, "dice"), mean(r, "nsd"), 1.0) for r in rows if r["dice_anterior"] != ""}


def summarise(folder: Path, names: list[str], metrics_file: str) -> list[dict]:
    """One row per condition (folder/<condition>/<metrics_file>), compared case by case with the control."""
    scores = {n: case_scores(folder / n / metrics_file) for n in names}
    control = scores[CONTROL]
    rows = []
    for n in names:
        cases = sorted(set(control) & set(scores[n]))
        a = np.array([control[c][0] for c in cases])
        b = np.array([scores[n][c][0] for c in cases])
        diff = b - a
        rows.append({"condition": n, "n": len(cases), "dice": round(float(b.mean()), 4),
                     "nsd": round(float(np.mean([scores[n][c][1] for c in cases])), 4),
                     "dice_change": round(float(diff.mean()), 4), "worse": int((diff < 0).sum()),
                     "worst_change": round(float(diff.min()), 4),
                     "p_wilcoxon": 1.0 if np.allclose(diff, 0) else round(float(wilcoxon(b, a).pvalue), 4),
                     "min_coverage": round(min(scores[n][c][2] for c in cases), 4)})
    with open(folder / "summary_shifted.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return rows


def report(run: Path, folder: Path, rows: list[dict], control: str) -> None:
    print(f"\n{run.name}: Dice by crop shift (control = {control})")
    print(f"{'condition':<10}{'n':>4}{'Dice':>8}{'change':>9}{'worse':>7}{'worst':>8}{'p':>8}{'NSD':>8}{'cover':>7}")
    for r in rows:
        print(f"{r['condition']:<10}{r['n']:>4}{r['dice']:>8.4f}{r['dice_change']:>+9.4f}{r['worse']:>7}"
              f"{r['worst_change']:>+8.3f}{r['p_wilcoxon']:>8.3f}{r['nsd']:>8.4f}{r['min_coverage']:>7.3f}")
    print(f"Saved {folder / 'summary_shifted.csv'}")


def main(runs: list[str], data_dir: str = defaults.DATA_DIR, out: str = OUT_DIR,
         shifts_mm: list[float] = (5, 10), axes: str = AXES, names: list[str] | None = None) -> dict[str, list[dict]]:
    folders = build(data_dir, out, shifts_mm, axes, names)
    results = {}
    for run in map(Path, runs):
        for name, folder in folders.items():
            evaluate.main(str(run), "test", data_dir=str(folder), out_dir=str(run / "shifted" / name))
        results[run.name] = rows = summarise(run / "shifted", list(folders), "metrics_test.csv")
        report(run, run / "shifted", rows, "padded, not shifted")
    return results


def main_external(runs: list[str], src: str, out: str = EXTERNAL_OUT, shifts_mm: list[float] = (5, 10),
                  subjects: list[str] | None = None, axes: str = AXES,
                  keep_largest: bool = False, names: list[str] | None = None) -> dict[str, list[dict]]:
    """The same test on the external dataset: the atlas box itself is moved, so the crop keeps real
    tissue all round (no padding) and the control is the usual hippo.external crop."""
    cases = {name: external.prepare(src, str(Path(out) / name), subjects, shift=moves)
             for name, moves in conditions(list(shifts_mm), axes, names)}
    folder = "shifted_external_largest" if keep_largest else "shifted_external"
    results = {}
    for run in map(Path, runs):
        for name, crops in cases.items():
            external.evaluate(str(run), crops, out_dir=str(run / folder / name), keep_largest=keep_largest)
        results[run.name] = rows = summarise(run / folder, list(cases), "metrics_external.csv")
        report(run, run / folder, rows, "the usual atlas-centred crop" + (", largest piece" if keep_largest else ""))
    return results


if __name__ == "__main__":
    p = defaults.make_parser("Score runs on MSD test crops shifted by a few mm, to test sensitivity to position.")
    p.add_argument("--run", nargs="+", default=[defaults.RUN_DIR], help="Run folder(s) with best.pt + config.yaml")
    p.add_argument("--shifts", nargs="+", type=float, default=[5, 10], help="Shifts in mm, applied + and -")
    p.add_argument("--out", help=f"Folder for the shifted cases (default {OUT_DIR}, or {EXTERNAL_OUT} with --external)")
    p.add_argument("--external", metavar="SRC", help="Shift the external dataset's atlas box instead (folder of s01/, ...)")
    p.add_argument("--subjects", nargs="+", help="With --external: only these subjects, e.g. s03 (default: all)")
    p.add_argument("--axes", default=AXES, help="Shift along these axes only, e.g. z (default: xyz)")
    p.add_argument("--conditions", nargs="+", help="Exact conditions instead of --axes/--shifts, e.g. z-5 z-10 "
                                                    "x-4_z-10 (\"_\" = together); the control is always added")
    p.add_argument("--largest-piece", action="store_true",
                   help="With --external: keep only each prediction's largest piece; -> <run>/shifted_external_largest/")
    a = p.parse_args()
    if a.largest_piece and not a.external:
        p.error("--largest-piece needs --external")
    if a.external:
        main_external(a.run, a.external, a.out or EXTERNAL_OUT, a.shifts, a.subjects, a.axes, a.largest_piece,
                      a.conditions)
    else:
        main(a.run, a.data_dir, a.out or OUT_DIR, a.shifts, a.axes, a.conditions)
