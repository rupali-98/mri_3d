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

With --largest-piece, each prediction keeps only its largest connected piece
(26-neighbour, whole hippocampus) before scoring: post-processing against the
small islands the model predicts around external hippocampi. It is opt-in and
written to its own folder, so the plain results stay comparable.

With --recentre, the box is placed by the model itself (test-time re-centring, no
labels used): a first prediction on the atlas box keeps its largest piece, the box
is moved to centre on that piece's centroid (same size), and the model predicts
again; only this second prediction is scored. A case whose first prediction is
empty keeps the atlas box. Note: this centres on the centre of mass, while MSD crops
are centred on the hippocampus's bounding box (2-3 mm higher), so see --msd-crop.

With --msd-crop, the external data is cropped the way the MSD crops were made: the
first prediction's bounding box (largest piece) plus the median margin of the run's
MSD training crops on every side (about 5-6 mm), so the box size varies per case as
in MSD. The margins come from MSD only; no external labels place the box.

Outputs in <run>/external/ (<run>/external_largest/ with --largest-piece;
<run>/external_recentred[_largest]/ with --recentre; <run>/external_msd_crop[_largest]/ with --msd-crop):
    metrics_external.csv   one row per hemisphere: Dice, NSD, HD95 (whole hippocampus), volumes
                           (--largest-piece adds pieces = pieces before, removed_ml = volume removed;
                           --recentre and --msd-crop add recentred and moved_mm, moved_x/y/z_mm = how far the
                           box centre moved from the atlas box; --msd-crop adds crop_x/y/z_mm = box size)
    summary_external.json  mean, median, std, volume agreement (absolute and relative error, ICC,
                           Bland-Altman bias and limits) and how the boxes were placed; also added to the
                           run's MLflow run as <folder>.<metric>, tagged with the post-processing and box
  crops in data/external_recentred/{first_pass,recentred}/ or data/external_msd_crop/{first_pass,msd_crop}/

Usage:
    python -m hippo.external --src data/mri_dataset --run runs/baseline
    python -m hippo.external --src data/mri_dataset --run runs/baseline --subjects s03
    python -m hippo.external --src data/mri_dataset --run runs/baseline --largest-piece
    python -m hippo.external --src data/mri_dataset --run runs/baseline --recentre --largest-piece
    python -m hippo.external --src data/mri_dataset --run runs/baseline --msd-crop --largest-piece
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
from scipy import ndimage

from . import defaults, ensemble, tracking
from .data import eval_transforms
from .metrics import summarise, whole_metrics
from .train import spacing_from_affine

OUT_DIR = "data/external"
RECENTRED_OUT = "data/external_recentred"
MSD_CROP_OUT = "data/external_msd_crop"
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


def world_box(img: nib.Nifti1Image, lo_mm, hi_mm) -> tuple[slice, ...]:
    """The voxel box whose voxel centres span world positions lo_mm..hi_mm (clipped to the volume)."""
    a, b = (nib.affines.apply_affine(np.linalg.inv(img.affine), p) for p in (lo_mm, hi_mm))
    start, stop = np.rint(np.minimum(a, b)).astype(int), np.rint(np.maximum(a, b)).astype(int) + 1
    return tuple(slice(max(int(s), 0), min(int(e), n)) for s, e, n in zip(start, stop, img.shape[:3]))


def prepare(src: str, out: str = OUT_DIR, subjects: list[str] | None = None, image: str = "t1w_standard",
            shift: list[tuple[int, float]] = (), centres: dict[str, np.ndarray] | None = None,
            bounds: dict[str, tuple] | None = None) -> list[dict]:
    """Crop every subject's left and right hippocampus; returns the case list (image, label, coverage,
    centre_mm = world position of the box centre).

    shift: (world axis 0/1/2 = x/y/z, mm) pairs move the box so the hippocampus sits that far towards
    +x (right), +y (anterior) or +z (superior) inside the crop (archive/shift.py --external); several
    pairs move it along several axes at once.
    centres: {case: world position (mm)} centres those cases' boxes there instead of on the atlas
    (test-time re-centring); other cases keep the atlas box.
    bounds: {case: (lowest, highest) world position (mm)} crops those cases to exactly that range
    instead (MSD-style crop; the box size then varies per case).
    """
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
        centre = centre.copy()
        for axis, mm in shift:
            # the box moves the opposite way to the hippocampus; find the voxel axis and sign of that world axis
            ornt = nib.io_orientation(ref.affine)
            v = int(np.flatnonzero(ornt[:, 0] == axis)[0])
            centre[v] -= ornt[v, 1] * mm / float(ref.header.get_zooms()[v])
        for s in names:
            img = nib.load(src / s / f"{s}_{image}_defaced_MNI.nii.gz")
            lbl = nib.load(src / s / f"{s}_hippolabels_{image}_{side}_MNI.nii.gz")
            if img.shape != ref.shape or not np.allclose(img.affine, ref.affine, atol=1e-3):
                raise SystemExit(f"{s}: {image} is not on the SPAM atlas grid; crops would be misplaced")
            name = f"{s}_{side}"
            here = centre
            if centres and name in centres:
                here = nib.affines.apply_affine(np.linalg.inv(img.affine), centres[name])
            box = world_box(img, *bounds[name]) if bounds and name in bounds else crop_box(here, img.shape, CROP_MM)
            y = np.rint(np.asarray(lbl.dataobj)).astype(np.uint8)
            save_crop(img, box, np.asarray(img.dataobj, dtype=np.float32)[box], out / "imagesTr" / f"{name}.nii.gz")
            save_crop(lbl, box, y[box], out / "labelsTr" / f"{name}.nii.gz")
            # share of the hippocampus inside the crop; below 1 means the box cut part of it off
            coverage = float((y[box] > 0).sum() / max((y > 0).sum(), 1))
            box_centre = nib.affines.apply_affine(img.affine, [(b.start + b.stop - 1) / 2 for b in box])
            cases.append({"case": name, "image": str(out / "imagesTr" / f"{name}.nii.gz"),
                          "label": str(out / "labelsTr" / f"{name}.nii.gz"), "coverage": round(coverage, 4),
                          "centre_mm": [round(float(c), 2) for c in box_centre]})
    cases.sort(key=lambda c: c["case"])
    (out / "cases.json").write_text(json.dumps(cases, indent=2))
    print(f"Cropped {len(cases)} hippocampi into {out}; lowest label coverage "
          f"{min(c['coverage'] for c in cases):.3f} ({min(cases, key=lambda c: c['coverage'])['case']})")
    return cases


def largest_piece(pred: np.ndarray) -> tuple[np.ndarray, int]:
    """The prediction with only its largest connected piece (26-neighbour, any label > 0) kept, and the
    number of pieces it had; labels inside the kept piece are unchanged."""
    pieces, n = ndimage.label(pred > 0, structure=np.ones((3, 3, 3)))
    if n <= 1:
        return pred, n
    keep = np.bincount(pieces.ravel())[1:].argmax() + 1
    return np.where(pieces == keep, pred, 0), n


@torch.no_grad()
def predictions(run_dir: str, cases: list[dict]):
    """(case, prediction (1, 1, X, Y, Z), batch) for each crop, with the run's own model and preprocessing."""
    run = Path(run_dir)
    cfg = yaml.safe_load((run / "config.yaml").read_text())
    device = defaults.pick_device()
    model = ensemble.load(run, device)  # one model, or the folds' averaged probabilities (hippo.ensemble)
    transforms = eval_transforms(cfg["data"].get("normalize", "zscore"))  # the run's own preprocessing
    loader = DataLoader(Dataset([{"image": c["image"], "label": c["label"]} for c in cases], transforms), batch_size=1)
    for case, batch in zip(cases, loader):
        yield case, model(batch["image"].to(device)).argmax(dim=1, keepdim=True).cpu(), batch


def first_pass(run_dir: str, cases: list[dict]) -> dict[str, dict]:
    """{case: {centre, lo, hi}}: world centroid and lowest/highest voxel position (mm) of the largest piece of
    the run's prediction; cases predicted empty are left out."""
    found = {}
    for case, pred, batch in predictions(run_dir, cases):
        kept, _ = largest_piece(pred[0, 0].numpy())
        if (kept > 0).any():
            affine = np.asarray(batch["image"].affine)
            affine = affine[0] if affine.ndim == 3 else affine  # batched: one affine per case
            world = nib.affines.apply_affine(affine, np.argwhere(kept > 0))
            found[case["case"]] = {"centre": world.mean(0), "lo": world.min(0), "hi": world.max(0)}
    return found


def msd_margins(split_file: str) -> np.ndarray:
    """Median margin (mm) between the hippocampus and the crop edge in the MSD training crops, per axis.

    MSD crops are the hippocampus's bounding box plus about this much on every side (the margins on
    opposite sides match within ~1 mm), so the average of the two sides is used on both.
    """
    margins = []
    for c in json.loads(Path(split_file).read_text())["train"]:
        img = nib.as_closest_canonical(nib.load(c["label"]))
        idx = np.argwhere(np.asarray(img.dataobj) > 0)
        low, high = idx.min(0), np.array(img.shape[:3]) - 1 - idx.max(0)
        margins.append((low + high) / 2 * np.array(img.header.get_zooms()[:3]))
    return np.median(margins, axis=0)


def box_columns(first: list[dict], second: list[dict], placed: set) -> dict[str, dict]:
    """Per case: whether the model placed the box, and how far its centre moved from the atlas box (mm)."""
    before = {c["case"]: np.array(c["centre_mm"]) for c in first}
    extra = {}
    for c in second:
        moved = np.array(c["centre_mm"]) - before[c["case"]]
        extra[c["case"]] = {"recentred": c["case"] in placed, "moved_mm": round(float(np.linalg.norm(moved)), 2),
                            **{f"moved_{a}_mm": round(float(m), 2) for a, m in zip("xyz", moved)}}
    print(f"Model placed {len(placed)} of {len(second)} boxes; median move "
          f"{np.median([e['moved_mm'] for e in extra.values()]):.1f} mm")
    return extra


def recentre(run_dir: str, src: str, out: str = RECENTRED_OUT, subjects: list[str] | None = None,
             keep_largest: bool = False) -> dict:
    """Test-time re-centring: predict on the atlas box, centre a same-sized box on the prediction's largest
    piece, predict again and score that; writes <run>/external_recentred[_largest]/."""
    first = prepare(src, str(Path(out) / "first_pass"), subjects)
    found = first_pass(run_dir, first)
    second = prepare(src, str(Path(out) / "recentred"), subjects, centres={k: v["centre"] for k, v in found.items()})
    folder = "external_recentred" + ("_largest" if keep_largest else "")
    return evaluate(run_dir, second, out_dir=str(Path(run_dir) / folder), keep_largest=keep_largest,
                    extra=box_columns(first, second, set(found)), box="re-centred on the first prediction")


def msd_crop(run_dir: str, src: str, out: str = MSD_CROP_OUT, subjects: list[str] | None = None,
             keep_largest: bool = False) -> dict:
    """Crop like MSD: predict on the atlas box, then crop to the prediction's bounding box (largest piece) plus
    the run's MSD training margins on every side, predict again and score that; writes
    <run>/external_msd_crop[_largest]/. No external labels are used to place the box."""
    cfg = yaml.safe_load((Path(run_dir) / "config.yaml").read_text())
    margin = msd_margins(cfg["data"]["split_file"])
    print(f"MSD margins (median of the run's training crops): x {margin[0]:.1f}, y {margin[1]:.1f}, "
          f"z {margin[2]:.1f} mm")
    first = prepare(src, str(Path(out) / "first_pass"), subjects)
    found = first_pass(run_dir, first)
    second = prepare(src, str(Path(out) / "msd_crop"), subjects,
                     bounds={k: (v["lo"] - margin, v["hi"] + margin) for k, v in found.items()})
    extra = box_columns(first, second, set(found))
    for c in second:  # the box size now follows the hippocampus, as in MSD
        shape = np.array(nib.load(c["image"]).shape[:3]) * np.array(nib.load(c["image"]).header.get_zooms()[:3])
        extra[c["case"]].update({f"crop_{a}_mm": round(float(s), 1) for a, s in zip("xyz", shape)})
    folder = "external_msd_crop" + ("_largest" if keep_largest else "")
    return evaluate(run_dir, second, out_dir=str(Path(run_dir) / folder), keep_largest=keep_largest, extra=extra,
                    box=f"MSD-style: first prediction's bounding box + MSD margins {np.round(margin, 1).tolist()} mm")


def evaluate(run_dir: str, cases: list[dict], out_dir: str | None = None, keep_largest: bool = False,
             extra: dict[str, dict] | None = None, box: str = "atlas") -> dict:
    """Score the run on the crops; out_dir defaults to <run>/external, or <run>/external_largest with
    keep_largest (archive/shift.py, recentre and msd_crop write elsewhere). `extra` adds columns per case;
    `box` says how the crops were placed (in the summary)."""
    run = Path(run_dir)
    cfg = yaml.safe_load((run / "config.yaml").read_text())
    rows = []
    for case, pred, batch in predictions(run_dir, cases):
        spacing = spacing_from_affine(batch["image"].affine)
        more = dict((extra or {}).get(case["case"], {}))
        if keep_largest:
            kept, n = largest_piece(pred[0, 0].numpy())
            removed = int((pred > 0).sum()) - int((kept > 0).sum())
            pred = torch.from_numpy(kept)[None, None]
            more.update(pieces=n, removed_ml=removed * float(np.prod(spacing)) / 1000)
        m = whole_metrics(pred, batch["label"], spacing, cfg["eval"]["nsd_tolerance_mm"])
        rows.append({"case": case["case"], **m, "label_coverage": case["coverage"], **more})

    out = Path(out_dir) if out_dir else run / ("external_largest" if keep_largest else "external")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "metrics_external.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary = {"n_cases": len(rows), "metric": "whole hippocampus (prediction > 0 vs any subfield label)",
               "post_processing": "largest piece only" if keep_largest else "none",
               "box": box}
    for k in ("dice", "nsd", "hd95"):
        vals = np.array([r[k] for r in rows])  # HD95 is nan where a prediction is empty
        summary[k] = {"mean": float(np.nanmean(vals)), "median": float(np.nanmedian(vals)),
                      "std": float(np.nanstd(vals))}
    summary["volume"] = {k: v for k, v in summarise(rows).items() if k not in ("dice", "nsd", "hd95")}
    (out / "summary_external.json").write_text(json.dumps(summary, indent=2))
    tracking.log_results(run, tracking.result_name(run, out), {**summarise(rows)},
                         {"largest_piece": keep_largest, "box": box, "labels": "whole hippocampus"})
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    p = defaults.make_parser("Evaluate a trained model on external whole-brain MNI data.")
    p.add_argument("--src", required=True, help="External dataset folder (s01/, s02/, ..., SPAM_*_MNI.nii.gz)")
    p.add_argument("--run", default=defaults.RUN_DIR, help="Run folder with best.pt + config.yaml, or an ensemble folder (hippo.ensemble)")
    p.add_argument("--subjects", nargs="+", help="Only these subjects, e.g. s03 (default: all)")
    p.add_argument("--out", default=OUT_DIR, help="Folder for the cropped cases")
    p.add_argument("--largest-piece", action="store_true",
                   help="Keep only each prediction's largest connected piece; results go to <run>/external_largest")
    p.add_argument("--recentre", action="store_true",
                   help="Test-time re-centring: predict, centre the box on the prediction, predict again; "
                        f"crops in {RECENTRED_OUT}, results in <run>/external_recentred[_largest]")
    p.add_argument("--msd-crop", action="store_true",
                   help="Crop like MSD: the first prediction's bounding box + MSD training margins, predict again; "
                        f"crops in {MSD_CROP_OUT}, results in <run>/external_msd_crop[_largest]")
    a = p.parse_args()
    if a.recentre and a.msd_crop:
        p.error("choose --recentre or --msd-crop, not both")
    if a.recentre:
        recentre(a.run, a.src, RECENTRED_OUT if a.out == OUT_DIR else a.out, a.subjects, a.largest_piece)
    elif a.msd_crop:
        msd_crop(a.run, a.src, MSD_CROP_OUT if a.out == OUT_DIR else a.out, a.subjects, a.largest_piece)
    else:
        evaluate(a.run, prepare(a.src, a.out, a.subjects), keep_largest=a.largest_piece)
