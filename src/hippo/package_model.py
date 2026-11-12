"""Package a trained experiment for serving: weights, settings and the facts the API needs, in one folder.

The API (hippo.api) must not depend on runs/ or on the training data, so this copies what it needs:
  - each fold's best.pt and config.yaml (fold0 ... fold4; the API averages their probabilities), or the
    single run's when the experiment has no folds or --single is given (runs/<name>/best.pt itself);
  - metadata.json: model name and version (a hash of the weights, so every response names the exact
    model), the MLflow run id of each model (to trace it back to its training run and scores), preprocessing, labels, the volume and edge-margin range of the training labels (for quality
    flags), the cross-validation results if hippo.results has written them, the intended use, and the
    two atlas boxes (left and right) as MNI world coordinates, so whole-brain MNI scans are cropped
    exactly as hippo.external crops them without shipping the atlas.

Outputs (default deploy/model/, git-ignored):
    fold<k>/best.pt, fold<k>/config.yaml   (or best.pt, config.yaml for a single run)
    metadata.json

Usage:
    python -m hippo.package_model --experiment runs/baseline --src data/mri_dataset --single   # the single run
    python -m hippo.package_model --experiment runs/baseline --src data/mri_dataset            # 5-fold average
    python -m hippo.package_model --experiment runs/baseline --src data/mri_dataset --out deploy/model --name "3D U-Net"
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import nibabel as nib
import yaml

from . import CLASS_NAMES, external, tracking
from .predict import training_range

OUT = "deploy/model"
INTENDED_USE = ("Research and education only. Not a medical device, not clinically validated, and not for "
                "diagnosis or any clinical decision.")


def model_runs(experiment, single: bool = False) -> list[Path]:
    """The fold runs of an experiment, or the experiment folder itself when it is a single run (or single=True)."""
    folds = [] if single else sorted(p.parent for p in Path(experiment).glob("fold*/best.pt"))
    if folds:
        return folds
    if (Path(experiment) / "best.pt").exists():
        return [Path(experiment)]
    raise SystemExit(f"No trained model in {experiment} (best.pt or fold*/best.pt)")


def mni_boxes(src) -> dict:
    """Left and right atlas boxes as world coordinates (mm) of their first and last voxel centres.

    The same box hippo.external.prepare cuts (CROP_MM around the SPAM atlas centre), so a scan on the MNI
    grid is cropped to exactly the same voxels; on another MNI grid, to the nearest voxels.
    """
    boxes = {}
    for side in ("L", "R"):
        centre, ref = external.atlas_centre(Path(src), side)
        box = external.crop_box(centre, ref.shape, external.CROP_MM)
        first = nib.affines.apply_affine(ref.affine, [b.start for b in box])
        last = nib.affines.apply_affine(ref.affine, [b.stop - 1 for b in box])
        boxes[side] = {"lo_mm": [round(float(v), 3) for v in first], "hi_mm": [round(float(v), 3) for v in last],
                       "size_vox": list(external.CROP_MM)}
    return boxes


def weights_hash(runs: list[Path]) -> str:
    digest = hashlib.sha256()
    for run in runs:
        digest.update((run / "best.pt").read_bytes())
    return digest.hexdigest()[:12]


def cv_results(experiment) -> list[dict]:
    """The experiment's cross-validation table (hippo.results), if it has been written."""
    path = Path(experiment) / "results" / "results.csv"
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return [{k: (round(float(v), 4) if k.endswith(("_mean", "_sd")) and v not in ("", "nan") else v)
                 for k, v in row.items()} for row in csv.DictReader(f)]


def main(experiment: str, src: str | None = None, out: str = OUT, name: str = "3D U-Net",
         single: bool = False) -> dict:
    runs = model_runs(experiment, single)
    configs = [yaml.safe_load((run / "config.yaml").read_text()) for run in runs]
    normalize = {c["data"].get("normalize", "zscore") for c in configs}
    if len(normalize) != 1:
        raise SystemExit(f"The runs use different preprocessing ({sorted(normalize)}); they can't be averaged")
    out = Path(out)
    if out.exists():
        shutil.rmtree(out)
    files = []
    for run, cfg in zip(runs, configs):
        target = out / run.name if run.name.startswith("fold") else out
        target.mkdir(parents=True, exist_ok=True)
        shutil.copy(run / "best.pt", target / "best.pt")
        cfg = {**cfg, "data": {k: v for k, v in cfg["data"].items() if k != "split_file"}}  # no local paths
        (target / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        files.append(str(target.relative_to(out)))
    # every case the fold models trained or validated on (the test split is not included)
    split = json.loads(Path(configs[0]["data"]["split_file"]).read_text())
    seen = {c["image"]: c for c in split["train"] + split["val"]}
    metadata = {
        "name": name, "version": weights_hash(runs), "experiment": Path(experiment).name,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "members": files, "mlflow_run_ids": [tracking.saved_run_id(run) for run in runs], "combination": "mean of class probabilities" if len(runs) > 1 else "single model",
        "normalize": normalize.pop(), "labels": {str(k): v for k, v in {0: "background", **CLASS_NAMES}.items()},
        "input": {"voxel_mm": 1.0, "crop_voxels_typical": [35, 50, 35], "contrast": "T1-weighted"},
        "training_label_range": training_range(list(seen.values())),
        "cross_validation": cv_results(experiment) if len(runs) > 1 else [],  # the folds' results, not this run's
        "intended_use": INTENDED_USE,
    }
    if src:
        metadata["mni_boxes"] = mni_boxes(src)
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Packaged {name} ({len(runs)} model{'s' if len(runs) > 1 else ''}, version {metadata['version']}) "
          f"into {out}" + ("" if src else "; no --src, so the whole-brain MNI endpoint will be off"))
    return metadata


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Package a trained experiment for the API.")
    p.add_argument("--experiment", required=True, help="Experiment folder with fold*/ or a single run, e.g. runs/baseline")
    p.add_argument("--src", help="MNI-HISUB25 folder with the SPAM atlas maps (for the whole-brain MNI endpoint)")
    p.add_argument("--out", default=OUT, help="Package folder")
    p.add_argument("--name", default="3D U-Net", help="Model name shown by the API")
    p.add_argument("--single", action="store_true", help="Package the single run in the experiment folder, not its folds")
    a = p.parse_args()
    main(a.experiment, a.src, a.out, a.name, a.single)
