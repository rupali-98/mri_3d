"""MLflow tracking shared by training and evaluation: one MLflow run per run folder (one fold).

Training starts the MLflow run and saves its id in <run>/mlflow_run_id.txt. hippo.evaluate,
hippo.external and archive/shift.py then add their results to that same run, so each fold holds its
config, fold, training curves and every evaluation, named by where the results were written
(test.dice_whole, external_largest.dice, ...) and tagged with the post-processing used. A run folder
from before this (no id file) gets a new MLflow run with its config the first time it is evaluated.
hippo.results --ablation reads them back as one table.

MLflow data lives in runs/mlflow.db and runs/mlartifacts/ (git-ignored) unless MLFLOW_TRACKING_URI says
otherwise. HIPPO_MLFLOW=0 turns logging off (the tests do this, except those that check the logging).
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import yaml

try:
    import mlflow
except ImportError:  # MLflow is optional; training and evaluation work without it.
    mlflow = None

EXPERIMENT = "hippo"
RUN_ID_FILE = "mlflow_run_id.txt"


def enabled() -> bool:
    return mlflow is not None and os.environ.get("HIPPO_MLFLOW", "1") != "0"


def inside_experiment(run_dir) -> bool:
    """A fold folder (runs/baseline/fold2) or an ensemble folder (runs/baseline/ensemble), not an experiment."""
    name = Path(run_dir).name
    return name.startswith("fold") or name == "ensemble"


def runs_root(run_dir) -> Path:
    """The folder holding all experiments (runs/): the parent of the experiment, also for a fold folder."""
    run = Path(run_dir).resolve()
    return run.parent.parent if inside_experiment(run) else run.parent


def experiment_name(run_dir) -> str:
    """runs/baseline/fold2 -> baseline; runs/baseline/ensemble -> baseline; runs/baseline -> baseline."""
    run = Path(run_dir)
    return run.parent.name if inside_experiment(run) else run.name


def setup(root: Path) -> None:
    """Point MLflow at <root>/mlflow.db with artifacts in <root>/mlartifacts, unless MLFLOW_TRACKING_URI is set."""
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri(f"sqlite:///{(Path(root) / 'mlflow.db').as_posix()}")
    name = os.environ.get("MLFLOW_EXPERIMENT_NAME", EXPERIMENT)
    if mlflow.get_experiment_by_name(name) is None:
        mlflow.create_experiment(name, artifact_location=(Path(root) / "mlartifacts").as_uri())
    mlflow.set_experiment(name)


def config_params(cfg: dict, run_dir) -> dict:
    """The settings that tell runs apart, flat, for the MLflow run's parameters."""
    t = {k: v for k, v in cfg["train"].items() if k != "out_dir"}
    data = cfg["data"]
    return {**cfg["model"], **t, "experiment": experiment_name(run_dir), "fold": data.get("fold", "none"),
            "labels": data.get("labels", "raw"), "normalize": data.get("normalize", "zscore"),
            "random_trim": data.get("random_trim")}


def start(run_dir, run_id: str | None = None):
    """Start (or continue) the run folder's MLflow run and remember its id there."""
    run_dir = Path(run_dir)
    setup(runs_root(run_dir))
    name = f"{experiment_name(run_dir)}/{run_dir.name}" if inside_experiment(run_dir) else run_dir.name
    active = None
    if run_id:
        try:  # continue the same MLflow run if it still exists
            active = mlflow.start_run(run_id=run_id)
        except Exception:
            active = None
    active = active or mlflow.start_run(run_name=name)
    (run_dir / RUN_ID_FILE).write_text(active.info.run_id)
    return active


def saved_run_id(run_dir) -> str | None:
    path = Path(run_dir) / RUN_ID_FILE
    return path.read_text().strip() if path.exists() else None


def result_name(run_dir, out_dir) -> str:
    """Name for a set of results by where they were written: "" for <run>/ itself, external_largest for
    <run>/external_largest/, shifted_external_largest.z+5 for <run>/shifted_external_largest/z+5/."""
    try:
        parts = Path(out_dir).resolve().relative_to(Path(run_dir).resolve()).parts
    except ValueError:
        parts = (Path(out_dir).name,)
    return ".".join(parts)


def log_results(run_dir, name: str, metrics: dict, flags: dict) -> None:
    """Add one evaluation's metrics (name.metric) and its settings (name.flag tags) to the run folder's MLflow run."""
    if not enabled():
        return
    run_dir = Path(run_dir)
    known = saved_run_id(run_dir)
    with start(run_dir, known):
        if not known:  # a run folder from before tracking: give its MLflow run the config
            cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
            mlflow.log_params({k: str(v) for k, v in config_params(cfg, run_dir).items()})
        mlflow.log_metrics({f"{name}.{k}": float(v) for k, v in metrics.items()
                            if isinstance(v, (int, float)) and math.isfinite(v)})
        mlflow.set_tags({f"{name}.{k}": str(v) for k, v in flags.items()})
