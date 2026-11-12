"""Ensemble of an experiment's fold models: their class probabilities averaged into one prediction.

The cross-validation tables (hippo.results on runs/<exp>) give the mean ± SD of the fold models scored one by
one; an ensemble is scored as one model, the way the API serves it (MLflow alias "ensemble"). make() writes
<experiment>/ensemble/ with fold0's config.yaml (fold: ensemble) and members.json (the fold folders), but no
weights. hippo.evaluate and hippo.external take it as --run like any run folder and write their results
inside it; hippo.results on it gives the ensemble's own table.

Usage:
    python -m hippo.ensemble --experiment runs/random_trim               # -> runs/random_trim/ensemble/
    python -m hippo.evaluate --run runs/random_trim/ensemble
    python -m hippo.external --src data/mri_dataset --run runs/random_trim/ensemble
    python -m hippo.results --experiment runs/random_trim/ensemble --name "Random position, 5-fold ensemble"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from .model import build_model

FOLDER = "ensemble"
MEMBERS = "members.json"


def make(experiment: str) -> Path:
    """Write <experiment>/ensemble/ (config.yaml + members.json) for the experiment's fold models."""
    exp = Path(experiment)
    folds = sorted(p for p in exp.glob("fold*") if (p / "best.pt").exists() and (p / "config.yaml").exists())
    if len(folds) < 2:
        raise SystemExit(f"{exp} has {len(folds)} trained fold folder(s); an ensemble needs at least 2")
    configs = [yaml.safe_load((f / "config.yaml").read_text()) for f in folds]
    if any(c["model"] != configs[0]["model"] or c["data"].get("normalize") != configs[0]["data"].get("normalize")
           for c in configs):
        raise SystemExit("The fold models differ in architecture or preprocessing; they can't be averaged")
    out = exp / FOLDER
    out.mkdir(exist_ok=True)
    cfg = configs[0]
    cfg["data"]["fold"] = FOLDER  # tells MLflow and hippo.results this is neither a fold nor a single run
    cfg["train"]["out_dir"] = str(out)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    (out / MEMBERS).write_text(json.dumps([f"../{f.name}" for f in folds], indent=2))
    return out


def members(run: str | Path) -> list[Path]:
    """The weight folders behind a run: itself for a trained run, the listed folds for an ensemble folder."""
    run = Path(run)
    if (run / "best.pt").exists():
        return [run]
    if (run / MEMBERS).exists():
        return [(run / m).resolve() for m in json.loads((run / MEMBERS).read_text())]
    raise SystemExit(f"{run} has neither best.pt nor {MEMBERS}")


def load(run: str | Path, device):
    """A function image -> class probabilities (batch, classes, X, Y, Z): one model, or the mean of the folds'."""
    models = []
    for folder in members(run):
        model = build_model(yaml.safe_load((folder / "config.yaml").read_text())).to(device)
        model.load_state_dict(torch.load(folder / "best.pt", map_location=device, weights_only=True))
        models.append(model.eval())
    return lambda image: torch.stack([torch.softmax(m(image), 1) for m in models]).mean(0)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Make an ensemble run folder from an experiment's fold models.")
    p.add_argument("--experiment", required=True, help="Experiment folder with fold0 ... fold4, e.g. runs/random_trim")
    print(f"Wrote {make(p.parse_args().experiment)}")
