"""Evaluate a trained model on the held-out test split.

Writes one row per case (Dice, NSD, predicted and true volumes) and a summary
with mean and median per metric, the same statistics the MSD paper reports.

Usage:
    python -m hippo.evaluate --run runs/baseline
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from monai.data import DataLoader, Dataset

from . import defaults
from .data import eval_transforms
from .metrics import case_metrics
from .model import build_model
from .train import spacing_from_affine


@torch.no_grad()
def main(run_dir: str, split: str = "test", data_dir: str | None = None) -> dict:
    run = Path(run_dir)
    cfg = yaml.safe_load((run / "config.yaml").read_text())
    if data_dir is not None:  # otherwise use the data the run was trained on
        cfg["data"]["split_file"] = str(Path(data_dir) / defaults.SPLIT_FILENAME)
    device = defaults.pick_device()

    model = build_model(cfg).to(device)
    model.load_state_dict(torch.load(run / "best.pt", map_location=device))
    model.eval()

    cases = json.loads(Path(cfg["data"]["split_file"]).read_text())[split]
    loader = DataLoader(Dataset(cases, eval_transforms(cfg["data"].get("normalize", "zscore"))), batch_size=1)

    rows = []
    for case, batch in zip(cases, loader):
        img, lbl = batch["image"].to(device), batch["label"]
        pred = model(img).argmax(dim=1, keepdim=True).cpu()
        m = case_metrics(pred, lbl, spacing_from_affine(batch["image"].affine), cfg["eval"]["nsd_tolerance_mm"])
        rows.append({"case": Path(case["image"]).name, **m})

    with open(run / f"metrics_{split}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    summary = {"n_cases": len(rows)}
    for k in rows[0]:
        if k.startswith(("dice", "nsd")):
            vals = np.array([r[k] for r in rows])
            summary[k] = {"mean": float(vals.mean()), "median": float(np.median(vals)), "std": float(vals.std())}
    (run / f"summary_{split}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    p = defaults.make_parser("Evaluate a trained model on a held-out split.")
    p.add_argument("--run", default=defaults.RUN_DIR)
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    # Without --data-dir / HIPPO_DATA_DIR / hippo.conf, use the location saved in the run.
    p.set_defaults(data_dir=None)
    a = p.parse_args()
    main(a.run, a.split, a.data_dir)
