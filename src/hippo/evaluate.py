"""Evaluate a trained model on the held-out test split.

Writes one row per case (Dice, NSD, predicted and true volumes) and a summary
with mean and median per metric, the same statistics the MSD paper reports.

Usage:
    python -m hippo.evaluate --run runs/baseline
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from monai.data import DataLoader, Dataset

from .data import eval_transforms
from .metrics import case_metrics
from .model import build_model
from .train import spacing_from_affine


@torch.no_grad()
def main(run_dir: str, split: str = "test") -> dict:
    run = Path(run_dir)
    cfg = yaml.safe_load((run / "config.yaml").read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(cfg).to(device)
    model.load_state_dict(torch.load(run / "best.pt", map_location=device))
    model.eval()

    cases = json.loads(Path(cfg["data"]["split_file"]).read_text())[split]
    loader = DataLoader(Dataset(cases, eval_transforms()), batch_size=1)

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
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="runs/baseline")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    a = p.parse_args()
    main(a.run, a.split)
