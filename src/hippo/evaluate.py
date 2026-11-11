"""Evaluate a trained model on the held-out test split.

Writes one row per case (Dice, NSD, HD95, predicted and true volumes; per part and for the whole
hippocampus) and a summary with mean and median per metric, the same statistics the MSD paper
reports, plus volume agreement per label (absolute and relative error, ICC, Bland-Altman bias and
limits). The summary is also added to the run's MLflow run as test.<metric>_<label>.

Every case is predicted. A case whose ground truth has no anterior or no
posterior can't be scored (NSD is undefined), so its Dice and NSD are left
blank and it is left out of the summary, which lists it under "dropped".
Its predicted volume is still written, and hippo.visualize still shows it.

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

from . import defaults, tracking
from .data import eval_transforms
from .metrics import LABELS, case_metrics, missing_parts, summarise
from .model import build_model
from .train import spacing_from_affine


@torch.no_grad()
def main(run_dir: str, split: str = "test", data_dir: str | None = None, out_dir: str | None = None) -> dict:
    """out_dir: where the CSV and summary go (default: the run folder), e.g. for archive/shift.py's test copies."""
    run = Path(run_dir)
    out = Path(out_dir) if out_dir else run
    out.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load((run / "config.yaml").read_text())
    if data_dir is not None:  # otherwise use the data the run was trained on
        cfg["data"]["split_file"] = str(Path(data_dir) / defaults.SPLIT_FILENAME)
    device = defaults.pick_device()

    model = build_model(cfg).to(device)
    model.load_state_dict(torch.load(run / "best.pt", map_location=device))
    model.eval()

    cases = json.loads(Path(cfg["data"]["split_file"]).read_text())[split]
    loader = DataLoader(Dataset(cases, eval_transforms(cfg["data"].get("normalize", "zscore"))), batch_size=1)

    rows, dropped = [], {}
    for case, batch in zip(cases, loader):
        img, lbl = batch["image"].to(device), batch["label"]
        pred = model(img).argmax(dim=1, keepdim=True).cpu()
        m = case_metrics(pred, lbl, spacing_from_affine(batch["image"].affine), cfg["eval"]["nsd_tolerance_mm"])
        name = Path(case["image"]).name
        if missing := missing_parts(lbl):
            dropped[name] = "no " + " or ".join(missing) + " in the ground truth"
            m = {k: "" if k.startswith(("dice", "nsd")) else v for k, v in m.items()}
        rows.append({"case": name, **m})

    with open(out / f"metrics_{split}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    scored = [r for r in rows if r["case"] not in dropped]
    summary = {"n_cases": len(scored), "n_dropped": len(dropped), "dropped": dropped}
    for k in rows[0]:
        if k.startswith(("dice", "nsd", "hd95")) and scored:
            vals = np.array([r[k] for r in scored])  # HD95 is nan where a prediction is empty
            summary[k] = {"mean": float(np.nanmean(vals)), "median": float(np.nanmedian(vals)),
                          "std": float(np.nanstd(vals))}
    summary["by_label"] = {label: summarise(scored, f"_{label}") for label in LABELS}
    (out / f"summary_{split}.json").write_text(json.dumps(summary, indent=2))
    name = tracking.result_name(run, out)
    tracking.log_results(run, split if name == "" else f"{name}.{split}",
                         {f"{k}_{label}": v for label, m in summary["by_label"].items() for k, v in m.items()},
                         {"post_processing": "none", "split": split})
    print(json.dumps(summary, indent=2))
    if dropped:
        print(f"{len(dropped)} of {len(rows)} cases not scored (predictions still made): "
              + "; ".join(f"{c}: {why}" for c, why in dropped.items()))
    return summary


if __name__ == "__main__":
    p = defaults.make_parser("Evaluate a trained model on a held-out split.")
    p.add_argument("--run", default=defaults.RUN_DIR)
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    # Without --data-dir / HIPPO_DATA_DIR / hippo.conf, use the location saved in the run.
    p.set_defaults(data_dir=None)
    a = p.parse_args()
    main(a.run, a.split, a.data_dir)
