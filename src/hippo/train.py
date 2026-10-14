"""Train the baseline 3D U-Net.

Usage:
    python -m hippo.train --config configs/baseline.yaml
"""
from __future__ import annotations

import argparse
import contextlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from monai.losses import DiceCELoss
from monai.utils import set_determinism

from .data import make_loaders
from .metrics import case_metrics
from .model import build_model

try:
    import mlflow
except ImportError:  # MLflow is optional; training works without it.
    mlflow = None


def spacing_from_affine(affine) -> tuple:
    a = np.asarray(affine)[0] if np.asarray(affine).ndim == 3 else np.asarray(affine)
    return tuple(np.sqrt((a[:3, :3] ** 2).sum(axis=0)).tolist())


@torch.no_grad()
def validate(model, loader, device, nsd_tol: float) -> dict:
    model.eval()
    rows = []
    for batch in loader:
        img, lbl = batch["image"].to(device), batch["label"].to(device)
        pred = model(img).argmax(dim=1, keepdim=True)
        rows.append(case_metrics(pred.cpu(), lbl.cpu(), spacing_from_affine(batch["image"].affine), nsd_tol))
    keys = rows[0].keys()
    return {k: float(np.mean([r[k] for r in rows])) for k in keys if k.startswith(("dice", "nsd"))}


def main(cfg_path: str, max_epochs: int | None = None) -> dict:
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    t = cfg["train"]
    epochs = max_epochs or t["epochs"]
    set_determinism(seed=t["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = t["amp"] and device.type == "cuda"

    out = Path(t["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(cfg_path, out / "config.yaml")

    splits = json.loads(Path(cfg["data"]["split_file"]).read_text())
    train_loader, val_loader = make_loaders(
        splits, t["batch_size"], cfg["data"]["num_workers"], cfg["data"]["cache_rate"]
    )

    model = build_model(cfg).to(device)
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    opt = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    run = mlflow.start_run(run_name=out.name) if mlflow else contextlib.nullcontext()
    best, history = -1.0, []
    with run:
        if mlflow:
            mlflow.log_params({**cfg["model"], **{k: v for k, v in t.items() if k != "out_dir"}})
        for epoch in range(1, epochs + 1):
            model.train()
            t0, losses = time.time(), []
            for batch in train_loader:
                img, lbl = batch["image"].to(device), batch["label"].to(device)
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    loss = loss_fn(model(img), lbl)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                losses.append(loss.item())
            sched.step()
            entry = {"epoch": epoch, "loss": float(np.mean(losses)), "sec": round(time.time() - t0, 1)}

            if epoch % t["val_every"] == 0 or epoch == epochs:
                val = validate(model, val_loader, device, cfg["eval"]["nsd_tolerance_mm"])
                entry.update(val)
                mean_dice = np.mean([v for k, v in val.items() if k.startswith("dice")])
                entry["mean_dice"] = float(mean_dice)
                if mean_dice > best:
                    best = mean_dice
                    torch.save(model.state_dict(), out / "best.pt")
            if mlflow:
                mlflow.log_metrics({k: v for k, v in entry.items() if k != "epoch"}, step=epoch)
            history.append(entry)
            print(json.dumps(entry))

        torch.save(model.state_dict(), out / "last.pt")
        (out / "history.json").write_text(json.dumps(history, indent=2))
        if mlflow:
            mlflow.log_artifacts(str(out))
    print(f"Best validation mean Dice: {best:.4f}")
    return {"best_mean_dice": best, "history": history}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/baseline.yaml")
    p.add_argument("--epochs", type=int, default=None, help="Override epochs (e.g. for a quick test)")
    a = p.parse_args()
    main(a.config, a.epochs)
