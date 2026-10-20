"""Train the baseline 3D U-Net.

Usage:
    python -m hippo.train --config configs/baseline.yaml
    python -m hippo.train --config configs/baseline.yaml --resume   # continue an interrupted run
"""
from __future__ import annotations

import contextlib
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from monai.losses import DiceCELoss
from monai.utils import set_determinism

from . import defaults
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


def save_atomic(obj, path: Path) -> None:
    """Write to a temp file then rename, so a crash mid-write never leaves a corrupt file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def check_disk(path: Path, min_free_gb: float) -> None:
    """Stop before training rather than crash hours in when the disk fills up."""
    free_gb = shutil.disk_usage(path).free / 1e9
    if free_gb < min_free_gb:
        raise SystemExit(
            f"Only {free_gb:.1f} GB free at {path.resolve()}; need {min_free_gb} GB "
            "(train.min_free_disk_gb in the config)"
        )


def setup_mlflow(out: Path) -> None:
    """Keep MLflow data next to the runs (git-ignored) unless MLFLOW_TRACKING_URI says otherwise."""
    root = out.resolve().parent
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri(f"sqlite:///{(root / 'mlflow.db').as_posix()}")
    name = os.environ.get("MLFLOW_EXPERIMENT_NAME", "hippo")
    if mlflow.get_experiment_by_name(name) is None:
        mlflow.create_experiment(name, artifact_location=(root / "mlartifacts").as_uri())
    mlflow.set_experiment(name)


def main(
    cfg_path: str,
    max_epochs: int | None = None,
    data_dir: str = defaults.DATA_DIR,
    resume: bool = False,
) -> dict:
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    cfg["data"]["split_file"] = str(Path(data_dir) / defaults.SPLIT_FILENAME)
    t = cfg["train"]
    epochs = max_epochs or t["epochs"]
    set_determinism(seed=t["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = t["amp"] and device.type == "cuda"

    out = Path(t["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    check_disk(out, t.get("min_free_disk_gb", 0))
    # Save the resolved config (including the data location) so evaluate uses the same data.
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    splits = json.loads(Path(cfg["data"]["split_file"]).read_text())
    train_loader, val_loader = make_loaders(
        splits, t["batch_size"], cfg["data"]["num_workers"], cfg["data"]["cache_rate"]
    )

    model = build_model(cfg).to(device)
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    opt = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # Full training state, rewritten every epoch; removed once the run finishes.
    ckpt_path = out / "checkpoint.pt"
    best, history, start_epoch, run_id = -1.0, [], 1, None
    if resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        sched.load_state_dict(ckpt["sched"])
        scaler.load_state_dict(ckpt["scaler"])
        best, history, run_id = ckpt["best"], ckpt["history"], ckpt["mlflow_run_id"]
        start_epoch = ckpt["epoch"] + 1
        print(f"Resuming from epoch {start_epoch} (best mean Dice so far {best:.4f})")
    elif resume:
        print(f"No checkpoint in {out}; starting from scratch")

    run = contextlib.nullcontext()
    if mlflow:
        setup_mlflow(out)
        try:  # continue the same MLflow run if it still exists
            run = mlflow.start_run(run_id=run_id) if run_id else mlflow.start_run(run_name=out.name)
        except Exception:
            run = mlflow.start_run(run_name=out.name)
    with run:
        if mlflow:
            run_id = mlflow.active_run().info.run_id
            if start_epoch == 1:
                mlflow.log_params({**cfg["model"], **{k: v for k, v in t.items() if k != "out_dir"}})
        for epoch in range(start_epoch, epochs + 1):
            # Reseed shuffling, augmentation and dropout from the epoch number, so a
            # resumed run sees exactly the batches an uninterrupted run would have.
            set_determinism(seed=t["seed"] + epoch)
            train_loader.dataset.transform.set_random_state(seed=t["seed"] + epoch)
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
            (out / "history.json").write_text(json.dumps(history, indent=2))
            save_atomic(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "opt": opt.state_dict(),
                    "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(),
                    "best": float(best),
                    "history": history,
                    "mlflow_run_id": run_id,
                },
                ckpt_path,
            )

        torch.save(model.state_dict(), out / "last.pt")
        ckpt_path.unlink(missing_ok=True)
        if mlflow:
            mlflow.log_artifacts(str(out))
    print(f"Best validation mean Dice: {best:.4f}")
    return {"best_mean_dice": best, "history": history}


if __name__ == "__main__":
    p = defaults.make_parser("Train the baseline 3D U-Net.")
    p.add_argument("--config", default=defaults.CONFIG, help="Experiment settings (YAML)")
    p.add_argument("--epochs", type=int, default=None, help="Override epochs (e.g. for a quick test)")
    p.add_argument("--resume", action="store_true", help="Continue from checkpoint.pt in the run folder")
    a = p.parse_args()
    main(a.config, a.epochs, a.data_dir, a.resume)
