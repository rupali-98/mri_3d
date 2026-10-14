"""Dataset download, patient-level splits and MONAI data loaders."""
from __future__ import annotations

import argparse
import json
import random
import tarfile
import urllib.request
from pathlib import Path

from monai.data import CacheDataset, DataLoader
from monai.transforms import (
    Compose,
    DivisiblePadd,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    RandAffined,
    RandFlipd,
    RandGaussianNoised,
    RandScaleIntensityd,
    RandShiftIntensityd,
)

# Public mirror used by the MONAI tutorials (CC-BY-SA 4.0 data).
MSD_URL = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task04_Hippocampus.tar"


def download(dest: Path) -> Path:
    """Download and extract Task04_Hippocampus into `dest`. Skips if present."""
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / "Task04_Hippocampus"
    if (target / "dataset.json").exists():
        print(f"Already present: {target}")
        return target
    tar_path = dest / "Task04_Hippocampus.tar"
    print(f"Downloading {MSD_URL} ...")
    urllib.request.urlretrieve(MSD_URL, tar_path)
    with tarfile.open(tar_path) as tar:
        tar.extractall(dest, filter="data")
    tar_path.unlink()
    return target


def make_splits(root: Path, out: Path, val_fraction: float, test_fraction: float, seed: int) -> dict:
    """Split the labelled MSD training cases into train/val/test.

    MSD's own test set has no public labels, so our held-out test set is carved
    from the labelled cases. Each file is one subject, so the split is
    subject-level and there is no leakage between sets.
    """
    meta = json.loads((root / "dataset.json").read_text())
    cases = [
        {"image": str(root / c["image"].lstrip("./")), "label": str(root / c["label"].lstrip("./"))}
        for c in meta["training"]
    ]
    cases.sort(key=lambda c: c["image"])
    random.Random(seed).shuffle(cases)
    n_test = round(len(cases) * test_fraction)
    n_val = round(len(cases) * val_fraction)
    splits = {
        "test": cases[:n_test],
        "val": cases[n_test : n_test + n_val],
        "train": cases[n_test + n_val :],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(splits, indent=2))
    print({k: len(v) for k, v in splits.items()})
    return splits


def base_transforms() -> list:
    keys = ["image", "label"]
    return [
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
        Orientationd(keys=keys, axcodes="RAS"),
        NormalizeIntensityd(keys="image", nonzero=True),
        # Crops vary slightly in size; pad so the U-Net's 3 downsamplings divide evenly.
        DivisiblePadd(keys=keys, k=8),
    ]


def train_transforms() -> Compose:
    keys = ["image", "label"]
    return Compose(
        base_transforms()
        + [
            RandAffined(
                keys=keys,
                prob=0.5,
                rotate_range=(0.2, 0.2, 0.2),
                scale_range=(0.1, 0.1, 0.1),
                mode=("bilinear", "nearest"),
                padding_mode="zeros",
            ),
            # Flip left-right only (axis 0 in RAS). Flipping the anterior-posterior
            # axis would make "anterior" and "posterior" labels swap places.
            RandFlipd(keys=keys, spatial_axis=0, prob=0.5),
            RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
            RandGaussianNoised(keys="image", std=0.05, prob=0.2),
            EnsureTyped(keys=keys),
        ]
    )


def eval_transforms() -> Compose:
    return Compose(base_transforms() + [EnsureTyped(keys=["image", "label"])])


def make_loaders(splits: dict, batch_size: int, num_workers: int, cache_rate: float):
    train_ds = CacheDataset(splits["train"], train_transforms(), cache_rate=cache_rate, num_workers=num_workers)
    val_ds = CacheDataset(splits["val"], eval_transforms(), cache_rate=cache_rate, num_workers=num_workers)
    # Cases differ in size after padding, so validation runs one volume at a time.
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              collate_fn=_pad_collate)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=num_workers)
    return train_loader, val_loader


def _pad_collate(batch):
    """Pad every volume in a batch to the largest shape so they can be stacked."""
    from monai.data import list_data_collate
    from monai.transforms import SpatialPad

    shape = [max(b["image"].shape[i] for b in batch) for i in range(1, 4)]
    pad = SpatialPad(spatial_size=shape)
    for b in batch:
        b["image"] = pad(b["image"])
        b["label"] = pad(b["label"])
    return list_data_collate(batch)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Download MSD Task04 and create splits.")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--test-fraction", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    root = download(Path(a.data_dir))
    make_splits(root, Path(a.data_dir) / "splits.json", a.val_fraction, a.test_fraction, a.seed)
