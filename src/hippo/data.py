"""Dataset download, patient-level splits and MONAI data loaders."""
from __future__ import annotations

import csv
import json
import random
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
from monai.data import CacheDataset, DataLoader
from monai.transforms import (
    Compose,
    DivisiblePadd,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    RandAffined,
    RandFlipd,
    RandGaussianNoised,
    Randomizable,
    RandScaleIntensityd,
    RandShiftIntensityd,
    SpatialCrop,
)

from . import defaults

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
    from the labelled cases. MSD gives no subject IDs, so the split is by file;
    files 2k-1 and 2k may be one person's left and right side (a known limitation).
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


def use_clean_labels(splits: dict, data_dir: str) -> dict:
    """Train on the hippo.clean output: cleaned labels, excluded cases left out.

    Excluded cases (e.g. a missing part) are also left out of validation, so they
    can't skew model selection; validation keeps the original labels otherwise.
    Test is unchanged: hippo.evaluate predicts every test case and leaves cases
    with a missing part out of the scores.
    """
    index_path = Path(data_dir) / "clean" / "cases.csv"
    if not index_path.exists():
        raise SystemExit(f"{index_path} not found: run python -m hippo.explore and python -m hippo.clean first")
    with open(index_path) as f:
        index = {r["case"]: r for r in csv.DictReader(f)}
    train = []
    for c in splits["train"]:
        row = index[Path(c["image"]).name.replace(".nii.gz", "")]
        if row["included"] == "True":
            train.append({"image": c["image"], "label": str(Path(data_dir) / row["label"])})
    val = [c for c in splits["val"] if index[Path(c["image"]).name.replace(".nii.gz", "")]["included"] == "True"]
    print(f"Clean labels: {len(train)} of {len(splits['train'])} training and "
          f"{len(val)} of {len(splits['val'])} validation cases kept")
    return {**splits, "train": train, "val": val}


NORMALIZE = ("zscore", "none")  # none = raw scanner intensities, for comparison only


class RandomTrimd(Randomizable, MapTransform):
    """Cut `trim` voxels per axis, split at random between the two sides, keeping the label whole."""

    def __init__(self, keys, trim, label_key: str = "label"):
        MapTransform.__init__(self, keys)
        self.trim = trim
        self.label_key = label_key

    def __call__(self, data):
        d = dict(data)
        shape = d[self.label_key].shape[1:]
        found = np.argwhere(np.asarray(d[self.label_key][0]) > 0)
        start, end = [], []
        for size, low, last, trim in zip(shape, found.min(0), found.max(0), self.trim):
            high = size - 1 - last
            total = min(trim, low + high)  # a tight crop gets trimmed less rather than losing label
            cut_low = self.R.randint(max(0, total - high), min(total, low) + 1)
            start.append(cut_low)
            end.append(size - (total - cut_low))
        crop = SpatialCrop(roi_start=start, roi_end=end)
        for key in self.key_iterator(d):
            d[key] = crop(d[key])
        return d


def base_transforms(normalize: str = "zscore", keys: tuple = ("image", "label"), trim=None) -> list:
    if normalize not in NORMALIZE:
        raise SystemExit(f"data.normalize must be one of {NORMALIZE}, not {normalize!r}")
    t = [
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
        Orientationd(keys=keys, axcodes="RAS"),
    ]
    if normalize == "zscore":
        t.append(NormalizeIntensityd(keys="image", nonzero=True))
    # After z-scoring, so a trimmed scan has the same intensities as at inference.
    if trim and any(trim):
        t.append(RandomTrimd(keys=keys, trim=trim))
    # Crops vary slightly in size; pad so the U-Net's 3 downsamplings divide evenly.
    return t + [DivisiblePadd(keys=keys, k=8)]


def train_transforms(normalize: str = "zscore", random_trim=None) -> Compose:
    keys = ["image", "label"]
    return Compose(
        base_transforms(normalize, trim=random_trim)
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


def eval_transforms(normalize: str = "zscore") -> Compose:
    return Compose(base_transforms(normalize) + [EnsureTyped(keys=["image", "label"])])


def make_loaders(splits: dict, batch_size: int, num_workers: int, cache_rate: float, normalize: str = "zscore",
                 random_trim=None):
    train_ds = CacheDataset(splits["train"], train_transforms(normalize, random_trim), cache_rate=cache_rate,
                            num_workers=num_workers)
    val_ds = CacheDataset(splits["val"], eval_transforms(normalize), cache_rate=cache_rate, num_workers=num_workers)
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
    p = defaults.make_parser("Download MSD Task04 and create splits.")
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--test-fraction", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    root = download(Path(a.data_dir))
    make_splits(root, Path(a.data_dir) / defaults.SPLIT_FILENAME, a.val_fraction, a.test_fraction, a.seed)
