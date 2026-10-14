"""Tiny synthetic stand-in for MSD Task04, with the same folder layout.

Each "subject" is a 32x40x32 volume with two touching ellipsoids labelled
1 (anterior) and 2 (posterior). Used only for tests, never for results.
"""
from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np


def make_fake_msd(root: Path, n: int = 12, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    task = root / "Task04_Hippocampus"
    (task / "imagesTr").mkdir(parents=True, exist_ok=True)
    (task / "labelsTr").mkdir(parents=True, exist_ok=True)
    zz, yy, xx = np.meshgrid(np.arange(32), np.arange(40), np.arange(32), indexing="ij")
    training = []
    for i in range(n):
        cx, cz = 16 + rng.integers(-2, 3), 16 + rng.integers(-2, 3)
        lbl = np.zeros((32, 40, 32), np.uint8)
        ant = ((zz - cz) / 5) ** 2 + ((yy - 14) / 7) ** 2 + ((xx - cx) / 5) ** 2 <= 1
        post = ((zz - cz) / 5) ** 2 + ((yy - 26) / 7) ** 2 + ((xx - cx) / 5) ** 2 <= 1
        lbl[ant], lbl[post] = 1, 2
        img = (lbl > 0) * 100.0 + lbl * 20.0 + rng.normal(0, 10, lbl.shape)
        name = f"hippocampus_{i:03d}.nii.gz"
        nib.save(nib.Nifti1Image(img.astype(np.float32), np.eye(4)), task / "imagesTr" / name)
        nib.save(nib.Nifti1Image(lbl, np.eye(4)), task / "labelsTr" / name)
        training.append({"image": f"./imagesTr/{name}", "label": f"./labelsTr/{name}"})
    (task / "dataset.json").write_text(json.dumps({"training": training}))
    return task
