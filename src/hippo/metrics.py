"""Segmentation metrics used in the Medical Segmentation Decathlon: Dice and NSD.

Both are reported per foreground class, for one case at a time, so results can
be compared with the challenge's published numbers.
"""
from __future__ import annotations

import numpy as np
import torch
from monai.metrics import compute_dice, compute_surface_dice
from monai.networks.utils import one_hot

from . import CLASS_NAMES, NUM_CLASSES


def _one_hot(label: torch.Tensor) -> torch.Tensor:
    """(B,1,H,W,D) integer labels -> (B,C,H,W,D) one-hot."""
    return one_hot(label.long(), num_classes=NUM_CLASSES, dim=1)


def missing_parts(label) -> list[str]:
    """Names of the parts (anterior, posterior) with no voxels in a label map."""
    return [name for cls, name in CLASS_NAMES.items() if not (label == cls).any()]


def case_metrics(pred: torch.Tensor, label: torch.Tensor, spacing_mm, nsd_tolerance_mm: float = 1.0) -> dict:
    """Dice, NSD and volumes for one case.

    pred, label: (1,1,H,W,D) integer label maps.
    spacing_mm: voxel size, e.g. (1.0, 1.0, 1.0).
    """
    p, y = _one_hot(pred), _one_hot(label)
    dice = compute_dice(p, y, include_background=False, ignore_empty=False)[0]
    nsd = compute_surface_dice(
        p, y,
        class_thresholds=[nsd_tolerance_mm] * (NUM_CLASSES - 1),
        include_background=False,
        spacing=list(map(float, spacing_mm)),
    )[0]
    voxel_ml = float(np.prod(spacing_mm)) / 1000.0
    out = {}
    for i, (cls, name) in enumerate(CLASS_NAMES.items()):
        out[f"dice_{name}"] = float(dice[i])
        out[f"nsd_{name}"] = float(nsd[i])
        out[f"vol_pred_ml_{name}"] = float((pred == cls).sum()) * voxel_ml
        out[f"vol_true_ml_{name}"] = float((label == cls).sum()) * voxel_ml
    return out
