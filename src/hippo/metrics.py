"""Segmentation and volume metrics: Dice, NSD and HD95 per part and for the whole hippocampus, and
volume agreement between predicted and true volumes.

Dice and NSD follow the Medical Segmentation Decathlon, so results can be compared with the challenge's
published numbers. HD95 is the 95th percentile of the distances between the two surfaces (mm); it is
blank (nan) when the prediction or the label is empty. "whole" merges anterior and posterior (label > 0).

Volume agreement over a set of cases: mean absolute error (ml), mean absolute relative error (% of the
true volume), ICC(A,1) (two-way, absolute agreement, single measure; McGraw & Wong 1996, the same as
Shrout & Fleiss ICC(2,1)) and the Bland-Altman bias with its 95% limits of agreement.
"""
from __future__ import annotations

import numpy as np
import torch
from monai.metrics import compute_dice, compute_hausdorff_distance, compute_surface_dice
from monai.networks.utils import one_hot

from . import CLASS_NAMES, NUM_CLASSES

LABELS = [*CLASS_NAMES.values(), "whole"]


def _one_hot(label: torch.Tensor) -> torch.Tensor:
    """(B,1,H,W,D) integer labels -> (B,C,H,W,D) one-hot."""
    return one_hot(label.long(), num_classes=NUM_CLASSES, dim=1)


def missing_parts(label) -> list[str]:
    """Names of the parts (anterior, posterior) with no voxels in a label map."""
    return [name for cls, name in CLASS_NAMES.items() if not (label == cls).any()]


def overlap(p: torch.Tensor, y: torch.Tensor, spacing_mm, nsd_tolerance_mm: float) -> tuple:
    """Dice, NSD and HD95 per foreground channel of one-hot (1,C,...) maps; HD95 is nan where one is empty."""
    spacing = list(map(float, spacing_mm))
    n = p.shape[1] - 1
    dice = compute_dice(p, y, include_background=False, ignore_empty=False)[0]
    nsd = compute_surface_dice(p, y, class_thresholds=[nsd_tolerance_mm] * n, include_background=False,
                               spacing=spacing)[0]
    empty = (p[0, 1:].flatten(1).sum(1) == 0) | (y[0, 1:].flatten(1).sum(1) == 0)
    hd95 = torch.full((n,), float("nan"))
    if not empty.all():  # MONAI warns and returns inf for an empty map; skip it instead
        hd95 = compute_hausdorff_distance(p, y, include_background=False, percentile=95, spacing=spacing)[0].float()
        hd95[empty] = float("nan")
    return dice, nsd, hd95


def whole_metrics(pred: torch.Tensor, label: torch.Tensor, spacing_mm, nsd_tolerance_mm: float = 1.0) -> dict:
    """Dice, NSD, HD95 and volumes of the whole hippocampus (any label > 0) for one case."""
    p, y = (one_hot((m > 0).long(), num_classes=2, dim=1) for m in (pred, label))
    dice, nsd, hd95 = overlap(p, y, spacing_mm, nsd_tolerance_mm)
    voxel_ml = float(np.prod(spacing_mm)) / 1000
    return {"dice": float(dice[0]), "nsd": float(nsd[0]), "hd95": float(hd95[0]),
            "vol_pred_ml": float((pred > 0).sum()) * voxel_ml, "vol_true_ml": float((label > 0).sum()) * voxel_ml}


def case_metrics(pred: torch.Tensor, label: torch.Tensor, spacing_mm, nsd_tolerance_mm: float = 1.0) -> dict:
    """Dice, NSD, HD95 and volumes for one case, per part and for the whole hippocampus.

    pred, label: (1,1,H,W,D) integer label maps.
    spacing_mm: voxel size, e.g. (1.0, 1.0, 1.0).
    """
    dice, nsd, hd95 = overlap(_one_hot(pred), _one_hot(label), spacing_mm, nsd_tolerance_mm)
    voxel_ml = float(np.prod(spacing_mm)) / 1000.0
    out = {}
    for i, (cls, name) in enumerate(CLASS_NAMES.items()):
        out[f"dice_{name}"] = float(dice[i])
        out[f"nsd_{name}"] = float(nsd[i])
        out[f"vol_pred_ml_{name}"] = float((pred == cls).sum()) * voxel_ml
        out[f"vol_true_ml_{name}"] = float((label == cls).sum()) * voxel_ml
        out[f"hd95_{name}"] = float(hd95[i])
    whole = whole_metrics(pred, label, spacing_mm, nsd_tolerance_mm)
    return {**out, **{f"{k}_whole": v for k, v in whole.items()}}


def icc_a1(ratings) -> float:
    """ICC(A,1) of an n cases x k raters array: two-way, absolute agreement, single measure."""
    y = np.asarray(ratings, dtype=float)
    n, k = y.shape
    grand = y.mean()
    ss_rows = k * ((y.mean(1) - grand) ** 2).sum()
    ss_cols = n * ((y.mean(0) - grand) ** 2).sum()
    ss_error = ((y - grand) ** 2).sum() - ss_rows - ss_cols
    ms_rows, ms_cols = ss_rows / (n - 1), ss_cols / (k - 1)
    ms_error = ss_error / ((n - 1) * (k - 1))
    return float((ms_rows - ms_error) / (ms_rows + (k - 1) * ms_error + k * (ms_cols - ms_error) / n))


def volume_agreement(pred_ml, true_ml) -> dict:
    """Absolute and relative volume error, ICC(A,1) and Bland-Altman bias and limits for paired volumes."""
    pred, true = np.asarray(pred_ml, dtype=float), np.asarray(true_ml, dtype=float)
    diff = pred - true
    spread = 1.96 * diff.std(ddof=1) if len(diff) > 1 else float("nan")
    return {"abs_vol_error_ml": float(np.abs(diff).mean()),
            "rel_vol_error_pct": float((np.abs(diff) / true).mean() * 100),
            "icc": icc_a1(np.column_stack([pred, true])) if len(diff) > 1 else float("nan"),
            "bias_ml": float(diff.mean()), "loa_low_ml": float(diff.mean() - spread),
            "loa_high_ml": float(diff.mean() + spread)}


def _number(value) -> float:
    """A metric from a CSV or a dict: blank (a case that could not be scored) and "nan" become nan."""
    return float("nan") if value in ("", None) else float(value)


def summarise(rows: list[dict], suffix: str = "") -> dict:
    """Mean Dice, NSD and HD95 and the volume agreement over cases, for one label.

    suffix picks the label's columns: "_anterior", "_posterior", "_whole" (hippo.evaluate) or "" (hippo.external).
    Cases left blank (not scored) are skipped; HD95 averages the cases where it is defined.
    """
    scored = [r for r in rows if _number(r.get(f"dice{suffix}")) == _number(r.get(f"dice{suffix}"))]  # not nan
    if not scored:
        return {"n": 0}
    out = {"n": len(scored)}
    for metric in ("dice", "nsd", "hd95"):
        values = np.array([_number(r.get(f"{metric}{suffix}")) for r in scored])
        out[metric] = float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")
    out.update(volume_agreement([_number(r[f"vol_pred_ml{suffix}"]) for r in scored],
                                [_number(r[f"vol_true_ml{suffix}"]) for r in scored]))
    return out
