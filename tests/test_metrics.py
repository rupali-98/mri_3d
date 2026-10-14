import torch

from hippo.metrics import case_metrics


def _blob():
    lbl = torch.zeros(1, 1, 20, 20, 20, dtype=torch.long)
    lbl[..., 5:10, 5:15, 5:15] = 1
    lbl[..., 10:15, 5:15, 5:15] = 2
    return lbl


def test_perfect_prediction_scores_one():
    lbl = _blob()
    m = case_metrics(lbl.clone(), lbl, (1.0, 1.0, 1.0))
    for name in ("anterior", "posterior"):
        assert abs(m[f"dice_{name}"] - 1.0) < 1e-6
        assert abs(m[f"nsd_{name}"] - 1.0) < 1e-6


def test_volume_uses_spacing():
    lbl = _blob()
    m = case_metrics(lbl, lbl, (2.0, 1.0, 1.0))  # 2 mm^3 voxels
    # anterior block: 5*10*10 = 500 voxels * 2 mm^3 = 1000 mm^3 = 1.0 ml
    assert abs(m["vol_true_ml_anterior"] - 1.0) < 1e-9


def test_shift_lowers_dice_and_nsd_within_tolerance():
    lbl = _blob()
    shifted = torch.roll(lbl, shifts=3, dims=-1)  # 3 voxel shift > 1 mm tolerance
    m = case_metrics(shifted, lbl, (1.0, 1.0, 1.0), nsd_tolerance_mm=1.0)
    assert m["dice_anterior"] < 0.9
    assert m["nsd_anterior"] < 0.9
    # a 1-voxel shift stays inside a 2 mm tolerance for NSD but still costs Dice
    one = torch.roll(lbl, shifts=1, dims=-1)
    m1 = case_metrics(one, lbl, (1.0, 1.0, 1.0), nsd_tolerance_mm=2.0)
    assert m1["nsd_anterior"] > m1["dice_anterior"]
