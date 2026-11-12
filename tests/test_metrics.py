import numpy as np
import pytest
import torch

from hippo.metrics import case_metrics, icc_a1, missing_parts, summarise, volume_agreement


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


def test_missing_parts():
    lbl = _blob()
    assert missing_parts(lbl) == []
    lbl[lbl == 2] = 0
    assert missing_parts(lbl) == ["posterior"]
    assert missing_parts(torch.zeros_like(lbl)) == ["anterior", "posterior"]


def test_hd95_and_whole_hippocampus():
    lbl = _blob()
    m = case_metrics(lbl.clone(), lbl, (1.0, 1.0, 1.0))
    assert m["hd95_anterior"] == 0 and m["dice_whole"] == pytest.approx(1.0)
    shifted = torch.roll(lbl, shifts=2, dims=-1)                   # every surface point moves 2 mm along z
    m = case_metrics(shifted, lbl, (1.0, 1.0, 1.0))
    assert m["hd95_anterior"] == pytest.approx(2.0) and m["hd95_whole"] == pytest.approx(2.0)
    # swapping the parts ruins the per-part scores but not the whole hippocampus
    swapped = torch.where(lbl == 1, 2, torch.where(lbl == 2, 1, 0))
    m = case_metrics(swapped, lbl, (1.0, 1.0, 1.0))
    assert m["dice_anterior"] == pytest.approx(0.0) and m["dice_whole"] == pytest.approx(1.0)
    assert m["vol_pred_ml_whole"] == m["vol_true_ml_whole"] == pytest.approx(1.0)


def test_hd95_is_nan_for_an_empty_prediction():
    lbl = _blob()
    m = case_metrics(torch.zeros_like(lbl), lbl, (1.0, 1.0, 1.0))
    assert np.isnan(m["hd95_anterior"]) and np.isnan(m["hd95_whole"])


def test_icc_matches_shrout_and_fleiss():
    # Shrout & Fleiss (1979), Table 2: 6 targets x 4 judges; their ICC(2,1) is 0.29
    ratings = [[9, 2, 5, 8], [6, 1, 3, 2], [8, 4, 6, 8], [7, 1, 2, 6], [10, 5, 6, 9], [6, 2, 4, 7]]
    assert icc_a1(ratings) == pytest.approx(0.29, abs=0.005)
    assert icc_a1([[1, 1], [2, 2], [3, 3]]) == pytest.approx(1.0)


def test_volume_agreement():
    true = np.array([3.0, 3.5, 4.0, 4.5])
    perfect = volume_agreement(true, true)
    assert perfect["abs_vol_error_ml"] == 0 and perfect["icc"] == pytest.approx(1.0) and perfect["bias_ml"] == 0
    over = volume_agreement(true + 0.5, true)                       # every prediction 0.5 ml too big
    assert over["abs_vol_error_ml"] == pytest.approx(0.5) and over["bias_ml"] == pytest.approx(0.5)
    assert over["rel_vol_error_pct"] == pytest.approx(np.mean(0.5 / true) * 100)
    assert over["icc"] < 1                                           # absolute agreement counts the offset
    assert over["loa_low_ml"] == pytest.approx(0.5) and over["loa_high_ml"] == pytest.approx(0.5)


def test_summarise_skips_unscored_cases():
    rows = [{"dice_whole": "0.8", "nsd_whole": "0.7", "hd95_whole": "2.0", "vol_pred_ml_whole": "3", "vol_true_ml_whole": "3"},
            {"dice_whole": "0.6", "nsd_whole": "0.5", "hd95_whole": "nan", "vol_pred_ml_whole": "4", "vol_true_ml_whole": "3"},
            {"dice_whole": "", "nsd_whole": "", "hd95_whole": "", "vol_pred_ml_whole": "9", "vol_true_ml_whole": "3"}]
    s = summarise(rows, "_whole")
    assert s["n"] == 2 and s["dice"] == pytest.approx(0.7) and s["hd95"] == pytest.approx(2.0)
    assert s["abs_vol_error_ml"] == pytest.approx(0.5)
    assert summarise([], "_whole") == {"n": 0}
