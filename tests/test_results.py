"""hippo.results: per-experiment table over folds and the Bland-Altman plot."""
import csv

import pytest
import yaml

from hippo import results


def _write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def _msd_row(case, dice, vol):
    row = {"case": case}
    for label in ("anterior", "posterior", "whole"):
        row.update({f"dice_{label}": dice, f"nsd_{label}": dice, f"hd95_{label}": 2.0,
                    f"vol_pred_ml_{label}": vol, f"vol_true_ml_{label}": 2.0})
    return row


def _experiment(tmp_path):
    exp = tmp_path / "exp"
    for fold, dice in ((0, 0.8), (1, 0.9)):
        run = exp / f"fold{fold}"
        run.mkdir(parents=True)
        (run / "config.yaml").write_text(yaml.safe_dump({"data": {"fold": fold}}))
        _write(run / "metrics_test.csv", [_msd_row("a", dice, 2.0), _msd_row("b", dice, 2.5), _msd_row("c", dice, 1.0)])
        _write(run / "external" / "metrics_external.csv",
               [{"case": c, "dice": dice - 0.1, "nsd": 0.7, "hd95": 3.0, "vol_pred_ml": 4.5, "vol_true_ml": 4.0}
                for c in ("s01_L", "s01_R", "s02_L")])
    return exp


def test_table_is_mean_and_sd_over_folds(tmp_path):
    rows = results.experiment_report(str(_experiment(tmp_path)), "3D U-Net")
    assert [(r["test set"], r["label"]) for r in rows] == [("MSD test", "anterior"), ("MSD test", "posterior"),
                                                           ("MSD test", "whole"), ("MNI-HISUB25, atlas box", "whole")]
    whole = rows[2]
    assert whole["folds"] == 2 and whole["cases"] == 3
    assert whole["dice_mean"] == pytest.approx(0.85) and whole["dice_sd"] == pytest.approx(0.0707, abs=1e-4)
    assert whole["abs_vol_error_ml_mean"] == pytest.approx((0 + 0.5 + 1.0) / 3)
    out = tmp_path / "exp" / "results"
    text = (out / "results.md").read_text()
    assert "## 3D U-Net" in text and "0.850 ± 0.071" in text and "Research use only" in text
    assert (out / "bland_altman.png").stat().st_size > 0 and (out / "results.csv").exists()


def test_single_run_and_older_csvs(tmp_path):
    run = tmp_path / "old"
    _write(run / "metrics_test.csv", [{"case": "a", "dice_anterior": 0.8, "nsd_anterior": 0.9,
                                       "vol_pred_ml_anterior": 1, "vol_true_ml_anterior": 1,
                                       "dice_posterior": 0.7, "nsd_posterior": 0.8,
                                       "vol_pred_ml_posterior": 1, "vol_true_ml_posterior": 1}])
    rows = results.scores(run)                                 # no fold folders: the folder itself
    assert [r["label"] for r in rows] == ["anterior", "posterior"]  # no whole columns yet: left out
    assert rows[0]["folds"] == 1 and rows[0]["dice_sd"] != rows[0]["dice_sd"]  # no SD for one run
    assert results.cell(rows[0]["hd95_mean"], rows[0]["hd95_sd"], 2) == "–"      # no HD95 in old files


def test_no_results_is_an_error(tmp_path):
    with pytest.raises(SystemExit, match="No results"):
        results.experiment_report(str(tmp_path))
