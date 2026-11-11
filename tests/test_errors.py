"""hippo.errors: error groups and buckets."""
import numpy as np
import pytest
import torch
import yaml

from hippo import errors
from hippo.data import make_splits
from hippo.model import build_model

from synthetic import make_fake_msd


def _label():
    lab = np.zeros((20, 20, 20), int)
    lab[5:15, 5:10, 5:15] = 1
    lab[5:15, 10:15, 5:15] = 2
    return lab


def test_error_groups():
    lab = _label()
    pred = lab.copy()
    pred[0:2, 0:2, 0:2] = 1                       # an island: 8 voxels
    pred[5:15, 5:15, 15] = 1                      # main piece one slice too long: 100 voxels
    pred[14, 5:15, 5:15] = 0                      # the outline stops one slice short: 100 voxels
    g = errors.error_groups(pred, lab, (1.0, 1.0, 1.0), parts=True)
    assert g["islands_ml"] == pytest.approx(0.008) and g["over_ml"] == pytest.approx(0.1)
    assert g["under_boundary_ml"] == pytest.approx(0.1) and g["under_chunks_ml"] == 0
    swapped = np.where(lab == 1, 2, np.where(lab == 2, 1, 0))
    assert errors.error_groups(swapped, lab, (1.0, 1.0, 1.0), parts=True)["split_ml"] == pytest.approx(1.0)
    assert errors.error_groups(swapped, lab, (1.0, 1.0, 1.0), parts=False)["split_ml"] == 0  # external: no parts
    chunk = lab.copy()
    chunk[5:15, 5:15, 5:9] = 0                    # a 4 mm deep region left out
    assert errors.error_groups(chunk, lab, (1.0, 1.0, 1.0), parts=True)["under_chunks_ml"] > 0


def test_bucket():
    groups = {"islands_ml": 0.3, "over_ml": 0.1, "under_boundary_ml": 0.05, "under_chunks_ml": 0.05, "split_ml": 0.0}
    assert errors.bucket(groups, 0.8, False) == ("false-positive islands", ["false-positive islands"])  # over is 20%
    assert errors.bucket(groups, 0.4, False)[0] == "complete failure (domain shift)"
    assert errors.bucket(groups, 0.9, True)[0] == "complete failure (domain shift)"   # nothing predicted


def test_main_writes_worst_cases(tmp_path, monkeypatch):
    data = tmp_path / "data"
    make_splits(make_fake_msd(data), data / "splits.json", 0.25, 0.25, seed=1)
    run = tmp_path / "run"
    run.mkdir()
    cfg = yaml.safe_load(open("configs/baseline.yaml"))
    cfg["model"]["channels"] = [8, 16, 32, 64]
    cfg["data"]["split_file"] = str(data / "splits.json")
    (run / "config.yaml").write_text(yaml.safe_dump(cfg))
    torch.save(build_model(cfg).state_dict(), run / "best.pt")
    rows = errors.main(str(run), worst=2, external_cases=str(tmp_path / "missing.json"))
    assert len(rows) == 2 and {r["test set"] for r in rows} == {"MSD test"}
    assert all(r["bucket"] in errors.BUCKETS.values() for r in rows)
    assert (run / "error_analysis" / "summary.md").read_text().startswith("# Error analysis")
