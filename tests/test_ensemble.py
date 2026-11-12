"""hippo.ensemble: an ensemble folder is scored like a run, with the fold models' probabilities averaged."""
import json

import pytest
import torch
import yaml

from hippo import ensemble, evaluate, tracking
from hippo.data import make_splits
from hippo.model import build_model

from synthetic import make_fake_msd


def _experiment(tmp_path, n_folds=2):
    """Fold folders with small untrained models (different weights) on fake data."""
    data = tmp_path / "data"
    make_splits(make_fake_msd(data, n=8), data / "splits.json", 0.25, 0.25, seed=1)
    cfg = yaml.safe_load(open("configs/baseline.yaml"))
    cfg["model"]["channels"] = [8, 16, 32, 64]
    cfg["data"]["split_file"] = str(data / "splits.json")
    exp = tmp_path / "runs" / "exp"
    for k in range(n_folds):
        fold = exp / f"fold{k}"
        fold.mkdir(parents=True)
        (fold / "config.yaml").write_text(yaml.safe_dump(cfg | {"data": cfg["data"] | {"fold": k}}))
        torch.manual_seed(k)
        torch.save(build_model(cfg).state_dict(), fold / "best.pt")
    return exp


def test_make_and_average(tmp_path):
    exp = _experiment(tmp_path)
    out = ensemble.make(str(exp))
    assert json.loads((out / "members.json").read_text()) == ["../fold0", "../fold1"]
    assert yaml.safe_load((out / "config.yaml").read_text())["data"]["fold"] == "ensemble"
    assert not (out / "best.pt").exists()                                   # no copied weights

    image = torch.randn(1, 1, 16, 16, 16)
    with torch.no_grad():
        mean = ensemble.load(out, "cpu")(image)
        parts = [ensemble.load(exp / f"fold{k}", "cpu")(image) for k in range(2)]
    assert torch.allclose(mean, (parts[0] + parts[1]) / 2, atol=1e-6)        # the mean of the folds' probabilities
    assert torch.allclose(mean.sum(1), torch.ones(1, 16, 16, 16), atol=1e-5)


def test_evaluate_takes_an_ensemble_folder(tmp_path):
    out = ensemble.make(str(_experiment(tmp_path)))
    summary = evaluate.main(str(out))
    assert summary["n_cases"] + summary["n_dropped"] == 2 and (out / "metrics_test.csv").exists()


def test_needs_trained_folds(tmp_path):
    exp = _experiment(tmp_path, n_folds=1)
    with pytest.raises(SystemExit, match="at least 2"):
        ensemble.make(str(exp))
    with pytest.raises(SystemExit, match="members.json"):
        ensemble.members(tmp_path / "nothing")


def test_mlflow_names_it_inside_its_experiment(tmp_path):
    assert tracking.experiment_name("runs/random_trim/ensemble") == "random_trim"
    assert tracking.inside_experiment("runs/random_trim/fold3") and not tracking.inside_experiment("runs/random_trim")
