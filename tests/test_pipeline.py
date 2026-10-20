"""End-to-end smoke test: splits -> train 2 epochs -> evaluate, on synthetic data."""
import json

import pytest
import torch
import yaml

from hippo import evaluate, train
from hippo.data import make_splits

from synthetic import make_fake_msd


def small_config(tmp_path, run="run", **train):
    cfg = yaml.safe_load(open("configs/baseline.yaml"))
    cfg["data"]["num_workers"] = 0
    cfg["model"]["channels"] = [8, 16, 32, 64]
    cfg["train"].update(epochs=2, batch_size=2, val_every=1, out_dir=str(tmp_path / run), min_free_disk_gb=0)
    cfg["train"].update(train)
    cfg_path = tmp_path / f"{run}.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    return cfg_path


def test_train_and_evaluate_end_to_end(tmp_path):
    root = make_fake_msd(tmp_path / "data")
    splits = make_splits(root, tmp_path / "data" / "splits.json", 0.25, 0.25, seed=1)

    # subject-level split: no file appears in two sets
    files = [c["image"] for s in splits.values() for c in s]
    assert len(files) == len(set(files)) == 12

    cfg_path = small_config(tmp_path)
    result = train.main(str(cfg_path), data_dir=str(tmp_path / "data"))
    assert (tmp_path / "run" / "best.pt").exists()
    assert len(result["history"]) == 2

    summary = evaluate.main(str(tmp_path / "run"))
    assert summary["n_cases"] == 3
    assert 0.0 <= summary["dice_anterior"]["mean"] <= 1.0
    assert (tmp_path / "run" / "metrics_test.csv").exists()
    json.loads((tmp_path / "run" / "summary_test.json").read_text())


def test_resume_after_crash(tmp_path, monkeypatch):
    root = make_fake_msd(tmp_path / "data")
    make_splits(root, tmp_path / "data" / "splits.json", 0.25, 0.25, seed=1)
    cfg_path = small_config(tmp_path)
    run = tmp_path / "run"

    # crash during epoch 2's validation, after epoch 1's checkpoint was written
    calls, real_validate = [], train.validate

    def crash_on_second_call(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("simulated crash")
        return real_validate(*args, **kwargs)

    monkeypatch.setattr(train, "validate", crash_on_second_call)
    with pytest.raises(RuntimeError):
        train.main(str(cfg_path), data_dir=str(tmp_path / "data"))
    assert (run / "checkpoint.pt").exists()
    assert [h["epoch"] for h in json.loads((run / "history.json").read_text())] == [1]

    monkeypatch.setattr(train, "validate", real_validate)
    result = train.main(str(cfg_path), data_dir=str(tmp_path / "data"), resume=True)
    assert [h["epoch"] for h in result["history"]] == [1, 2]
    assert (run / "last.pt").exists()
    assert not (run / "checkpoint.pt").exists()

    # the resumed run ends with exactly the weights of a run that was never interrupted
    train.main(str(small_config(tmp_path, run="uninterrupted")), data_dir=str(tmp_path / "data"))
    resumed = torch.load(run / "last.pt", weights_only=True)
    reference = torch.load(tmp_path / "uninterrupted" / "last.pt", weights_only=True)
    assert all(torch.equal(resumed[k], reference[k]) for k in reference)


def test_refuses_to_start_when_disk_is_full(tmp_path):
    cfg_path = small_config(tmp_path, min_free_disk_gb=10**9)
    with pytest.raises(SystemExit, match="GB free"):
        train.main(str(cfg_path), data_dir=str(tmp_path / "data"))
