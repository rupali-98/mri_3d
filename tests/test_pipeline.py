"""End-to-end smoke test: splits -> train 2 epochs -> evaluate, on synthetic data."""
import csv
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch
import yaml

from hippo import evaluate, train
from hippo.data import make_splits
from hippo.model import build_model

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

    # file-level split: no file appears in two sets
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


def set_labels(cfg_path, labels):
    cfg = yaml.safe_load(cfg_path.read_text())
    cfg["data"]["labels"] = labels
    cfg_path.write_text(yaml.safe_dump(cfg))


class StopBeforeTraining(Exception):
    pass


def test_train_uses_clean_labels(tmp_path, monkeypatch):
    data = tmp_path / "data"
    make_splits(make_fake_msd(data), data / "splits.json", 0.25, 0.25, seed=1)
    splits = json.loads((data / "splits.json").read_text())
    name = lambda c: c["image"].split("/")[-1].replace(".nii.gz", "")
    changed, dropped = name(splits["train"][0]), name(splits["train"][1])

    # a minimal hippo.clean index: one training label replaced, one case excluded
    (data / "clean").mkdir()
    with open(data / "clean" / "cases.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["case", "included", "label"])
        w.writeheader()
        for c in sum(splits.values(), []):
            label = f"clean/labelsTr/{name(c)}.nii.gz" if name(c) == changed else c["label"]
            w.writerow({"case": name(c), "included": name(c) != dropped, "label": label})

    # catch what train.main hands to the loaders, then stop: no need to train to check the wiring
    seen = {}

    def capture(splits, *args, **kwargs):
        seen.update(splits)
        raise StopBeforeTraining

    monkeypatch.setattr(train, "make_loaders", capture)
    cfg_path = small_config(tmp_path)
    set_labels(cfg_path, "clean")
    with pytest.raises(StopBeforeTraining):
        train.main(str(cfg_path), data_dir=str(data))

    labels = {name(c): c["label"] for c in seen["train"]}
    assert len(labels) == len(splits["train"]) - 1 and dropped not in labels
    assert labels[changed] == str(data / "clean" / "labelsTr" / f"{changed}.nii.gz")
    assert seen["val"] == splits["val"] and seen["test"] == splits["test"]
    assert yaml.safe_load((tmp_path / "run" / "config.yaml").read_text())["data"]["labels"] == "clean"

    # raw (the default) leaves the training labels alone
    set_labels(cfg_path, "raw")
    with pytest.raises(StopBeforeTraining):
        train.main(str(cfg_path), data_dir=str(data))
    assert seen["train"] == splits["train"]


def test_train_rejects_unknown_labels(tmp_path):
    data = tmp_path / "data"
    make_splits(make_fake_msd(data), data / "splits.json", 0.25, 0.25, seed=1)
    cfg_path = small_config(tmp_path)
    set_labels(cfg_path, "cleaned")
    with pytest.raises(SystemExit, match="raw or clean"):
        train.main(str(cfg_path), data_dir=str(data))


@pytest.mark.parametrize("normalize, expected", [("zscore", "zscore"), ("none", "none"), (None, "zscore")])
def test_evaluate_uses_run_normalize(tmp_path, monkeypatch, normalize, expected):
    data = tmp_path / "data"
    make_splits(make_fake_msd(data), data / "splits.json", 0.25, 0.25, seed=1)
    cfg = yaml.safe_load(open("configs/baseline.yaml"))
    cfg["model"]["channels"] = [8, 16, 32, 64]
    cfg["data"]["split_file"] = str(data / "splits.json")
    cfg["data"].pop("normalize", None)
    if normalize is not None:  # None: a run from before the setting existed
        cfg["data"]["normalize"] = normalize
    run = tmp_path / "run"
    run.mkdir()
    (run / "config.yaml").write_text(yaml.safe_dump(cfg))
    torch.save(build_model(cfg).state_dict(), run / "best.pt")   # untrained: only the preprocessing matters

    seen, real = [], evaluate.eval_transforms
    monkeypatch.setattr(evaluate, "eval_transforms", lambda n: seen.append(n) or real(n))
    evaluate.main(str(run))
    assert seen == [expected]


def test_evaluate_predicts_but_does_not_score_case_with_missing_part(tmp_path):
    data = tmp_path / "data"
    splits = make_splits(make_fake_msd(data), data / "splits.json", 0.25, 0.25, seed=1)
    gone = splits["test"][0]["label"]                # remove the posterior from one test label
    y = nib.load(gone)
    nib.save(nib.Nifti1Image(np.where(np.asarray(y.dataobj) == 2, 0, y.dataobj).astype(np.uint8), y.affine), gone)
    cfg = yaml.safe_load(open("configs/baseline.yaml"))
    cfg["model"]["channels"] = [8, 16, 32, 64]
    cfg["data"]["split_file"] = str(data / "splits.json")
    run = tmp_path / "run"
    run.mkdir()
    (run / "config.yaml").write_text(yaml.safe_dump(cfg))
    torch.save(build_model(cfg).state_dict(), run / "best.pt")

    summary = evaluate.main(str(run))
    case = Path(gone).name
    assert (summary["n_cases"], summary["n_dropped"]) == (2, 1)
    assert summary["dropped"] == {case: "no posterior in the ground truth"}
    assert all(np.isfinite(v["mean"]) for k, v in summary.items() if k.startswith(("dice", "nsd")))
    with open(run / "metrics_test.csv") as f:
        rows = {r["case"]: r for r in csv.DictReader(f)}
    assert len(rows) == 3                             # every test case was predicted
    assert rows[case]["dice_posterior"] == "" and rows[case]["nsd_anterior"] == ""
    assert rows[case]["vol_pred_ml_anterior"] != "" and float(rows[case]["vol_true_ml_posterior"]) == 0


def test_refuses_to_start_when_disk_is_full(tmp_path):
    cfg_path = small_config(tmp_path, min_free_disk_gb=10**9)
    with pytest.raises(SystemExit, match="GB free"):
        train.main(str(cfg_path), data_dir=str(tmp_path / "data"))
