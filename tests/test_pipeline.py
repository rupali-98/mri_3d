"""End-to-end smoke test: splits -> train 2 epochs -> evaluate, on synthetic data."""
import json

import yaml

from hippo import evaluate, train
from hippo.data import make_splits

from synthetic import make_fake_msd


def test_train_and_evaluate_end_to_end(tmp_path):
    root = make_fake_msd(tmp_path / "data")
    splits = make_splits(root, tmp_path / "data" / "splits.json", 0.25, 0.25, seed=1)

    # subject-level split: no file appears in two sets
    files = [c["image"] for s in splits.values() for c in s]
    assert len(files) == len(set(files)) == 12

    cfg = yaml.safe_load(open("configs/baseline.yaml"))
    cfg["data"].update(root=str(root), split_file=str(tmp_path / "data" / "splits.json"), num_workers=0)
    cfg["model"]["channels"] = [8, 16, 32, 64]
    cfg["train"].update(epochs=2, batch_size=2, val_every=1, out_dir=str(tmp_path / "run"))
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    result = train.main(str(cfg_path))
    assert (tmp_path / "run" / "best.pt").exists()
    assert len(result["history"]) == 2

    summary = evaluate.main(str(tmp_path / "run"))
    assert summary["n_cases"] == 3
    assert 0.0 <= summary["dice_anterior"]["mean"] <= 1.0
    assert (tmp_path / "run" / "metrics_test.csv").exists()
    json.loads((tmp_path / "run" / "summary_test.json").read_text())
