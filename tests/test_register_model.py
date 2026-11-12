"""hippo.register_model: packages in MLflow's registry, read back by alias, served by the API."""
import json
import shutil

import pytest
import torch
import yaml
from fastapi.testclient import TestClient
from mlflow.artifacts import download_artifacts

from hippo import register_model
from hippo.api import create_app
from hippo.data import make_splits
from hippo.model import build_model

from synthetic import make_fake_msd


def _experiment(tmp_path, seed=0):
    """A single trained run (untrained, small) on fake data; `seed` changes the weights."""
    data = tmp_path / "data"
    if not (data / "splits.json").exists():
        make_splits(make_fake_msd(data, n=8), data / "splits.json", 0.25, 0.25, seed=1)
    cfg = yaml.safe_load(open("configs/baseline.yaml"))
    cfg["model"]["channels"] = [8, 16, 32, 64]
    cfg["data"]["split_file"] = str(data / "splits.json")
    run = tmp_path / "runs" / "exp"
    run.mkdir(parents=True, exist_ok=True)
    (run / "config.yaml").write_text(yaml.safe_dump(cfg))
    torch.manual_seed(seed)
    torch.save(build_model(cfg).state_dict(), run / "best.pt")
    return run


@pytest.fixture
def store(tmp_path, monkeypatch):
    """A fresh MLflow database and a local folder standing in for the gs:// bucket."""
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    (tmp_path / "runs").mkdir()
    return {"root": str(tmp_path / "runs"), "artifacts": (tmp_path / "bucket").as_uri()}


def test_register_alias_and_show(tmp_path, store):
    run = _experiment(tmp_path)
    first = register_model.register(str(run), alias="champion", artifact_root=store["artifacts"], root=store["root"])
    assert first["version"] == 1 and first["uri"].startswith(store["artifacts"])
    assert register_model.show("champion", root=store["root"]) == first["catalog_uri"]          # the readable folder
    assert register_model.show("champion", field="source", root=store["root"]) == first["uri"]  # MLflow's copy
    assert register_model.show("champion", field="weights_version", root=store["root"]) == first["weights_version"]

    _experiment(tmp_path, seed=1)                                          # retrained: new weights
    second = register_model.register(str(run), root=store["root"])         # the experiment remembers its store
    assert second["version"] == 2 and second["weights_version"] != first["weights_version"]
    assert register_model.show("champion", field="version", root=store["root"]) == "1"   # alias not moved
    register_model.register(str(run), alias="champion", root=store["root"])
    assert register_model.show("champion", field="version", root=store["root"]) == "3"


def test_artifact_store_rules(tmp_path, store):
    run = _experiment(tmp_path)
    with pytest.raises(SystemExit, match="--artifact-root"):
        register_model.register(str(run), root=store["root"])              # first time needs the bucket
    register_model.register(str(run), artifact_root=store["artifacts"], root=store["root"])
    with pytest.raises(SystemExit, match="already stores its artifacts"):
        register_model.register(str(run), artifact_root=(tmp_path / "other").as_uri(), root=store["root"])
    with pytest.raises(SystemExit, match="alias"):
        register_model.show("champion", root=store["root"])


def test_downloaded_package_serves(tmp_path, store):
    """What the cloud build does: download the version's artifacts and serve them."""
    run = _experiment(tmp_path)
    reg = register_model.register(str(run), alias="champion", artifact_root=store["artifacts"], root=store["root"])
    fetched = tmp_path / "deploy"
    shutil.copytree(download_artifacts(artifact_uri=register_model.show("champion", root=store["root"])),
                    fetched / "model")
    assert {p.name for p in (fetched / "model").iterdir()} == {"best.pt", "config.yaml", "metadata.json", "MODEL_CARD.md"}
    with TestClient(create_app(str(fetched / "model"))) as client:
        assert client.get("/model").json()["version"] == reg["weights_version"]


def test_bucket_is_readable(tmp_path, store):
    """Readable folder per version, and an index at the store's root that follows the production alias."""
    run = _experiment(tmp_path)
    first = register_model.register(str(run), alias="champion", artifact_root=store["artifacts"], root=store["root"])
    assert first["uri"].endswith(f"/exp_single_{first['weights_version']}")
    _experiment(tmp_path, seed=1)
    second = register_model.register(str(run), alias="random_trim", root=store["root"])

    bucket = tmp_path / "bucket"
    index = json.loads((bucket / "registry.json").read_text())
    assert index["production_alias"] == "champion"
    rows = {r["version"]: r for r in index["versions"]}
    assert rows[1]["production"] and rows[1]["aliases"] == ["champion"]
    assert not rows[2]["production"] and rows[2]["aliases"] == ["random_trim"]
    assert rows[2]["experiment"] == "exp" and rows[2]["kind"] == "single run"
    assert "**PRODUCTION**" in (bucket / "REGISTRY.md").read_text()

    register_model.promote(2, "champion", root=store["root"])               # the index follows the alias
    rows = {r["version"]: r for r in json.loads((bucket / "registry.json").read_text())["versions"]}
    assert rows[2]["production"] and not rows[1]["production"]

    text = register_model.table(register_model.versions(root=store["root"]), live=first["weights_version"])
    line = next(l for l in text.splitlines() if l.strip().startswith("1 "))
    assert "LIVE" in line and "PRODUCTION" not in line                      # alias moved, service not redeployed yet
    assert second["uri"] in text


def test_kind_names():
    assert register_model.kind(1) == "single run" and register_model.kind("5") == "5-fold ensemble"
    assert register_model.folder_name({"experiment": "random_trim", "members": list("abcde"),
                                       "version": "5bf19bf02440"}) == "random_trim_5fold_5bf19bf02440"


def test_promote_rolls_back(tmp_path, store):
    run = _experiment(tmp_path)
    first = register_model.register(str(run), alias="champion", artifact_root=store["artifacts"], root=store["root"])
    _experiment(tmp_path, seed=1)
    register_model.register(str(run), alias="champion", root=store["root"])
    register_model.promote(1, "champion", root=store["root"])
    assert register_model.show("champion", root=store["root"]) == first["catalog_uri"]
    with pytest.raises(SystemExit, match="no version 9"):
        register_model.promote(9, "champion", root=store["root"])



def test_missing_bucket_stops_before_mlflow_writes(tmp_path, store, monkeypatch):
    run = _experiment(tmp_path)
    monkeypatch.setattr(register_model, "bucket_exists", lambda name: name == "real-bucket")
    with pytest.raises(SystemExit, match="gs://BUCKET doesn't exist"):
        register_model.register(str(run), artifact_root="gs://BUCKET/mlflow", root=store["root"])
    assert register_model.tracking.mlflow.get_experiment_by_name(register_model.EXPERIMENT) is None  # nothing created
    register_model.check_store("gs://real-bucket/mlflow")          # an existing bucket passes
    register_model.check_store(store["artifacts"])                  # local stores aren't checked

    def offline(name):
        raise OSError("Project was not passed and could not be determined from the environment.")
    monkeypatch.setattr(register_model, "bucket_exists", offline)
    with pytest.raises(SystemExit, match="Can't check the bucket gs://real-bucket: Project was not passed"):
        register_model.check_store("gs://real-bucket/mlflow")


def test_build_list(tmp_path, store):
    run = _experiment(tmp_path)
    first = register_model.register(str(run), alias="champion", artifact_root=store["artifacts"], root=store["root"])
    _experiment(tmp_path, seed=1)
    second = register_model.register(str(run), alias="random_trim", root=store["root"])
    value = register_model.build_list(["champion", "random_trim"], root=store["root"])
    assert value == (f"champion={first['catalog_uri']}={first['weights_version']};"
                     f"random_trim={second['catalog_uri']}={second['weights_version']}")
    with pytest.raises(SystemExit, match="letters, digits"):
        register_model.build_list(["bad name"], root=store["root"])
    with pytest.raises(SystemExit, match="alias 'missing'"):
        register_model.build_list(["missing"], root=store["root"])
