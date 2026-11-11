"""hippo.model_catalog: the readable catalogue, through register_model on a local folder standing in for the bucket."""
import json
from pathlib import Path

import pytest

from hippo import model_catalog, register_model

from test_register_model import _experiment, store  # noqa: F401  (fixture)


def _catalog(tmp_path):
    return tmp_path / "models" / "hippo-unet"


def test_layout_card_metadata_and_aliases(tmp_path, store):
    run = _experiment(tmp_path)
    first = register_model.register(str(run), alias="champion", artifact_root=store["artifacts"], root=store["root"])
    same = register_model.register(str(run), alias="copy", root=store["root"])           # same weights again
    _experiment(tmp_path, seed=1)
    other = register_model.register(str(run), alias="random_trim", root=store["root"])
    cat = _catalog(tmp_path)
    folder = f"exp_single_{first['weights_version']}"
    assert first["catalog_uri"] == (cat / folder).as_uri() == same["catalog_uri"]       # one folder per weights
    assert sorted(p.name for p in cat.iterdir()) == sorted(
        [".object_metadata.json", "README.md", "aliases", "catalog.json", folder, f"exp_single_{other['weights_version']}"])
    assert {p.name for p in (cat / folder).iterdir()} == {"MODEL_CARD.md", "best.pt", "config.yaml", "metadata.json"}

    card = (cat / folder / "MODEL_CARD.md").read_text()
    assert "**PRODUCTION**" in card and "| Registry versions | 1, 2 |" in card and "`champion`, `copy`" in card
    assert "Research and education only" in card and "None in the package (a single run)" in card

    meta = json.loads((cat / ".object_metadata.json").read_text())[f"{folder}/best.pt"]
    assert all(isinstance(v, str) for v in meta.values())                                  # Cloud Storage needs text
    assert {k: meta[k] for k in ("model", "registry-versions", "aliases", "production", "kind")} == {
        "model": "hippo-unet", "registry-versions": "1,2", "aliases": "champion,copy", "production": "true",
        "kind": "single run"}

    assert json.loads((cat / "aliases" / "champion.json").read_text())["folder"] == folder
    readme = (cat / "README.md").read_text()
    assert f"[{folder}]({folder}/MODEL_CARD.md)" in readme and "**PRODUCTION**" in readme
    assert len(json.loads((cat / "catalog.json").read_text())["models"]) == 2

    register_model.promote(3, "champion", root=store["root"])                             # production moves
    assert json.loads((cat / "aliases" / "champion.json").read_text())["version"] == 3
    assert "**PRODUCTION**" not in (cat / folder / "MODEL_CARD.md").read_text()
    assert json.loads((cat / ".object_metadata.json").read_text())[f"{folder}/best.pt"]["production"] == "false"


def test_removed_alias_file_goes_and_weights_are_not_uploaded_twice(tmp_path, store, monkeypatch):
    run = _experiment(tmp_path)
    register_model.register(str(run), alias="old", artifact_root=store["artifacts"], root=store["root"])
    cat = _catalog(tmp_path)
    assert (cat / "aliases" / "old.json").exists()
    register_model.tracking.mlflow.MlflowClient().delete_registered_model_alias("hippo-unet", "old")
    uploads = []
    original = model_catalog.LocalStore.upload
    monkeypatch.setattr(model_catalog.LocalStore, "upload", lambda self, *a: uploads.append(a[1]) or original(self, *a))
    register_model.publish_catalog(root=store["root"])
    assert not (cat / "aliases" / "old.json").exists()                                    # stale alias removed
    assert uploads == []                                                                  # weights already there


@pytest.mark.parametrize("location, expected", [
    ("gs://bucket/mlflow", "gs://bucket/models/m"),
    ("gs://bucket", "gs://bucket/models/m"),
    ("gs://bucket/a/b", "gs://bucket/a/models/m"),
    ("file:///tmp/x/bucket", "file:///tmp/x/models/m"),
])
def test_catalog_root(location, expected):
    assert model_catalog.catalog_root(location, "m") == expected


def test_row_folder():
    row = {"kind": "5-fold ensemble", "experiment": "random_trim", "weights_version": "a2d551dad98a"}
    assert model_catalog.row_folder(row) == "random_trim_5fold_a2d551dad98a"
    assert model_catalog.row_folder(row | {"kind": "single run"}) == "random_trim_single_a2d551dad98a"
    assert model_catalog.row_folder(row | {"experiment": "?"}) is None


def test_gcs_store_paths_and_metadata():
    """GcsStore against a fake client: object names under the prefix, metadata set before upload."""
    class Blob:
        def __init__(self, bucket, name):
            self.bucket, self.name, self.metadata = bucket, name, None
        def upload_from_string(self, text, content_type=None):
            self.bucket.objects[self.name] = (text, dict(self.metadata or {}), content_type)
        def upload_from_filename(self, path):
            self.bucket.objects[self.name] = (Path(path).read_bytes(), dict(self.metadata or {}), None)
        def exists(self):
            return self.name in self.bucket.objects
        def download_as_text(self):
            return self.bucket.objects[self.name][0]
        def patch(self):
            data, _, ctype = self.bucket.objects[self.name]
            self.bucket.objects[self.name] = (data, dict(self.metadata), ctype)
        def delete(self):
            del self.bucket.objects[self.name]

    class Bucket:
        objects = {}
        def blob(self, name):
            return Blob(self, name)
        get_blob = blob

    class Client:
        bucket_ = Bucket()
        def bucket(self, name):
            assert name == "hippo-models-rupali"
            return self.bucket_
        def list_blobs(self, bucket, prefix):
            return [Blob(bucket, n) for n in bucket.objects if n.startswith(prefix)]

    store = model_catalog.GcsStore("gs://hippo-models-rupali/models/hippo-unet", client=Client())
    store.write_text("# hi", "README.md", {"model": "hippo-unet"})
    store.write_text("{}", "aliases/champion.json", {"alias": "champion"})
    objects = Client.bucket_.objects
    assert objects["models/hippo-unet/README.md"] == ("# hi", {"model": "hippo-unet"}, "text/markdown")
    assert objects["models/hippo-unet/aliases/champion.json"][2] == "application/json"
    assert store.list("aliases/") == ["aliases/champion.json"] and store.exists("README.md")
    store.set_metadata("README.md", {"model": "hippo-unet", "production": "true"})
    assert objects["models/hippo-unet/README.md"][1]["production"] == "true"
    assert store.uri("README.md") == "gs://hippo-models-rupali/models/hippo-unet/README.md"
    store.delete("aliases/champion.json")
    assert store.list("aliases/") == []
