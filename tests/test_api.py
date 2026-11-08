"""hippo.api and hippo.package_model: the HTTP service on a tiny untrained model and fake data."""
import base64
import gzip
import json

import nibabel as nib
import numpy as np
import pytest
import torch
import yaml
from fastapi.testclient import TestClient

from hippo import package_model
from hippo.api import create_app
from hippo.data import make_splits
from hippo.model import build_model

from synthetic import make_fake_msd


def _package(tmp_path, folds=False, boxes=True):
    """A packaged model (untrained, small) plus the fake MSD task it was 'trained' on."""
    data = tmp_path / "data"
    task = make_fake_msd(data, n=8)
    make_splits(task, data / "splits.json", 0.25, 0.25, seed=1)
    cfg = yaml.safe_load(open("configs/baseline.yaml"))
    cfg["model"]["channels"] = [8, 16, 32, 64]
    cfg["data"]["split_file"] = str(data / "splits.json")
    runs = [tmp_path / "exp"]
    if folds:  # two fold folders; the packager only needs their weights and configs
        runs = [tmp_path / "exp" / f"fold{k}" for k in range(2)]
    for run in runs:
        run.mkdir(parents=True)
        (run / "config.yaml").write_text(yaml.safe_dump(cfg))
        torch.save(build_model(cfg).state_dict(), run / "best.pt")
    out = tmp_path / "model"
    meta = package_model.main(str(tmp_path / "exp"), out=str(out))
    if boxes:  # what --src would add: one 16 x 24 x 16 mm box per side, inside the fake brain below
        meta["mni_boxes"] = {"L": {"lo_mm": [10.0, 10.0, 10.0], "hi_mm": [25.0, 33.0, 25.0], "size_vox": [16, 24, 16]},
                             "R": {"lo_mm": [40.0, 10.0, 10.0], "hi_mm": [55.0, 33.0, 25.0], "size_vox": [16, 24, 16]}}
        (out / "metadata.json").write_text(json.dumps(meta))
    return out, task


def _nifti(shape=(32, 40, 32), zooms=(1.0, 1.0, 1.0)) -> bytes:
    data = np.random.default_rng(0).random(shape).astype(np.float32)
    image = nib.Nifti1Image(data, np.diag([*zooms, 1.0]))
    return gzip.compress(image.to_bytes())


def _post(client, path, content, **params):
    return client.post(path, files={"file": ("scan.nii.gz", content)}, params=params)


def test_package_holds_what_serving_needs(tmp_path):
    out, _ = _package(tmp_path, folds=True, boxes=False)
    meta = json.loads((out / "metadata.json").read_text())
    assert meta["members"] == ["fold0", "fold1"] and meta["combination"] == "mean of class probabilities"
    assert len(meta["version"]) == 12 and meta["normalize"] == "zscore"
    assert {"vol_min", "vol_max", "margin_min"} <= set(meta["training_label_range"])
    assert "Research" in meta["intended_use"]
    assert "split_file" not in yaml.safe_load((out / "fold0" / "config.yaml").read_text())["data"]  # no local paths


def test_single_run_option(tmp_path):
    _package(tmp_path, folds=True, boxes=False)
    (tmp_path / "exp" / "best.pt").write_bytes((tmp_path / "exp" / "fold0" / "best.pt").read_bytes())
    (tmp_path / "exp" / "config.yaml").write_text((tmp_path / "exp" / "fold0" / "config.yaml").read_text())
    meta = package_model.main(str(tmp_path / "exp"), out=str(tmp_path / "single"), single=True)
    assert meta["members"] == ["."] and meta["combination"] == "single model" and meta["cross_validation"] == []


def test_health_model_and_crop(tmp_path):
    out, task = _package(tmp_path)
    with TestClient(create_app(str(out))) as client:
        assert client.get("/health").json() == {"status": "ok"}
        info = client.get("/model").json()
        assert info["name"] == "3D U-Net" and info["endpoints"] == {"crop": True, "mni": True}
        upload = (task / "imagesTr" / "hippocampus_000.nii.gz").read_bytes()
        r = _post(client, "/segment/crop", upload, include_mask="true")
        assert r.status_code == 200
        body = r.json()
        assert body["model"]["version"] == info["version"] and "not for diagnosis" in body["intended_use"]
        h = body["hippocampi"][0]
        assert set(h["volumes_ml"]) == {"anterior", "posterior", "whole"}
        assert {"pieces", "edge_margin_mm", "flags", "needs_review"} <= set(h["qc"])
        mask = nib.Nifti1Image.from_bytes(gzip.decompress(base64.b64decode(h["mask_nifti_gz_base64"])))
        scan = nib.load(task / "imagesTr" / "hippocampus_000.nii.gz")
        assert mask.shape == scan.shape and np.allclose(mask.affine, scan.affine)   # lines up with the upload
        assert h["volumes_ml"]["whole"] == pytest.approx((np.asarray(mask.dataobj) > 0).sum() / 1000, abs=1e-3)


@pytest.mark.parametrize("content, why", [
    (_nifti(shape=(130, 40, 32)), "too large for a hippocampus crop"),
    (_nifti(zooms=(2.0, 2.0, 2.0)), "voxel size"),
    (gzip.compress(nib.Nifti1Image(np.zeros((8, 8, 8, 2), np.float32), np.eye(4)).to_bytes()), "3D image"),
    (b"not an image", "not a readable NIfTI"),
])
def test_crop_refuses_what_the_model_was_not_made_for(tmp_path, content, why):
    out, _ = _package(tmp_path)
    with TestClient(create_app(str(out))) as client:
        r = _post(client, "/segment/crop", content)
    assert r.status_code == 422 and why in json.dumps(r.json()["detail"])


def test_upload_limit_and_api_key(tmp_path):
    out, task = _package(tmp_path)
    upload = (task / "imagesTr" / "hippocampus_000.nii.gz").read_bytes()
    with TestClient(create_app(str(out), max_mb=1e-4)) as client:
        assert _post(client, "/segment/crop", upload).status_code == 413
    with TestClient(create_app(str(out), api_key="secret")) as client:
        assert client.get("/health").status_code == 200                     # liveness stays open
        assert client.get("/model").status_code == 401
        assert _post(client, "/segment/crop", upload).status_code == 401
        assert client.get("/model", headers={"X-API-Key": "secret"}).status_code == 200


def test_mni_scan_gives_both_sides(tmp_path):
    out, _ = _package(tmp_path)
    brain = _nifti(shape=(64, 48, 40))
    with TestClient(create_app(str(out))) as client:
        r = _post(client, "/segment/mni", brain, include_mask="true")
        assert r.status_code == 200
        body = r.json()
        assert [h["side"] for h in body["hippocampi"]] == ["left", "right"]
        assert "not checked" in body["warnings"][0] and body["post_processing"] == "none"
        mask = nib.Nifti1Image.from_bytes(gzip.decompress(base64.b64decode(body["hippocampi"][0]["mask_nifti_gz_base64"])))
        assert mask.shape == (16, 24, 16) and np.allclose(mask.affine[:3, 3], [10, 10, 10])  # the left box, in place
        largest = _post(client, "/segment/mni", brain, largest="true").json()
        assert largest["post_processing"] == "largest piece"
        assert all(h["qc"]["pieces"] <= 1 and "removed_ml" in h for h in largest["hippocampi"])
        # a crop is not a whole-brain scan: the boxes don't fit inside it
        r = _post(client, "/segment/mni", _nifti(shape=(32, 40, 32)))
        assert r.status_code == 422 and "outside the image" in r.json()["detail"]


def test_mni_needs_the_boxes(tmp_path):
    out, _ = _package(tmp_path, boxes=False)
    with TestClient(create_app(str(out))) as client:
        assert client.get("/model").json()["endpoints"]["mni"] is False
        assert _post(client, "/segment/mni", _nifti(shape=(64, 48, 40))).status_code == 501
