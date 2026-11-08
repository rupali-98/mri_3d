"""hippo.predict: masks on the image's own grid, label-free checks, no labels needed."""
import csv

import nibabel as nib
import numpy as np
import pytest
import torch
import yaml
from monai.transforms import Compose

from hippo import predict
from hippo.data import base_transforms, make_splits
from hippo.model import build_model

from synthetic import make_fake_msd


def _save_image(path, shape=(31, 41, 30), origin=(0.0, 0.0, 0.0)):
    affine = np.eye(4)
    affine[:3, 3] = origin
    data = np.ones(shape, np.float32)
    data[3, 5, 7] = 50                                   # a voxel whose place can be checked
    nib.save(nib.Nifti1Image(data, affine), path)
    return path


def _run(tmp_path):
    data = tmp_path / "data"
    root = make_fake_msd(data)
    make_splits(root, data / "splits.json", 0.25, 0.25, seed=1)
    (root / "imagesTs").mkdir()
    _save_image(root / "imagesTs" / "hippocampus_101.nii.gz")
    (root / "imagesTs" / "._hippocampus_101.nii.gz").write_bytes(b"macOS metadata")
    run = tmp_path / "run"
    run.mkdir()
    cfg = yaml.safe_load(open("configs/baseline.yaml"))
    cfg["model"]["channels"] = [8, 16, 32, 64]
    cfg["data"]["split_file"] = str(data / "splits.json")
    (run / "config.yaml").write_text(yaml.safe_dump(cfg))
    torch.save(build_model(cfg).state_dict(), run / "best.pt")
    return run, root


@pytest.mark.parametrize("origin", [(0.0, 0.0, 0.0), (12.0, -8.0, 3.0)])
def test_mask_returns_to_the_image_grid(tmp_path, origin):
    path = _save_image(tmp_path / "hippocampus_001.nii.gz", origin=origin)
    loaded = Compose(base_transforms("none", keys=("image",)))({"image": str(path)})["image"]
    assert loaded.shape[1:] == (32, 48, 32)              # padded, so cutting it back is really tested
    mask = (loaded.as_tensor()[0].numpy() == 50).astype(np.uint8)
    back = predict.to_image_grid(mask, loaded.affine, nib.load(path))
    assert back.shape == (31, 41, 30)
    assert back[3, 5, 7] == 1 and back.sum() == 1


def test_imagests_masks_and_csv_without_labels(tmp_path):
    run, root = _run(tmp_path)
    rows = predict.main(str(run))
    assert [r["case"] for r in rows] == ["hippocampus_101"]  # the "._" file is skipped
    saved = nib.load(run / "predictions_imagesTs" / "hippocampus_101.nii.gz")
    image = nib.load(root / "imagesTs" / "hippocampus_101.nii.gz")
    assert saved.shape == image.shape
    assert np.array_equal(saved.affine, image.affine)
    assert set(np.unique(np.asarray(saved.dataobj))) <= {0, 1, 2}
    with open(run / "predictions_imagesTs.csv") as f:
        assert next(csv.DictReader(f))["partner_in"] == "imagesTs"  # partner 102 is not in the fake splits


def test_test_set_is_predicted_too(tmp_path):
    run, _ = _run(tmp_path)
    assert len(predict.main(str(run), "test")) == 3
    assert (run / "predictions_test.csv").exists()


def test_flags():
    limits = {"vol_min": 2.0, "vol_max": 5.0, "margin_min": 3.0}
    good = {"vol_anterior_ml": 2.0, "vol_posterior_ml": 1.5, "vol_total_ml": 3.5, "pieces": 1, "edge_margin_mm": 4.0}
    assert predict.flags(good, limits) == ""
    assert "volume" in predict.flags({**good, "vol_total_ml": 6.0}, limits)
    assert "pieces" in predict.flags({**good, "pieces": 2}, limits)
    assert "edge" in predict.flags({**good, "edge_margin_mm": 1.0}, limits)
    assert "part missing" in predict.flags({**good, "vol_posterior_ml": 0.0}, limits)
    assert "edge" in predict.flags({**good, "edge_margin_mm": float("nan")}, limits)  # an empty mask


def test_partner_in():
    splits = {"train": [{"image": "a/hippocampus_002.nii.gz"}], "test": [{"image": "a/hippocampus_003.nii.gz"}]}
    assert predict.partner_in(1, splits) == "train"      # 1 pairs with 2
    assert predict.partner_in(4, splits) == "test"       # 4 pairs with 3
    assert predict.partner_in(6, splits) == "imagesTs"


def test_rejects_unknown_image_set(tmp_path):
    with pytest.raises(SystemExit, match="--images"):
        predict.main(str(tmp_path), "train")
