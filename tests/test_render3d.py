"""Masks saved by hippo.visualize line up with the MRI in hippo.render3d."""
import nibabel as nib
import numpy as np
import pytest

from hippo import render3d, visualize


def world_centroid(path, value):
    data, spacing, origin = render3d.load_ras(str(path))
    return np.argwhere(np.rint(data) == value).mean(0) * spacing + origin


def make_case(tmp_path):
    """Odd-sized volume (gets padded) stored with a flipped x axis (gets reoriented)."""
    shape, affine = (30, 37, 29), np.diag([-1.0, 1.0, 1.0, 1.0])
    affine[0, 3] = 29.0
    lbl = np.zeros(shape, np.uint8)
    lbl[5:12, 8:18, 10:20], lbl[5:12, 18:30, 10:20] = 1, 2
    img = lbl * 50.0 + 10.0
    nib.save(nib.Nifti1Image(img.astype(np.float32), affine), tmp_path / "img.nii.gz")
    nib.save(nib.Nifti1Image(lbl, affine), tmp_path / "lbl.nii.gz")
    return tmp_path / "img.nii.gz", tmp_path / "lbl.nii.gz"


def test_saved_mask_lines_up_with_original(tmp_path):
    image, label = make_case(tmp_path)
    # save the label exactly the way visualize --save-nifti saves predictions
    d = visualize.load(image, label)
    assert d["label"].shape[1:] != nib.load(label).shape  # padding really happened
    affine = np.asarray(d["image"].affine, dtype=np.float64)
    nib.save(nib.Nifti1Image(d["label"][0].numpy().astype(np.uint8), affine), tmp_path / "saved.nii.gz")

    for part in (1, 2):
        assert np.allclose(world_centroid(label, part), world_centroid(tmp_path / "saved.nii.gz", part))


def test_build_scenes(tmp_path):
    pytest.importorskip("vedo")
    image, label = make_case(tmp_path)
    assert len(render3d.build_scenes(str(image))) == 1

    scenes = render3d.build_scenes(str(image), str(label), str(label))
    assert len(scenes) == 2  # ground truth and prediction side by side
    for objects in scenes:
        assert len(objects) == 4  # MRI volume, anterior, posterior, text
