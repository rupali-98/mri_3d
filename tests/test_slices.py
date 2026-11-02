"""hippo.slices draws Slicer-oriented slices, lines masks up by world position and classifies errors."""
import io

import nibabel as nib
import numpy as np
from PIL import Image

from hippo import slices


def save(path, data, origin=(0.0, 0.0, 0.0), spacing=1.0):
    affine = np.diag([spacing, spacing, spacing, 1.0])
    affine[:3, 3] = origin
    nib.save(nib.Nifti1Image(data, affine), path)
    return path


def test_on_grid_places_crop_by_position(tmp_path):
    crop = np.zeros((4, 4, 4), np.uint8)
    crop[1, 2, 3] = 2
    path = save(tmp_path / "crop.nii.gz", crop, origin=(10.0, 20.0, 30.0))
    out = slices.on_grid(path, (40, 40, 40), np.ones(3), np.zeros(3))
    assert out.shape == (40, 40, 40) and out.sum() == 2 and out[11, 22, 33] == 2   # outside the crop: 0


def test_surfaces_in_world_mm():
    labels = np.zeros((10, 10, 10), np.int32)
    labels[2:5, 3:6, 4:8] = 1                   # a box of voxels 2-4, 3-5, 4-7
    labels[0:2, 0:2, 0:2] = 2                   # touches the edge: still a closed surface
    out = slices.surfaces(labels, (2.0, 1.0, 1.0), (100.0, 0.0, -50.0),
                          {1: ("anterior", "orangered"), 2: ("posterior", "steelblue"), 3: ("none", "red")})
    assert [(name, colour) for name, colour, _, _ in out] == [("anterior", "#ff4500"), ("posterior", "#4682b4")]
    verts = out[0][2]
    # the surface sits half a voxel outside the voxel centres, at their world position
    assert np.allclose(verts.min(0), [100 + 2 * 1.5, 2.5, -50 + 3.5]) and np.allclose(verts.max(0), [100 + 2 * 4.5, 5.5, -50 + 7.5])
    assert out[1][2].min(0).tolist() == [100 - 1.0, -0.5, -50 - 0.5]


def test_error_map():
    truth = np.array([1, 1, 2, 0, 2, 0])
    pred = np.array([1, 2, 0, 1, 2, 0])
    assert slices.error_map(pred, truth).tolist() == [1, 2, 3, 4, 1, 0]
    assert slices.error_map(pred, truth, parts=False).tolist() == [1, 1, 3, 4, 1, 0]   # whole hippocampus only


def test_render_radiological_orientation(tmp_path):
    img = np.zeros((20, 10, 6), np.float32)
    img[15:, :, :] = 100                     # high x = patient's right
    img[:, 8:, :] += 50                      # high y = anterior
    scan = slices.Scan(save(tmp_path / "img.nii.gz", img, spacing=2.0))
    png = scan.render(slices.VIEWS[0], scan.start, contrast=(0, 100), crosshair=False, scale=1.0)
    axial = np.asarray(Image.open(io.BytesIO(png)).convert("L"))
    assert axial.shape == (20, 40)           # rows = y (10 x 2 mm), columns = x (20 x 2 mm)
    assert axial[12, 2] > axial[12, 30]      # right on the screen's left
    assert axial[12, 30] > axial[-2, 30]     # anterior at the top
