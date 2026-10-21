"""hippo.explore flags planted data problems and leaves clean cases alone."""
import csv
import json
import shutil

import nibabel as nib
import numpy as np

from hippo import explore

from synthetic import make_fake_msd


def edit(path, image=None, affine=None):
    """Rewrite a NIfTI file with new data and/or affine."""
    old = nib.load(path)
    data = np.asarray(old.dataobj) if image is None else image
    nib.save(nib.Nifti1Image(data, old.affine if affine is None else affine), path)


def test_explore_flags_planted_defects(tmp_path):
    task = make_fake_msd(tmp_path / "data", n=10)
    img = lambda i: task / "imagesTr" / f"hippocampus_{i:03d}.nii.gz"
    lbl = lambda i: task / "labelsTr" / f"hippocampus_{i:03d}.nii.gz"
    data = lambda p: np.asarray(nib.load(p).dataobj).copy()

    x = data(img(0)); x[5, 5, 5] = np.nan; edit(img(0), x)                  # NaN voxel
    edit(lbl(1), affine=np.diag([1.0, 1, 1, 1]) + np.eye(4, k=3) * 5)      # label shifted 5 mm
    y = data(lbl(2)); y[y == 1] = 3; edit(lbl(2), y)                       # invalid label value
    y = data(lbl(3)); y[y == 2] = 0; edit(lbl(3), y)                       # posterior missing
    y = data(lbl(4)); y[2, 2, 2] = 1; edit(lbl(4), y)                      # stray 1-voxel island
    edit(img(5), (data(img(5)) * 400).astype(np.float32))                  # different intensity scale
    shutil.copy(img(7), img(6))                                            # duplicate image

    summary = explore.main(str(tmp_path / "data"), str(tmp_path / "report"), "configs/baseline.yaml",
                           figures=False)
    with open(tmp_path / "report" / "cases.csv") as f:
        cases = {r["case"]: r for r in csv.DictReader(f)}
    status = lambda i: cases[f"hippocampus_{i:03d}"]["status"]
    issues = lambda i: cases[f"hippocampus_{i:03d}"]["issues"]

    assert status(0) == "FAIL" and "nonfinite" in issues(0)
    assert status(1) == "FAIL" and "label_position" in issues(1)
    assert status(2) == "FAIL" and "label_values" in issues(2)
    assert status(3) == "FAIL" and "missing_part" in issues(3)
    assert status(4) == "WARN" and "fragmented" in issues(4)
    assert cases["hippocampus_004"]["piece_sizes_anterior"].endswith("+1")  # island size is reported
    assert cases["hippocampus_005"]["intensity_group"] != cases["hippocampus_008"]["intensity_group"]
    assert "duplicate" in issues(6) + issues(7)
    assert status(8) == status(9) == "OK"

    assert summary["status"]["FAIL"] == 4
    assert len(summary["intensity_groups"]) == 2
    assert json.loads((tmp_path / "report" / "summary.json").read_text()) == summary


def test_explore_figures(tmp_path):
    make_fake_msd(tmp_path / "data", n=3)
    explore.main(str(tmp_path / "data"), str(tmp_path / "report"), "configs/baseline.yaml", figures=True)
    assert (tmp_path / "report" / "figures" / "distributions.png").stat().st_size > 0
