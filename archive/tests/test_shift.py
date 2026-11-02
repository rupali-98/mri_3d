"""archive/shift.py moves the test crops without losing anatomy, keeps world positions, and scores every condition."""
import csv
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch
import yaml

from hippo.data import make_splits
from hippo.model import build_model

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "archive"))  # for shift
sys.path.insert(0, str(ROOT / "tests"))    # for synthetic and test_external's make_external
import shift  # noqa: E402
from synthetic import make_fake_msd  # noqa: E402


def test_conditions():
    names = [n for n, _ in shift.conditions([5, 10])]
    assert names == ["none", "x+5", "x-5", "x+10", "x-10", "y+5", "y-5", "y+10", "y-10",
                     "z+5", "z-5", "z+10", "z-10"]


def test_conditions_on_some_axes():
    assert [n for n, _ in shift.conditions([5, 10], "z")] == ["none", "z+5", "z-5", "z+10", "z-10"]


def test_named_and_combined_conditions():
    assert shift.conditions([], names=["z-5", "x-4_z-10"]) == [("none", []), ("z-5", [(2, -5.0)]),
                                                                ("x-4_z-10", [(0, -4.0), (2, -10.0)])]
    for bad in ["z10", "q-4", "x-4_x-5", "x-4z-10"]:
        with pytest.raises(ValueError):
            shift.parse(bad)


def test_external_combined_shift_moves_box_along_both_axes(tmp_path):
    from test_external import make_external
    from hippo import external

    src = tmp_path / "ext"
    make_external(src, subjects=("s01",))
    crops = {n: external.prepare(str(src), str(tmp_path / n), shift=moves)
             for n, moves in shift.conditions([], names=["y+3_z-2"])}
    centre = lambda n: np.argwhere(np.asarray(nib.load(crops[n][0]["label"]).dataobj) > 0).mean(0)
    assert np.allclose(centre("y+3_z-2") - centre("none"), [0, 3, -2])


def test_external_largest_piece_has_its_own_folder(tmp_path):
    from test_external import make_external

    src = tmp_path / "ext"
    make_external(src, subjects=("s01",))
    cfg = yaml.safe_load(open("configs/baseline.yaml"))
    cfg["model"]["channels"] = [8, 16, 32, 64]
    run = tmp_path / "run"
    run.mkdir()
    (run / "config.yaml").write_text(yaml.safe_dump(cfg))
    torch.save(build_model(cfg).state_dict(), run / "best.pt")

    results = shift.main_external([str(run)], str(src), str(tmp_path / "shifted_external"), shifts_mm=[3],
                                  axes="z", keep_largest=True)
    assert [r["condition"] for r in results["run"]] == ["none", "z+3", "z-3"]
    assert not (run / "shifted_external").exists()                 # plain results are not touched
    header = (run / "shifted_external_largest" / "z-3" / "metrics_external.csv").read_text().splitlines()[0]
    assert header.endswith(",pieces,removed_ml")


def test_shift_volume_moves_content_and_keeps_world_position():
    data = np.zeros((6, 7, 8), np.uint8)
    data[1, 2, 3] = 5
    affine = np.diag([2.0, 1.0, 1.0, 1.0])
    affine[:3, 3] = [10, 20, 30]
    img = nib.Nifti1Image(data, affine)

    out = shift.shift_volume(img, axis=0, shift_vox=3, pad=4)
    moved = np.asarray(out.dataobj)
    assert moved.shape == (14, 15, 16)                      # padded by 4 on every side
    assert np.argwhere(moved == 5).tolist() == [[1 + 4 + 3, 2 + 4, 3 + 4]]
    assert moved.sum() == 5                                 # nothing lost, nothing added
    world = lambda a, ijk: nib.affines.apply_affine(a, ijk)
    assert np.allclose(world(out.affine, [8, 6, 7]), world(affine, [1, 2, 3]))  # same anatomy, same place

    control = np.asarray(shift.shift_volume(img, axis=None, shift_vox=0, pad=4).dataobj)
    assert np.argwhere(control == 5).tolist() == [[5, 6, 7]]


def test_main_builds_and_scores_each_condition(tmp_path):
    data = tmp_path / "data"
    splits = make_splits(make_fake_msd(data), data / "splits.json", 0.25, 0.25, seed=1)
    cfg = yaml.safe_load(open("configs/baseline.yaml"))
    cfg["model"]["channels"] = [8, 16, 32, 64]
    cfg["data"]["split_file"] = str(data / "splits.json")
    run = tmp_path / "run"
    run.mkdir()
    (run / "config.yaml").write_text(yaml.safe_dump(cfg))
    torch.save(build_model(cfg).state_dict(), run / "best.pt")   # untrained: exercises the path only

    results = shift.main([str(run)], str(data), str(tmp_path / "shifted"), shifts_mm=[3])
    rows = {r["condition"]: r for r in results["run"]}
    assert list(rows) == ["none", "x+3", "x-3", "y+3", "y-3", "z+3", "z-3"]
    assert rows["none"]["dice_change"] == 0 and rows["none"]["p_wilcoxon"] == 1.0
    assert all(r["n"] == len(splits["test"]) for r in rows.values())
    assert not (run / "metrics_test.csv").exists()          # the run's own test results are not touched

    # each label keeps all its voxels after the shift (padding stops it falling off the edge)
    original = sum(int((np.asarray(nib.load(c["label"]).dataobj) > 0).sum()) for c in splits["test"])
    moved = sum(int((np.asarray(nib.load(p).dataobj) > 0).sum())
                for p in (tmp_path / "shifted" / "z-3" / "labelsTr").glob("*.nii.gz"))
    assert moved == original
    with open(run / "shifted" / "summary_shifted.csv") as f:
        assert len(list(csv.DictReader(f))) == 7


def test_external_box_moves_and_control_matches_hippo_external(tmp_path):
    from test_external import make_external
    from hippo import external

    src = tmp_path / "ext"
    make_external(src, subjects=("s01",))
    cfg = yaml.safe_load(open("configs/baseline.yaml"))
    cfg["model"]["channels"] = [8, 16, 32, 64]
    run = tmp_path / "run"
    run.mkdir()
    (run / "config.yaml").write_text(yaml.safe_dump(cfg))
    torch.save(build_model(cfg).state_dict(), run / "best.pt")

    results = shift.main_external([str(run)], str(src), str(tmp_path / "shifted_external"), shifts_mm=[3])
    rows = {r["condition"]: r for r in results["run"]}
    assert list(rows) == ["none", "x+3", "x-3", "y+3", "y-3", "z+3", "z-3"]
    assert rows["none"]["dice_change"] == 0 and all(r["n"] == 2 and r["min_coverage"] == 1.0 for r in rows.values())
    assert not (run / "external").exists()                 # the run's own external results are not touched

    # same anatomy, box moved 3 mm the other way: the hippocampus sits 3 voxels further along +y in the crop
    # (y, because the small synthetic volume leaves the x box against its edge, where it can't move)
    crop = lambda cond: nib.load(tmp_path / "shifted_external" / cond / "labelsTr" / "s01_L.nii.gz")
    centre = lambda img: np.argwhere(np.asarray(img.dataobj) > 0).mean(0)
    none, moved = crop("none"), crop("y+3")
    assert np.allclose(centre(moved) - centre(none), [0, 3, 0])
    assert np.allclose(moved.affine[:3, 3] - none.affine[:3, 3], [0, -3, 0])
    # the control is exactly the usual hippo.external crop
    usual = external.prepare(str(src), str(tmp_path / "usual"), ["s01"])
    assert np.array_equal(np.asarray(nib.load(usual[0]["label"]).dataobj), np.asarray(crop("none").dataobj))
