"""data.normalize: zscore rescales each scan, none keeps raw intensities. data.random_trim: random sub-crop in training."""
import json

import nibabel as nib
import numpy as np
import pytest
import torch

from hippo.data import RandomTrimd, eval_transforms, fold_file, make_folds, make_loaders, make_splits, train_transforms

from synthetic import make_fake_msd


def test_normalize_options(tmp_path):
    task = make_fake_msd(tmp_path / "data", n=1)
    case = {"image": str(task / "imagesTr" / "hippocampus_000.nii.gz"),
            "label": str(task / "labelsTr" / "hippocampus_000.nii.gz")}
    raw = np.asarray(nib.load(case["image"]).dataobj, dtype=np.float32)

    kept = eval_transforms("none")(case)["image"].numpy()
    assert np.isclose(kept.max(), raw.max())             # padding adds zeros only; values unchanged

    z = eval_transforms("zscore")(case)["image"].numpy()
    assert z.mean() == pytest.approx(0.0, abs=0.1)        # ~0 and ~1, not exact: padding adds a few zeros
    assert z.std() == pytest.approx(1.0, abs=0.2)

    with pytest.raises(SystemExit, match="data.normalize"):
        eval_transforms("minmax")


def _trim_case(first=8, last=11, size=20):
    """Channel-first image and label along z; the image holds its own z index, so a cut can be traced."""
    label = np.zeros((1, 4, 4, size), np.uint8)
    label[:, :, :, first:last + 1] = 1
    image = np.broadcast_to(np.arange(size, dtype=np.float32), (1, 4, 4, size)).copy()
    return {"image": torch.from_numpy(image), "label": torch.from_numpy(label)}


def test_random_trim_moves_the_label_and_keeps_it_whole():
    trim = RandomTrimd(keys=["image", "label"], trim=(0, 0, 5)).set_random_state(seed=0)
    starts = set()
    for _ in range(100):
        out = trim(_trim_case())
        assert out["image"].shape == (1, 4, 4, 15)              # only z shrinks
        assert out["label"].sum() == 4 * 4 * 4                  # label voxels all kept
        start = int(out["image"][0, 0, 0, 0])
        starts.add(start)
        assert np.array_equal(np.nonzero(np.asarray(out["label"][0, 0, 0]))[0], np.arange(8, 12) - start)  # image and label cut alike
    assert starts == {0, 1, 2, 3, 4, 5}                         # every position is reachable


def test_random_trim_never_cuts_label_in_a_tight_crop():
    trim = RandomTrimd(keys=["image", "label"], trim=(0, 0, 5)).set_random_state(seed=0)
    for _ in range(50):
        out = trim(_trim_case(first=1, last=17, size=20))       # margins 1 and 2: only 3 can go
        assert out["image"].shape[-1] == 17
        assert out["label"].sum() == 4 * 4 * 17


def test_random_trim_is_off_unless_asked_for():
    has_trim = lambda compose: any(isinstance(t, RandomTrimd) for t in compose.transforms)
    assert not has_trim(train_transforms())
    assert not has_trim(train_transforms(random_trim=[0, 0, 0]))
    assert has_trim(train_transforms(random_trim=[0, 0, 5]))
    assert not has_trim(eval_transforms())                       # validation and test always see whole crops


def test_random_trim_batches_of_different_sizes_load(tmp_path):
    task = make_fake_msd(tmp_path / "data", n=4)
    cases = [{"image": str(task / "imagesTr" / f"hippocampus_{i:03d}.nii.gz"),
              "label": str(task / "labelsTr" / f"hippocampus_{i:03d}.nii.gz")} for i in range(4)]
    train_loader, _ = make_loaders({"train": cases, "val": cases[:1]}, 2, 0, 1.0, random_trim=[0, 0, 5])
    batch = next(iter(train_loader))
    assert batch["image"].shape[-1] % 8 == 0                     # padded after the trim, so the U-Net can divide it
    assert batch["image"].shape[-1] < 32 + 8


def test_folds_cover_train_and_val_once_and_keep_test(tmp_path):
    task = make_fake_msd(tmp_path / "data", n=12)
    split_file = tmp_path / "data" / "splits.json"
    splits = make_splits(task, split_file, 0.25, 0.25, seed=1)
    folds = make_folds(split_file, 3, seed=1)
    pool = sorted(c["image"] for c in splits["train"] + splits["val"])
    vals = [c["image"] for f in folds for c in f["val"]]
    assert sorted(vals) == pool                                  # every non-test case validates exactly once
    for i, f in enumerate(folds):
        assert not {c["image"] for c in f["train"]} & {c["image"] for c in f["val"]}
        assert sorted(c["image"] for c in f["train"] + f["val"]) == pool
        assert f["test"] == splits["test"]                       # the test split is the same in every fold
        assert json.loads(fold_file(tmp_path / "data", i).read_text()) == f
    assert make_folds(split_file, 3, seed=1) == folds            # fixed: the same folds every time
