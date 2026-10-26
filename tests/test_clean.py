"""hippo.clean removes only small stray pieces, excludes bad cases and never touches raw data."""
import csv
import hashlib

import nibabel as nib
import numpy as np
import yaml

from hippo import clean, compare, explore
from hippo.data import make_splits, use_clean_labels

from synthetic import make_fake_msd


def test_remove_small_pieces():
    y = np.zeros((20, 20, 20), np.uint8)
    y[5:10, 5:10, 5:10] = 1        # main anterior piece (125 voxels)
    y[10, 10, 10] = 1              # touches the main piece only by a corner: same piece (26-neighbour)
    y[15, 15, 15] = 1              # floating single voxel: removed
    y[1:4, 14:17, 1:3] = 1         # separate 18-voxel piece: >= 10, kept and reported
    y[12:17, 2:7, 12:17] = 2       # posterior, one piece: untouched

    cleaned, removed, kept = clean.remove_small_pieces(y, min_voxels=10)

    assert cleaned[15, 15, 15] == 0
    assert cleaned[10, 10, 10] == 1
    assert (cleaned[1:4, 14:17, 1:3] == 1).all()
    assert (cleaned[5:10, 5:10, 5:10] == 1).all()
    assert np.array_equal(cleaned == 2, y == 2)
    assert removed == [{"part": "anterior", "pieces": 1, "voxels": 1}]
    assert kept == [{"part": "anterior", "sizes": [18]}]


def test_clean_end_to_end(tmp_path):
    data = tmp_path / "data"
    task = make_fake_msd(data, n=6)
    lbl = lambda i: task / "labelsTr" / f"hippocampus_{i:03d}.nii.gz"

    def rewrite(i, y):
        nib.save(nib.Nifti1Image(y, nib.load(lbl(i)).affine), lbl(i))

    y1 = np.asarray(nib.load(lbl(1)).dataobj).copy()
    y1[2, 2, 2] = 1                                              # stray island -> cleaned
    rewrite(1, y1)
    y2 = np.asarray(nib.load(lbl(2)).dataobj).copy()
    y2[y2 == 1] = 3                                              # invalid label -> FAIL -> excluded
    rewrite(2, y2)
    raw = {p: hashlib.md5(p.read_bytes()).hexdigest() for p in task.rglob("*.nii.gz")}

    cfg = yaml.safe_load(open("configs/baseline.yaml"))
    cfg["clean"]["exclude"] = ["hippocampus_003"]
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    explore.main(str(data), str(tmp_path / "explore"), str(cfg_path), figures=False)
    summary = clean.main(str(data), str(tmp_path / "explore"), str(tmp_path / "clean"), str(cfg_path))
    with open(tmp_path / "clean" / "cases.csv") as f:
        index = {r["case"]: r for r in csv.DictReader(f)}

    assert summary["included"] == 4 and summary["labels_changed"] == 1
    assert index["hippocampus_002"]["included"] == "False" and "label_values" in index["hippocampus_002"]["reason"]
    assert index["hippocampus_003"]["included"] == "False" and "clean.exclude" in index["hippocampus_003"]["reason"]
    assert "removed 1 voxels" in index["hippocampus_001"]["changes"]
    cleaned = nib.load(index["hippocampus_001"]["label"]).get_fdata()
    expected = y1.astype(float)
    expected[2, 2, 2] = 0                                        # only the island changed
    assert np.array_equal(cleaned, expected)
    assert index["hippocampus_000"]["label"] == "Task04_Hippocampus/labelsTr/hippocampus_000.nii.gz"  # unchanged case keeps original
    assert raw == {p: hashlib.md5(p.read_bytes()).hexdigest() for p in task.rglob("*.nii.gz")}   # raw data untouched


def test_use_clean_labels_changes_only_training(tmp_path):
    data = tmp_path / "data"
    task = make_fake_msd(data, n=12)
    cfg = yaml.safe_load(open("configs/baseline.yaml"))
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    explore.main(str(data), str(tmp_path / "explore"), str(cfg_path), figures=False)
    clean.main(str(data), str(tmp_path / "explore"), str(data / "clean"), str(cfg_path))
    splits = make_splits(task, data / "splits.json", 0.25, 0.25, seed=1)

    # pretend clean changed one training label and excluded another
    with open(data / "clean" / "cases.csv") as f:
        rows = list(csv.DictReader(f))
    changed, dropped = [c["image"].split("/")[-1].replace(".nii.gz", "") for c in splits["train"][:2]]
    for r in rows:
        if r["case"] == changed:
            r["label"] = f"clean/labelsTr/{changed}.nii.gz"
        if r["case"] == dropped:
            r["included"] = "False"
    with open(data / "clean" / "cases.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    out = use_clean_labels(splits, str(data))

    assert out["val"] == splits["val"] and out["test"] == splits["test"]
    assert len(out["train"]) == len(splits["train"]) - 1
    labels = {c["image"].split("/")[-1].replace(".nii.gz", ""): c["label"] for c in out["train"]}
    assert dropped not in labels
    assert labels[changed] == str(data / "clean" / "labelsTr" / f"{changed}.nii.gz")


def test_compare_is_paired(tmp_path):
    def write(run, dice):
        run.mkdir()
        with open(run / "metrics_test.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["case", "dice_anterior", "vol_pred_ml"])
            w.writeheader()
            w.writerows({"case": f"c{i}", "dice_anterior": d, "vol_pred_ml": 1.0} for i, d in enumerate(dice))

    write(tmp_path / "a", [0.80, 0.85, 0.90])
    write(tmp_path / "b", [0.80, 0.86, 0.88])
    write(tmp_path / "same", [0.80, 0.85, 0.90])

    r = compare.main(str(tmp_path / "a"), str(tmp_path / "b"))["metrics"]["dice_anterior"]
    assert np.isclose(r["mean_diff"], (0.01 - 0.02) / 3)
    assert (r["better"], r["worse"]) == (1, 1)
    assert compare.main(str(tmp_path / "a"), str(tmp_path / "same"))["metrics"]["dice_anterior"]["p_wilcoxon"] == 1.0
