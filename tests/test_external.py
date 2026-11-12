"""hippo.external crops each hippocampus around the atlas, keeps it in place, and scores the whole hippocampus."""
import csv
import json

import nibabel as nib
import numpy as np
import pytest
import torch
import yaml

from hippo import external
from hippo.model import build_model

SHAPE = (80, 90, 80)
AFFINE = np.array([[1.0, 0, 0, -40], [0, 1, 0, -50], [0, 0, 1, -40], [0, 0, 0, 1]])  # 1 mm, MNI-like origin


def ellipsoid(centre, radii):
    g = np.mgrid[tuple(slice(0, n) for n in SHAPE)]
    return sum(((g[i] - centre[i]) / radii[i]) ** 2 for i in range(3)) <= 1


def make_external(root, subjects=("s01", "s02")):
    """Two hippocampi per subject (left/right), three subfields each, and matching SPAM maps."""
    root.mkdir(parents=True)
    centres = {"L": (20, 40, 30), "R": (60, 40, 30)}
    for side, c in centres.items():
        for i, sub in enumerate(external.SUBFIELDS):
            prob = ellipsoid((c[0], c[1] + 4 * (i - 1), c[2]), (5, 6, 5)).astype(np.float32)
            nib.save(nib.Nifti1Image(prob, AFFINE), root / f"SPAM_{sub}_{side}_MNI.nii.gz")
    rng = np.random.default_rng(0)
    for s in subjects:
        (root / s).mkdir()
        img = rng.normal(100, 5, SHAPE).astype(np.float32)
        for side, c in centres.items():
            y = np.zeros(SHAPE, np.int16)
            for i in range(3):  # subfields stacked along the long (y) axis
                y[ellipsoid((c[0], c[1] + 4 * (i - 1), c[2]), (5, 4, 5)) & (y == 0)] = i + 1
            img[y > 0] = 60
            nib.save(nib.Nifti1Image(y, AFFINE), root / s / f"{s}_hippolabels_t1w_standard_{side}_MNI.nii.gz")
        nib.save(nib.Nifti1Image(img.astype(np.int16), AFFINE), root / s / f"{s}_t1w_standard_defaced_MNI.nii.gz")
    return centres


def world_centroid(path, mask_fn):
    im = nib.load(path)
    return nib.affines.apply_affine(im.affine, np.argwhere(mask_fn(np.asarray(im.dataobj))).mean(0))


def test_prepare_crops_in_place(tmp_path):
    src = tmp_path / "ext"
    make_external(src)
    cases = external.prepare(str(src), str(tmp_path / "crops"))

    assert [c["case"] for c in cases] == ["s01_L", "s01_R", "s02_L", "s02_R"]
    for c in cases:
        crop = nib.load(c["label"])
        assert crop.shape == external.CROP_MM
        assert c["coverage"] == 1.0                           # whole hippocampus inside the box
        assert set(np.unique(np.asarray(crop.dataobj))) == {0, 1, 2, 3}
        s, side = c["case"].split("_")
        full = src / s / f"{s}_hippolabels_t1w_standard_{side}_MNI.nii.gz"
        assert np.allclose(world_centroid(c["label"], lambda y: y > 0), world_centroid(full, lambda y: y > 0))

    assert len(external.prepare(str(src), str(tmp_path / "one"), subjects=["s02"])) == 2


def test_prepare_centres_box_on_given_point(tmp_path):
    src = tmp_path / "ext"
    make_external(src, subjects=("s01",))
    atlas = {c["case"]: c for c in external.prepare(str(src), str(tmp_path / "atlas"))}
    point = np.array(atlas["s01_L"]["centre_mm"]) + [0, 5, -3]
    moved = {c["case"]: c for c in external.prepare(str(src), str(tmp_path / "moved"), centres={"s01_L": point})}
    # centred on the point to the nearest whole voxel (an even-sized box can't do better)
    box_move = np.array(moved["s01_L"]["centre_mm"]) - atlas["s01_L"]["centre_mm"]
    assert np.allclose(moved["s01_L"]["centre_mm"], point, atol=1.0) and box_move[0] == 0
    assert moved["s01_R"]["centre_mm"] == atlas["s01_R"]["centre_mm"]           # others keep the atlas box
    inside = lambda c: np.argwhere(np.asarray(nib.load(c["label"]).dataobj) > 0).mean(0)
    assert np.allclose(inside(moved["s01_L"]) - inside(atlas["s01_L"]), -box_move)  # anatomy moves the other way


def test_recentre_follows_the_first_prediction(tmp_path, monkeypatch):
    src = tmp_path / "ext"
    make_external(src, subjects=("s01",))
    run = tmp_path / "run"
    run.mkdir()
    cfg = yaml.safe_load(open("configs/baseline.yaml"))
    cfg["model"]["channels"] = [8, 16, 32, 64]
    (run / "config.yaml").write_text(yaml.safe_dump(cfg))
    torch.save(build_model(cfg).state_dict(), run / "best.pt")
    # a stand-in model that predicts the label 4 mm too high, plus a stray island: the box must follow the main piece
    real = external.predictions

    def shifted_label(run_dir, cases):
        for case, _, batch in real(run_dir, cases):
            pred = torch.roll((batch["label"] > 0).long(), 4, dims=4)
            pred[0, 0, 1, 1, 1] = 1
            yield case, pred, batch
    monkeypatch.setattr(external, "predictions", shifted_label)

    summary = external.recentre(str(run), str(src), str(tmp_path / "recentred"), keep_largest=True)
    assert summary["box"] == "re-centred on the first prediction" and summary["n_cases"] == 2
    with open(run / "external_recentred_largest" / "metrics_external.csv") as f:
        rows = {r["case"]: r for r in csv.DictReader(f)}
    for r in rows.values():
        assert r["recentred"] == "True" and float(r["moved_z_mm"]) == 4.0 and float(r["moved_y_mm"]) == 0.0
        assert int(r["pieces"]) == 2                                              # the island was removed again
    assert not (run / "external").exists()


def test_world_box():
    img = nib.Nifti1Image(np.zeros(SHAPE, np.uint8), AFFINE)            # voxel i is at world i - 40 (x)
    assert external.world_box(img, [-30, -50, -40], [-20.2, -41, -31]) == (slice(10, 21), slice(0, 10), slice(0, 10))
    assert external.world_box(img, [-45, -55, 30], [-38, -48, 60]) == (slice(0, 3), slice(0, 3), slice(70, 80))  # clipped


def test_msd_margins(tmp_path):
    cases = []
    for i, (lo, hi) in enumerate([((3, 5, 4), (15, 24, 16)), ((5, 5, 2), (17, 24, 14))]):
        y = np.zeros((20, 30, 20), np.uint8)
        y[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = 1
        nib.save(nib.Nifti1Image(y, np.eye(4)), tmp_path / f"c{i}.nii.gz")
        cases.append({"label": str(tmp_path / f"c{i}.nii.gz")})
    (tmp_path / "splits.json").write_text(json.dumps({"train": cases}))
    # case 0: low 3,5,4 high 5,6,4 -> 4, 5.5, 4; case 1: low 5,5,2 high 3,6,6 -> 4, 5.5, 4
    assert np.allclose(external.msd_margins(str(tmp_path / "splits.json")), [4, 5.5, 4])


def test_msd_crop_is_prediction_box_plus_msd_margins(tmp_path, monkeypatch):
    src = tmp_path / "ext"
    make_external(src, subjects=("s01",))
    run = tmp_path / "run"
    run.mkdir()
    cfg = yaml.safe_load(open("configs/baseline.yaml"))
    cfg["model"]["channels"] = [8, 16, 32, 64]
    cfg["data"]["split_file"] = "unused: margins are patched"
    (run / "config.yaml").write_text(yaml.safe_dump(cfg))
    torch.save(build_model(cfg).state_dict(), run / "best.pt")
    monkeypatch.setattr(external, "msd_margins", lambda split_file: np.array([4.0, 5.0, 3.0]))
    real = external.predictions

    def perfect(run_dir, cases):  # a stand-in model that predicts the label, plus a stray island
        for case, _, batch in real(run_dir, cases):
            pred = (batch["label"] > 0).long()
            pred[0, 0, 1, 1, 1] = 1
            yield case, pred, batch
    monkeypatch.setattr(external, "predictions", perfect)

    summary = external.msd_crop(str(run), str(src), str(tmp_path / "msd"), keep_largest=True)
    assert summary["box"].startswith("MSD-style") and summary["dice"]["mean"] == 1.0
    for c in json.loads((tmp_path / "msd" / "msd_crop" / "cases.json").read_text()):
        y = np.asarray(nib.load(c["label"]).dataobj) > 0
        idx = np.argwhere(y)
        assert c["coverage"] == 1.0
        assert idx.min(0).tolist() == [4, 5, 3]                                   # margin below each axis
        assert (np.array(y.shape) - 1 - idx.max(0)).tolist() == [4, 5, 3]        # and the same above
    with open(run / "external_msd_crop_largest" / "metrics_external.csv") as f:
        row = next(csv.DictReader(f))
    assert row["recentred"] == "True" and float(row["crop_z_mm"]) == 5 * 2 + 1 + 2 * 3   # ellipsoid z extent 11 + 2 x 3


def test_box_stays_inside_volume():
    box = external.crop_box(np.array([2.0, 88.0, 40.0]), SHAPE, (48, 56, 56))
    assert [(b.start, b.stop) for b in box] == [(0, 48), (34, 90), (12, 68)]  # shifted in x and y, centred in z


def test_whole_metrics_merges_parts():
    label = torch.zeros(1, 1, 16, 16, 16, dtype=torch.long)
    label[0, 0, 4:12, 4:8, 4:12], label[0, 0, 4:12, 8:12, 4:12] = 1, 3        # two subfields
    pred = (label > 0).long() * 2                                             # same region, other label values
    m = external.whole_metrics(pred, label, (1.0, 1.0, 1.0), 1.0)
    assert m["dice"] == 1.0 and m["nsd"] == 1.0 and m["vol_pred_ml"] == m["vol_true_ml"]


def test_evaluate_writes_report(tmp_path):
    src = tmp_path / "ext"
    make_external(src, subjects=("s01",))
    cases = external.prepare(str(src), str(tmp_path / "crops"))
    run = tmp_path / "run"
    run.mkdir()
    cfg = yaml.safe_load(open("configs/baseline.yaml"))
    cfg["model"]["channels"] = [8, 16, 32, 64]
    (run / "config.yaml").write_text(yaml.safe_dump(cfg))
    torch.save(build_model(cfg).state_dict(), run / "best.pt")   # untrained: exercises the path only

    summary = external.evaluate(str(run), cases)
    assert summary["n_cases"] == 2 and 0.0 <= summary["dice"]["mean"] <= 1.0
    assert (run / "external" / "metrics_external.csv").read_text().startswith("case,dice,nsd")


def test_largest_piece_keeps_biggest_and_its_labels():
    pred = np.zeros((12, 12, 12), np.int64)
    pred[1:6, 1:6, 1:4], pred[1:6, 1:6, 4:6] = 1, 2      # main piece, two labels touching
    pred[9, 9, 9] = 1                                     # island
    pred[10, 10, 10] = 2                                  # touches the island diagonally: same piece
    kept, n = external.largest_piece(pred)
    assert n == 2 and kept[9, 9, 9] == 0 and kept[10, 10, 10] == 0
    assert np.array_equal(kept[:8, :8, :8], pred[:8, :8, :8])          # main piece untouched, labels kept
    _, n = external.largest_piece(pred * (np.arange(12) < 8)[:, None, None])
    assert n == 1
    assert external.largest_piece(np.zeros((4, 4, 4), np.int64))[1] == 0


def test_evaluate_largest_piece_writes_its_own_report(tmp_path):
    src = tmp_path / "ext"
    make_external(src, subjects=("s01",))
    cases = external.prepare(str(src), str(tmp_path / "crops"))
    run = tmp_path / "run"
    run.mkdir()
    cfg = yaml.safe_load(open("configs/baseline.yaml"))
    cfg["model"]["channels"] = [8, 16, 32, 64]
    (run / "config.yaml").write_text(yaml.safe_dump(cfg))
    torch.save(build_model(cfg).state_dict(), run / "best.pt")

    summary = external.evaluate(str(run), cases, keep_largest=True)
    assert summary["post_processing"] == "largest piece only"
    assert not (run / "external").exists()                            # the plain results are not touched
    header = (run / "external_largest" / "metrics_external.csv").read_text().splitlines()[0]
    assert header == "case,dice,nsd,hd95,vol_pred_ml,vol_true_ml,label_coverage,pieces,removed_ml"


@pytest.mark.parametrize("normalize, expected", [("zscore", "zscore"), ("none", "none"), (None, "zscore")])
def test_evaluate_uses_run_normalize(tmp_path, monkeypatch, normalize, expected):
    src = tmp_path / "ext"
    make_external(src, subjects=("s01",))
    cases = external.prepare(str(src), str(tmp_path / "crops"))
    run = tmp_path / "run"
    run.mkdir()
    cfg = yaml.safe_load(open("configs/baseline.yaml"))
    cfg["model"]["channels"] = [8, 16, 32, 64]
    cfg["data"].pop("normalize", None)
    if normalize is not None:  # None: a run from before the setting existed
        cfg["data"]["normalize"] = normalize
    (run / "config.yaml").write_text(yaml.safe_dump(cfg))
    torch.save(build_model(cfg).state_dict(), run / "best.pt")

    seen, real = [], external.eval_transforms
    monkeypatch.setattr(external, "eval_transforms", lambda n: seen.append(n) or real(n))
    external.evaluate(str(run), cases)
    assert seen == [expected]


def test_prepare_rejects_unknown_subject(tmp_path):
    src = tmp_path / "ext"
    make_external(src)
    with pytest.raises(SystemExit, match="s09"):
        external.prepare(str(src), str(tmp_path / "crops"), subjects=["s01", "s09"])


@pytest.mark.parametrize("change", ["shifted", "resized"])
def test_prepare_rejects_image_off_atlas_grid(tmp_path, change):
    src = tmp_path / "ext"
    make_external(src)
    path = src / "s02" / "s02_t1w_standard_defaced_MNI.nii.gz"
    img = nib.load(path)
    data, affine = np.asarray(img.dataobj), img.affine.copy()
    if change == "shifted":
        affine[0, 3] += 2                      # same size, moved 2 mm: a crop would land beside the hippocampus
    else:
        data = data[:-1]                       # same origin, one slice fewer
    nib.save(nib.Nifti1Image(data, affine), path)
    with pytest.raises(SystemExit, match="s02: .* not on the SPAM atlas grid"):
        external.prepare(str(src), str(tmp_path / "crops"))


def test_coverage_counts_label_outside_the_box(tmp_path):
    src = tmp_path / "ext"
    make_external(src)
    path = src / "s01" / "s01_hippolabels_t1w_standard_L_MNI.nii.gz"
    lbl = nib.load(path)
    y = np.asarray(lbl.dataobj).copy()
    inside = int((y > 0).sum())
    # the left box spans x 0-47 (atlas centre x 20); put 2x2x2 extra label voxels at x 70-71, outside it
    y[70:72, 40:42, 30:32] = 1
    nib.save(nib.Nifti1Image(y, lbl.affine), path)

    cases = {c["case"]: c for c in external.prepare(str(src), str(tmp_path / "crops"))}
    assert cases["s01_L"]["coverage"] == round(inside / (inside + 8), 4) < 1
    assert all(cases[c]["coverage"] == 1.0 for c in ("s01_R", "s02_L", "s02_R"))
    crop = np.asarray(nib.load(cases["s01_L"]["label"]).dataobj)
    assert int((crop > 0).sum()) == inside     # the crop holds exactly the part inside the box
