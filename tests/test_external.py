"""hippo.external crops each hippocampus around the atlas, keeps it in place, and scores the whole hippocampus."""
import nibabel as nib
import numpy as np
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
