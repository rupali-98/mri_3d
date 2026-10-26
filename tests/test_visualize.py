"""The visualisation script renders and reports volumes, with and without labels."""
import nibabel as nib
import numpy as np
import pytest
import torch
import yaml

from hippo import visualize
from hippo.model import build_model

from synthetic import make_fake_msd


def test_visualize_label_prediction_and_unlabelled(tmp_path):
    task = make_fake_msd(tmp_path / "data", n=1)
    image = task / "imagesTr" / "hippocampus_000.nii.gz"
    label = task / "labelsTr" / "hippocampus_000.nii.gz"

    # a run folder with an (untrained) model, enough to exercise the prediction path
    run = tmp_path / "run"
    run.mkdir()
    cfg = yaml.safe_load(open("configs/baseline.yaml"))
    cfg["model"]["channels"] = [8, 16, 32, 64]
    (run / "config.yaml").write_text(yaml.safe_dump(cfg))
    torch.save(build_model(cfg).state_dict(), run / "best.pt")

    out = tmp_path / "with_label.png"
    vols = visualize.main(str(image), str(label), runs=str(run), out=str(out), montage_step=3, save_nifti=True)
    assert out.stat().st_size > 0
    assert (tmp_path / "with_label_sagittal.png").stat().st_size > 0
    pred = nib.load(tmp_path / "with_label_pred.nii.gz")
    assert set(np.unique(pred.get_fdata())) <= {0, 1, 2}
    assert set(vols) == {"ground truth", "prediction"}
    # synthetic ellipsoids: 1 mm voxels, each part roughly (4/3)·π·5·7·5 mm³ ≈ 0.7 ml
    assert 0.5 < vols["ground truth"]["anterior"] < 0.9
    assert 0.5 < vols["ground truth"]["posterior"] < 0.9

    out = tmp_path / "unlabelled.png"
    assert visualize.main(str(image), out=str(out)) == {}
    assert out.stat().st_size > 0


def test_visualize_compares_runs(tmp_path, monkeypatch):
    task = make_fake_msd(tmp_path / "data", n=1)
    image = task / "imagesTr" / "hippocampus_000.nii.gz"
    label = task / "labelsTr" / "hippocampus_000.nii.gz"
    cfg = yaml.safe_load(open("configs/baseline.yaml"))
    cfg["model"]["channels"] = [8, 16, 32, 64]
    for name, normalize in [("zscored", "zscore"), ("raw", "none")]:
        run = tmp_path / name
        run.mkdir()
        (run / "config.yaml").write_text(yaml.safe_dump({**cfg, "data": {**cfg["data"], "normalize": normalize}}))
        torch.save(build_model(cfg).state_dict(), run / "best.pt")

    # record the intensities each run's model is given
    seen, real_predict = {}, visualize.predict

    def recording_predict(run, x):
        seen[run.name] = float(x.max())
        return real_predict(run, x)

    monkeypatch.setattr(visualize, "predict", recording_predict)

    vols = visualize.main(str(image), str(label), runs=[str(tmp_path / "zscored"), str(tmp_path / "raw")],
                          out=str(tmp_path / "cmp.png"), save_nifti=True)
    assert list(vols) == ["ground truth", "zscored", "raw"]
    raw_max = float(np.asarray(nib.load(image).dataobj).max())
    assert np.isclose(seen["raw"], raw_max) and seen["zscored"] < raw_max  # each run gets its own preprocessing
    assert (tmp_path / "cmp_pred_zscored.nii.gz").exists() and (tmp_path / "cmp_pred_raw.nii.gz").exists()


def test_visualize_html(tmp_path):
    pytest.importorskip("plotly")
    task = make_fake_msd(tmp_path / "data", n=1)
    image = task / "imagesTr" / "hippocampus_000.nii.gz"
    label = task / "labelsTr" / "hippocampus_000.nii.gz"

    visualize.main(str(image), str(label), out=str(tmp_path / "case.png"), html=True)
    page = (tmp_path / "case_3d.html").read_text(encoding="utf-8")
    assert "ground truth: anterior" in page and "ground truth: posterior" in page
