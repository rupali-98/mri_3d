"""streamlit_dashboard/compare_slices.py runs headless: predicts the chosen runs, shows one row per mask,
and follows the controls (plane sliders in the sidebar)."""
import json
import shutil
from pathlib import Path

import torch
import yaml
from streamlit.testing.v1 import AppTest

from hippo.model import build_model

from synthetic import make_fake_msd

APP = Path(__file__).resolve().parents[1] / "streamlit_dashboard" / "compare_slices.py"


def test_app_predicts_and_compares(tmp_path, monkeypatch):
    # a tiny project: fake MSD data, three untrained runs, and the app in streamlit_dashboard/ so it finds them
    make_fake_msd(tmp_path / "data", n=2)
    cfg = yaml.safe_load(open("configs/baseline.yaml"))
    cfg["model"]["channels"] = [8, 16, 32, 64]
    for name in ("baseline", "other_a", "other_b"):
        run = tmp_path / "runs" / name
        run.mkdir(parents=True)
        (run / "config.yaml").write_text(yaml.safe_dump(cfg))
        torch.save(build_model(cfg).state_dict(), run / "best.pt")
    (tmp_path / "streamlit_dashboard").mkdir()
    shutil.copy(APP, tmp_path / "streamlit_dashboard")
    monkeypatch.chdir(tmp_path)  # the app changes directory to its project root; this puts it back afterwards

    at = AppTest.from_file(str(tmp_path / "streamlit_dashboard" / "compare_slices.py"), default_timeout=60).run()
    assert not at.exception
    image = "data/Task04_Hippocampus/imagesTr/hippocampus_001.nii.gz"
    at.selectbox(key="image").select(image).run()
    assert at.selectbox(key=f"truth::{image}").value == image.replace("imagesTr", "labelsTr")  # guessed

    assert at.multiselect[0].value == ["runs/baseline", "runs/other_a", "runs/other_b"]  # all ticked
    at.multiselect[0].unselect("runs/other_a")
    at.button[0].click().run()
    assert not at.exception and not at.error
    text = " ".join(m.value for m in at.markdown)
    assert text.index("**ground truth**") < text.index("**baseline: Dice whole") < text.index("**other_b: Dice whole")
    assert "other_a" not in text
    assert text.count("data:image/png") == 9                          # 3 rows (truth, baseline, other_b) x 3 views
    assert (tmp_path / "runs/figures/notebook/hippocampus_001_pred_other_b.nii.gz").exists()

    assert [s.key for s in at.sidebar.slider if (s.key or "").startswith("pos")] == ["pos2", "pos1", "pos0"]  # planes in sidebar
    at.slider(key="pos2").set_value(3).run()                           # axial slider
    assert "Axial slice 4 /" in " ".join(c.value for c in at.caption)

    assert not at.get("plotly_chart")                                   # 3D is off until ticked
    next(c for c in at.sidebar.checkbox if c.label == "3D surfaces").check().run()
    assert not at.exception and len(at.get("plotly_chart")) == 3        # one 3D view per row
    # hover gives mm and slider values; on the crosshair they are the sliders' own values
    traces = json.loads(at.get("plotly_chart")[0].proto.spec)["data"]
    assert all("mm" in t["hovertemplate"] and "sliders:" in t["hovertemplate"] for t in traces)
    cross = next(t for t in traces if t["name"] == "crosshair")
    assert cross["customdata"] == [[at.slider(key=f"pos{a}").value for a in range(3)]]
    assert cross["customdata"][0][2] == 3

    at.radio[0].set_value("Errors").run()
    assert not at.exception and "missed" in " ".join(m.value for m in at.markdown)
    assert len(at.get("plotly_chart")) == 3

    # largest piece is for external crops only: greyed out for an MSD image
    assert next(c for c in at.sidebar.checkbox if c.label == "Largest piece only").disabled

    # ... and available for an external crop (here a copy of the fake case under data/external/)
    for kind in ("imagesTr", "labelsTr"):
        (tmp_path / "data/external" / kind).mkdir(parents=True)
        shutil.copy(tmp_path / "data/Task04_Hippocampus" / kind / "hippocampus_001.nii.gz",
                    tmp_path / "data/external" / kind / "s01_L.nii.gz")
    at = AppTest.from_file(str(tmp_path / "streamlit_dashboard" / "compare_slices.py"), default_timeout=60).run()
    at.selectbox(key="image").select("data/external/imagesTr/s01_L.nii.gz").run()
    at.multiselect[0].set_value(["runs/baseline"])
    at.button[0].click().run()
    box = next(c for c in at.sidebar.checkbox if c.label == "Largest piece only")
    assert not box.disabled
    box.check().run()
    assert not at.exception and "**baseline (largest of " in " ".join(m.value for m in at.markdown)

    # an ASHS label for the image is picked as the second reference: its own row, and each run is scored against it
    ashs = tmp_path / "data/ashs/labelsTr"
    ashs.mkdir(parents=True)
    shutil.copy(tmp_path / "data/Task04_Hippocampus/labelsTr/hippocampus_001.nii.gz", ashs)
    at = AppTest.from_file(str(tmp_path / "streamlit_dashboard" / "compare_slices.py"), default_timeout=60).run()
    at.selectbox(key="image").select(image).run()
    assert at.selectbox(key=f"second::{image}").value == "data/ashs/labelsTr/hippocampus_001.nii.gz"
    at.multiselect[0].set_value(["runs/baseline"])
    at.button[0].click().run()
    assert not at.exception and not at.error
    text = " ".join(m.value for m in at.markdown)
    assert "**ASHS-PMC-T1: vs ground truth Dice whole 1.000" in text            # same file as the ground truth here
    assert "| vs ASHS-PMC-T1 Dice whole" in text and text.count("data:image/png") == 9

    # an archive/shift.py copy of the same case gets its own prediction file instead of overwriting the original's
    shifted = tmp_path / "data/shifted/y-5"
    for kind in ("imagesTr", "labelsTr"):
        (shifted / kind).mkdir(parents=True)
        shutil.copy(tmp_path / "data/Task04_Hippocampus" / kind / "hippocampus_001.nii.gz", shifted / kind)
    at = AppTest.from_file(str(tmp_path / "streamlit_dashboard" / "compare_slices.py"), default_timeout=60).run()
    at.text_input[0].input("shifted y-5").run()
    assert at.selectbox(key="image").options == ["data/shifted/y-5/imagesTr/hippocampus_001.nii.gz"]
    at.multiselect[0].set_value(["runs/baseline"])
    at.button[0].click().run()
    assert not at.exception and not at.error
    assert (tmp_path / "runs/figures/notebook/hippocampus_001_shifted_y-5_pred_baseline.nii.gz").exists()
