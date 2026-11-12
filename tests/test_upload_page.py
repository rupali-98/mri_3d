"""streamlit_dashboard/upload_page.py: what the cloud page shows before and after a result.

AppTest can't upload a file, so a result from the real API (tiny model) is put into the page's session
state, as the Segment button would.
"""
from pathlib import Path

import pytest
import streamlit as st

from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from hippo import api_client
from hippo.api import create_app

from test_api import _nifti, _package

PAGE = str(Path(__file__).resolve().parents[1] / "streamlit_dashboard" / "upload_page.py")
OFFERED = {"default": "champion", "models": [
    {"key": "champion", "name": "3D U-Net", "version": "aaaa", "combination": "single model", "default": True},
    {"key": "ensemble", "name": "3D U-Net (ensemble)", "version": "bbbb", "combination": "mean of class probabilities",
     "default": False}]}


@pytest.fixture(autouse=True)
def offered_models(monkeypatch):
    """The page asks the API for its models on every run; answer for it (no server in these tests)."""
    monkeypatch.setattr(api_client, "models", lambda url, key, session=None: OFFERED)
    st.cache_data.clear()  # the page caches the list for 10 minutes, also across tests in one process


def _result(tmp_path, kind, largest=False):
    out, task = _package(tmp_path)
    upload = (task / "imagesTr" / "hippocampus_000.nii.gz").read_bytes() if kind == "crop" else _nifti(shape=(64, 48, 40))
    with TestClient(create_app(str(out))) as c:
        result = api_client.segment("http://testserver", None, upload, "scan.nii.gz", kind, largest, session=c)
    return {"results": {"champion": result}, "scan": upload, "name": "scan.nii.gz", "scans": {}}


def test_not_configured(monkeypatch):
    monkeypatch.delenv("HIPPO_API_URL", raising=False)
    at = AppTest.from_file(PAGE, default_timeout=30).run()
    assert not at.exception and "HIPPO_API_URL" in at.error[0].value


def test_before_an_upload(monkeypatch):
    monkeypatch.setenv("HIPPO_API_URL", "http://testserver")
    at = AppTest.from_file(PAGE, default_timeout=30).run()
    assert not at.exception
    assert at.button[0].label == "Segment" and at.button[0].disabled                  # nothing uploaded yet
    assert "Research use only" in at.caption[0].value
    assert not next(t for t in at.toggle if t.label == "Largest piece only").disabled  # offered for crops too


def test_crop_result(tmp_path, monkeypatch):
    monkeypatch.setenv("HIPPO_API_URL", "http://testserver")
    at = AppTest.from_file(PAGE, default_timeout=30)
    at.session_state["shown"] = _result(tmp_path, "crop")
    at.run()
    assert not at.exception
    assert [m.label for m in at.metric] == ["Anterior volume", "Posterior volume", "Whole volume"]
    assert len(at.get("image")) == 3                                                    # axial, coronal, sagittal
    assert at.get("download_button")
    next(t for t in at.toggle if t.label == "3D view").set_value(True).run()
    assert not at.exception and at.get("plotly_chart")


def test_whole_brain_result_has_both_sides(tmp_path, monkeypatch):
    monkeypatch.setenv("HIPPO_API_URL", "http://testserver")
    at = AppTest.from_file(PAGE, default_timeout=30)
    at.session_state["shown"] = _result(tmp_path, "mni")
    at.run()
    assert not at.exception
    side = next(s for s in at.get("button_group") if s.label == "Hippocampus")
    assert side.options == ["left", "right"]
    assert "not checked" in at.info[0].value                                           # the MNI warning
    side.set_value("right").run()
    assert not at.exception and len(at.get("image")) == 3


def test_largest_piece_result_says_what_was_removed(tmp_path, monkeypatch):
    monkeypatch.setenv("HIPPO_API_URL", "http://testserver")
    at = AppTest.from_file(PAGE, default_timeout=30)
    at.session_state["shown"] = _result(tmp_path, "crop", largest=True)
    at.run()
    assert not at.exception
    assert "post-processing: largest piece" in at.caption[1].value.lower()
    assert any("Largest piece only:" in i.value and "ml removed" in i.value for i in at.info)


def test_models_box_lists_the_default_first(monkeypatch):
    monkeypatch.setenv("HIPPO_API_URL", "http://testserver")
    at = AppTest.from_file(PAGE, default_timeout=30).run()
    assert not at.exception
    box = next(b for b in at.multiselect if b.label == "Models")
    assert box.options[0].startswith("champion: 3D U-Net (single model) · aaaa · default")
    assert box.value == ["champion"] and len(box.options) == 2


def test_two_models_side_by_side(tmp_path, monkeypatch):
    from test_api import _versions
    out, task = _versions(tmp_path)
    upload = (task / "imagesTr" / "hippocampus_000.nii.gz").read_bytes()
    with TestClient(create_app(str(out))) as c:
        results = {k: api_client.segment("http://testserver", None, upload, "scan.nii.gz", "crop", session=c, model=k)
                   for k in ("champion", "other")}
    monkeypatch.setenv("HIPPO_API_URL", "http://testserver")
    at = AppTest.from_file(PAGE, default_timeout=30)
    at.session_state["shown"] = {"results": results, "scan": upload, "name": "scan.nii.gz", "scans": {}}
    at.run()
    assert not at.exception
    table = at.dataframe[0].value
    assert list(table["Model"]) == ["champion", "other"]
    assert list(table["Dice vs champion, whole"]) == [1.0, 1.0]       # same weights in the test: identical masks
    assert len(at.get("image")) == 6                                   # 3 views x 2 models
    assert len(at.get("download_button")) == 2
    assert not at.metric                                               # the table replaces the single-model metrics
    next(t for t in at.toggle if t.label == "3D view").set_value(True).run()
    assert not at.exception and len(at.get("plotly_chart")) == 2


def test_example_scan_is_offered_and_scored(tmp_path, monkeypatch):
    """With HIPPO_EXAMPLES_URI the sidebar offers examples; a result for one is scored against its label."""
    from hippo import examples
    from synthetic import make_fake_msd
    make_fake_msd(tmp_path / "data", n=1)
    case = next(e for e in examples.EXAMPLES if e["key"] == "msd_008") | {
        "image": "Task04_Hippocampus/imagesTr/hippocampus_000.nii.gz",
        "label": "Task04_Hippocampus/labelsTr/hippocampus_000.nii.gz"}
    examples.prepare(str(tmp_path / "examples"), str(tmp_path / "data"), [case])
    monkeypatch.setenv("HIPPO_API_URL", "http://testserver")
    monkeypatch.setenv("HIPPO_EXAMPLES_URI", str(tmp_path / "examples"))

    at = AppTest.from_file(PAGE, default_timeout=30).run()
    assert not at.exception
    box = next(s for s in at.selectbox if s.label == "Example scan")
    assert box.value == "upload" and at.button[0].disabled                  # nothing chosen yet
    box.set_value("msd_008").run()
    assert not at.exception and not at.button[0].disabled                   # the example can be segmented
    assert any("CC BY-SA 4.0" in c.value for c in at.sidebar.caption)       # licence shown with the example

    shown = _result(tmp_path / "result", "crop")
    shown |= {"label": (tmp_path / "examples" / "msd_008" / "label.nii.gz").read_bytes(), "parts": True,
              "example": case["title"]}
    at = AppTest.from_file(PAGE, default_timeout=30)
    at.session_state["shown"] = shown
    at.run()
    assert not at.exception
    scored = next(d.value for d in at.dataframe if "Dice, whole" in d.value.columns)
    assert list(scored.columns) == ["Model", "Dice, whole", "Dice, anterior", "Dice, posterior"]  # MSD: parts too
    assert 0 <= scored["Dice, whole"][0] <= 1

    # the manual label is drawn as its own row above the prediction, and in the 3D view
    assert len(at.get("image")) == 6                                         # 3 views x (label + model)
    assert any("Manual label (ground truth)" in m.value for m in at.markdown)
    next(t for t in at.toggle if t.label == "3D view").set_value(True).run()
    assert not at.exception and len(at.get("plotly_chart")) == 2
    next(t for t in at.toggle if t.label == "Show the manual label").set_value(False).run()
    assert not at.exception and len(at.get("image")) == 3


def test_subfield_label_keeps_its_own_colours(tmp_path, monkeypatch):
    """An external example's label has subfields 1-3: drawn in their colours, and the caption says so."""
    import gzip
    import nibabel as nib
    import numpy as np
    shown = _result(tmp_path, "crop")
    scan = api_client.read_scan(shown["scan"])
    label = np.zeros(scan.shape, np.uint8)
    label[10:14, 12:20, 10:14], label[14:18, 12:20, 10:14], label[18:22, 12:20, 10:14] = 1, 2, 3
    shown |= {"label": gzip.compress(nib.Nifti1Image(label, scan.affine).to_bytes()), "parts": False,
              "example": "MNI-HISUB25 s03, left (external)"}
    monkeypatch.setenv("HIPPO_API_URL", "http://testserver")
    at = AppTest.from_file(PAGE, default_timeout=30)
    at.session_state["shown"] = shown
    at.run()
    assert not at.exception
    header = next(m.value for m in at.markdown if "Manual label (ground truth)" in m.value)
    assert "CA1-3" in header and "subiculum" in header                      # the subfield legend
    assert any("label is in subfields" in c.value for c in at.caption)
    scored = next(d.value for d in at.dataframe if "Dice, whole" in d.value.columns)
    assert list(scored.columns) == ["Model", "Dice, whole"]                  # whole only for subfield labels


def test_examples_unavailable_keeps_uploads(monkeypatch, tmp_path):
    monkeypatch.setenv("HIPPO_API_URL", "http://testserver")
    monkeypatch.setenv("HIPPO_EXAMPLES_URI", str(tmp_path / "nothing-here"))
    at = AppTest.from_file(PAGE, default_timeout=30).run()
    assert not at.exception and "couldn't be loaded" in at.sidebar.warning[0].value
    assert not any(s.label == "Example scan" for s in at.selectbox)


def test_model_service_down(monkeypatch):
    def down(url, key, session=None):
        raise api_client.ApiError("The model service couldn't be reached (ConnectionError). Try again in a minute.")
    monkeypatch.setattr(api_client, "models", down)
    monkeypatch.setenv("HIPPO_API_URL", "http://testserver")
    at = AppTest.from_file(PAGE, default_timeout=30).run()
    assert not at.exception and "couldn't be reached" in at.sidebar.error[0].value
