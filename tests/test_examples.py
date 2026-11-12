"""hippo.examples: the upload page's example scans, copied with a manifest and read back from the store."""
import json

import pytest

from hippo import examples

from synthetic import make_fake_msd


def _cases():
    """Two of the listed examples, pointed at the fake MSD task (same fields as the real list)."""
    real = {e["key"]: e for e in examples.EXAMPLES}
    return [real["msd_008"] | {"image": "Task04_Hippocampus/imagesTr/hippocampus_000.nii.gz",
                               "label": "Task04_Hippocampus/labelsTr/hippocampus_000.nii.gz"},
            real["s03_L"] | {"image": "Task04_Hippocampus/imagesTr/hippocampus_001.nii.gz",
                             "label": "Task04_Hippocampus/labelsTr/hippocampus_001.nii.gz"}]


def test_prepare_and_read_back(tmp_path):
    make_fake_msd(tmp_path / "data", n=2)
    manifest = examples.prepare(str(tmp_path / "out"), str(tmp_path / "data"), _cases())
    listed = examples.catalogue(str(tmp_path / "out"))
    assert [e["key"] for e in listed] == ["msd_008", "s03_L"] and manifest.name == "examples.json"
    assert listed[0]["image"] == "msd_008/image.nii.gz" and listed[0]["parts"] and not listed[1]["parts"]
    assert all("CC BY" in e["licence"] for e in listed)                       # licence travels with the scan
    source = tmp_path / "data" / "Task04_Hippocampus" / "imagesTr" / "hippocampus_000.nii.gz"
    assert examples.read(str(tmp_path / "out"), listed[0]["image"]) == source.read_bytes()


def test_missing_source_says_which(tmp_path):
    with pytest.raises(SystemExit, match="msd_008"):
        examples.prepare(str(tmp_path / "out"), str(tmp_path / "empty"), _cases()[:1])


def test_the_real_list():
    """Two MSD crops (one test case), and one external subject's left and right side."""
    by_key = {e["key"]: e for e in examples.EXAMPLES}
    assert by_key["msd_008"]["split"] == "test" and by_key["msd_007"]["split"] == "validation"
    assert by_key["s03_L"]["side"] == "left" and by_key["s03_R"]["side"] == "right"
    assert all(e["kind"] == "crop" for e in examples.EXAMPLES)
    assert json.dumps(examples.EXAMPLES)                                     # the manifest can hold it
