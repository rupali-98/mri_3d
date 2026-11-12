"""hippo.api_client: the upload page's calls to the API, run against the real API on a tiny model."""
import gzip

import nibabel as nib
import numpy as np
import pytest
from fastapi.testclient import TestClient

from hippo import api_client
from hippo.api import create_app

from test_api import _nifti, _package

URL = "http://testserver"  # TestClient's own address


@pytest.fixture
def client(tmp_path):
    out, task = _package(tmp_path)
    with TestClient(create_app(str(out), api_key="secret")) as c:
        yield c, task


def test_crop_round_trip(client):
    c, task = client
    upload = (task / "imagesTr" / "hippocampus_000.nii.gz").read_bytes()
    result = api_client.segment(URL, "secret", upload, "hippocampus_000.nii.gz", "crop", session=c)
    h = result["hippocampi"][0]
    mask = api_client.mask_image(h)
    scan = api_client.read_scan(upload)
    region = api_client.scan_region(scan, mask)
    assert region.shape == scan.shape and np.allclose(region.affine, scan.affine)     # a crop: the whole scan
    assert np.array_equal(region.get_fdata(), scan.get_fdata())
    assert api_client.mask_bytes(h)[:2] == b"\x1f\x8b"                                # a .nii.gz file


def test_mni_region_is_the_atlas_box(client):
    c, _ = client
    brain = _nifti(shape=(64, 48, 40))
    result = api_client.segment(URL, "secret", brain, "brain.nii.gz", "mni", largest=True, session=c)
    assert result["post_processing"] == "largest piece"
    left = result["hippocampi"][0]
    region = api_client.scan_region(api_client.read_scan(brain), api_client.mask_image(left))
    full = api_client.read_scan(brain).get_fdata()
    assert region.shape == (16, 24, 16)
    assert np.array_equal(region.get_fdata(), full[10:26, 10:34, 10:26])               # the left box, from the brain


@pytest.mark.parametrize("key, content, kind, message", [
    ("wrong", None, "crop", "API key was refused"),
    ("secret", _nifti(shape=(130, 40, 32)), "crop", "The scan can't be used: [130, 40, 32] voxels is too large"),
    ("secret", b"not an image", "crop", "The scan can't be used: not a readable NIfTI"),
    ("secret", _nifti(shape=(32, 40, 32)), "mni", "atlas box lies partly outside"),
])
def test_refusals_read_well(client, key, content, kind, message):
    c, task = client
    content = content or (task / "imagesTr" / "hippocampus_000.nii.gz").read_bytes()
    with pytest.raises(api_client.ApiError, match=message.replace("[", r"\[").replace("]", r"\]")):
        api_client.segment(URL, key, content, "scan.nii.gz", kind, session=c)


def test_unreachable_service():
    with pytest.raises(api_client.ApiError, match="couldn't be reached"):
        api_client.segment("http://127.0.0.1:9", None, b"x", "scan.nii", "crop")      # nothing listens on port 9
    with pytest.raises(ValueError, match="kind"):
        api_client.segment(URL, None, b"x", "scan.nii", "brain")


def test_read_scan_and_settings(tmp_path, monkeypatch):
    image = nib.Nifti1Image(np.arange(24, dtype=np.float32).reshape(2, 3, 4), np.eye(4))
    plain = image.to_bytes()
    for data in (plain, gzip.compress(plain)):
        assert api_client.read_scan(data).shape == (2, 3, 4)
    monkeypatch.setenv("HIPPO_API_URL", "https://example.run.app/")
    monkeypatch.delenv("HIPPO_API_KEY", raising=False)
    (tmp_path / "key").write_text("from-file\n")
    monkeypatch.setenv("HIPPO_API_KEY_FILE", str(tmp_path / "key"))
    assert api_client.settings() == ("https://example.run.app", "from-file")
    monkeypatch.setenv("HIPPO_API_KEY", "from-env")
    assert api_client.settings()[1] == "from-env"


def test_largest_is_sent_for_crops_too(client):
    c, task = client
    upload = (task / "imagesTr" / "hippocampus_000.nii.gz").read_bytes()
    result = api_client.segment(URL, "secret", upload, "scan.nii.gz", "crop", largest=True, session=c)
    assert result["post_processing"] == "largest piece" and "removed_ml" in result["hippocampi"][0]


def test_models_and_choice(tmp_path):
    from test_api import _versions
    out, task = _versions(tmp_path)
    with TestClient(create_app(str(out), api_key="secret")) as c:
        listing = api_client.models(URL, "secret", session=c)
        assert listing["default"] == "champion" and len(listing["models"]) == 2
        upload = (task / "imagesTr" / "hippocampus_000.nii.gz").read_bytes()
        assert api_client.segment(URL, "secret", upload, "s.nii.gz", session=c, model="other")["model"]["key"] == "other"
        with pytest.raises(api_client.ApiError, match="Unknown model: no model called 'nope'"):
            api_client.segment(URL, "secret", upload, "s.nii.gz", session=c, model="nope")
        with pytest.raises(api_client.ApiError, match="API key was refused"):
            api_client.models(URL, "wrong", session=c)


def test_agreement():
    a = np.zeros((6, 6, 6), np.uint8)
    a[1:4, 1:4, 1:3] = 1
    a[1:4, 1:4, 3:5] = 2
    b = a.copy()
    b[1:4, 1:4, 3:5] = 1                                                    # posterior drawn as anterior
    image = lambda x: nib.Nifti1Image(x, np.eye(4))
    same = api_client.agreement(image(a), image(a))
    assert same == {"whole": 1.0, "anterior": 1.0, "posterior": 1.0}
    swapped = api_client.agreement(image(a), image(b))
    assert swapped["whole"] == 1.0 and swapped["posterior"] == 0.0 and swapped["anterior"] == pytest.approx(2 / 3)
    with pytest.raises(ValueError, match="different grids"):
        api_client.agreement(image(a), nib.Nifti1Image(a, np.diag([2.0, 1, 1, 1])))


def test_accuracy_against_subfield_labels():
    """External labels are subfields (1-3): only the whole hippocampus can be scored against them."""
    pred = np.zeros((6, 6, 6), np.uint8)
    pred[1:4, 1:4, 1:3], pred[1:4, 1:4, 3:5] = 1, 2
    truth = (pred > 0).astype(np.uint8) * 3                                   # all one subfield
    image = lambda x: nib.Nifti1Image(x, np.eye(4))
    assert api_client.accuracy(image(pred), image(truth), parts=False) == {"whole": 1.0}
    assert set(api_client.accuracy(image(pred), image(pred))) == {"whole", "anterior", "posterior"}
