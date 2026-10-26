"""data.normalize: zscore rescales each scan, none keeps raw intensities."""
import nibabel as nib
import numpy as np
import pytest

from hippo.data import eval_transforms

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
