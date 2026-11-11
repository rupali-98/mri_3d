"""Client for the hippocampus API (hippo.api), used by the cloud upload page.

The upload page (streamlit_dashboard/upload_page.py) doesn't run the model itself: it sends the visitor's
scan to the deployed API, whose model comes from the MLflow bucket, and draws what comes back. This module
holds that logic without Streamlit or PyTorch, so it is light and can be tested on its own:
  - models(): the models the API offers (GET /models), to choose from;
  - segment(): POST the scan to /segment/crop or /segment/mni with the API key (and the chosen model),
    return the JSON, and turn the API's refusals (401, 404, 413, 422, ...) into a readable ApiError;
  - mask_image(): the base64 NIfTI mask of one hippocampus from that JSON;
  - scan_region(): the part of the uploaded scan the mask covers (the whole crop, or the atlas box of a
    whole-brain MNI scan), on the mask's grid, so the two can be drawn together.
The local research page (streamlit_dashboard/compare_slices.py) doesn't use this module.

Settings (environment variables, read by settings()):
    HIPPO_API_URL       the API's address, e.g. https://hippo-api-....run.app
    HIPPO_API_KEY       its key, or HIPPO_API_KEY_FILE: a file holding it (kept on the server, never shown)
"""
from __future__ import annotations

import base64
import gzip
import os
from pathlib import Path

import nibabel as nib
import numpy as np
import requests

KINDS = {"crop": "/segment/crop", "mni": "/segment/mni"}
TIMEOUT_S = 120  # the first request after a pause waits for the API's cold start


class ApiError(Exception):
    """The API refused the request or couldn't be reached; the message is meant for the page."""


def settings() -> tuple[str | None, str | None]:
    """The API address and key from the environment (the key may be in a file, as in compose.yaml)."""
    key = os.environ.get("HIPPO_API_KEY")
    if not key and os.environ.get("HIPPO_API_KEY_FILE"):
        key = Path(os.environ["HIPPO_API_KEY_FILE"]).read_text().strip() or None
    url = os.environ.get("HIPPO_API_URL")
    return (url.rstrip("/") if url else None), key


def _refusal(r) -> ApiError:
    """The API's error response as a sentence for the page."""
    try:
        detail = r.json().get("detail", r.text)
    except ValueError:
        detail = r.text
    detail = "; ".join(detail) if isinstance(detail, list) else str(detail)
    reason = {401: "The page's API key was refused (ask the owner to check HIPPO_API_KEY)",
              404: "Unknown model", 413: "The file is too large",
              422: "The scan can't be used"}.get(r.status_code, f"Error {r.status_code}")
    return ApiError(f"{reason}: {detail}")


def models(api_url: str, api_key: str | None, session=requests) -> dict:
    """{"default": name, "models": [{key, name, version, combination, default, ...}]} from GET /models."""
    try:
        r = session.get(api_url + "/models", headers={"X-API-Key": api_key} if api_key else {}, timeout=TIMEOUT_S)
    except requests.RequestException as e:
        raise ApiError(f"The model service couldn't be reached ({e.__class__.__name__}). Try again in a minute.") from None
    if r.status_code != 200:
        raise _refusal(r)
    return r.json()


def segment(api_url: str, api_key: str | None, data: bytes, filename: str, kind: str = "crop",
            largest: bool = False, session=requests, model: str | None = None) -> dict:
    """Send one scan to the API and return its JSON (with masks); raises ApiError with the reason.

    model: a name from models(); None uses the API's default.
    """
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {sorted(KINDS)}, not {kind!r}")
    params = {"include_mask": "true"}
    if largest:
        params["largest"] = "true"
    if model:
        params["model"] = model
    try:
        r = session.post(api_url + KINDS[kind], files={"file": (filename, data)}, params=params,
                         headers={"X-API-Key": api_key} if api_key else {}, timeout=TIMEOUT_S)
    except requests.RequestException as e:
        raise ApiError(f"The model service couldn't be reached ({e.__class__.__name__}). Try again in a minute.") from None
    if r.status_code == 200:
        return r.json()
    raise _refusal(r)


def mask_image(hippocampus: dict) -> nib.Nifti1Image:
    """The mask the API returned for one hippocampus (on the crop's grid, with its affine)."""
    raw = gzip.decompress(base64.b64decode(hippocampus["mask_nifti_gz_base64"]))
    return nib.Nifti1Image.from_bytes(raw)


def mask_bytes(hippocampus: dict) -> bytes:
    """The mask as a .nii.gz file, for a download button."""
    return base64.b64decode(hippocampus["mask_nifti_gz_base64"])


def agreement(mask_a: nib.Nifti1Image, mask_b: nib.Nifti1Image) -> dict:
    """Dice between two models' masks of the same scan (whole, anterior, posterior); 1 = identical.

    Agreement, not accuracy: neither mask is the truth. Both come from the same upload and box, so the grids match.
    """
    a, b = (np.rint(np.asarray(m.dataobj)).astype(np.uint8) for m in (mask_a, mask_b))
    if a.shape != b.shape or not np.allclose(mask_a.affine, mask_b.affine):
        raise ValueError("the masks are on different grids")
    dice = lambda x, y: float(2 * (x & y).sum() / (x.sum() + y.sum())) if (x.sum() + y.sum()) else 1.0
    return {"whole": dice(a > 0, b > 0), "anterior": dice(a == 1, b == 1), "posterior": dice(a == 2, b == 2)}


def accuracy(mask: nib.Nifti1Image, label: nib.Nifti1Image, parts: bool = True) -> dict:
    """Dice of a predicted mask against a manual label on the same grid (an example scan's ground truth).

    parts=False for labels without anterior/posterior (MNI-HISUB25 subfields): only the whole hippocampus is scored.
    """
    dice = agreement(mask, label)
    return dice if parts else {"whole": dice["whole"]}


def read_scan(data: bytes) -> nib.Nifti1Image:
    """An uploaded .nii or .nii.gz file as a NIfTI image, in memory."""
    return nib.Nifti1Image.from_bytes(gzip.decompress(data) if data[:2] == b"\x1f\x8b" else data)


def scan_region(scan: nib.Nifti1Image, mask: nib.Nifti1Image) -> nib.Nifti1Image:
    """The part of the scan under the mask, on the mask's grid: the whole crop, or one atlas box of a brain.

    The API cut the mask's grid out of this same scan, so the offset is a whole number of voxels.
    """
    start = np.rint(np.linalg.inv(scan.affine) @ mask.affine[:, 3])[:3].astype(int)
    if (start < 0).any() or (start + np.array(mask.shape) > np.array(scan.shape[:3])).any():
        raise ValueError("the mask doesn't lie inside the scan")
    region = tuple(slice(s, s + n) for s, n in zip(start, mask.shape))
    return nib.Nifti1Image(np.asarray(scan.dataobj, dtype=np.float32)[region], mask.affine)
