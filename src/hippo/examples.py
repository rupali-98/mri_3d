"""Example scans for the cloud upload page, with their manual labels, so a visitor without files can try it.

Four hippocampus crops, stored in the bucket (gs://BUCKET/examples/) and read by the page at run time
(HIPPO_EXAMPLES_URI), so they can change without rebuilding the page:
  - hippocampus_008 (MSD test) and hippocampus_007 (MSD validation): a possible left/right pair. MSD records
    neither side nor subject; files 2k-1 and 2k probably are one person's two sides, but that is a guess. No
    MSD test case has its partner in test, so this pair has one test and one validation case: neither was
    used to train the weights (validation only chose the checkpoint). 008 is near the test mean (anterior
    Dice 0.894, posterior 0.917 for the 3D U-Net).
  - s03_L and s03_R (MNI-HISUB25): one person's left and right hippocampus, atlas-box crops as hippo.external
    makes them. Their labels are subfields, not anterior/posterior, so only the whole hippocampus is scored.

Licences (shown on the page): MSD Task04 CC BY-SA 4.0; MNI-HISUB25 CC BY-NC-SA 3.0 (Kulaga-Yoskovitz et al.,
Scientific Data 2015), so non-commercial use with attribution only.

This module is copied into the upload page's image: keep it free of PyTorch and MONAI.

Outputs: <out>/<key>/image.nii.gz and label.nii.gz per example, and <out>/examples.json (titles, side,
split, licence, whether the label has anterior/posterior parts).

Usage:
    python -m hippo.examples --out deploy/examples                           # copy the 4 crops + examples.json
    gcloud storage cp --recursive deploy/examples/* gs://BUCKET/examples/    # you run this: it publishes data
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

MSD_LICENCE = "MSD Task04 Hippocampus (Vanderbilt), CC BY-SA 4.0."
HISUB_LICENCE = ("MNI-HISUB25 (Kulaga-Yoskovitz et al., Scientific Data 2015, NITRC), CC BY-NC-SA 3.0: "
                 "non-commercial use only.")

# Paths are relative to the data folder (defaults.DATA_DIR)
EXAMPLES = [
    {"key": "msd_008", "title": "MSD hippocampus_008 (test case)", "dataset": "MSD Task04", "split": "test",
     "side": "not recorded (possible partner: msd_007)", "parts": True, "kind": "crop",
     "image": "Task04_Hippocampus/imagesTr/hippocampus_008.nii.gz",
     "label": "Task04_Hippocampus/labelsTr/hippocampus_008.nii.gz",
     "note": "In distribution: a held-out MSD test crop. MSD doesn't record the side; files 007 and 008 are probably "
             "one person's two sides.", "licence": MSD_LICENCE},
    {"key": "msd_007", "title": "MSD hippocampus_007 (validation case)", "dataset": "MSD Task04", "split": "validation",
     "side": "not recorded (possible partner: msd_008)", "parts": True, "kind": "crop",
     "image": "Task04_Hippocampus/imagesTr/hippocampus_007.nii.gz",
     "label": "Task04_Hippocampus/labelsTr/hippocampus_007.nii.gz",
     "note": "In distribution: an MSD validation crop (not trained on; it helped choose the checkpoint). Probably the "
             "other side of hippocampus_008.", "licence": MSD_LICENCE},
    {"key": "s03_L", "title": "MNI-HISUB25 s03, left (external)", "dataset": "MNI-HISUB25", "split": "external test",
     "side": "left", "parts": False, "kind": "crop",
     "image": "external/imagesTr/s03_L.nii.gz", "label": "external/labelsTr/s03_L.nii.gz",
     "note": "External: another scanner and labelling protocol, cropped around the atlas centre. The label is in "
             "subfields, so only the whole hippocampus is scored.", "licence": HISUB_LICENCE},
    {"key": "s03_R", "title": "MNI-HISUB25 s03, right (external)", "dataset": "MNI-HISUB25", "split": "external test",
     "side": "right", "parts": False, "kind": "crop",
     "image": "external/imagesTr/s03_R.nii.gz", "label": "external/labelsTr/s03_R.nii.gz",
     "note": "External: the same person's right hippocampus. The label is in subfields, so only the whole "
             "hippocampus is scored.", "licence": HISUB_LICENCE},
]
MANIFEST = "examples.json"


def prepare(out: str, data_dir: str = "data", examples: list[dict] = EXAMPLES) -> Path:
    """Copy each example's image and label to <out>/<key>/ and write examples.json; returns the manifest path."""
    out = Path(out)
    listed = []
    for e in examples:
        folder = out / e["key"]
        folder.mkdir(parents=True, exist_ok=True)
        for part in ("image", "label"):
            source = Path(data_dir) / e[part]
            if not source.exists():
                raise SystemExit(f"{source} is missing (example {e['key']}); run hippo.data / hippo.external first")
            shutil.copyfile(source, folder / f"{part}.nii.gz")
        listed.append({k: v for k, v in e.items() if k not in ("image", "label")}
                      | {"image": f"{e['key']}/image.nii.gz", "label": f"{e['key']}/label.nii.gz"})
    (out / MANIFEST).write_text(json.dumps({"examples": listed}, indent=2))
    return out / MANIFEST


def read(uri: str, name: str) -> bytes:
    """One file from the examples store: a gs://bucket/prefix (read with the service's own account) or a folder."""
    if uri.startswith("gs://"):
        from google.cloud import storage  # only needed for the bucket
        bucket, _, prefix = uri[len("gs://"):].partition("/")
        path = f"{prefix.strip('/')}/{name}" if prefix.strip("/") else name
        return storage.Client().bucket(bucket).blob(path).download_as_bytes()
    return (Path(uri) / name).read_bytes()


def catalogue(uri: str) -> list[dict]:
    """The examples listed in the store's examples.json."""
    return json.loads(read(uri, MANIFEST))["examples"]


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Copy the upload page's example scans and labels into one folder.")
    p.add_argument("--out", default="deploy/examples", help="Folder to write (then copy it to the bucket)")
    p.add_argument("--data-dir", default="data", help="Folder holding Task04_Hippocampus/ and external/")
    a = p.parse_args()
    print(f"Wrote {prepare(a.out, a.data_dir)}")
