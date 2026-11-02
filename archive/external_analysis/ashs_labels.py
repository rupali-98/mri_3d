"""ASHS-PMC-T1 anterior/posterior labels cut to each external crop box, for the Streamlit page.

Reads the ASHS segmentation of each subject in data/ashs/<sNN>/result/layer_002_*.nii.gz
(ITK-SNAP DSS output, whole head, 0.5 x 1 x 0.5 mm) and keeps the part inside that
hemisphere's crop box from data/external/cases.json, at ASHS's own resolution.
Labels become 1 = anterior, 2 = posterior (ASHS: left 1/2, right 101/102); every
other ASHS structure (ERC, BA35/36, PHC, ...) is dropped.

Outputs: data/ashs/labelsTr/<sNN>_{L,R}.nii.gz; the page lines them up with the crop
by world position, so pick one as "Second reference" (or "Ground truth").

Usage (from the project root):
    PYTHONPATH=src python archive/external_analysis/ashs_labels.py
"""
import glob
import json
from pathlib import Path

import nibabel as nib
import numpy as np

OUT = Path("data/ashs/labelsTr")
ASHS = {"L": (1, 2), "R": (101, 102)}  # (anterior, posterior)


def crop(seg_img, crop_img) -> nib.Nifti1Image:
    """The part of the ASHS grid inside the crop box, keeping its voxel size."""
    seg_img = nib.as_closest_canonical(seg_img)
    seg = np.rint(np.asarray(seg_img.dataobj)).astype(np.int32)
    sp, org = np.array(seg_img.header.get_zooms()[:3]), seg_img.affine[:3, 3]
    cs, co = np.array(crop_img.header.get_zooms()[:3]), crop_img.affine[:3, 3]
    lo, hi = co - cs / 2, co + (np.array(crop_img.shape[:3]) - 0.5) * cs  # outer faces of the crop, world mm
    box = tuple(slice(max(int(np.ceil((lo[a] - org[a]) / sp[a])), 0),
                      min(int(np.floor((hi[a] - org[a]) / sp[a])) + 1, seg.shape[a])) for a in range(3))
    affine = seg_img.affine.copy()
    affine[:3, 3] = org + np.array([b.start for b in box]) * sp
    return nib.Nifti1Image(seg[box], affine)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for case in json.load(open("data/external/cases.json")):
        subject, side = case["case"].split("_")
        found = glob.glob(f"data/ashs/{subject}/result/layer_002_*.nii.gz")
        if not found:
            continue  # ASHS not run on this subject
        part = crop(nib.load(found[0]), nib.load(case["image"]))
        seg = np.asarray(part.dataobj)
        ant, post = ASHS[side]
        labels = np.where(seg == ant, 1, np.where(seg == post, 2, 0)).astype(np.uint8)
        nib.save(nib.Nifti1Image(labels, part.affine), OUT / f"{case['case']}.nii.gz")
        print(f"{case['case']}: anterior {(labels == 1).sum()}, posterior {(labels == 2).sum()} voxels "
              f"({np.prod(part.header.get_zooms()[:3]):.2f} mm³ each)")


if __name__ == "__main__":
    main()
