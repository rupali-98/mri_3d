"""Interactive 3D rendering of an MRI volume with vedo.

The MRI is volume-rendered (translucent), the hippocampus parts are solid
surfaces. With both a label and a prediction, they are shown side by side with
a linked camera. Needs a screen (WSL on Windows 11 works); see
requirements-viewer.txt for a light install without PyTorch.

Archived (see archive/README.md); run it from the project root with PYTHONPATH=src.

Usage:
    python archive/render3d.py --case hippocampus_222
    python archive/render3d.py --case hippocampus_222 --pred runs/figures/hippocampus_222_pred.nii.gz
    python archive/render3d.py --image scan.nii.gz [--label mask.nii.gz] [--pred pred.nii.gz]
    python archive/render3d.py --case hippocampus_135 --pred runs/figures/compare/hippocampus_135_pred_*.nii.gz

Make a prediction file first with:
    python -m hippo.visualize --case hippocampus_222 --run runs/baseline --save-nifti
or one per run (named <case>_pred_<run>.nii.gz), to compare runs side by side:
    python -m hippo.visualize --case hippocampus_135 --run runs/baseline runs/no_norm --save-nifti
"""
from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np

from hippo import CLASS_NAMES, defaults
from hippo.slices import load_ras

COLORS = {1: "orangered", 2: "steelblue"}  # anterior, posterior
HELP = "drag: rotate · scroll: zoom · shift+drag: pan · q: quit"


def mri_volume(path: str):
    from vedo import Volume

    data, spacing, origin = load_ras(path)
    vol = Volume(data.astype(np.float32), spacing=spacing, origin=origin)
    lo, hi = np.percentile(data[data > 0], [40, 99.5])
    vol.cmap("bone", vmin=lo, vmax=hi)
    # dark voxels fully transparent, bright tissue faintly visible
    vol.alpha([(lo, 0.0), ((lo + hi) / 2, 0.05), (hi, 0.25)])
    return vol


def label_surfaces(path: str, name: str) -> tuple[list, str]:
    """One smooth surface per hippocampus part, plus a line with their volumes."""
    from vedo import Volume

    data, spacing, origin = load_ras(path)
    data = np.rint(data).astype(np.uint8)
    voxel_ml = float(np.prod(spacing)) / 1000
    surfaces, volumes = [], []
    for value, part in CLASS_NAMES.items():
        if not (data == value).any():
            continue
        part_vol = Volume((data == value).astype(np.uint8), spacing=spacing, origin=origin)
        surfaces.append(part_vol.isosurface(0.5).smooth().c(COLORS[value]).alpha(0.9))
        volumes.append(f"{part} {(data == value).sum() * voxel_ml:.3f} ml")
    summary = f"{name}: " + ", ".join(volumes)
    print(summary)
    return surfaces, summary


def pred_name(path: str, n_preds: int) -> str:
    """Panel title: the run name from <case>_pred_<run>.nii.gz when comparing runs, else "prediction"."""
    if n_preds == 1:
        return "prediction"
    stem = Path(path).name.replace(".nii.gz", "")
    return stem.split("_pred_", 1)[1] if "_pred_" in stem else stem


def build_scenes(image: str, label: str | None = None, pred: str | list[str] | None = None) -> list[list]:
    """Objects for each panel: one panel per mask (or just the MRI if there are none)."""
    from vedo import Text2D

    case = Path(image).name.replace(".nii.gz", "")
    preds = [pred] if isinstance(pred, str) else list(pred or [])
    masks = ([("ground truth", label)] if label else []) + [(pred_name(p, len(preds)), p) for p in preds]
    vol = mri_volume(image)
    if not masks:
        return [[vol, Text2D(f"{case}\n{HELP}", pos="top-left", s=0.8, c="white")]]
    scenes = []
    for i, (name, path) in enumerate(masks):
        surfaces, summary = label_surfaces(path, name)
        text = Text2D(f"{case}\n{summary}\norange = anterior, blue = posterior\n{HELP}",
                      pos="top-left", s=0.7, c="white")
        scenes.append([vol if i == 0 else vol.clone(), *surfaces, text])
    return scenes


def main(image: str, label: str | None = None, pred: str | list[str] | None = None) -> None:
    from vedo import Plotter

    scenes = build_scenes(image, label, pred)
    plt = Plotter(N=len(scenes), sharecam=True, axes=1, bg="black", title="3D NIfTI Visualization")
    for i, objects in enumerate(scenes):  # one panel (renderer) per scene, camera shared
        plt.at(i).show(*objects, viewup="z")
    plt.interactive().close()


if __name__ == "__main__":
    p = defaults.make_parser("Interactive 3D rendering of an MRI with its label and/or prediction.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--case", help="Case name in the dataset, e.g. hippocampus_222")
    src.add_argument("--image", help="Path to any NIfTI image")
    p.add_argument("--label", help="Label NIfTI (with --image)")
    p.add_argument("--pred", nargs="+", help="Prediction NIfTI(s), e.g. from hippo.visualize --save-nifti; "
                                             "several are shown side by side, one panel each")
    a = p.parse_args()
    if a.case:
        a.image, a.label = defaults.find_case(a.case, a.data_dir)
    main(str(a.image), str(a.label) if a.label else None, a.pred)
