"""Show a 3D MRI volume with its label and/or model prediction.

Draws sagittal, coronal and axial slices through the hippocampus, plus a 3D
surface of each mask, and prints the volume of each part in ml.

Usage:
    python -m hippo.visualize --case hippocampus_222                      # MRI + ground truth
    python -m hippo.visualize --case hippocampus_222 --run runs/baseline  # + model prediction
    python -m hippo.visualize --case hippocampus_002 --run runs/baseline  # imagesTs: prediction only
    python -m hippo.visualize --image scan.nii.gz [--label mask.nii.gz] --show
    python -m hippo.visualize --case hippocampus_222 --run runs/baseline --html  # + rotatable 3D view
    python -m hippo.visualize --case hippocampus_222 --run runs/baseline --montage  # + every 3rd sagittal slice
    python -m hippo.visualize --case hippocampus_222 --run runs/baseline --save-nifti  # prediction as NIfTI
    python -m hippo.visualize --case hippocampus_135 --run runs/baseline runs/no_norm --html  # compare runs
"""
from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import yaml
from monai.transforms import Compose

from . import CLASS_NAMES, defaults
from .data import base_transforms
from .model import build_model

COLORS = {1: "#e4572e", 2: "#2e86ab"}  # anterior, posterior
OUT_DIR = "runs/figures"


def load(image: Path, label: Path | None, normalize: str = "zscore") -> dict:
    """Load in RAS orientation, normalised and padded exactly as the model sees it in training."""
    keys = ["image", "label"] if label else ["image"]
    t = Compose(base_transforms(normalize, keys))
    return t({"image": str(image), **({"label": str(label)} if label else {})})


@torch.no_grad()
def predict(run: Path, image: torch.Tensor) -> np.ndarray:
    cfg = yaml.safe_load((run / "config.yaml").read_text())
    model = build_model(cfg)
    model.load_state_dict(torch.load(run / "best.pt", map_location="cpu", weights_only=True))
    model.eval()
    return model(image[None]).argmax(1)[0].numpy()


def volumes_ml(mask: np.ndarray, spacing) -> dict:
    voxel_ml = float(np.prod(spacing)) / 1000
    return {name: round(float((mask == c).sum()) * voxel_ml, 3) for c, name in CLASS_NAMES.items()}


def dice(pred: np.ndarray, truth: np.ndarray) -> dict:
    return {name: round(2 * float(((pred == c) & (truth == c)).sum())
                        / max(float((pred == c).sum() + (truth == c).sum()), 1), 3)
            for c, name in CLASS_NAMES.items()}


def draw_surface(ax, mask: np.ndarray, spacing) -> None:
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from skimage.measure import marching_cubes

    for c, color in COLORS.items():
        if (mask == c).sum() < 2:
            continue
        verts, faces, _, _ = marching_cubes(np.pad(mask == c, 1).astype(np.uint8), 0.5, spacing=spacing)
        ax.add_collection3d(Poly3DCollection(verts[faces], facecolor=color, edgecolor="none", alpha=0.7))
    extent = np.array(mask.shape) * spacing
    ax.set(xlim=(0, extent[0]), ylim=(0, extent[1]), zlim=(0, extent[2]), xlabel="R", ylabel="A", zlabel="S")
    ax.set_box_aspect(extent)
    ax.view_init(elev=20, azim=-60)
    ax.tick_params(labelsize=6)


def save_montage(img, masks: dict, spacing, centre_x: int, step: int, cmap, path: Path, title: str) -> None:
    """Every `step`-th sagittal slice across the hippocampus, aligned so the middle slice is included."""
    import matplotlib.pyplot as plt

    fg = next(iter(masks.values()), np.zeros_like(img)) > 0
    lo, hi = (np.argwhere(fg)[:, 0].min(), np.argwhere(fg)[:, 0].max()) if fg.any() else (0, img.shape[0] - 1)
    xs = list(range(centre_x - (centre_x - lo) // step * step, hi + 1, step))
    rows = [("MRI", None)] + list(masks.items())
    aspect = spacing[2] / spacing[1]
    fig, axes = plt.subplots(len(rows), len(xs), figsize=(2.3 * len(xs), 2.4 * len(rows)), squeeze=False)
    for r, (row_name, mask) in enumerate(rows):
        for c, x in enumerate(xs):
            ax = axes[r][c]
            ax.imshow(img[x].T, cmap="gray", origin="lower", aspect=aspect)
            if mask is not None:
                ax.imshow(mask[x].T, cmap=cmap, vmin=0, vmax=2, alpha=0.55, origin="lower",
                          aspect=aspect, interpolation="nearest")
            label = f"x={x}" + (" (middle)" if x == centre_x else "")
            ax.set_title(f"{row_name}\n{label}" if c == 0 else f"\n{label}", fontsize=8)
            ax.axis("off")
    fig.suptitle(f"{title}: sagittal slices every {step} (P ← → A; orange = anterior, blue = posterior)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def save_html(img: np.ndarray, masks: dict, spacing, path: Path, title: str) -> None:
    """Interactive 3D view (rotate, zoom, toggle parts) as a self-contained HTML file."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise SystemExit("--html needs plotly: pip install plotly (or rebuild the Docker image)")

    def coords(box):  # physical coordinates (mm) of every voxel in a box of the volume
        grid = np.mgrid[tuple(slice(a, b) for a, b in box)].astype(np.float32)
        return [(grid[i] * spacing[i]).ravel() for i in range(3)]

    x, y, z = coords([(0, s) for s in img.shape])
    tissue = img[img != 0]  # ignore the zero padding
    traces = [go.Volume(
        x=x, y=y, z=z, value=img.ravel().astype(np.float32),
        isomin=float(np.percentile(tissue, 60)), isomax=float(tissue.max()),
        opacity=0.07, surface_count=12, colorscale="gray", showscale=False, name="MRI", showlegend=True,
    )]
    for i, (mask_name, mask) in enumerate(masks.items()):
        fg = np.argwhere(mask > 0)
        if not len(fg):
            continue
        # crop to the hippocampus (plus a margin) so the file stays small
        box = list(zip(np.maximum(fg.min(0) - 2, 0), np.minimum(fg.max(0) + 3, mask.shape)))
        x, y, z = coords(box)
        sub = mask[tuple(slice(a, b) for a, b in box)]
        for c, part in CLASS_NAMES.items():
            traces.append(go.Isosurface(
                x=x, y=y, z=z, value=(sub == c).ravel().astype(np.float32), isomin=0.5, isomax=1,
                surface_count=1, colorscale=[[0, COLORS[c]], [1, COLORS[c]]], showscale=False,
                name=f"{mask_name}: {part}", showlegend=True,
                visible=True if i == 0 else "legendonly",  # first mask shown; click legend for others
            ))
    fig = go.Figure(traces)
    fig.update_layout(
        title=f"{title}  (drag to rotate, scroll to zoom, click legend to show/hide)",
        height=750, legend=dict(itemsizing="constant"),
        scene=dict(aspectmode="data", xaxis_title="R (mm)", yaxis_title="A (mm)", zaxis_title="S (mm)"),
    )
    fig.write_html(path)


def main(
    image: str,
    label: str | None = None,
    runs: str | list[str] | None = None,
    out: str | None = None,
    show: bool = False,
    html: bool = False,
    montage_step: int = 0,
    save_nifti: bool = False,
) -> dict:
    import matplotlib

    if not show:
        matplotlib.use("Agg")  # containers have no display; just save the figure
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    d = load(Path(image), Path(label) if label else None)
    img = d["image"][0].numpy()
    spacing = tuple(float(s) for s in d["image"].pixdim[:3])
    masks = {}
    if label:
        masks["ground truth"] = d["label"][0].numpy().astype(int)
    runs = [runs] if isinstance(runs, str) else list(runs or [])
    for run in runs:
        # each run sees the image preprocessed the way it was trained (the figure shows the z-scored one)
        normalize = yaml.safe_load((Path(run) / "config.yaml").read_text())["data"].get("normalize", "zscore")
        x = d["image"] if normalize == "zscore" else load(Path(image), None, normalize)["image"]
        masks["prediction" if len(runs) == 1 else Path(run).name] = predict(Path(run), x)
    titles = {name: name for name in masks}
    if label:
        for name in list(masks)[1:]:
            scores = dice(masks[name], masks["ground truth"])
            titles[name] = f"{name} (Dice " + ", ".join(f"{k[:4]} {v:.2f}" for k, v in scores.items()) + ")"

    # slice through the centre of the hippocampus (or the volume, if there is no mask)
    fg = next(iter(masks.values()), np.zeros_like(img)) > 0
    centre = np.argwhere(fg).mean(0).round().astype(int) if fg.any() else np.array(img.shape) // 2

    # (title, 2D slice taken from a 3D array, (horizontal, vertical) spacing)
    views = [
        ("sagittal  (P ← → A)", lambda v: v[centre[0]].T, (spacing[1], spacing[2])),
        ("coronal  (L ← → R)", lambda v: v[:, centre[1], :].T, (spacing[0], spacing[2])),
        ("axial  (L ← → R)", lambda v: v[:, :, centre[2]].T, (spacing[0], spacing[1])),
    ]
    cmap = ListedColormap(["none", COLORS[1], COLORS[2]])
    rows = [("MRI", None)] + list(masks.items())
    fig = plt.figure(figsize=(14, 3.6 * len(rows)))
    results = {}
    for r, (row_name, mask) in enumerate(rows):
        for c, (view_name, cut, (sx, sy)) in enumerate(views):
            ax = fig.add_subplot(len(rows), 4, r * 4 + c + 1)
            ax.imshow(cut(img), cmap="gray", origin="lower", aspect=sy / sx)
            if mask is not None:
                ax.imshow(cut(mask), cmap=cmap, vmin=0, vmax=2, alpha=0.55, origin="lower",
                          aspect=sy / sx, interpolation="nearest")
            ax.set_title(f"{titles.get(row_name, row_name)}: {view_name}" if c == 0 else view_name, fontsize=9)
            ax.axis("off")
        if mask is not None:
            draw_surface(fig.add_subplot(len(rows), 4, r * 4 + 4, projection="3d"), mask, spacing)
            results[row_name] = volumes_ml(mask, spacing)
            print(f"{row_name:>13}: " + ", ".join(f"{k} {v} ml" for k, v in results[row_name].items())
                  + titles[row_name][len(row_name):])

    name = Path(image).name.replace(".nii.gz", "")
    fig.suptitle(f"{name}   shape {img.shape}, spacing {tuple(round(s, 2) for s in spacing)} mm   "
                 f"(orange = anterior, blue = posterior)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.25 / fig.get_figheight()))  # leave room for the suptitle
    out_path = Path(out) if out else Path(OUT_DIR) / f"{name}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    print(f"Saved {out_path}")
    if save_nifti:
        if not runs:
            raise SystemExit("--save-nifti needs --run (there is no prediction to save)")
        # the affine of the loaded image (RAS, including padding) keeps the mask aligned with the MRI
        affine = np.asarray(d["image"].affine, dtype=np.float64)
        for mask_name in list(masks)[1 if label else 0:]:
            suffix = "" if mask_name == "prediction" else f"_{mask_name}"
            nifti_path = out_path.with_name(f"{out_path.stem}_pred{suffix}.nii.gz")
            nib.save(nib.Nifti1Image(masks[mask_name].astype(np.uint8), affine), nifti_path)
            print(f"Saved {nifti_path}")
    if montage_step:
        montage_path = out_path.with_name(f"{out_path.stem}_sagittal.png")
        save_montage(img, masks, spacing, int(centre[0]), montage_step, cmap, montage_path, name)
        print(f"Saved {montage_path}")
    if html:
        html_path = out_path.with_name(f"{out_path.stem}_3d.html")
        save_html(img, masks, spacing, html_path, name)
        print(f"Saved {html_path} (open in any browser)")
    if show:
        plt.show()
    plt.close(fig)
    return results


if __name__ == "__main__":
    p = defaults.make_parser("Visualise a 3D MRI volume with its label and/or prediction.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--case", help="Case name in the dataset, e.g. hippocampus_222")
    src.add_argument("--image", help="Path to any NIfTI image")
    p.add_argument("--label", help="Label NIfTI (with --image)")
    p.add_argument("--run", nargs="+", help="Run folder(s) with best.pt + config.yaml, to add each model's "
                                            "prediction; several runs are shown one row each, by folder name")
    p.add_argument("--out", help=f"Output PNG (default {OUT_DIR}/<case>.png)")
    p.add_argument("--show", action="store_true", help="Also open an interactive window (needs a display)")
    p.add_argument("--html", action="store_true", help="Also save a rotatable 3D view as <out>_3d.html")
    p.add_argument("--montage", type=int, nargs="?", const=3, default=0, metavar="STEP",
                   help="Also save every STEP-th sagittal slice (default 3) as <out>_sagittal.png")
    p.add_argument("--save-nifti", action="store_true",
                   help="Also save the prediction as <out>_pred.nii.gz (needs --run); open it with render3d")
    a = p.parse_args()
    if a.case:
        a.image, a.label = defaults.find_case(a.case, a.data_dir)
    main(str(a.image), str(a.label) if a.label else None, a.run, a.out, a.show, a.html, a.montage,
         a.save_nifti)
