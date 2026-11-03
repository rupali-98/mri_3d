"""Compare predictions slice by slice (3D Slicer style), as a Streamlit page.

Pick a hippocampus crop and its ground truth, tick the runs and click Predict and show. Each run predicts
with its own preprocessing (z-score or raw intensities). You get one row per
mask (ground truth, an optional second reference such as ASHS, then each run), each with
axial, coronal and sagittal views; the three plane sliders in the sidebar move the
same slice in every row. Ticking "3D surfaces" adds a rotatable 3D view of each
mask below the slices (same colours, same Labels/Errors switch).

Predictions are saved to runs/figures/notebook/<case>_pred_<run>.nii.gz (the folder
keeps its old name), so ITK-SNAP or 3D Slicer can open them. Research use only, not for diagnosis.

Usage (from the project root):
    PYTHONPATH=src .venv/bin/streamlit run streamlit_dashboard/compare_slices.py
"""
from __future__ import annotations

import base64
import os
import sys
from pathlib import Path

# Work from the project root (so data/ and runs/ resolve) and import hippo from src/
ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import nibabel as nib
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import yaml

from hippo import visualize
from hippo.external import largest_piece
from hippo.slices import (ERRORS, NONE, VIEWS, Scan, error_map, guess_mask, label_colours, legend_html,
                          list_nifti, mm_per_px, on_grid, rgb_colours, surfaces)

OUT = Path("runs/figures/notebook")  # predictions are saved here (name kept so earlier files are still found)
MAX_CROP = 128  # longest side (voxels) accepted as a hippocampus crop
PRED_COLOURS = rgb_colours(label_colours({1, 2}))  # anterior orange, posterior blue
ERROR_COLOURS = rgb_colours(ERRORS)
PANELS_PER_LINE = 3  # 3D views side by side
ASHS = Path("data/ashs/labelsTr")  # ASHS-PMC-T1 anterior/posterior on the external crops (archive/external_analysis/ashs_labels.py)
LIGHT = dict(ambient=0.45, diffuse=0.8, specular=0.25, roughness=0.6)
# 3D hover: the part, its world position and the slices to move the plane sliders to
HOVER = ("<b>%{fullData.name}</b><br>x %{x:.1f}  y %{y:.1f}  z %{z:.1f} mm<br>"
         "sliders: sagittal %{customdata[0]}, coronal %{customdata[1]}, axial %{customdata[2]}<extra></extra>")


def predict(run: str, image: str) -> Path:
    """Save the run's prediction for `image` as NIfTI (aligned with the MRI by its affine); returns the path."""
    run = Path(run)
    normalize = yaml.safe_load((run / "config.yaml").read_text())["data"].get("normalize", "zscore")
    d = visualize.load(Path(image), None, normalize)
    mask = visualize.predict(run, d["image"])
    OUT.mkdir(parents=True, exist_ok=True)
    case = Path(image).name.split(".nii")[0]
    parts = Path(image).parts
    if len(parts) >= 4 and parts[-4] != "data":
        # data/<set>/<condition>/imagesTr/<case> (archive/shift.py, external --recentre): one file per condition,
        # so it doesn't overwrite the prediction of the usual crop of the same case
        case += f"_{parts[-4]}_{parts[-3]}"
    path = OUT / f"{case}_pred_{run.name}.nii.gz"
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), np.asarray(d["image"].affine, dtype=np.float64)), path)
    return path


@st.cache_data(show_spinner=False, max_entries=64)
def prediction(run: str, image: str, stamp: tuple) -> np.ndarray:
    """The run's prediction on the image grid; `stamp` (file times) re-predicts after retraining."""
    scan = load_scan(image)
    return on_grid(predict(run, image), scan.data.shape, scan.spacing, scan.origin)


@st.cache_resource(show_spinner=False, max_entries=16)
def load_scan(image: str) -> Scan:
    return Scan(image)


def dice_text(pred: np.ndarray, truth: np.ndarray, parts: bool) -> str:
    """Whole-hippocampus Dice, plus anterior/posterior when the ground truth has those labels."""
    p, t = pred > 0, truth > 0
    text = f"Dice whole {2 * (p & t).sum() / max(p.sum() + t.sum(), 1):.3f}"
    if parts:
        text += ", " + ", ".join(f"{name} {d:.3f}" for name, d in visualize.dice(pred, truth).items())
    return text


def reference_name(path: str) -> str:
    """Row name for the second reference: ASHS for its crops, else the file name."""
    return "ASHS-PMC-T1" if Path(path).parent == ASHS else Path(path).name.split(".nii")[0]


def external_crop(image: str) -> bool:
    """True for crops of the external dataset (data/external*/, data/shifted_external/), False for MSD."""
    return any(part.startswith("external") or part == "shifted_external" for part in Path(image).parts)


def slice_numbers(scan: Scan, points: np.ndarray) -> np.ndarray:
    """The sagittal, coronal and axial plane-slider values (voxel index) nearest each point in mm."""
    idx = np.rint((points - np.asarray(scan.origin)) / np.asarray(scan.spacing)).astype(int)
    # a list, not an array: plotly would pack an array as binary, which hover text can't always read
    return np.clip(idx, 0, np.asarray(scan.data.shape) - 1).tolist()


def figure_3d(scan: Scan, labels: np.ndarray, colours: dict, ghost: np.ndarray | None, point) -> go.Figure:
    """Rotatable surfaces of one mask, with an optional faint ground truth and the crosshair point."""
    def mesh(v, f, **style):
        return go.Mesh3d(x=v[:, 0], y=v[:, 1], z=v[:, 2], i=f[:, 0], j=f[:, 1], k=f[:, 2], showlegend=True,
                         customdata=slice_numbers(scan, v), hovertemplate=HOVER, **style)

    traces = [mesh(v, f, color=c, name=name, lighting=LIGHT,
                   opacity=0.3 if name == "correct" else 1.0)  # see-through, so the errors stand out
              for name, c, v, f in surfaces(labels, scan.spacing, scan.origin, colours)]
    if ghost is not None:
        traces += [mesh(v, f, color="#8c8c8c", opacity=0.15, name="ground truth")
                   for _, _, v, f in surfaces((ghost > 0).astype(int), scan.spacing, scan.origin,
                                              {1: ("ground truth", "grey")})]
    if point is not None:
        traces.append(go.Scatter3d(x=[point[0]], y=[point[1]], z=[point[2]], mode="markers", name="crosshair",
                                   marker=dict(size=4, color="#8c8c8c", symbol="diamond"),
                                   customdata=slice_numbers(scan, np.array([point])), hovertemplate=HOVER))
    # every panel shows the whole crop at the same scale and camera, so they can be compared
    lo = np.asarray(scan.origin) - np.asarray(scan.spacing) / 2
    hi = lo + np.asarray(scan.data.shape) * np.asarray(scan.spacing)
    ratio = (hi - lo) / (hi - lo).max()
    axis = lambda a, title: dict(range=[lo[a], hi[a]], title=title, showbackground=False)
    fig = go.Figure(traces)
    fig.update_layout(
        height=420, margin=dict(l=0, r=0, t=0, b=0), uirevision="keep",  # keeps the rotation when a slider moves
        legend=dict(orientation="h", yanchor="bottom", y=0, x=0),
        scene=dict(xaxis=axis(0, "x: L → R (mm)"), yaxis=axis(1, "y: P → A (mm)"), zaxis=axis(2, "z: I → S (mm)"),
                   aspectmode="manual", aspectratio=dict(x=ratio[0], y=ratio[1], z=ratio[2]),
                   camera=dict(eye=dict(x=-1.5, y=1.1, z=0.8), up=dict(x=0, y=0, z=1))),
    )
    return fig


def png_html(png: bytes, colour: str) -> str:
    """A slice as an inline image with its view's colour as the border (exact pixel size, unlike st.image)."""
    data = base64.b64encode(png).decode()
    return f"<img src='data:image/png;base64,{data}' style='border:2px solid {colour};margin:2px'>"


st.set_page_config(page_title="Compare slices", layout="wide")
st.title("Compare predictions slice by slice")
st.caption("Research use only, not for diagnosis.")

# Hippocampus crops only: MSD images*/ folders and the external crops in data/external/imagesTr
# (whole-brain scans such as data/mri_dataset are left out; the model was trained on crops)
files = [f for f in list_nifti() if Path(f).parent.name.startswith("images")]
truths = [f for f in list_nifti() if Path(f).parent.name.startswith("labels")]
runs_found = sorted(str(p.parent) for p in Path("runs").glob("*/best.pt"))

with st.sidebar:
    st.header("Image and runs")
    terms = st.text_input("Filter", placeholder="e.g.  hippocampus_135   or   s03_L").lower().split()
    image = st.selectbox("Image", [f for f in files if all(t in f.lower() for t in terms)], key="image")
    guess = guess_mask(image) if image else NONE
    options = [NONE] + truths + ([guess] if guess not in truths and guess != NONE else [])
    # keyed by image, so picking another image starts from its own label file
    truth_path = st.selectbox("Ground truth", options, index=options.index(guess), key=f"truth::{image}")
    # an independent tool's labels, e.g. ASHS anterior/posterior for an external crop (picked when there is one)
    ashs = str(ASHS / Path(image).name) if image else NONE
    second_options = [NONE] + truths
    second_path = st.selectbox("Second reference", second_options,
                               index=second_options.index(ashs) if ashs in truths else 0, key=f"second::{image}",
                               help="Shown as its own row, scored against the ground truth; each run is also "
                                    "scored against it. ASHS-PMC-T1 crops are in data/ashs/labelsTr")
    runs = st.multiselect("Runs", runs_found, default=runs_found, format_func=lambda r: Path(r).name)
    if st.button("Predict and show", type="primary", disabled=not (image and runs)):
        st.session_state["shown"] = {"image": image, "truth": truth_path, "second": second_path, "runs": runs}
        for _, axis, _, _ in VIEWS:  # new image: sliders start again at the hippocampus
            st.session_state.pop(f"pos{axis}", None)

    st.header("Display")
    shown = st.session_state.get("shown")
    has_truth = bool(shown) and shown["truth"] != NONE
    colour = st.radio("Colour", ["Labels", "Errors"], horizontal=True, disabled=not has_truth,
                      help="Errors needs a ground truth")
    errors = colour == "Errors" and has_truth
    contrast = st.slider("Contrast %", 0, 100, (1, 99))
    opacity = st.slider("Opacity", 0.0, 1.0, 0.5, 0.05)
    crosshair = st.checkbox("Crosshair", value=True)
    size = st.slider("Size px", 120, 480, 240, 20)
    external = bool(shown) and external_crop(shown["image"])
    largest = st.checkbox("Largest piece only", value=False, disabled=not external,
                          help="Post-processing for external crops only (the MSD results use the plain prediction): "
                               "keep each prediction's largest connected piece (as hippo.external --largest-piece); "
                               "Dice is recomputed") and external  # a tick left over from an external image is ignored
    view3d = st.checkbox("3D surfaces", value=False, help="A rotatable 3D view of each mask, below the slices")
    ghost = st.checkbox("Ground truth as a ghost in 3D", value=True, disabled=not (view3d and has_truth),
                        help="The ground truth as a faint grey surface in each run's 3D view (Labels colour only)")

if not shown:
    st.markdown(
        "Pick an image (and its ground truth) in the sidebar, choose the runs "
        "and click **Predict and show**. Then move through the slices with the **Planes** sliders in the sidebar.\n\n"
        "- **Labels**: orange = anterior, blue = posterior.\n"
        "- **Errors** (against the ground truth): green = correct, yellow = wrong part, blue = missed, "
        "red = extra. External crops (subfield labels) compare the whole hippocampus only, so there is no yellow.\n"
        "- Ground truth row: anterior/posterior (MSD), or gold = CA1-3, green = subiculum, purple = CA4-DG "
        "(external, names inferred from the atlas).\n"
        "- **3D surfaces** (Display): a rotatable 3D view of each mask below the slices, with the ground truth "
        "as a faint grey ghost in each run's view.\n"
        "- **Largest piece only** (Display, external crops only): post-processing that keeps each prediction's "
        "largest connected piece; the row title says how many pieces there were, and Dice is recomputed. "
        "It is greyed out for MSD images, whose results always use the plain prediction.\n"
        "- **Second reference** (sidebar): another set of labels shown as its own row, e.g. ASHS-PMC-T1 "
        "anterior/posterior for s03 and s17 (picked automatically for those external crops). Its row is scored "
        "against the ground truth, and each run's title adds its Dice against it (anterior/posterior too). "
        "ASHS is another method, not ground truth; Errors colours still compare with the ground truth.\n\n"
        "Images listed: MSD `imagesTr`/`imagesTs`, external crops in `data/external/imagesTr` "
        "(made by `python -m hippo.external`) and shifted MSD test crops in `data/shifted/<condition>/imagesTr` "
        "(made by `PYTHONPATH=src python archive/shift.py`; type e.g. `shifted/y-5` in the filter) and shifted external crops in "
        "`data/shifted_external/<condition>/imagesTr` (`archive/shift.py --external`; e.g. `shifted_external/y-5 s03_L`) "
        "and boxes re-centred on the model's first prediction in `data/external_recentred/recentred/imagesTr` "
        "(`hippo.external --recentre`; e.g. `recentred/ s03_L`). Whole-brain scans are not listed (the model expects crops): open them in ITK-SNAP or 3D Slicer."
    )
    if not runs_found:
        st.warning("No trained runs found (runs/*/best.pt).")
    st.stop()

try:
    scan = load_scan(shown["image"])
    if max(scan.data.shape) > MAX_CROP:
        raise ValueError(f"{scan.data.shape} voxels looks like a whole-brain scan; the model expects a "
                         "hippocampus crop (MSD, or data/external/imagesTr)")
    truth = on_grid(shown["truth"], scan.data.shape, scan.spacing, scan.origin) if shown["truth"] != NONE else None
    rows = []  # (title, labels, colours, error map or None)
    if truth is not None:
        values = set(np.unique(truth).tolist()) - {0}
        parts = values <= {1, 2}
        rows.append(("ground truth", truth, rgb_colours(label_colours(values)), None))
    second = shown.get("second", NONE)
    second = on_grid(second, scan.data.shape, scan.spacing, scan.origin) if second != NONE else None
    if second is not None:
        name2 = reference_name(shown["second"])
        parts2 = set(np.unique(second).tolist()) - {0} <= {1, 2}
        title = name2 + (f": vs ground truth {dice_text(second, truth, parts and parts2)}" if truth is not None else "")
        rows.append((title, second, rgb_colours(label_colours(set(np.unique(second).tolist()) - {0})),
                     error_map(second, truth, parts and parts2) if truth is not None else None))
    for run in shown["runs"]:
        stamp = (os.path.getmtime(Path(run) / "best.pt"), os.path.getmtime(shown["image"]))
        with st.spinner(f"Predicting with {Path(run).name} ..."):
            pred = prediction(run, shown["image"], stamp)
        name = Path(run).name
        if largest:
            pred, pieces = largest_piece(pred)
            name += f" (largest of {pieces} pieces)"
        title = name + (f": {dice_text(pred, truth, parts)}" if truth is not None else "")
        if second is not None:
            title += f"{':' if truth is None else ' |'} vs {name2} {dice_text(pred, second, parts2)}"
        rows.append((title, pred, PRED_COLOURS, error_map(pred, truth, parts) if truth is not None else None))
except Exception as e:  # show the problem instead of a long traceback
    st.error(f"Could not show this file: {e}")
    st.stop()

shape, spacing = scan.data.shape, scan.spacing
st.markdown(f"**{Path(shown['image']).name}**: {shape[0]} x {shape[1]} x {shape[2]} voxels, "
            f"{spacing[0]:.2g} x {spacing[1]:.2g} x {spacing[2]:.2g} mm")

# start at the ground truth when there is one, else at the first prediction
voxels = np.argwhere(rows[0][1] > 0)
start = [int(c) for c in voxels[np.abs(voxels - voxels.mean(0)).sum(1).argmin()]] if len(voxels) \
    else [n // 2 for n in shape]
pos = list(start)
with st.sidebar:
    st.header("Planes")
    for name, axis, _, view_colour in VIEWS:
        st.markdown(f"<span style='color:{view_colour}'>■</span> **{name}**", unsafe_allow_html=True)
        pos[axis] = st.slider(name, 0, shape[axis] - 1, start[axis], key=f"pos{axis}", label_visibility="collapsed")
st.caption(" | ".join(f"{name} slice {pos[axis] + 1} / {shape[axis]} ({'xyz'[axis]} = {scan.mm(axis, pos[axis]):.1f} mm)"
                      for name, axis, _, _ in VIEWS))
st.markdown(legend_html(ERROR_COLOURS if errors else PRED_COLOURS), unsafe_allow_html=True)

scale = mm_per_px(shape, spacing, size)
for title, labels, colours, error in rows:
    labels, colours = (error, ERROR_COLOURS) if errors and error is not None else (labels, colours)
    pictures = "".join(png_html(scan.render(view, pos, contrast, opacity, crosshair, scale,
                                            labels=labels, colours=colours), view[3]) for view in VIEWS)
    st.markdown(f"**{title}**<br>{pictures}", unsafe_allow_html=True)

st.code(scan.describe(pos), language=None)
st.caption("Label at the crosshair: " + ", ".join(f"{t.split(':')[0]} {int(lab[tuple(pos)])}" for t, lab, _, _ in rows))

if view3d:
    st.subheader("3D")
    st.caption("Drag to rotate, scroll to zoom, right-drag to pan; click a legend entry to hide it. Each view rotates "
               "on its own and keeps its rotation while you move the plane sliders. The grey diamond is the crosshair. "
               "Hover over a surface for its position (mm) and the sagittal/coronal/axial slider values that "
               "show that spot in the slices. "
               + ("Errors: correct is see-through, so missed (blue), extra (red) and wrong part (yellow) stand out."
                  if errors else ""))
    point = [scan.mm(a, pos[a]) for a in range(3)] if crosshair else None
    for start_row in range(0, len(rows), PANELS_PER_LINE):
        for col, (title, labels, colours, error) in zip(st.columns(PANELS_PER_LINE),
                                                        rows[start_row:start_row + PANELS_PER_LINE]):
            labels, colours = (error, ERROR_COLOURS) if errors and error is not None else (labels, colours)
            faint = truth if ghost and truth is not None and not errors and title != "ground truth" else None
            with col:
                st.markdown(f"**{title}**")
                st.plotly_chart(figure_3d(scan, labels, colours, faint, point), key=f"3d::{title.split(':')[0]}")
