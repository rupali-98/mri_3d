"""Upload one scan and see its hippocampus segmentation: the page for the cloud.

A visitor uploads one T1 NIfTI (a hippocampus crop, or a whole-brain scan in MNI space) and clicks
Segment. The page sends it to the deployed API (hippo.api), whose model comes from the MLflow bucket,
and shows the anterior, posterior and whole volumes, the quality flags, the axial, coronal and sagittal
slices with the mask (as in the research page), an optional 3D view, and a download of the mask.
When the API holds several models (MLflow aliases), the Models box chooses one or several: with several,
the same scan is sent to each and the page shows a comparison table (volumes, pieces, flags, and Dice
agreement with the first model picked, the reference), one row of slices per model with shared plane
sliders, 3D views and a mask download per model. The API key stays on the server (HIPPO_API_KEY); visitors
never see it. Nothing is stored.

Visitors without a scan can pick an example instead (hippo.examples: two MSD crops and the left and right
crops of one MNI-HISUB25 subject, read from the bucket). Examples come with their manual labels, so the page
also scores each model against them (Dice); an uploaded scan has no ground truth, so there Dice is only
agreement between models.

Unlike the research page (compare_slices.py), it needs no data/, runs/ or PyTorch. Research use only, not for
diagnosis.

Settings: HIPPO_API_URL (the API's address) and HIPPO_API_KEY or HIPPO_API_KEY_FILE (see hippo.api_client);
HIPPO_EXAMPLES_URI (gs://BUCKET/examples or a local folder from hippo.examples; without it, no examples).

Usage (from the project root):
    HIPPO_API_URL=http://127.0.0.1:8000 PYTHONPATH=src streamlit run streamlit_dashboard/upload_page.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import nibabel as nib
import plotly.graph_objects as go
import streamlit as st

from hippo import api_client, examples
from hippo.slices import VIEWS, Scan, label_colours, mm_per_px, rgb_colours, surfaces

CROP, MNI = "Hippocampus crop", "Whole brain in MNI space"
MAX_MB = 30  # the API's limit (Cloud Run takes at most 32 MiB per request)
COLOURS = rgb_colours(label_colours({1, 2}))  # anterior orangered, posterior steelblue
LIGHT = dict(ambient=0.45, diffuse=0.8, specular=0.25, roughness=0.6)


def scan_for(hippocampus: dict, upload: bytes) -> Scan:
    """The uploaded scan under this hippocampus's mask, as a Scan the slice views can draw."""
    mask = api_client.mask_image(hippocampus)
    region = api_client.scan_region(api_client.read_scan(upload), mask)
    with tempfile.TemporaryDirectory() as folder:  # Scan reads files; they are gone after this
        nib.save(region, Path(folder) / "scan.nii.gz")
        nib.save(mask, Path(folder) / "mask.nii.gz")
        return Scan(str(Path(folder) / "scan.nii.gz"), str(Path(folder) / "mask.nii.gz"))


@st.cache_data(ttl=600, show_spinner=False)
def available_models(api_url: str, api_key: str | None) -> dict:
    """The API's models (GET /models), asked once every 10 minutes rather than on every click."""
    return api_client.models(api_url, api_key)


@st.cache_data(ttl=600, show_spinner=False)
def example_list(uri: str) -> list[dict]:
    """The example scans on offer (examples.json in the bucket), asked once every 10 minutes."""
    return examples.catalogue(uri)


@st.cache_data(show_spinner=False)
def example_files(uri: str, image: str, label: str) -> tuple[bytes, bytes]:
    """One example's scan and manual label (small crops, kept once downloaded)."""
    return examples.read(uri, image), examples.read(uri, label)


def model_label(m: dict) -> str:
    """How a model is named in the dropdown, e.g. "champion: 3D U-Net (single model) · aa4f93ffd511"."""
    return f"{m['key']}: {m['name']} ({m['combination']}) · {m['version']}" + (" · default" if m["default"] else "")


def figure_3d(scan: Scan) -> go.Figure:
    """Rotatable surfaces of the anterior and posterior parts, in world mm."""
    traces = [go.Mesh3d(x=v[:, 0], y=v[:, 1], z=v[:, 2], i=f[:, 0], j=f[:, 1], k=f[:, 2], color=c, name=name,
                        showlegend=True, lighting=LIGHT, hovertemplate=f"<b>{name}</b><extra></extra>")
              for name, c, v, f in surfaces(scan.labels, scan.spacing, scan.origin, COLOURS)]
    fig = go.Figure(traces)
    fig.update_layout(height=420, margin=dict(l=0, r=0, t=0, b=0), legend=dict(orientation="h", y=0),
                      scene=dict(aspectmode="data", xaxis_title="x: L → R (mm)", yaxis_title="y: P → A (mm)",
                                 zaxis_title="z: I → S (mm)", camera=dict(eye=dict(x=-1.5, y=1.1, z=0.8))))
    return fig


st.set_page_config(page_title="Hippocampus segmentation", layout="wide")
st.title("Hippocampus segmentation")
st.caption("Research use only, not for diagnosis. Upload public research scans only, never identifiable patient data.")

api_url, api_key = api_client.settings()
if not api_url:
    st.error("This page isn't connected to the model service: set HIPPO_API_URL (and HIPPO_API_KEY).",
             title="Not configured")
    st.stop()

examples_uri = os.environ.get("HIPPO_EXAMPLES_URI")
with st.sidebar:
    st.header("Scan")
    offered_examples = []
    if examples_uri:
        try:
            offered_examples = example_list(examples_uri)
        except Exception as e:  # the page still works with uploads
            st.warning(f"The example scans couldn't be loaded ({type(e).__name__}).", title="No examples")
    by_example = {e["key"]: e for e in offered_examples}
    source = st.selectbox("Example scan", ["upload"] + list(by_example),
                          format_func=lambda k: "None: upload my own" if k == "upload" else by_example[k]["title"],
                          help="Try the page without a file of your own: public research crops with their manual "
                               "labels, so the result is also scored against the label") if by_example else "upload"
    example = by_example.get(source)
    if example:
        data, label = example_files(examples_uri, example["image"], example["label"])
        name = f"{example['key']}.nii.gz"
        kind = CROP if example["kind"] == "crop" else MNI
        st.caption(f"{example['note']} Side: {example['side']}. Data: {example['licence']}")
    else:
        upload = st.file_uploader("T1 scan (NIfTI)", type=["nii", "gz"], max_upload_size=MAX_MB,
                                  help="One 1 mm T1-weighted scan, .nii or .nii.gz, up to 30 MB")
        kind = st.segmented_control("Scan type", [CROP, MNI], default=CROP, required=True,
                                    help="A crop around one hippocampus (about 35 × 50 × 35 voxels, as in MSD), or a "
                                         "whole-brain scan already registered to MNI space (both sides are segmented)")
        data, name, label = (upload.getvalue(), upload.name, None) if upload else (None, None, None)
    try:
        offered = available_models(api_url, api_key)
    except api_client.ApiError as e:
        st.error(str(e), title="The model service isn't available")
        st.stop()
    by_key = {m["key"]: m for m in offered["models"]}
    keys = [offered["default"]] + [k for k in by_key if k != offered["default"]]  # the default first
    chosen = st.multiselect("Models", keys, default=keys[:1], format_func=lambda k: model_label(by_key[k]),
                            help="The models this service holds (MLflow aliases). Pick several to compare them on the "
                                 "same scan; the first one picked is the reference") if len(keys) > 1 else keys
    largest = st.toggle("Largest piece only",
                        help="Keep only the mask's largest connected piece, removing small islands. "
                             "Post-processing: off by default, and not used in the reported MSD results")
    if st.button("Segment", type="primary", disabled=data is None or not chosen, width="stretch"):
        results, failed = {}, []
        with st.spinner("Segmenting (the first request after a pause can take a few seconds) ..."):
            for key in chosen:
                try:
                    results[key] = api_client.segment(api_url, api_key, data, name,
                                                      "mni" if kind == MNI else "crop", largest, model=key)
                except api_client.ApiError as e:
                    failed.append(f"{key}: {e}")
        for message in failed:
            st.error(message, title="Couldn't segment with this model")
        if results:
            st.session_state["shown"] = {"results": results, "scan": data, "name": name, "scans": {},
                                         "label": label, "parts": bool(example and example["parts"]),
                                         "example": example["title"] if example else None}
        else:
            st.session_state.pop("shown", None)

shown = st.session_state.get("shown")
if not shown:
    st.markdown(
        "1. **Upload** one T1 scan in the sidebar (NIfTI, 1 mm voxels), or pick an **example scan** "
        "(public research crops with their manual labels).\n"
        "2. **Choose the scan type:** a hippocampus crop, or a whole brain registered to MNI space.\n"
        "3. **Choose one model, or several to compare** (when the service holds more than one).\n"
        "4. Click **Segment**. You get the anterior, posterior and whole hippocampus volume, quality flags, "
        "the slices with the mask, a 3D view and the mask to download; with several models, a comparison table "
        "and one row of slices per model.\n\n"
        "Colours: **orange** = anterior (head), **blue** = posterior (body and tail). The models are 3D U-Nets "
        "trained on MSD Task04; they work best on scans like those.")
    st.stop()

results = shown["results"]
order = list(results)
first = results[order[0]]
for warning in first.get("warnings", []):
    st.info(warning)
st.caption(f"Post-processing: {first['post_processing']} · " + " · ".join(
    f"{k}: version {r['model']['version']}, {r['seconds']} s" for k, r in results.items()))

sides = [h["side"] for h in first["hippocampi"]]
side = st.segmented_control("Hippocampus", sides, default=sides[0], required=True) if len(sides) > 1 else sides[0]
picked = {k: r["hippocampi"][sides.index(side)] for k, r in results.items()}
for k, h in picked.items():  # built once per model and side, not on every slider move
    if (k, side) not in shown["scans"]:
        shown["scans"][(k, side)] = scan_for(h, shown["scan"])
scans = {k: shown["scans"][(k, side)] for k in picked}
stem = shown["name"].split(".nii")[0]

if len(order) == 1:
    hippocampus = picked[order[0]]
    for col, name in zip(st.columns(3), ("anterior", "posterior", "whole")):
        col.metric(f"{name.capitalize()} volume", f"{hippocampus['volumes_ml'][name]:.2f} ml")
    qc = hippocampus["qc"]
    if qc["needs_review"]:
        st.warning("; ".join(qc["flags"]) + ". Check the mask in the slices before using these volumes.",
                   title="Needs review")
    else:
        st.success(f"No quality flags (one piece, {qc['edge_margin_mm']} mm from the crop edge).")
    if "removed_ml" in hippocampus:  # largest piece only: say what was removed, and what the plain mask had
        before = hippocampus["qc_before_post_processing"]
        st.info(f"Largest piece only: {hippocampus['removed_ml']:.2f} ml removed from {before['pieces']} pieces. "
                + ("Before post-processing: " + "; ".join(before["flags"]) + "." if before["flags"] else ""))
else:
    reference = order[0]
    masks = {k: api_client.mask_image(h) for k, h in picked.items()}
    table = []
    for k, h in picked.items():
        agree = api_client.agreement(masks[reference], masks[k])
        table.append({"Model": k, "Version": results[k]["model"]["version"],
                      "Anterior (ml)": h["volumes_ml"]["anterior"], "Posterior (ml)": h["volumes_ml"]["posterior"],
                      "Whole (ml)": h["volumes_ml"]["whole"], "Pieces": h["qc"]["pieces"],
                      f"Dice vs {reference}, whole": round(agree["whole"], 3),
                      f"Dice vs {reference}, anterior": round(agree["anterior"], 3),
                      f"Dice vs {reference}, posterior": round(agree["posterior"], 3),
                      "Removed (ml)": h.get("removed_ml", 0.0), "Flags": "; ".join(h["qc"]["flags"]) or "none"})
    st.subheader("Comparison")
    st.dataframe(table, hide_index=True, alt=f"Volumes, quality flags and agreement of {len(order)} models")
    st.caption(f"Dice here is agreement with the reference model ({reference}), not accuracy: there is no ground "
               "truth for an uploaded scan. Volumes in ml; a whole-brain scan's volumes are in MNI space.")

if shown.get("label"):  # an example scan: its manual label is the ground truth
    truth = api_client.read_scan(shown["label"])
    scored = []
    for k, h in picked.items():
        dice = api_client.accuracy(api_client.mask_image(h), truth, shown["parts"])
        scored.append({"Model": k} | {f"Dice, {part}": round(v, 3) for part, v in dice.items()})
    st.subheader("Against the manual label")
    st.dataframe(scored, hide_index=True, alt=f"Dice of each model against the manual label of {shown['example']}")
    st.caption(f"{shown['example']}: Dice against its manual label (1 = perfect overlap). "
               + ("" if shown["parts"] else "This label is in subfields, not anterior/posterior, so only the whole "
                                            "hippocampus is scored; its protocol includes tissue the MSD labels "
                                            "leave out, so expect lower Dice than in distribution."))

with st.sidebar:
    st.header("Display")
    opacity = st.slider("Mask opacity", 0.0, 1.0, 0.5, 0.05)
    size = st.slider("Size (px)", 160, 480, 240 if len(order) > 1 else 280, 20)
    st.header("Planes")
    base = scans[order[0]]  # every model's mask is on the same grid, so one set of sliders moves all rows
    pos = list(base.start)
    for view_name, axis, _, _ in VIEWS:
        pos[axis] = st.slider(view_name, 0, base.data.shape[axis] - 1, base.start[axis], key=f"pos{axis}::{side}")

scale = mm_per_px(base.data.shape, base.spacing, size)
for k in order:
    h = picked[k]
    if len(order) > 1:
        st.markdown(f"**{k}** · {results[k]['model']['name']} · whole {h['volumes_ml']['whole']:.2f} ml"
                    + (f" · :orange[needs review: {'; '.join(h['qc']['flags'])}]" if h["qc"]["needs_review"] else ""))
    for col, view in zip(st.columns(3), VIEWS):
        col.image(scans[k].render(view, pos, opacity=opacity, scale=scale), caption=view[0],
                  alt=f"{view[0]} slice of the uploaded scan with the hippocampus predicted by {k}")
    st.download_button(f"Download the {k} mask (.nii.gz)", api_client.mask_bytes(h),
                       file_name=f"{stem}_{side.split()[0]}_{k}_mask.nii.gz", mime="application/gzip",
                       icon=":material/download:", key=f"download::{k}::{side}")
st.caption("Orange = anterior, blue = posterior. Radiological orientation, as in 3D Slicer: the patient's right "
           "is on the screen's left in the axial and coronal views. " + base.describe(pos))

if st.toggle("3D view"):
    for start in range(0, len(order), 3):
        for col, k in zip(st.columns(3), order[start:start + 3]):
            with col:
                st.markdown(f"**{k}**")
                st.plotly_chart(figure_3d(scans[k]), key=f"3d::{k}::{side}",
                                alt=f"Rotatable 3D surfaces of the hippocampus predicted by {k}")

st.caption("Research use only, not for diagnosis.")
