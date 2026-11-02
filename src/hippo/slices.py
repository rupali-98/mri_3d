"""Axial, coronal and sagittal slices of a NIfTI scan with mask overlays, drawn like 3D Slicer.

Used by the Streamlit page streamlit_dashboard/compare_slices.py (ground truth and the
predictions of several runs side by side). Slices are drawn as PNG images, so the page
can update them quickly while a slider moves.

Orientation follows 3D Slicer's default (radiological): in the axial and coronal
views the patient's right is on the left of the screen; the sagittal view has
anterior on the left. Each view is labelled with its edge letters.

surfaces() gives each label's surface in world mm, for the 3D view of streamlit_dashboard/compare_slices.py.
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from matplotlib.colors import to_hex, to_rgb
import nibabel as nib
from PIL import Image, ImageDraw, ImageFont

NONE = "(none)"
# (name, axis the view cuts, edge letters left/right/top/bottom, colour as in 3D Slicer)
VIEWS = [("Axial", 2, "RLAP", "#e8605a"), ("Coronal", 1, "RLSI", "#6cc46c"), ("Sagittal", 0, "APSI", "#f0c050")]
AXIS_COLOUR = {axis: colour for _, axis, _, colour in VIEWS}

# label value -> (name, colour), chosen by which values a mask contains
LABEL_SETS = [
    ({1, 2}, {1: ("anterior", "orangered"), 2: ("posterior", "steelblue")}),
    ({1, 2, 3}, {1: ("CA1-3", "gold"), 2: ("subiculum", "mediumseagreen"), 3: ("CA4-DG", "orchid")}),
]
PALETTE = ["orangered", "steelblue", "gold", "mediumseagreen", "orchid", "turquoise", "salmon", "khaki"]
# error_map() values
ERRORS = {1: ("correct", "limegreen"), 2: ("wrong part", "gold"), 3: ("missed", "dodgerblue"), 4: ("extra", "red")}

try:
    FONT = ImageFont.load_default(size=14)
except TypeError:  # Pillow < 10.1 has one fixed-size font
    FONT = ImageFont.load_default()


def list_nifti(dirs=("data", "runs")) -> list[str]:
    # skips macOS "._" metadata files, which share the name but are not images
    return sorted(str(p) for d in dirs for p in Path(d).rglob("*")
                  if p.name.endswith((".nii.gz", ".nii")) and not p.name.startswith("._"))


def guess_mask(image: str) -> str:
    """The label file next to an image in an imagesTr/ or imagesTs/ folder (MSD, external crops), if any."""
    p = Path(image)
    for images, labels in [("imagesTr", "labelsTr"), ("imagesTs", "labelsTs")]:
        if p.parent.name == images and (p.parent.parent / labels / p.name).exists():
            return str(p.parent.parent / labels / p.name)
    return NONE


def load_ras(path: str) -> tuple[np.ndarray, tuple, tuple]:
    """Array in RAS orientation, voxel size (mm) and world position of the first voxel.

    Placing every file by its own origin lines up masks with a different size
    (e.g. the padded prediction) with the MRI.
    """
    img = nib.as_closest_canonical(nib.load(path))
    spacing = tuple(float(s) for s in img.header.get_zooms()[:3])
    return np.asarray(img.dataobj), spacing, tuple(float(o) for o in img.affine[:3, 3])


def load(path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Array in RAS orientation, voxel size (mm) and world position of the first voxel."""
    data, spacing, origin = load_ras(str(path))
    return np.asarray(data, dtype=np.float32), np.array(spacing), np.array(origin)


def is_label_map(data: np.ndarray) -> bool:
    """Whole numbers with only a few distinct values: a mask, not an MRI."""
    sample = data[::4, ::4, ::4]
    return bool(np.all(sample == np.rint(sample))) and len(np.unique(sample)) <= 20


def label_colours(values: set) -> dict:
    for known, names in LABEL_SETS:
        if values and values <= known:
            return {v: names[v] for v in sorted(values)}
    return {v: (f"label {v}", PALETTE[i % len(PALETTE)]) for i, v in enumerate(sorted(values))}


def rgb_colours(names: dict) -> dict:
    """{value: (name, colour)} with colours as RGB arrays (0-1)."""
    return {v: (name, np.array(to_rgb(c))) for v, (name, c) in names.items()}


def on_grid(path, shape, spacing, origin) -> np.ndarray:
    """A mask moved onto the image grid by world position (nearest voxel); 0 where it does not reach.

    Lines up crops, padded predictions and whole-brain scans of the same space.
    """
    m, m_spacing, m_origin = load(path)
    m = np.rint(m).astype(np.int32)
    idx = [np.rint((origin[a] + np.arange(shape[a]) * spacing[a] - m_origin[a]) / m_spacing[a]).astype(int)
           for a in range(3)]
    inside = [((i >= 0) & (i < m.shape[a])).reshape([-1 if b == a else 1 for b in range(3)])
              for a, i in enumerate(idx)]
    out = m[np.ix_(*[np.clip(i, 0, m.shape[a] - 1) for a, i in enumerate(idx)])]
    return out * (inside[0] & inside[1] & inside[2])


def error_map(pred: np.ndarray, truth: np.ndarray, parts: bool = True) -> np.ndarray:
    """Each voxel as an ERRORS value (0 = background in both).

    parts=False compares the whole hippocampus only (labels that are not anterior/posterior).
    """
    p, t = (pred, truth) if parts else ((pred > 0).astype(int), (truth > 0).astype(int))
    out = np.zeros(pred.shape, np.uint8)
    out[(p > 0) & (t > 0) & (p == t)] = 1
    out[(p > 0) & (t > 0) & (p != t)] = 2
    out[(p == 0) & (t > 0)] = 3
    out[(p > 0) & (t == 0)] = 4
    return out


def surfaces(labels: np.ndarray, spacing, origin, colours: dict) -> list[tuple[str, str, np.ndarray, np.ndarray]]:
    """(name, hex colour, vertices in world mm, triangles) of each label value's surface, for a 3D view."""
    from skimage.measure import marching_cubes
    out = []
    for value, (name, colour) in colours.items():
        mask = labels == value
        if mask.sum() < 2:
            continue
        # padded so a label touching the edge still gets a closed surface; the pad is taken off the vertices
        verts, faces, _, _ = marching_cubes(np.pad(mask, 1).astype(np.uint8), 0.5, spacing=tuple(spacing))
        out.append((name, to_hex(colour), verts - np.asarray(spacing) + np.asarray(origin), faces))
    return out


def mm_per_px(shape, spacing, size: int) -> float:
    """Scale at which the scan's longest side (mm) is `size` pixels; the same for all views."""
    return max(n * s for n, s in zip(shape, spacing)) / size


class Scan:
    """One image (+ optional mask) and how to draw any of its slices as a PNG."""

    def __init__(self, image, mask=NONE):
        self.data, self.spacing, self.origin = load(image)
        if is_label_map(self.data):  # a label file picked as the image: colour it
            self.labels = np.rint(self.data).astype(np.int32)
        elif mask and mask != NONE:
            self.labels = on_grid(mask, self.data.shape, self.spacing, self.origin)
        else:
            self.labels = None
        values = set(np.unique(self.labels).tolist()) - {0} if self.labels is not None else set()
        self.colours = rgb_colours(label_colours(values))
        tissue = self.data[self.data > self.data.min()]
        self.levels = np.percentile(tissue if tissue.size else self.data, np.arange(101))
        # start at the labelled voxel nearest the mask's centre, or at the centre of the scan
        if values:
            voxels = np.argwhere(self.labels > 0)
            self.start = [int(c) for c in voxels[np.abs(voxels - voxels.mean(0)).sum(1).argmin()]]
        else:
            self.start = [n // 2 for n in self.data.shape]

    def mm(self, axis: int, i: int) -> float:
        return float(self.origin[axis] + i * self.spacing[axis])

    def describe(self, pos) -> str:
        x, y, z = (self.mm(a, pos[a]) for a in range(3))
        text = f"x {x:.1f}  y {y:.1f}  z {z:.1f} mm   intensity {self.data[tuple(pos)]:.4g}"
        if self.labels is not None:
            v = int(self.labels[tuple(pos)])
            text += f"   label {v}" + (f" ({self.colours[v][0]})" if v in self.colours else "")
        return text

    def render(self, view, pos, contrast=(1, 99), opacity=0.5, crosshair=True, scale=1.0,
               labels=None, colours=None) -> bytes:
        """PNG bytes of the slice through voxel `pos` for one of VIEWS, oriented as its edge letters say.

        contrast: darkest and brightest grey as percentiles of the tissue intensities.
        scale: mm per pixel (see mm_per_px). labels/colours: overlay other than the scan's own mask
        (e.g. a prediction or an error_map), on the same grid.
        """
        _, axis, letters, _ = view
        labels, colours = (self.labels, self.colours) if labels is None else (labels, colours)
        cols_axis, rows_axis = [a for a in range(3) if a != axis]
        lo, hi = self.levels[contrast[0]], self.levels[contrast[1]]
        grey = np.clip((np.take(self.data, pos[axis], axis) - lo) / max(hi - lo, 1e-6), 0, 1)
        rgb = np.repeat(grey[..., None], 3, -1)
        if labels is not None and opacity > 0:
            lab = np.take(labels, pos[axis], axis)
            for v, (_, c) in colours.items():
                rgb[lab == v] = (1 - opacity) * rgb[lab == v] + opacity * c
        # rows top to bottom = high to low (A or S first); columns left to right = high to low (R or A first)
        rgb = rgb.transpose(1, 0, 2)[::-1, ::-1]
        n_cols, n_rows = self.data.shape[cols_axis], self.data.shape[rows_axis]
        size = (max(1, round(n_cols * self.spacing[cols_axis] / scale)),
                max(1, round(n_rows * self.spacing[rows_axis] / scale)))
        img = Image.fromarray((rgb * 255).astype(np.uint8)).resize(size, Image.NEAREST)
        draw = ImageDraw.Draw(img)
        if crosshair:  # where the other two views cut this one
            col = (n_cols - 1 - pos[cols_axis] + 0.5) * size[0] / n_cols
            row = (n_rows - 1 - pos[rows_axis] + 0.5) * size[1] / n_rows
            draw.line([(col, 0), (col, size[1])], fill=AXIS_COLOUR[cols_axis])
            draw.line([(0, row), (size[0], row)], fill=AXIS_COLOUR[rows_axis])
        left, right, top, bottom = letters
        for text, xy, anchor in [(left, (4, size[1] / 2), "lm"), (right, (size[0] - 4, size[1] / 2), "rm"),
                                 (top, (size[0] / 2, 3), "mt"), (bottom, (size[0] / 2, size[1] - 3), "mb")]:
            draw.text(xy, text, fill="white", font=FONT, anchor=anchor, stroke_width=2, stroke_fill="black")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()


def legend_html(colours: dict) -> str:
    """Colour swatches for a {value: (name, rgb)} dict, for an ipywidgets.HTML."""
    return " ".join(f"<span style='background:rgb{tuple(int(255 * x) for x in c)};padding:0 8px'>&nbsp;</span>"
                    f" {n} &nbsp;" for _, (n, c) in colours.items())
