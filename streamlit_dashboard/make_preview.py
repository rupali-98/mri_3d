"""Draw a preview of the Streamlit page for the README, with the page's own slice-drawing code.

Same rows as compare_slices.py shows for one crop, each as axial, coronal and sagittal views
through the ground truth's centre, with the Dice the row titles give:
  - external crop (e.g. s03_L): ground truth, ASHS (second reference, if there is one), the
    run's prediction, its error map (whole hippocampus);
  - MSD case (hippocampus_NNN): ground truth, the run's prediction with anterior/posterior
    Dice, its error map per part (yellow = wrong part). Plain prediction, no post-processing.
It is a still picture made from the same functions, not a browser screenshot.

Outputs: docs/figures/streamlit_preview.png (or --out)

Usage (from the project root):
    PYTHONPATH=src .venv/bin/python streamlit_dashboard/make_preview.py [--case s03_L] [--run runs/baseline]
    PYTHONPATH=src .venv/bin/python streamlit_dashboard/make_preview.py --case hippocampus_135 --out docs/figures/streamlit_preview_msd.png
"""
import argparse
import io
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont, ImageOps

from hippo import visualize
from hippo.slices import ERRORS, VIEWS, Scan, error_map, label_colours, mm_per_px, on_grid, rgb_colours

OUT = Path("docs/figures/streamlit_preview.png")
ASHS = Path("data/ashs/labelsTr")
FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"  # macOS; Pillow's built-in font otherwise


def dice(a, b) -> float:
    return 2 * (a & b).sum() / max(a.sum() + b.sum(), 1)


def predict(run: Path, image: str, scan: Scan) -> np.ndarray:
    """The run's prediction with its own preprocessing, placed on the image grid by world position (as the page does)."""
    normalize = yaml.safe_load((run / "config.yaml").read_text())["data"].get("normalize", "zscore")
    d = visualize.load(Path(image), None, normalize)
    mask = visualize.predict(run, d["image"]).astype(np.uint8)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "pred.nii.gz"
        nib.save(nib.Nifti1Image(mask, np.asarray(d["image"].affine, dtype=np.float64)), path)
        return on_grid(path, scan.data.shape, scan.spacing, scan.origin)


def main(case: str, run: Path, name: str, out: Path = OUT):
    msd = case.startswith("hippocampus_")  # MSD case: anterior/posterior labels, scored per part
    folder = "data/Task04_Hippocampus" if msd else "data/external"
    image, label = f"{folder}/imagesTr/{case}.nii.gz", f"{folder}/labelsTr/{case}.nii.gz"
    scan = Scan(image)
    grid = (scan.data.shape, scan.spacing, scan.origin)
    truth = on_grid(label, *grid)
    pred = predict(run, image, scan)
    if msd:
        rows = [("ground truth (MSD: anterior orange, posterior blue)", truth, rgb_colours(label_colours({1, 2}))),
                (f"{name}: Dice whole {dice(pred > 0, truth > 0):.3f}, anterior {dice(pred == 1, truth == 1):.3f}, "
                 f"posterior {dice(pred == 2, truth == 2):.3f}", pred, rgb_colours(label_colours({1, 2}))),
                (f"{name}, Errors colour: green correct, yellow wrong part, blue missed, red extra",
                 error_map(pred, truth, parts=True), rgb_colours(ERRORS))]
        return draw(scan, truth, rows, case, name, out)
    rows = [("ground truth (external subfields)", truth, rgb_colours(label_colours(set(np.unique(truth)) - {0})))]
    if (ASHS / f"{case}.nii.gz").exists():
        ashs = on_grid(str(ASHS / f"{case}.nii.gz"), *grid)
        rows.append((f"ASHS-PMC-T1 (second reference): vs ground truth Dice whole {dice(ashs > 0, truth > 0):.3f}",
                     ashs, rgb_colours(label_colours({1, 2}))))
        vs_ashs = (f" | vs ASHS whole {dice(pred > 0, ashs > 0):.3f}, anterior {dice(pred == 1, ashs == 1):.3f}, "
                   f"posterior {dice(pred == 2, ashs == 2):.3f}")
    else:
        vs_ashs = ""
    rows.append((f"{name}: Dice whole {dice(pred > 0, truth > 0):.3f}{vs_ashs}", pred, rgb_colours(label_colours({1, 2}))))
    rows.append((f"{name}, Errors colour: green correct, blue missed, red extra", error_map(pred, truth, parts=False),
                 rgb_colours(ERRORS)))
    draw(scan, truth, rows, case, name, out)


def draw(scan: Scan, truth: np.ndarray, rows: list, case: str, name: str, out: Path):
    """The rows as the page shows them, through the ground truth's centre, saved to `out`."""
    voxels = np.argwhere(truth > 0)
    pos = [int(c) for c in voxels[np.abs(voxels - voxels.mean(0)).sum(1).argmin()]]  # where the page starts
    scale = mm_per_px(scan.data.shape, scan.spacing, 240)
    # each slice as the page shows it: exact pixel size, a 2 px border in its view's colour
    tiles = [[ImageOps.expand(Image.open(io.BytesIO(scan.render(v, pos, (1, 99), 0.5, True, scale,
                                                                 labels=labels, colours=colours))).convert("RGB"),
                              border=2, fill=v[3]) for v in VIEWS] for _, labels, colours in rows]
    pad, gap, title_h = 16, 6, 26
    width = pad * 2 + sum(t.width for t in tiles[0]) + gap * 2
    height = pad + 30 + sum(title_h + max(t.height for t in row) + gap for row in tiles) + 20
    page = Image.new("RGB", (width, height), "white")
    pen = ImageDraw.Draw(page)
    font = lambda size: ImageFont.truetype(FONT, size) if Path(FONT).exists() else ImageFont.load_default()
    pen.text((pad, pad), f"Streamlit page preview: {case}  ({name})", fill="#222", font=font(16))
    y = pad + 30
    for (title, _, _), row in zip(rows, tiles):
        pen.text((pad, y + 4), title, fill="#222", font=font(13))
        y += title_h
        x = pad
        for tile in row:
            page.paste(tile, (x, y))
            x += tile.width + gap
        y += max(t.height for t in row) + gap
    pen.text((pad, y + 2), "Research use only, not for diagnosis.", fill="#777", font=font(12))
    out.parent.mkdir(parents=True, exist_ok=True)
    page.save(out)
    print(f"Saved {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--case", default="s03_L")
    p.add_argument("--run", default="runs/baseline")
    p.add_argument("--name", default="3D U-Net", help="Run name shown in the row titles")
    p.add_argument("--out", default=str(OUT), help="Output PNG")
    a = p.parse_args()
    main(a.case, Path(a.run), a.name, Path(a.out))
