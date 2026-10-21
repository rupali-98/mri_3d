"""Explore the dataset and check its quality before cleaning or splitting.

Measures every case (labelled and unlabelled), flags problems and writes a
report. Raw data is only read, never changed.

    FAIL  unusable as is (bad file, misaligned or invalid label)
    WARN  usable, but worth a look (duplicate, fragmented label, unusual values)

Outputs in data/explore/:
    cases.csv      one row per case: measurements, status, issues
    summary.json   counts, flagged cases, intensity groups, current split balance
    figures/       distributions.png + a review figure for every flagged case

Usage:
    python -m hippo.explore                 # report + figures
    python -m hippo.explore --no-figures    # numbers only (faster)
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import nibabel as nib
import numpy as np
import yaml
from scipy import ndimage

from . import CLASS_NAMES, defaults

OUT_DIR = "data/explore"
# 26-neighbour connectivity: voxels touching by face, edge or corner belong to the same piece
CONNECTIVITY = np.ones((3, 3, 3), bool)


def list_cases(task: Path) -> list[dict]:
    """All cases in dataset.json; unlabelled (imagesTs) ones have label None."""
    meta = json.loads((task / "dataset.json").read_text())
    cases = [{"image": task / c["image"].lstrip("./"), "label": task / c["label"].lstrip("./")}
             for c in meta["training"]]
    cases += [{"image": task / (c if isinstance(c, str) else c["image"]).lstrip("./"), "label": None}
              for c in meta.get("test", [])]
    for c in cases:
        c["case"] = c["image"].name.replace(".nii.gz", "")
    return cases


def measure(image: Path, label: Path | None) -> tuple[dict, list, str | None]:
    """Measurements for one case, its problems as (severity, code, detail), and an image hash."""
    row, issues = {}, []
    try:
        im = nib.load(image)
        x = np.asarray(im.dataobj, dtype=np.float64)
    except Exception as e:
        return row, [("FAIL", "unreadable", f"image: {e}")], None
    finite = x[np.isfinite(x)]
    row.update(
        shape="x".join(map(str, x.shape)),
        spacing="x".join(f"{s:.3g}" for s in im.header.get_zooms()[:3]),
        orientation="".join(nib.aff2axcodes(im.affine)),
        intensity_mean=round(float(finite.mean()), 1),
        intensity_p99=round(float(np.percentile(finite, 99)), 1),
    )
    if finite.size < x.size:
        issues.append(("FAIL", "nonfinite", f"{x.size - finite.size} NaN/inf voxels"))
    if finite.std() == 0:
        issues.append(("FAIL", "blank", "image has a single value"))
    digest = hashlib.md5(np.nan_to_num(x).astype(np.float32).tobytes()).hexdigest()
    if label is None:
        return row, issues, digest

    try:
        lb = nib.load(label)
        raw = np.asarray(lb.dataobj)
    except Exception as e:
        return row, issues + [("FAIL", "unreadable", f"label: {e}")], digest
    values = set(np.unique(raw).tolist())
    if not values <= {0, *CLASS_NAMES}:
        issues.append(("FAIL", "label_values", f"unexpected values {sorted(values - {0, *CLASS_NAMES})}"))
    if raw.shape != x.shape:
        return row, issues + [("FAIL", "label_shape", f"label {raw.shape} vs image {x.shape}")], digest
    if not np.allclose(lb.affine, im.affine, atol=1e-3):
        issues.append(("FAIL", "label_position", "label and image affines differ"))

    y = np.rint(raw).astype(int)
    voxel_ml = float(np.prod(im.header.get_zooms()[:3])) / 1000
    for value, part in CLASS_NAMES.items():
        mask = y == value
        pieces, n = ndimage.label(mask, structure=CONNECTIVITY)
        sizes = sorted(np.bincount(pieces.ravel())[1:].tolist(), reverse=True)
        row[f"vol_{part}_ml"] = round(mask.sum() * voxel_ml, 3)
        row[f"pieces_{part}"] = n
        row[f"piece_sizes_{part}"] = "+".join(map(str, sizes))  # voxels, largest first
        if n == 0:
            issues.append(("FAIL", "missing_part", f"no {part} label"))
    fg = y > 0
    edge = np.zeros_like(fg)
    edge[[0, -1]] = edge[:, [0, -1]] = edge[:, :, [0, -1]] = True
    if (fg & edge).any():
        issues.append(("WARN", "touches_edge", "hippocampus reaches the crop border (may be cut off)"))
    if fg.any() and finite.std() > 0:
        ring = ndimage.binary_dilation(fg, iterations=3) & ~fg
        row["contrast"] = round(float((x[fg].mean() - x[ring].mean()) / x.std()), 3)
    if row["vol_posterior_ml"] > 0:
        row["ratio_ant_post"] = round(row["vol_anterior_ml"] / row["vol_posterior_ml"], 3)
    return row, issues, digest


def intensity_groups(p99: dict, gap_decades: float) -> dict:
    """Group cases whose intensity scales are >= 10**gap_decades apart (e.g. different scanners/exports)."""
    order = sorted(p99, key=p99.get)
    groups, g = {}, 0
    for prev, case in zip([None] + order, order):
        if prev is not None and np.log10(max(p99[case], 1e-6)) - np.log10(max(p99[prev], 1e-6)) >= gap_decades:
            g += 1
        groups[case] = g
    return groups


def flag_outliers(rows: list[dict], keys: list[str], z: float) -> None:
    """WARN values far from the median (robust z: median/MAD, with a 5% floor so near-identical data isn't flagged)."""
    for k in keys:
        ok = [r for r in rows if r["status"] != "FAIL" and k in r]
        if len(ok) < 3:
            continue
        v = np.array([r[k] for r in ok], float)
        med = np.median(v)
        scale = max(1.4826 * np.median(np.abs(v - med)), 0.05 * abs(med), 1e-9)
        for r, value in zip(ok, v):
            if abs(value - med) / scale > z:
                r["_issues"].append(("WARN", f"unusual_{k}", f"{k} {value:g} (median {med:.3g})"))


def plot_distributions(rows: list[dict], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labelled = [r for r in rows if "vol_anterior_ml" in r]
    splits = sorted({r["split"] for r in labelled})
    panels = [("vol_anterior_ml", "anterior volume (ml)"), ("vol_posterior_ml", "posterior volume (ml)"),
              ("ratio_ant_post", "anterior / posterior"), ("contrast", "hippocampus contrast")]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (k, title) in zip(axes.flat, panels):
        data = [[r[k] for r in labelled if r["split"] == s and k in r] for s in splits]
        ax.hist(data, bins=25, stacked=True, label=splits)
        ax.set_title(title)
        ax.legend(fontsize=8)
    ax = axes.flat[4]
    groups = sorted({r["intensity_group"] for r in rows if "intensity_group" in r})
    ax.hist([[np.log10(max(r["intensity_p99"], 1e-6)) for r in rows if r.get("intensity_group") == g] for g in groups],
            bins=40, stacked=True, label=[f"group {g}" for g in groups])
    ax.set_title("intensity scale: log10(99th percentile)")
    ax.legend(fontsize=8)
    ax = axes.flat[5]
    status = Counter(r["status"] for r in rows)
    ax.bar(list(status), list(status.values()), color=[{"OK": "#4caf50", "WARN": "#f2c14e", "FAIL": "#d62728"}[s]
                                                        for s in status])
    ax.set_title("case status")
    fig.suptitle(f"Dataset overview: {len(rows)} cases ({len(labelled)} labelled)")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def review_figures(rows: list[dict], cases: dict, folder: Path) -> None:
    """Slice + sagittal-montage figure for every flagged case, using hippo.visualize."""
    from .visualize import main as render

    folder.mkdir(parents=True, exist_ok=True)
    for old in folder.glob("*.png"):  # figures of cases no longer flagged would be misleading
        old.unlink()
    for r in rows:
        if r["status"] == "OK":
            continue
        c = cases[r["case"]]
        try:
            render(str(c["image"]), str(c["label"]) if c["label"] else None,
                   out=str(folder / f"{r['case']}.png"), montage_step=3)
        except Exception as e:  # a FAIL case may be impossible to draw
            print(f"  no figure for {r['case']}: {e}")


def main(data_dir: str = defaults.DATA_DIR, out: str = OUT_DIR, config: str = defaults.CONFIG,
         figures: bool = True) -> dict:
    qc = yaml.safe_load(Path(config).read_text())["qc"]
    task = Path(data_dir) / "Task04_Hippocampus"
    split_file = Path(data_dir) / defaults.SPLIT_FILENAME
    split_of = ({Path(c["image"]).name.replace(".nii.gz", ""): s
                 for s, cs in json.loads(split_file.read_text()).items() for c in cs}
                if split_file.exists() else {})
    cases = {c["case"]: c for c in list_cases(task)}

    rows, seen = [], {}
    for name, c in cases.items():
        row, issues, digest = measure(c["image"], c["label"])
        if digest in seen:
            issues.append(("WARN", "duplicate", f"same image data as {seen[digest]}"))
        seen.setdefault(digest, name)
        for value, part in CLASS_NAMES.items():
            if row.get(f"pieces_{part}", 0) > qc["max_pieces"]:
                issues.append(("WARN", "fragmented",
                               f"{part} in {row[f'pieces_{part}']} pieces ({row[f'piece_sizes_{part}']} voxels)"))
        rows.append({"case": name, "labelled": c["label"] is not None, "split": split_of.get(name, "-"),
                     **row, "_issues": issues,
                     "status": "FAIL" if any(s == "FAIL" for s, _, _ in issues) else "OK"})

    # dataset-level checks: odd geometry, intensity scale groups, outliers
    for k in ["spacing", "orientation"]:
        common = Counter(r[k] for r in rows if k in r).most_common(1)[0][0]
        for r in rows:
            if k in r and r[k] != common:
                r["_issues"].append(("WARN", f"odd_{k}", f"{k} {r[k]} (most cases {common})"))
    groups = intensity_groups({r["case"]: r["intensity_p99"] for r in rows if "intensity_p99" in r},
                              qc["intensity_gap_decades"])
    for r in rows:
        if r["case"] in groups:
            r["intensity_group"] = groups[r["case"]]
    flag_outliers(rows, ["vol_anterior_ml", "vol_posterior_ml", "ratio_ant_post", "contrast"], qc["outlier_z"])
    for r in rows:
        issues = r.pop("_issues")
        r["status"] = ("FAIL" if any(s == "FAIL" for s, _, _ in issues)
                       else "WARN" if issues else "OK")
        r["issues"] = "; ".join(f"{s} {code}: {detail}" for s, code, detail in issues)
        r["issue_codes"] = sorted({code for _, code, _ in issues})

    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(k for r in rows for k in r if k != "issue_codes"))
    with open(out_dir / "cases.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    labelled = [r for r in rows if r["labelled"]]
    summary = {
        "n_cases": len(rows),
        "n_labelled": len(labelled),
        "status": dict(Counter(r["status"] for r in rows)),
        "issue_counts": dict(Counter(code for r in rows for code in r["issue_codes"])),
        "fail": {r["case"]: r["issues"] for r in rows if r["status"] == "FAIL"},
        "warn": {r["case"]: r["issues"] for r in rows if r["status"] == "WARN"},
        "intensity_groups": {
            str(g): {"n": sum(1 for r in rows if r.get("intensity_group") == g),
                     "p99_range": [min(r["intensity_p99"] for r in rows if r.get("intensity_group") == g),
                                   max(r["intensity_p99"] for r in rows if r.get("intensity_group") == g)]}
            for g in sorted(set(groups.values()))
        },
        "current_split_balance": {
            s: {"n": len(rs),
                "vol_anterior_ml": round(float(np.mean([r["vol_anterior_ml"] for r in rs])), 3),
                "vol_posterior_ml": round(float(np.mean([r["vol_posterior_ml"] for r in rs])), 3),
                "intensity_groups": dict(Counter(str(r.get("intensity_group")) for r in rs))}
            for s in sorted({r["split"] for r in labelled})
            for rs in [[r for r in labelled if r["split"] == s and "vol_anterior_ml" in r]] if rs
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"{summary['n_cases']} cases ({summary['n_labelled']} labelled): {summary['status']}")
    print(f"issues: {summary['issue_counts']}")
    for g, info in summary["intensity_groups"].items():
        print(f"intensity group {g}: {info['n']} cases, 99th percentile {info['p99_range'][0]:g}-{info['p99_range'][1]:g}")
    for s, b in summary["current_split_balance"].items():
        print(f"split {s}: {b}")
    for status in ["fail", "warn"]:
        for case, text in summary[status].items():
            print(f"  {case}: {text}")

    if figures:
        (out_dir / "figures").mkdir(exist_ok=True)
        plot_distributions(rows, out_dir / "figures" / "distributions.png")
        review_figures(rows, cases, out_dir / "figures" / "review")
    print(f"Report in {out_dir}/")
    return summary


if __name__ == "__main__":
    p = defaults.make_parser("Explore the dataset and report data-quality problems.")
    p.add_argument("--config", default=defaults.CONFIG, help="Experiment settings (YAML) with a qc: section")
    p.add_argument("--out", default=OUT_DIR, help="Report folder")
    p.add_argument("--no-figures", action="store_true", help="Skip plots and per-case review figures")
    a = p.parse_args()
    main(a.data_dir, a.out, a.config, not a.no_figures)
