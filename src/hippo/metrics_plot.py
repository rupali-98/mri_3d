"""Plot each run's Dice and NSD on MSD test, the external dataset and their shifted crops.

One panel per metric. For every run, each case is a dot (blue circles = MSD
test, orange diamonds = external, green squares = shifted MSD test, yellow
triangles = shifted external) and the bar with a number is the mean.

The datasets are not scored on the same labels:
  - MSD test (labels anterior/posterior): the mean of the anterior and posterior
    score of each case, from <run>/metrics_test.csv (hippo.evaluate); cases
    hippo.evaluate could not score (blank Dice/NSD) are left out;
  - internal_anterior / internal_posterior (orangered / steelblue triangles): MSD test per
    part, same file; only drawn when named in --datasets (the README in-distribution figure);
  - external (subfield labels): the whole hippocampus, from
    <run>/external/metrics_external.csv (hippo.external);
  - external_largest (dark orange triangles; same labels): the same after keeping each
    prediction's largest connected piece, from <run>/external_largest/metrics_external.csv
    (hippo.external --largest-piece);
  - external_msd_crop_largest (green plus signs; same labels): the MSD-style box with the largest piece kept,
    from <run>/external_msd_crop_largest/metrics_external.csv (hippo.external --msd-crop --largest-piece);
    only drawn when named in --datasets;
  - shifted (same labels as MSD test): each test case's score averaged over all
    its shifted copies, from <run>/shifted/<condition>/metrics_test.csv
    (archive/shift.py); the padded, unshifted control is left out;
  - external_shifted (yellow triangles; same labels as external): each external
    hippocampus averaged over its shifted atlas boxes, from
    <run>/shifted_external/<condition>/metrics_external.csv (archive/shift.py --external);
    the control (the usual crop) is the external group itself.
So compare runs within a dataset; the gap between MSD test and external mixes the
domain shift with the different definition. A dataset whose files are missing is skipped.

--curve draws something else: each run's mean Dice and NSD against the hippocampus's height in the
external atlas box, largest piece kept, from <run>/shifted_external_largest/ (archive/shift.py).

Outputs (default runs/figures/):
    metrics_internal_external.png   the figure
    metrics_internal_external.csv   the same numbers as a table: run, dataset, metric, n, mean, median, std
    with --curve: the same two files, with position_mm in place of dataset

Usage:
    python -m hippo.metrics_plot --runs runs/baseline runs/clean_labels runs/no_norm
    python -m hippo.metrics_plot --runs runs/baseline --names "3D U-Net" --datasets internal external external_largest --out docs/figures/baseline_msd_vs_external
    python -m hippo.metrics_plot --runs runs/baseline --names "3D U-Net" --datasets internal_anterior internal_posterior --out docs/figures/msd_test_parts
    python -m hippo.metrics_plot --runs runs/baseline runs/random_trim --names "3D U-Net" "Random position" --datasets internal external_largest external_msd_crop_largest --decimals 3 --out docs/figures/random_trim_scores
    python -m hippo.metrics_plot --runs runs/baseline runs/random_trim --names "3D U-Net" "Random position" --curve --out docs/figures/random_trim_position_curve
--datasets keeps only some groups (the README figure);
--names labels the runs in the figure and table (default: the run folder names);
--decimals sets the digits of the mean labels (default 2).
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np

from . import defaults

OUT = "runs/figures/metrics_internal_external"
METRICS = [("dice", "Dice"), ("nsd", "NSD (1 mm)")]
# (key, legend name, colour, marker); colours are the first four slots of a colourblind-checked palette
DATASETS = [("internal", "MSD test (mean of anterior/posterior)", "#2a78d6", "o"),
            ("external", "External (whole hippocampus)", "#eb6834", "D"),
            # a darker step of the external hue: the same data after post-processing, not a new dataset
            ("external_largest", "External, largest piece only (post-processing)", "#9c3a12", "v"),
            ("shifted", "MSD test, shifted crops (mean over shifts)", "#1baf7a", "s"),
            ("external_shifted", "External, shifted boxes (mean over shifts)", "#eda100", "^"),
            ("external_msd_crop_largest", "External, MSD-style box, largest piece only", "#1baf7a", "P"),
            # MSD test per part, in the label colours of the slice views (anterior orangered, posterior steelblue)
            ("internal_anterior", "MSD test, anterior", "#e0502a", "^"),
            ("internal_posterior", "MSD test, posterior", "#4682b4", "v")]
ON_REQUEST = {"internal_anterior", "internal_posterior", "external_msd_crop_largest"}  # drawn only when named in --datasets
# archive/shift.py condition -> how far the hippocampus sits above its usual place in the box (mm)
CURVE = [("z-10", -10), ("z-5", -5), ("none", 0), ("z+5", 5), ("z+10", 10)]
CURVE_RUNS = [("#2a78d6", "o"), ("#eb6834", "D")]  # in run order, from the same palette as DATASETS
CONTROL = "none"  # archive/shift.py's padded, unshifted copy: not a shift, so left out of the shifted group
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def case_scores(run: Path) -> dict:
    """{dataset: {metric: per-case array}}; a dataset is missing if its CSV is not there yet."""
    scores = {}
    if (run / "metrics_test.csv").exists():
        rows = [r for r in read_csv(run / "metrics_test.csv") if r["dice_anterior"] != ""]
        scores["internal"] = {m: np.array([(float(r[f"{m}_anterior"]) + float(r[f"{m}_posterior"])) / 2 for r in rows])
                              for m, _ in METRICS}
        for part in ("anterior", "posterior"):
            scores[f"internal_{part}"] = {m: np.array([float(r[f"{m}_{part}"]) for r in rows]) for m, _ in METRICS}
    # external_largest: hippo.external --largest-piece; external_msd_crop_largest: also --msd-crop
    for ds in ("external", "external_largest", "external_msd_crop_largest"):
        if (run / ds / "metrics_external.csv").exists():
            rows = read_csv(run / ds / "metrics_external.csv")
            scores[ds] = {m: np.array([float(r[m]) for r in rows]) for m, _ in METRICS}
    for ds, folder, name in [("shifted", "shifted", "metrics_test.csv"),
                             ("external_shifted", "shifted_external", "metrics_external.csv")]:
        paths = sorted(p for p in (run / folder).glob(f"*/{name}") if p.parent.name != CONTROL)
        if paths:
            # one dot per case: its score averaged over every shifted copy
            per_case = {}
            for path in paths:
                for r in read_csv(path):
                    if "dice" in r:  # external: whole hippocampus
                        per_case.setdefault(r["case"], []).append([float(r[m]) for m, _ in METRICS])
                    elif r["dice_anterior"] != "":
                        per_case.setdefault(r["case"], []).append(
                            [(float(r[f"{m}_anterior"]) + float(r[f"{m}_posterior"])) / 2 for m, _ in METRICS])
            means = np.array([np.mean(v, axis=0) for _, v in sorted(per_case.items())])
            scores[ds] = {m: means[:, k] for k, (m, _) in enumerate(METRICS)}
    return scores


def summary_rows(scores: dict) -> list[dict]:
    return [{"run": run, "dataset": ds, "metric": m, "n": len(v), "mean": round(float(v.mean()), 4),
             "median": round(float(np.median(v)), 4), "std": round(float(v.std()), 4)}
            for run, by_ds in scores.items() for ds, by_m in by_ds.items() for m, v in by_m.items()]


def curve_scores(run: Path) -> dict:
    """{metric: per-case array for each CURVE position}; empty unless every position has been scored."""
    paths = [run / "shifted_external_largest" / condition / "metrics_external.csv" for condition, _ in CURVE]
    if not all(p.exists() for p in paths):
        return {}
    rows = [read_csv(p) for p in paths]
    return {m: [np.array([float(r[m]) for r in position]) for position in rows] for m, _ in METRICS}


def plot_curve(curves: dict) -> plt.Figure:
    """Mean Dice and NSD against the hippocampus's height in the box, one line per run."""
    fig, axes = plt.subplots(1, len(METRICS), figsize=(10.5, 4.6))
    x = [mm for _, mm in CURVE]
    for ax, (metric, title) in zip(axes, METRICS):
        for (run, curve), (colour, marker) in zip(curves.items(), CURVE_RUNS):
            means = [v.mean() for v in curve[metric]]
            ax.plot(x, means, color=colour, marker=marker, lw=2, markersize=7, zorder=3,
                    path_effects=[pe.withStroke(linewidth=4, foreground="white")])  # keeps crossing lines apart
            ax.annotate(f"{means[-1]:.3f}", (x[-1], means[-1]), xytext=(9, 0), textcoords="offset points",
                        va="center", fontsize=9, color=INK)
        ax.set_title(title, loc="left", fontsize=11, color=INK)
        ax.set_xticks(x, [f"{mm:+d}" if mm else "0" for mm in x], color=INK)
        ax.set_xlim(x[0] - 1, x[-1] + 3)  # room for the end labels
        ax.set_xlabel("Hippocampus height in the box, from its usual place (mm)", fontsize=9, color=MUTED)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors=MUTED, length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
    handles = [plt.Line2D([], [], color=c, marker=mk, lw=2, markersize=6, label=run)
               for run, (c, mk) in zip(curves, CURVE_RUNS)]
    fig.legend(handles=handles, loc="upper left", ncol=2, frameon=False, fontsize=9, bbox_to_anchor=(0.01, 0.99))
    fig.text(0.01, 0.01, f"Mean of {len(next(iter(curves.values()))[METRICS[0][0]][0])} external hippocampi, whole "
             "hippocampus, atlas box, largest piece kept. The vertical axes do not start at 0.\n"
             "Negative: box moved up, less tissue below the hippocampus. Positive: box moved down, more tissue below. "
             "Research use only.", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.07, 1, 0.93))
    return fig


def plot(scores: dict, decimals: int = 2) -> plt.Figure:
    """The figure for {run: case_scores(run)}; the caller shows or saves it. Only datasets present are drawn."""
    runs = list(scores)
    shown = [d for d in DATASETS if any(d[0] in s for s in scores.values())]  # legend and spacing follow these
    fig, axes = plt.subplots(1, len(METRICS), figsize=(max(4, 1.75 * len(shown)) * len(METRICS), 4.8), sharey=True)
    rng = np.random.default_rng(0)  # fixed jitter, so the figure is the same every time
    for ax, (metric, title) in zip(axes, METRICS):
        for i, run in enumerate(runs):
            for j, (ds, _, colour, marker) in enumerate(shown):
                if ds not in scores[run]:
                    continue
                v = scores[run][ds][metric]
                step = 0.84 / len(shown)  # the groups of one run share 0.84 of the space between runs
                x = i + (j - (len(shown) - 1) / 2) * step
                ax.scatter(x + rng.uniform(-0.3, 0.3, len(v)) * step, v, s=14, color=colour, marker=marker,
                           alpha=0.45, linewidths=0, zorder=2)
                ax.plot([x - 0.4 * step, x + 0.4 * step], [v.mean()] * 2, color=colour, lw=2.5,
                        solid_capstyle="round", zorder=3)
                ax.annotate(f"{v.mean():.{decimals}f}", (x, v.mean()), xytext=(0, 6), textcoords="offset points",
                            ha="center", fontsize=9, color=INK, zorder=4,
                            path_effects=[pe.withStroke(linewidth=3, foreground="white")])  # readable over dots
        ax.set_title(title, loc="left", fontsize=11, color=INK)
        ax.set_xticks(range(len(runs)), runs, color=INK)
        ax.set_ylim(0, 1.04)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors=MUTED, length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
    handles = [plt.Line2D([], [], color=c, marker=mk, ls="", markersize=6, label=name) for _, name, c, mk in shown]
    fig.legend(handles=handles, loc="upper left", ncol=2, frameon=False, fontsize=9, bbox_to_anchor=(0.01, 0.99))
    note = "Dots: one per case. Bar and number: mean."
    if any(ds.startswith("external") for ds, *_ in shown):
        note += " MSD test and external use different label definitions, so compare runs within a dataset."
    if any(ds.endswith("shifted") for ds, *_ in shown):
        note += ("\nShifted: each case's crop moved ±5/±10 mm per axis (archive/shift.py; MSD padded with zeros, "
                 "external box moved on the whole brain), averaged over the shifts.")
    fig.text(0.01, 0.01, note + " Research use only.", fontsize=8, color=MUTED, wrap=True)
    fig.tight_layout(rect=(0, 0.08, 1, 0.9))
    return fig


def main_curve(runs: list[str], labels: list[str], out: Path) -> list[dict]:
    """Figure and table of the position curve; at most two runs, so the lines stay readable."""
    if len(runs) > len(CURVE_RUNS):
        raise SystemExit(f"--curve compares at most {len(CURVE_RUNS)} runs")
    curves = {label: c for r, label in zip(runs, labels) if (c := curve_scores(Path(r)))}
    if not curves:
        raise SystemExit("No position curve found; run archive/shift.py --external ... --axes z --largest-piece first")
    fig = plot_curve(curves)
    fig.savefig(out.with_suffix(".png"), dpi=150)
    plt.close(fig)
    rows = [{"run": run, "position_mm": mm, "metric": m, "n": len(v), "mean": round(float(v.mean()), 4),
             "median": round(float(np.median(v)), 4), "std": round(float(v.std()), 4)}
            for run, curve in curves.items() for m, _ in METRICS for (_, mm), v in zip(CURVE, curve[m])]
    with open(out.with_suffix(".csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"Saved {out.with_suffix('.png')} and {out.with_suffix('.csv')}")
    return rows


def main(runs: list[str], out: str = OUT, datasets: list[str] | None = None,
         names: list[str] | None = None, curve: bool = False, decimals: int = 2) -> list[dict]:
    """Figure and table for the runs; `datasets` (e.g. ["internal", "external"]) keeps only those groups,
    `names` labels the runs (one per run; default: the folder names)."""
    if names and len(names) != len(runs):
        raise SystemExit(f"--names needs one name per run ({len(runs)}), got {len(names)}")
    labels = names or [Path(r).name for r in runs]
    if curve:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        return main_curve(runs, labels, Path(out))
    scores = {label: s for r, label in zip(runs, labels) if (s := case_scores(Path(r)))}
    keep = (lambda d: d in datasets) if datasets else (lambda d: d not in ON_REQUEST)
    scores = {run: kept for run, s in scores.items() if (kept := {d: v for d, v in s.items() if keep(d)})}
    if not scores:
        raise SystemExit("No metrics found; run hippo.evaluate, hippo.external and/or archive/shift.py on these runs first")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig = plot(scores, decimals)
    fig.savefig(out.with_suffix(".png"), dpi=150)
    plt.close(fig)
    rows = summary_rows(scores)
    with open(out.with_suffix(".csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(f"{r['run']:>13}  {r['dataset']:<16}  {r['metric']:<4}  n={r['n']:<3} mean {r['mean']:.3f}  "
              f"median {r['median']:.3f}  std {r['std']:.3f}")
    print(f"Saved {out.with_suffix('.png')} and {out.with_suffix('.csv')}")
    return rows


if __name__ == "__main__":
    matplotlib.use("Agg")  # files only; set here, not on import, so interactive use keeps its own backend
    p = defaults.make_parser("Plot Dice and NSD per run on the MSD test split and the external dataset.")
    p.add_argument("--runs", nargs="+", required=True, help="Run folders, e.g. runs/baseline runs/no_norm")
    p.add_argument("--out", default=OUT, help="Output path without extension (.png and .csv are written)")
    p.add_argument("--datasets", nargs="+", choices=[d for d, *_ in DATASETS],
                   help="Only these groups, e.g. internal external (default: all that exist)")
    p.add_argument("--names", nargs="+", help='Labels for the runs, one per run, e.g. "3D U-Net" (default: folder names)')
    p.add_argument("--curve", action="store_true", help="Draw Dice and NSD against the hippocampus's height in the box")
    p.add_argument("--decimals", type=int, default=2, help="Digits of the mean labels")
    a = p.parse_args()
    main(a.runs, a.out, a.datasets, a.names, a.curve, a.decimals)
