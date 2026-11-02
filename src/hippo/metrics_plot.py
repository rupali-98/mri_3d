"""Plot each run's Dice and NSD on MSD test, the external dataset and their shifted crops.

One panel per metric. For every run, each case is a dot (blue circles = MSD
test, orange diamonds = external, green squares = shifted MSD test, yellow
triangles = shifted external) and the bar with a number is the mean.

The datasets are not scored on the same labels:
  - MSD test (labels anterior/posterior): the mean of the anterior and posterior
    score of each case, from <run>/metrics_test.csv (hippo.evaluate); cases
    hippo.evaluate could not score (blank Dice/NSD) are left out;
  - external (subfield labels): the whole hippocampus, from
    <run>/external/metrics_external.csv (hippo.external);
  - external_largest (dark orange triangles; same labels): the same after keeping each
    prediction's largest connected piece, from <run>/external_largest/metrics_external.csv
    (hippo.external --largest-piece);
  - shifted (same labels as MSD test): each test case's score averaged over all
    its shifted copies, from <run>/shifted/<condition>/metrics_test.csv
    (archive/shift.py); the padded, unshifted control is left out;
  - external_shifted (yellow triangles; same labels as external): each external
    hippocampus averaged over its shifted atlas boxes, from
    <run>/shifted_external/<condition>/metrics_external.csv (archive/shift.py --external);
    the control (the usual crop) is the external group itself.
So compare runs within a dataset; the gap between MSD test and external mixes the
domain shift with the different definition. A dataset whose files are missing is skipped.

Outputs (default runs/figures/):
    metrics_internal_external.png   the figure
    metrics_internal_external.csv   the same numbers as a table: run, dataset, metric, n, mean, median, std

Usage:
    python -m hippo.metrics_plot --runs runs/baseline runs/clean_labels runs/no_norm
    python -m hippo.metrics_plot --runs runs/baseline --names "3D U-Net" --datasets internal external external_largest --out docs/figures/baseline_msd_vs_external
--datasets keeps only some groups (the README figure);
--names labels the runs in the figure and table (default: the run folder names).
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
            ("external_shifted", "External, shifted boxes (mean over shifts)", "#eda100", "^")]
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
    for ds in ("external", "external_largest"):  # external_largest: hippo.external --largest-piece
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


def plot(scores: dict) -> plt.Figure:
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
                ax.annotate(f"{v.mean():.2f}", (x, v.mean()), xytext=(0, 6), textcoords="offset points",
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
    note = ("Dots: one per case. Bar and number: mean. MSD test and external use different label definitions, "
            "so compare runs within a dataset.")
    if any(ds.endswith("shifted") for ds, *_ in shown):
        note += ("\nShifted: each case's crop moved ±5/±10 mm per axis (archive/shift.py; MSD padded with zeros, "
                 "external box moved on the whole brain), averaged over the shifts.")
    fig.text(0.01, 0.01, note + " Research use only.", fontsize=8, color=MUTED, wrap=True)
    fig.tight_layout(rect=(0, 0.08, 1, 0.9))
    return fig


def main(runs: list[str], out: str = OUT, datasets: list[str] | None = None,
         names: list[str] | None = None) -> list[dict]:
    """Figure and table for the runs; `datasets` (e.g. ["internal", "external"]) keeps only those groups,
    `names` labels the runs (one per run; default: the folder names)."""
    if names and len(names) != len(runs):
        raise SystemExit(f"--names needs one name per run ({len(runs)}), got {len(names)}")
    labels = names or [Path(r).name for r in runs]
    scores = {label: s for r, label in zip(runs, labels) if (s := case_scores(Path(r)))}
    if datasets:
        scores = {run: kept for run, s in scores.items() if (kept := {d: v for d, v in s.items() if d in datasets})}
    if not scores:
        raise SystemExit("No metrics found; run hippo.evaluate, hippo.external and/or archive/shift.py on these runs first")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig = plot(scores)
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
    a = p.parse_args()
    main(a.runs, a.out, a.datasets, a.names)
