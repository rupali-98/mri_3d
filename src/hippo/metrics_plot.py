"""Plot each run's Dice and NSD on the internal (MSD test) and external datasets.

One panel per metric. For every run, each case is a dot (blue circles = MSD
test, orange diamonds = external) and the bar with a number is the mean.

The two datasets are not scored on the same labels:
  - MSD test (labels anterior/posterior): the mean of the anterior and posterior
    score of each case, from <run>/metrics_test.csv (hippo.evaluate);
  - external (subfield labels): the whole hippocampus, from
    <run>/external/metrics_external.csv (hippo.external).
So compare runs within a dataset; the gap between datasets mixes the domain
shift with the different definition.

Outputs (default runs/figures/):
    metrics_internal_external.png   the figure
    metrics_internal_external.csv   the same numbers as a table: run, dataset, metric, n, mean, median, std

Usage:
    python -m hippo.metrics_plot --runs runs/baseline runs/clean_labels runs/no_norm
or interactively in notebooks/metrics_plot.ipynb.
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
# (key, legend name, colour, marker); colours are the first two slots of a colourblind-checked palette
DATASETS = [("internal", "MSD test (mean of anterior/posterior)", "#2a78d6", "o"),
            ("external", "External (whole hippocampus)", "#eb6834", "D")]
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def case_scores(run: Path) -> dict:
    """{dataset: {metric: per-case array}}; a dataset is missing if its CSV is not there yet."""
    scores = {}
    if (run / "metrics_test.csv").exists():
        rows = read_csv(run / "metrics_test.csv")
        scores["internal"] = {m: np.array([(float(r[f"{m}_anterior"]) + float(r[f"{m}_posterior"])) / 2 for r in rows])
                              for m, _ in METRICS}
    if (run / "external" / "metrics_external.csv").exists():
        rows = read_csv(run / "external" / "metrics_external.csv")
        scores["external"] = {m: np.array([float(r[m]) for r in rows]) for m, _ in METRICS}
    return scores


def summary_rows(scores: dict) -> list[dict]:
    return [{"run": run, "dataset": ds, "metric": m, "n": len(v), "mean": round(float(v.mean()), 4),
             "median": round(float(np.median(v)), 4), "std": round(float(v.std()), 4)}
            for run, by_ds in scores.items() for ds, by_m in by_ds.items() for m, v in by_m.items()]


def plot(scores: dict) -> plt.Figure:
    """The figure for {run: case_scores(run)}; the caller shows or saves it."""
    runs = list(scores)
    fig, axes = plt.subplots(1, len(METRICS), figsize=(5.2 * len(METRICS), 4.6), sharey=True)
    rng = np.random.default_rng(0)  # fixed jitter, so the figure is the same every time
    for ax, (metric, title) in zip(axes, METRICS):
        for i, run in enumerate(runs):
            for j, (ds, _, colour, marker) in enumerate(DATASETS):
                if ds not in scores[run]:
                    continue
                v = scores[run][ds][metric]
                x = i + (j - 0.5) * 0.38
                ax.scatter(x + rng.uniform(-0.1, 0.1, len(v)), v, s=14, color=colour, marker=marker, alpha=0.45,
                           linewidths=0, zorder=2)
                ax.plot([x - 0.15, x + 0.15], [v.mean()] * 2, color=colour, lw=2.5, solid_capstyle="round", zorder=3)
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
    handles = [plt.Line2D([], [], color=c, marker=mk, ls="", markersize=6, label=name) for _, name, c, mk in DATASETS]
    fig.legend(handles=handles, loc="upper left", ncol=2, frameon=False, fontsize=9, bbox_to_anchor=(0.01, 0.99))
    fig.text(0.01, 0.01, "Dots: one per case. Bar and number: mean. MSD test and external use different label "
             "definitions, so compare runs within a dataset. Research use only.", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    return fig


def main(runs: list[str], out: str = OUT) -> list[dict]:
    scores = {Path(r).name: s for r in runs if (s := case_scores(Path(r)))}
    if not scores:
        raise SystemExit("No metrics found; run hippo.evaluate and/or hippo.external on these runs first")
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
        print(f"{r['run']:>13}  {r['dataset']:<8}  {r['metric']:<4}  n={r['n']:<3} mean {r['mean']:.3f}  "
              f"median {r['median']:.3f}  std {r['std']:.3f}")
    print(f"Saved {out.with_suffix('.png')} and {out.with_suffix('.csv')}")
    return rows


if __name__ == "__main__":
    matplotlib.use("Agg")  # files only; set here, not on import, so notebooks keep inline figures
    p = defaults.make_parser("Plot Dice and NSD per run on the MSD test split and the external dataset.")
    p.add_argument("--runs", nargs="+", required=True, help="Run folders, e.g. runs/baseline runs/no_norm")
    p.add_argument("--out", default=OUT, help="Output path without extension (.png and .csv are written)")
    a = p.parse_args()
    main(a.runs, a.out)
