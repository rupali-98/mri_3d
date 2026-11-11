"""One results table per experiment, a Bland-Altman plot, and the ablation table across experiments.

An experiment is a folder of fold runs (runs/<name>/fold0 ... fold4, from hippo.train --fold) or a single
run folder; when a folder holds fold runs, only those are used (runs/baseline/fold* rather than the
single run saved in runs/baseline itself). For each test set and label the table gives Dice, HD95 (mm), NSD (1 mm), absolute and
relative volume error and ICC(A,1): each fold model is scored on the whole test set, and the table shows
the mean ± SD over the folds (a single run shows its own value). Test sets, all test-only:
  - MSD test (the fixed 39-case split): anterior, posterior and whole hippocampus (labels merged);
  - MNI-HISUB25: whole hippocampus only (its labels are subfields, not anterior/posterior), on the atlas
    box, the atlas box with the largest piece kept, and the MSD-style box with the largest piece kept.
A test set is left out when its files are missing (run hippo.evaluate / hippo.external first).

The Bland-Altman plot compares predicted with true volume per case; with folds, each case's predicted
volume is the mean over the fold models.

--ablation reads every evaluated run back from MLflow (src/hippo/tracking.py) and writes one row per
experiment: its settings, number of folds and the main metrics (mean ± SD over folds).

Outputs:
    <experiment>/results/results.md      the table (Markdown)
    <experiment>/results/results.csv     the same, one row per test set and label, mean and SD columns
    <experiment>/results/bland_altman.png
    runs/ablation.md, runs/ablation.csv  with --ablation

Usage:
    python -m hippo.results --experiment runs/baseline
    python -m hippo.results --experiment runs/baseline --name "3D U-Net"
    python -m hippo.results --ablation
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from . import tracking
from .metrics import summarise
from .metrics_plot import GRID, INK, MUTED

# (name, folder in the run, CSV, labels, post-processing, colour per label)
TEST_SETS = [
    ("MSD test", "", "metrics_test.csv", ("anterior", "posterior", "whole"), "none",
     {"anterior": "#e0502a", "posterior": "#4682b4", "whole": "#2a78d6"}),
    ("MNI-HISUB25, atlas box", "external", "metrics_external.csv", ("whole",), "none", {"whole": "#eb6834"}),
    ("MNI-HISUB25, atlas box", "external_largest", "metrics_external.csv", ("whole",), "largest piece",
     {"whole": "#9c3a12"}),
    ("MNI-HISUB25, MSD-style box", "external_msd_crop_largest", "metrics_external.csv", ("whole",), "largest piece",
     {"whole": "#1baf7a"}),
]
COLUMNS = [("dice", "Dice", 3), ("hd95", "HD95 (mm)", 2), ("nsd", "NSD (1 mm)", 3),
           ("abs_vol_error_ml", "Abs. volume error (ml)", 3), ("rel_vol_error_pct", "Rel. volume error (%)", 1),
           ("icc", "ICC", 3)]
# ablation columns: (heading, MLflow metric, digits)
ABLATION = [("MSD test Dice, anterior", "test.dice_anterior", 3), ("MSD test Dice, posterior", "test.dice_posterior", 3),
            ("MSD test Dice, whole", "test.dice_whole", 3), ("MSD test HD95, whole (mm)", "test.hd95_whole", 2),
            ("MSD test NSD, whole", "test.nsd_whole", 3), ("MSD test volume ICC, whole", "test.icc_whole", 3),
            ("External Dice, atlas box", "external.dice", 3),
            ("External Dice, atlas box + largest piece", "external_largest.dice", 3),
            ("External Dice, MSD-style box + largest piece", "external_msd_crop_largest.dice", 3),
            ("External HD95, atlas box (mm)", "external.hd95", 2), ("External volume ICC, atlas box", "external.icc", 3)]
SETTINGS = ["labels", "normalize", "random_trim"]


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def fold_runs(experiment) -> list[Path]:
    """The experiment's fold folders (fold0, fold1, ...), or the folder itself when it is a single run."""
    folds = sorted(p for p in Path(experiment).glob("fold*") if (p / "config.yaml").exists())
    return folds or [Path(experiment)]


def mean_sd(values: list[float]) -> tuple[float, float]:
    """Mean and SD (n - 1) over folds, skipping missing values; SD is nan for a single value."""
    v = np.array([x for x in values if x == x], dtype=float)  # x == x is False for nan
    if not len(v):
        return float("nan"), float("nan")
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else float("nan")


def scores(experiment) -> list[dict]:
    """One row per test set and label: number of folds and cases, and mean and SD of each column over folds."""
    runs, rows = fold_runs(experiment), []
    for name, folder, file, labels, post, _ in TEST_SETS:
        found = [read_csv(run / folder / file) for run in runs if (run / folder / file).exists()]
        if not found:
            continue
        for label in labels:
            suffix = f"_{label}" if folder == "" else ""
            per_fold = [summarise(cases, suffix) for cases in found]
            if not any(s["n"] for s in per_fold):
                continue  # e.g. whole-hippocampus columns missing in an evaluation from before they existed
            row = {"test set": name, "post-processing": post, "label": label, "folds": len(found),
                   "cases": max(s["n"] for s in per_fold)}
            for key, _, _ in COLUMNS:
                row[f"{key}_mean"], row[f"{key}_sd"] = mean_sd([s.get(key, float("nan")) for s in per_fold])
            rows.append(row)
    return rows


def cell(mean: float, sd: float, digits: int) -> str:
    if mean != mean:
        return "–"
    return f"{mean:.{digits}f}" + (f" ± {sd:.{digits}f}" if sd == sd else "")


def markdown(rows: list[dict], title: str) -> str:
    """The table as Markdown, with a note on how to read it."""
    head = ["Test set", "Post-processing", "Label", "Folds", "Cases", *(h for _, h, _ in COLUMNS)]
    lines = [f"## {title}", "", "| " + " | ".join(head) + " |", "|" + "---|" * 5 + "---:|" * len(COLUMNS)]
    for r in rows:
        lines.append("| " + " | ".join([r["test set"], r["post-processing"], r["label"], str(r["folds"]),
                                        str(r["cases"]), *(cell(r[f"{k}_mean"], r[f"{k}_sd"], d)
                                                           for k, _, d in COLUMNS)]) + " |")
    lines += ["", "Mean ± SD over the fold models, each scored on the whole test set (a single run shows its own "
              "value). Whole = anterior and posterior merged. MNI-HISUB25 has subfield labels, so only the whole "
              "hippocampus is scored there. Relative volume error is |predicted − true| / true. "
              "ICC = ICC(A,1), absolute agreement. Research use only, not for diagnosis.", ""]
    return "\n".join(lines)


def volumes(experiment) -> list[tuple]:
    """(test set, post-processing, label, colour, predicted ml, true ml) per panel; predicted = mean over folds."""
    runs, panels = fold_runs(experiment), []
    for name, folder, file, labels, post, colours in TEST_SETS:
        found = [{r["case"]: r for r in read_csv(run / folder / file)} for run in runs
                 if (run / folder / file).exists()]
        if not found:
            continue
        cases = sorted(set.intersection(*(set(f) for f in found)))
        for label in labels:
            suffix = f"_{label}" if folder == "" else ""
            if f"vol_pred_ml{suffix}" not in found[0][cases[0]]:
                continue
            pred = np.array([np.mean([float(f[c][f"vol_pred_ml{suffix}"]) for f in found]) for c in cases])
            true = np.array([float(found[0][c][f"vol_true_ml{suffix}"]) for c in cases])
            panels.append((name, post, label, colours[label], pred, true))
    return panels


def bland_altman(panels: list[tuple], folds: int) -> plt.Figure:
    """One Bland-Altman panel per test set and label: difference against mean volume, bias and 95% limits."""
    cols = min(3, len(panels))
    rows = math.ceil(len(panels) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 3.9 * rows + 0.6), squeeze=False)
    for ax, (name, post, label, colour, pred, true) in zip(axes.flat, panels):
        mean, diff = (pred + true) / 2, pred - true
        bias, spread = diff.mean(), 1.96 * diff.std(ddof=1)
        ax.scatter(mean, diff, s=18, color=colour, alpha=0.6, linewidths=0, zorder=3)
        for y, style, text in [(bias, "-", f"bias {bias:+.2f}"), (bias + spread, "--", f"+1.96 SD {bias + spread:+.2f}"),
                               (bias - spread, "--", f"−1.96 SD {bias - spread:+.2f}")]:
            ax.axhline(y, color=INK if style == "-" else MUTED, lw=1.2, ls=style, zorder=2)
            ax.annotate(text, (1, y), xycoords=("axes fraction", "data"), xytext=(-4, 3), textcoords="offset points",
                        ha="right", fontsize=8, color=INK)
        ax.axhline(0, color=GRID, lw=0.8, zorder=1)
        ax.set_title(f"{name}, {label}" + ("" if post == "none" else f"\n{post}"), loc="left", fontsize=10, color=INK)
        ax.set_xlabel("Mean of predicted and true volume (ml)", fontsize=8, color=MUTED)
        ax.set_ylabel("Predicted − true (ml)", fontsize=8, color=MUTED)
        ax.grid(color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(colors=MUTED, labelsize=8, length=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
    for ax in list(axes.flat)[len(panels):]:
        ax.axis("off")
    note = "One dot per case. Solid line: mean difference (bias); dashed: 95% limits of agreement. "
    note += f"Predicted volume: mean of {folds} fold models. " if folds > 1 else ""
    fig.text(0.01, 0.01, note + "MNI-HISUB25 volumes are in MNI space. Research use only.", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


def experiment_report(experiment: str, name: str | None = None) -> list[dict]:
    """Write the experiment's table and Bland-Altman plot to <experiment>/results/."""
    rows = scores(experiment)
    if not rows:
        raise SystemExit(f"No results in {experiment}: run hippo.evaluate and hippo.external on its runs first")
    out = Path(experiment) / "results"
    out.mkdir(parents=True, exist_ok=True)
    title = name or Path(experiment).name
    (out / "results.md").write_text(markdown(rows, title))
    with open(out / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    fig = bland_altman(volumes(experiment), len(fold_runs(experiment)))
    fig.savefig(out / "bland_altman.png", dpi=150)
    plt.close(fig)
    print(markdown(rows, title))
    print(f"Saved {out / 'results.md'}, results.csv and bland_altman.png")
    return rows


def ablation_rows(root: str = "runs") -> list[dict]:
    """One row per experiment from MLflow: settings, folds and mean ± SD of the ABLATION metrics."""
    tracking.setup(Path(root))
    runs = tracking.mlflow.search_runs(output_format="list")  # the experiment set by tracking.setup
    groups = {}
    for run in runs:
        params, metrics = run.data.params, run.data.metrics
        if "experiment" not in params or not any(key in metrics for _, key, _ in ABLATION):
            continue  # a training run from before tracking, or a run not evaluated yet
        # a fold run and a single run of the same config (runs/baseline/fold0 and runs/baseline) are kept apart
        kind = "single run" if params.get("fold", "none") == "none" else "cross-validation"
        groups.setdefault((params["experiment"], kind), []).append((params, metrics))
    rows = []
    for (experiment, kind), members in sorted(groups.items()):
        row = {"experiment": experiment, "kind": kind, "folds": len(members),
               **{s: members[0][0].get(s, "") for s in SETTINGS}}
        for heading, key, _ in ABLATION:
            row[f"{heading} mean"], row[f"{heading} sd"] = mean_sd([m.get(key, float("nan")) for _, m in members])
        rows.append(row)
    return rows


def ablation(root: str = "runs") -> list[dict]:
    """Write <root>/ablation.md and ablation.csv from MLflow."""
    rows = ablation_rows(root)
    if not rows:
        raise SystemExit("No evaluated runs in MLflow yet: run hippo.evaluate and hippo.external first")
    head = ["Experiment", "Kind", "Folds", *SETTINGS, *(h for h, _, _ in ABLATION)]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * (3 + len(SETTINGS)) + "---:|" * len(ABLATION)]
    for r in rows:
        lines.append("| " + " | ".join([r["experiment"], r["kind"], str(r["folds"]), *(str(r[s]) for s in SETTINGS),
                                        *(cell(r[f"{h} mean"], r[f"{h} sd"], d) for h, _, d in ABLATION)]) + " |")
    text = "\n".join(lines) + ("\n\nFrom MLflow. Mean ± SD over each experiment's folds. MSD test and external use "
                               "different label definitions, so compare experiments within a column. "
                               "Research use only.\n")
    (Path(root) / "ablation.md").write_text(text)
    with open(Path(root) / "ablation.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(text)
    print(f"Saved {Path(root) / 'ablation.md'} and ablation.csv")
    return rows


if __name__ == "__main__":
    matplotlib.use("Agg")  # files only
    p = argparse.ArgumentParser(description="Results table per experiment, Bland-Altman plot and ablation table.")
    p.add_argument("--experiment", help="Experiment folder (with fold0 ... fold4, or a single run), e.g. runs/baseline")
    p.add_argument("--name", help='Title of the table, e.g. "3D U-Net" (default: the folder name)')
    p.add_argument("--ablation", action="store_true", help="Write runs/ablation.md from MLflow instead")
    p.add_argument("--root", default="runs", help="Folder holding the experiments and mlflow.db (for --ablation)")
    a = p.parse_args()
    if a.ablation:
        ablation(a.root)
    elif a.experiment:
        experiment_report(a.experiment, a.name)
    else:
        p.error("give --experiment or --ablation")
