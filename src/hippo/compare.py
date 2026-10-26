"""Compare two evaluated runs case by case on the same split.

Both runs are scored on the same cases, so the comparison is paired: for each
metric, the mean per-case difference (B - A), how many cases got better or worse,
and a Wilcoxon signed-rank p-value. A small mean difference with a large p-value
is noise between training runs, not an effect of the change.

Usage (run python -m hippo.evaluate on both runs first):
    python -m hippo.compare runs/baseline runs/clean_labels
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon


def load(run: Path, split: str) -> dict:
    path = run / f"metrics_{split}.csv"
    if not path.exists():
        raise SystemExit(f"{path} not found: run python -m hippo.evaluate --run {run} --split {split} first")
    with open(path) as f:
        return {r["case"]: r for r in csv.DictReader(f)}


def main(run_a: str, run_b: str, split: str = "test") -> dict:
    a, b = load(Path(run_a), split), load(Path(run_b), split)
    if a.keys() != b.keys():
        raise SystemExit(f"The runs were evaluated on different cases; compare runs that share {split} cases")
    cases = sorted(a)
    metrics = [k for k in next(iter(a.values())) if k.startswith(("dice", "nsd"))]

    result = {"split": split, "n_cases": len(cases), "a": run_a, "b": run_b, "metrics": {}}
    for m in metrics:
        va = np.array([float(a[c][m]) for c in cases])
        vb = np.array([float(b[c][m]) for c in cases])
        diff = vb - va
        # wilcoxon is undefined when every case is identical; that is no difference at all
        p = 1.0 if np.allclose(diff, 0) else float(wilcoxon(vb, va).pvalue)
        result["metrics"][m] = {"mean_a": float(va.mean()), "mean_b": float(vb.mean()),
                                "mean_diff": float(diff.mean()), "better": int((diff > 0).sum()),
                                "worse": int((diff < 0).sum()), "p_wilcoxon": p}

    print(f"{split} split, {len(cases)} cases.  A = {run_a}   B = {run_b}")
    print(f"{'metric':<16}{'A':>8}{'B':>8}{'B-A':>9}{'better':>8}{'worse':>7}{'p':>8}")
    for m, r in result["metrics"].items():
        print(f"{m:<16}{r['mean_a']:>8.4f}{r['mean_b']:>8.4f}{r['mean_diff']:>+9.4f}"
              f"{r['better']:>8}{r['worse']:>7}{r['p_wilcoxon']:>8.3f}")
    out = Path(run_b) / f"compare_{split}_vs_{Path(run_a).name}.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"Saved {out}")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Paired per-case comparison of two evaluated runs.")
    p.add_argument("run_a", help="Reference run, e.g. runs/baseline")
    p.add_argument("run_b", help="Run with the change, e.g. runs/clean_labels")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    a = p.parse_args()
    main(a.run_a, a.run_b, a.split)
