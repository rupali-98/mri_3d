"""Plain vs largest-piece Dice/NSD on the external crops (none, z-5, z-10), left and right separately."""
import csv
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

RUNS, CONDS = ["baseline", "clean_labels", "no_norm"], ["none", "z-5", "z-10"]


def load(path):
    return {r["case"]: r for r in csv.DictReader(open(path))} if path.exists() else None


def p(a, b):
    return 1.0 if np.allclose(a, b) else wilcoxon(a, b).pvalue


for run in RUNS:
    print(f"\n===== {run}")
    for side in "LR":
        print(f"-- {side}")
        base = None
        for c in CONDS:
            plain = load(Path(f"runs/{run}/shifted_external/{c}/metrics_external.csv"))
            post = load(Path(f"runs/{run}/shifted_external_largest/{c}/metrics_external.csv"))
            if not plain or not post:
                print(f"{c:5} missing results")
                continue
            ks = sorted(k for k in plain if k.endswith(side))
            g = lambda d, m: np.array([float(d[k][m]) for k in ks])
            line = f"{c:5}"
            for m in ("dice", "nsd"):
                a, b = g(plain, m), g(post, m)
                line += (f" | {m} {a.mean():.3f} -> {b.mean():.3f} ({(b - a).mean():+.4f}, "
                         f"better {int((b > a).sum())} worse {int((b < a).sum())}, p {p(b, a):.2g})")
            pieces, removed = g(post, "pieces"), g(post, "removed_ml")
            vr = g(post, "vol_pred_ml") / g(post, "vol_true_ml")
            line += (f" | pieces median {np.median(pieces):.0f} (max {pieces.max():.0f}, one piece {int((pieces == 1).sum())})"
                     f" removed {removed.mean():.3f} ml, vol ratio {np.median(g(plain, 'vol_pred_ml') / g(plain, 'vol_true_ml')):.3f}"
                     f" -> {np.median(vr):.3f}")
            print(line)
            if c == "none":
                base = {m: g(post, m) for m in ("dice", "nsd")}
            elif base is not None:
                print("      after post-processing, vs none: " + ", ".join(
                    f"{m} {(g(post, m) - base[m]).mean():+.4f} (worse {int((g(post, m) < base[m]).sum())}, "
                    f"p {p(g(post, m), base[m]):.2g})" for m in ("dice", "nsd")))
