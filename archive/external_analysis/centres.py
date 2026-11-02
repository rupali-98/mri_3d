"""Predicted vs true hippocampus centre (world mm) per case, external atlas crops and MSD test (baseline)."""
import csv, json, sys
import numpy as np, nibabel as nib
from scipy.stats import wilcoxon
sys.path.insert(0, "src")
from hippo import external

OUT = "runs/analysis/external/centres.csv"  # per-case offsets (folder made below)
import os; os.makedirs(os.path.dirname(OUT), exist_ok=True)

def measure(cases, group):
    rows = []
    for case, pred, batch in external.predictions("runs/baseline", cases):
        A = np.asarray(batch["image"].affine); A = A[0] if A.ndim == 3 else A
        p, y = pred[0, 0].numpy() > 0, batch["label"][0, 0].numpy() > 0
        kept = external.largest_piece(p.astype(int))[0] > 0
        world = lambda m: nib.affines.apply_affine(A, np.argwhere(m))
        wp, wk, wy = world(p), world(kept), world(y)
        name = case.get("case") or case["image"].split("/")[-1].split(".nii")[0]
        r = {"group": group, "case": name, "side": name[-1] if group == "external" else "-"}
        for a, ax in enumerate("xyz"):
            r[f"d{ax}"] = wp[:, a].mean() - wy[:, a].mean()            # plain prediction
            r[f"d{ax}_largest"] = wk[:, a].mean() - wy[:, a].mean()    # largest piece
            r[f"lo_{ax}"] = wk[:, a].min() - wy[:, a].min()            # lower edge (largest piece)
            r[f"hi_{ax}"] = wk[:, a].max() - wy[:, a].max()            # upper edge
        rows.append(r)
    return rows

ext = json.load(open("data/external/cases.json"))
msd = json.load(open("data/splits.json"))["test"]
rows = measure(ext, "external") + measure(msd, "msd")
with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def show(sel, title):
    print(f"\n== {title} (n={len(sel)})")
    for kind in ("", "_largest"):
        line = f"  centre{kind or ' (plain)':10}"
        for ax in "xyz":
            d = np.array([r[f'd{ax}{kind}'] for r in sel])
            line += (f" | {ax}: mean {d.mean():+.2f} sd {d.std():.2f} med {np.median(d):+.2f} "
                     f"[{d.min():+.1f},{d.max():+.1f}] +{int((d>0).sum())}/-{int((d<0).sum())} p {wilcoxon(d).pvalue:.1g}")
        print(line)
    for ax in "xyz":
        lo = np.array([r[f'lo_{ax}'] for r in sel]); hi = np.array([r[f'hi_{ax}'] for r in sel])
        print(f"  edges {ax}: lower edge {np.median(lo):+.1f} mm (mean {lo.mean():+.2f}), upper edge {np.median(hi):+.1f} mm (mean {hi.mean():+.2f})")
e = [r for r in rows if r["group"] == "external"]
show([r for r in e if r["side"] == "L"], "external LEFT")
show([r for r in e if r["side"] == "R"], "external RIGHT")
show([r for r in rows if r["group"] == "msd"], "MSD test (reference)")
print("\nper-case written to", OUT)
