"""Dice/NSD per hemisphere (L, R) for the external crops: none vs z-5 vs z-10, change and Wilcoxon p, left vs right."""
import csv, numpy as np
from scipy.stats import wilcoxon
R=["baseline","clean_labels","no_norm"]; C=["none","z-5","z-10"]
def load(r,c):
    return {x["case"]:x for x in csv.DictReader(open(f"runs/{r}/shifted_external/{c}/metrics_external.csv"))}
def p(a,b): d=np.array(a)-np.array(b); return 1.0 if np.allclose(d,0) else wilcoxon(a,b).pvalue
# check none == external/ csv
for r in R:
    e={x["case"]:x for x in csv.DictReader(open(f"runs/{r}/external/metrics_external.csv"))}; n=load(r,"none")
    print(r,"none==external:", all(abs(float(e[k]["dice"])-float(n[k]["dice"]))<1e-6 for k in n))
for r in R:
    D={c:load(r,c) for c in C}
    print(f"\n===== {r}")
    for side in "LR":
        ks=sorted(k for k in D["none"] if k.endswith(side))
        print(f"-- side {side} n={len(ks)}")
        for c in C:
            g=lambda m:np.array([float(D[c][k][m]) for k in ks])
            d,n=g("dice"),g("nsd"); vr=g("vol_pred_ml")/g("vol_true_ml"); cov=g("label_coverage")
            line=f"{c:5} Dice {d.mean():.3f}±{d.std():.3f} med {np.median(d):.3f} min {d.min():.3f}({ks[d.argmin()]})  NSD {n.mean():.3f}±{n.std():.3f} med {np.median(n):.3f} min {n.min():.3f}({ks[n.argmin()]})  vol {np.median(vr):.3f} over {int((vr>1).sum())}  cov {cov.min():.3f}"
            if c!="none":
                d0=np.array([float(D['none'][k]['dice']) for k in ks]); n0=np.array([float(D['none'][k]['nsd']) for k in ks])
                line+=f"\n      dDice {(d-d0).mean():+.4f} worse {int(((d-d0)<0).sum())} p {p(d,d0):.2g} | dNSD {(n-n0).mean():+.4f} worse {int(((n-n0)<0).sum())} p {p(n,n0):.2g}"
            print(line)
    # L vs R paired by subject
    subs=sorted({k[:-2] for k in D["none"]})
    for c in C:
        for m in ["dice","nsd"]:
            L=np.array([float(D[c][s+"_L"][m]) for s in subs]); Rr=np.array([float(D[c][s+"_R"][m]) for s in subs])
            print(f"L-R {c:5} {m:4}: mean diff {(L-Rr).mean():+.4f}, L higher {int((L>Rr).sum())}/25, p {p(L,Rr):.2g}")
    for c in ["z-5","z-10"]:
        for m in ["dice","nsd"]:
            dL=np.array([float(D[c][s+"_L"][m])-float(D["none"][s+"_L"][m]) for s in subs]); dR=np.array([float(D[c][s+"_R"][m])-float(D["none"][s+"_R"][m]) for s in subs])
            print(f"gain L vs R {c:5} {m:4}: L {dL.mean():+.4f} R {dR.mean():+.4f}, p {p(dL,dR):.2g}")
