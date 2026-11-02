"""Score baseline predictions against the external label with and without the subiculum (part 2)."""
import json, sys
import numpy as np, torch, yaml
sys.path.insert(0, "src")
from hippo import external, train

cfg = yaml.safe_load(open("runs/baseline/config.yaml")); tol = cfg["eval"]["nsd_tolerance_mm"]
cases = json.load(open("data/external/cases.json"))
T = lambda a: torch.from_numpy(a.astype(np.int64))[None, None]
rows, inner_shift = [], []
for case, pred, batch in external.predictions("runs/baseline", cases):
    sp = train.spacing_from_affine(batch["image"].affine)
    y = batch["label"][0, 0].numpy().astype(int)
    p_plain = pred[0, 0].numpy() > 0
    p_large = external.largest_piece(pred[0, 0].numpy())[0] > 0
    left = case["case"].endswith("L")
    # how far part 2 sits towards the inner side compared with part 1 (mm; + = inner)
    c1, c2 = np.argwhere(y == 1)[:, 0].mean(), np.argwhere(y == 2)[:, 0].mean()
    inner_shift.append((c2 - c1) * sp[0] * (1 if left else -1))
    cx = np.argwhere(y > 0)[:, 0].mean()
    inner = (np.arange(y.shape[0]) > cx) if left else (np.arange(y.shape[0]) < cx)
    labels = {"full label": y > 0,
              "without subiculum (part 2)": (y > 0) & (y != 2),
              "without inner half of subiculum": (y > 0) & ~((y == 2) & inner[:, None, None])}
    r = {"case": case["case"]}
    for name, lab in labels.items():
        for pn, p in (("plain", p_plain), ("largest", p_large)):
            m = external.whole_metrics(T(p), T(lab), sp, tol)
            r[(name, pn)] = (m["dice"], m["nsd"], m["vol_pred_ml"] / m["vol_true_ml"])
    rows.append(r)
print(f"part 2 lies on the inner side of part 1 by median {np.median(inner_shift):.1f} mm (inner in {sum(s > 0 for s in inner_shift)}/50)\n")
from scipy.stats import wilcoxon
for pn in ("plain", "largest"):
    print(f"-- prediction: {pn}")
    base = np.array([r[("full label", pn)][0] for r in rows])
    for name in ("full label", "without subiculum (part 2)", "without inner half of subiculum"):
        d = np.array([r[(name, pn)][0] for r in rows]); n = np.array([r[(name, pn)][1] for r in rows])
        v = np.median([r[(name, pn)][2] for r in rows])
        extra = "" if name == "full label" else f"  vs full: better {int((d > base).sum())}/50, p {wilcoxon(d, base).pvalue:.2g}"
        for s in "LR":
            pass
        print(f"   {name:34} Dice {d.mean():.3f}  NSD {n.mean():.3f}  pred/true volume {v:.2f}{extra}")
