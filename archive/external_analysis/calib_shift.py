"""Mean prediction-vs-label centre offset on 20 subjects -> shift the predictions of the held-out subjects."""
import json, sys
import numpy as np, nibabel as nib, torch, yaml
from scipy import ndimage
from scipy.stats import wilcoxon
sys.path.insert(0, "src")
from hippo import external

cfg = yaml.safe_load(open("runs/baseline/config.yaml")); tol = cfg["eval"]["nsd_tolerance_mm"]
cases = json.load(open("data/external/cases.json"))
data = {}
for case, pred, batch in external.predictions("runs/baseline", cases):
    A = np.asarray(batch["image"].affine); A = A[0] if A.ndim == 3 else A
    assert np.allclose(A[:3, :3], np.diag(np.diag(A)[:3])) and (np.diag(A)[:3] > 0).all()  # RAS, axis-aligned
    p = external.largest_piece(pred[0, 0].numpy())[0] > 0
    y = batch["label"][0, 0].numpy() > 0
    sp = np.diag(A)[:3]
    off = (np.argwhere(p).mean(0) - np.argwhere(y).mean(0)) * sp           # prediction - label, mm
    data[case["case"]] = dict(p=p, y=y, sp=sp, off=off)

def score(p, y, sp):
    m = external.whole_metrics(torch.from_numpy(p.astype(np.int64))[None, None],
                               torch.from_numpy(y.astype(np.int64))[None, None], sp, tol)
    return m["dice"], m["nsd"]

def shifted(p, shift_mm, sp, whole):
    s = shift_mm / sp
    if whole:
        return ndimage.shift(p.astype(np.uint8), np.rint(s), order=0) > 0
    return ndimage.shift(p.astype(np.float32), s, order=1) >= 0.5           # sub-voxel: interpolate, threshold

def run(train_subj, test_subj, label):
    rec = {side: -np.mean([data[f"{s}_{side}"]["off"] for s in train_subj], 0) for side in "LR"}
    rows = []
    for s in test_subj:
        for side in "LR":
            d = data[f"{s}_{side}"]
            base = score(d["p"], d["y"], d["sp"])
            whole = score(shifted(d["p"], rec[side], d["sp"], True), d["y"], d["sp"])
            frac = score(shifted(d["p"], rec[side], d["sp"], False), d["y"], d["sp"])
            rows.append((f"{s}_{side}", base, whole, frac))
    return rec, rows

subjects = sorted({k[:3] for k in data})
print("== Calibrate on s01-s20, test on s21-s25")
rec, rows = run(subjects[:20], subjects[20:], "fixed")
for side in "LR":
    print(f"  recommended shift {side} (x, y, z mm): {np.round(rec[side], 2)}  -> whole voxels {np.rint(rec[side]).astype(int)}")
for name, i in (("no shift", 1), ("whole-voxel shift", 2), ("sub-voxel shift", 3)):
    d = np.array([r[i][0] for r in rows]); n = np.array([r[i][1] for r in rows])
    print(f"  {name:18} Dice {d.mean():.4f}  NSD {n.mean():.4f}")
b = np.array([r[1][0] for r in rows]); w = np.array([r[2][0] for r in rows]); f = np.array([r[3][0] for r in rows])
print(f"  per case Dice change (whole): " + ", ".join(f"{r[0]} {r[2][0]-r[1][0]:+.3f}" for r in rows))
print(f"  whole: better {int((w>b).sum())}/10, p {wilcoxon(w,b).pvalue:.2g};  sub-voxel: better {int((f>b).sum())}/10, p {wilcoxon(f,b).pvalue:.2g}")

print("\n== 5-fold check (20 subjects calibrate, 5 test; every subject tested once)")
allrows, recs = [], []
for k in range(5):
    test = subjects[k::5]; train = [s for s in subjects if s not in test]
    r, rows = run(train, test, k); allrows += rows; recs.append(r)
for side in "LR":
    R = np.array([r[side] for r in recs])
    print(f"  shift {side} across folds: mean {np.round(R.mean(0), 2)}, range x {R[:,0].min():+.2f}..{R[:,0].max():+.2f}, "
          f"y {R[:,1].min():+.2f}..{R[:,1].max():+.2f}, z {R[:,2].min():+.2f}..{R[:,2].max():+.2f}")
for side in ["L", "R", "both"]:
    sel = [r for r in allrows if side == "both" or r[0].endswith(side)]
    for i, name in ((2, "whole-voxel"), (3, "sub-voxel")):
        for j, m in ((0, "Dice"), (1, "NSD")):
            a = np.array([r[1][j] for r in sel]); c = np.array([r[i][j] for r in sel])
            print(f"  {side:4} {name:11} {m:4} {a.mean():.4f} -> {c.mean():.4f} ({(c-a).mean():+.4f}; better {int((c>a).sum())}, worse {int((c<a).sum())}, p {wilcoxon(c,a).pvalue:.2g})")

print("\n== Sign check: centre offset (prediction - label, mm) before and after the shift, held-out s21-s25")
rec, _ = run(subjects[:20], subjects[20:], "fixed")
for s in subjects[20:]:
    for side in "LR":
        d = data[f"{s}_{side}"]; q = shifted(d["p"], rec[side], d["sp"], False)
        after = (np.argwhere(q).mean(0) - np.argwhere(d["y"]).mean(0)) * d["sp"]
        # opposite direction, for comparison
        r = shifted(d["p"], -rec[side], d["sp"], False)
        print(f"  {s}_{side}: before {np.round(d['off'],2)}  after {np.round(after,2)}  "
              f"Dice no shift {score(d['p'], d['y'], d['sp'])[0]:.3f}, shifted {score(q, d['y'], d['sp'])[0]:.3f}, opposite {score(r, d['y'], d['sp'])[0]:.3f}")
