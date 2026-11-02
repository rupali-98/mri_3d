"""Label (yellow) and baseline prediction (cyan) outlines on 1 mm T1, 0.4 mm T1 and 0.4 mm T2, in world mm."""
import json, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, nibabel as nib, numpy as np
sys.path.insert(0, "src")
from hippo import external
OUT = "runs/analysis/external/gtcheck"     # figures
import os; os.makedirs(OUT, exist_ok=True)
SRC = "data/mri_dataset"

class Vol:
    def __init__(self, path):
        im = nib.load(path); self.d = np.asarray(im.dataobj, dtype=np.float32)
        self.o, self.s = im.affine[:3, 3], np.diag(im.affine)[:3]
    def coords(self, a): return self.o[a] + np.arange(self.d.shape[a]) * self.s[a]
    def idx(self, a, mm): return int(np.clip(round((mm - self.o[a]) / self.s[a]), 0, self.d.shape[a] - 1))
    def plane(self, axis, mm):
        """2D slice at world mm along axis; returns (data[h, v], h coords, v coords)."""
        i = self.idx(axis, mm); sl = [slice(None)] * 3; sl[axis] = i
        h, v = [a for a in range(3) if a != axis][0], 2
        return self.d[tuple(sl)], self.coords(h), self.coords(v)

cases = {c["case"]: c for c in json.load(open("data/external/cases.json"))}
want = ["s03_L", "s03_R", "s17_L", "s17_R"]
preds = {}
for case, pred, batch in external.predictions("runs/baseline", [cases[w] for w in want]):
    A = np.asarray(batch["image"].affine); A = A[0] if A.ndim == 3 else A
    p = nib.Nifti1Image((pred[0, 0].numpy() > 0).astype(np.uint8), A)
    nib.save(p, f"{OUT}/{case['case']}_pred.nii.gz"); preds[case["case"]] = Vol(f"{OUT}/{case['case']}_pred.nii.gz")

for name in want:
    s, side = name.split("_")
    imgs = [("T1 1 mm", Vol(f"{SRC}/{s}/{s}_t1w_standard_defaced_MNI.nii.gz"), Vol(f"{SRC}/{s}/{s}_hippolabels_t1w_standard_{side}_MNI.nii.gz")),
            ("T1 0.4 mm", Vol(f"{SRC}/{s}/{s}_t1w_hires_defaced_MNI.nii.gz"), Vol(f"{SRC}/{s}/{s}_hippolabels_hres_{side}_MNI.nii.gz")),
            ("T2 0.4 mm", Vol(f"{SRC}/{s}/{s}_t2w_hires_defaced_MNI.nii.gz"), Vol(f"{SRC}/{s}/{s}_hippolabels_hres_{side}_MNI.nii.gz"))]
    lab = imgs[0][2]; w = np.argwhere(lab.d > 0) * lab.s + lab.o
    lo, hi, c = w.min(0) - 7, w.max(0) + 7, w.mean(0)
    ys = [w[:, 1].min() + f * (w[:, 1].max() - w[:, 1].min()) for f in (0.2, 0.5, 0.8)]
    views = [(1, y, f"coronal y={y:.0f} ({t})") for y, t in zip(ys, ["tail", "body", "head"])] + [(0, c[0], f"sagittal x={c[0]:.0f}")]
    fig, axes = plt.subplots(3, 4, figsize=(17, 12))
    for r, (title, img, gt) in enumerate(imgs):
        for col, (axis, mm, vt) in enumerate(views):
            ax = axes[r, col]; h = 0 if axis == 1 else 1
            d, hc, vc = img.plane(axis, mm)
            hm, vm = (hc >= lo[h]) & (hc <= hi[h]), (vc >= lo[2]) & (vc <= hi[2])
            d = d[np.ix_(hm, vm)]
            vmin, vmax = np.percentile(d, [1, 99])
            ax.imshow(d.T, origin="lower", cmap="gray", vmin=vmin, vmax=vmax,
                      extent=[hc[hm][0], hc[hm][-1], vc[vm][0], vc[vm][-1]], interpolation="nearest")
            for m, colour in ((gt, "yellow"), (preds[name], "cyan")):
                md, mh, mv = m.plane(axis, mm)
                if (md > 0).any():
                    ax.contour(mh, mv, (md > 0).T.astype(float), levels=[0.5], colors=colour, linewidths=1.1)
            ax.set_xlim(hi[h], lo[h]) if axis == 1 else ax.set_xlim(hi[h], lo[h])  # coronal: R on left; sagittal: A on left
            ax.set_ylim(lo[2], hi[2]); ax.set_aspect("equal")
            ax.set_title(f"{title}: {vt}", fontsize=9)
            ax.set_xlabel("x mm (R ← → L)" if axis == 1 else "y mm (A ← → P)", fontsize=8); ax.set_ylabel("z mm (S ↑)", fontsize=8)
            ax.tick_params(labelsize=7); ax.grid(color="white", alpha=0.15, lw=0.5)
    fig.suptitle(f"{name}: label (yellow) vs baseline prediction (cyan)", fontsize=13)
    fig.tight_layout(); fig.savefig(f"{OUT}/{name}.png", dpi=110); plt.close(fig)
    print("saved", f"{OUT}/{name}.png")
