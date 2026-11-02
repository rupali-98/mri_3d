"""Baseline model vs ASHS-PMC-T1 (anterior/posterior hippocampus) vs MNI-HISUB25 labels, on s03 and s17.

Everything is compared on ASHS's own 0.5 x 1 x 0.5 mm grid, restricted to the external crop box:
the model's prediction and the external label (1 mm) are placed on it by world position.
Needs the ASHS results in data/ashs/<sNN>/result/ (ITK-SNAP DSS downloads).

Outputs: runs/analysis/external/ashs/ashs_compare.json and the saved predictions; prints per case and the means.

Usage (from the project root):
    PYTHONPATH=src python archive/external_analysis/ashs_compare.py
"""
import glob, json, sys
import numpy as np, nibabel as nib
sys.path.insert(0, "src")
from hippo import external
from hippo.slices import on_grid

OUT = "runs/analysis/external/ashs"
ASHS = {"L": (1, 2), "R": (101, 102)}          # (anterior, posterior)
cases = [c for c in json.load(open("data/external/cases.json")) if c["case"][:3] in ("s03", "s17")]

# model predictions on the usual atlas crops, saved so they can be placed by world position
import os; os.makedirs(OUT, exist_ok=True)
for case, pred, batch in external.predictions("runs/baseline", cases):
    A = np.asarray(batch["image"].affine); A = A[0] if A.ndim == 3 else A
    nib.save(nib.Nifti1Image(pred[0, 0].numpy().astype(np.uint8), A), f"{OUT}/{case['case']}_pred.nii.gz")

def dice(a, b): return 2 * (a & b).sum() / max(a.sum() + b.sum(), 1)

rows = []
for c in cases:
    s, side = c["case"].split("_")
    seg_img = nib.as_closest_canonical(nib.load(glob.glob(f"data/ashs/{s}/result/layer_002_*.nii.gz")[0]))
    seg = np.rint(np.asarray(seg_img.dataobj)).astype(int)
    sp, org = np.array(seg_img.header.get_zooms()[:3]), seg_img.affine[:3, 3]
    # keep the part of the ASHS grid inside this hemisphere's crop box
    crop = nib.load(c["image"]); cs = np.array(crop.header.get_zooms()[:3]); co = crop.affine[:3, 3]
    lo, hi = co - cs / 2, co + (np.array(crop.shape) - 0.5) * cs
    box = tuple(slice(max(int(np.ceil((lo[a] - org[a]) / sp[a])), 0), min(int(np.floor((hi[a] - org[a]) / sp[a])) + 1, seg.shape[a]))
                for a in range(3))
    seg = seg[box]
    o2 = org + np.array([b.start for b in box]) * sp
    ant, post = ASHS[side]
    a_ant, a_post = seg == ant, seg == post
    m = on_grid(f"{OUT}/{c['case']}_pred.nii.gz", seg.shape, sp, o2)       # model: 1 anterior, 2 posterior
    m = np.rint(np.asarray(m)).astype(int)
    mp = m                                                                  # plain prediction (main result)
    m = external.largest_piece(m)[0]                                        # main piece: geometry without islands
    gt = np.asarray(on_grid(c["label"], seg.shape, sp, o2)) > 0              # MNI-HISUB25 whole hippocampus
    m_ant, m_post, m_all, a_all = m == 1, m == 2, m > 0, a_ant | a_post
    vml = np.prod(sp) / 1000
    ys = o2[1] + np.arange(seg.shape[1]) * sp[1]
    def boundary(ant_mask, post_mask):
        """Front/back boundary (mm): halfway between the back of the anterior part and the front of the posterior part."""
        ya = ys[np.flatnonzero(ant_mask.any((0, 2)))]; yp = ys[np.flatnonzero(post_mask.any((0, 2)))]
        return (ya.min() + yp.max()) / 2 if len(ya) and len(yp) else np.nan
    xs = o2[0] + np.arange(seg.shape[0]) * sp[0]
    def edges(mask):
        x = xs[np.flatnonzero(mask.any((1, 2)))]; y = ys[np.flatnonzero(mask.any((0, 2)))]
        inner, outer = (x.max(), x.min()) if side == "L" else (x.min(), x.max())   # RAS: left inner side = larger x
        return inner, outer, y.max(), y.min()                                       # inner, outer, front, back
    r = dict(case=c["case"], dice_whole_model_gt_plain=dice(mp > 0, gt),
             dice_ant_plain=dice(mp == 1, a_ant), dice_post_plain=dice(mp == 2, a_post),
             dice_ant=dice(m_ant, a_ant), dice_post=dice(m_post, a_post),
             dice_whole_model_ashs=dice(m_all, a_all), dice_whole_ashs_gt=dice(a_all, gt), dice_whole_model_gt=dice(m_all, gt),
             vol_model_ant=m_ant.sum() * vml, vol_ashs_ant=a_ant.sum() * vml,
             vol_model_post=m_post.sum() * vml, vol_ashs_post=a_post.sum() * vml,
             vol_gt=gt.sum() * vml,
             boundary_model=boundary(m_ant, m_post), boundary_ashs=boundary(a_ant, a_post),
             ashs_touches_box=bool(a_all[0].any() or a_all[-1].any() or a_all[:, 0].any() or a_all[:, -1].any()
                                   or a_all[:, :, 0].any() or a_all[:, :, -1].any()))
    em, ea, eg = edges(m_all), edges(a_all), edges(gt)
    for k, name in enumerate(("inner", "outer", "front", "back")):
        r[f"{name}_model_minus_gt"] = em[k] - eg[k]
        r[f"{name}_ashs_minus_gt"] = ea[k] - eg[k]
    rows.append(r)

for r in rows:
    side = r["case"][-1]
    sgn = -1 if side == "L" else 1   # inner-edge difference expressed as + = reaches further inwards
    print(f"\n== {r['case']}")
    print(f"  (plain prediction: anterior {r['dice_ant_plain']:.3f}, posterior {r['dice_post_plain']:.3f}, whole vs GT {r['dice_whole_model_gt_plain']:.3f})")
    print(f"  anterior  Dice model vs ASHS {r['dice_ant']:.3f}   volume model {r['vol_model_ant']:.2f} ml, ASHS {r['vol_ashs_ant']:.2f} ml")
    print(f"  posterior Dice model vs ASHS {r['dice_post']:.3f}   volume model {r['vol_model_post']:.2f} ml, ASHS {r['vol_ashs_post']:.2f} ml")
    print(f"  front/back boundary y: model {r['boundary_model']:.1f} mm, ASHS {r['boundary_ashs']:.1f} mm "
          f"(model {'further forward' if r['boundary_model'] > r['boundary_ashs'] else 'further back'} by {abs(r['boundary_model'] - r['boundary_ashs']):.1f} mm)")
    print(f"  whole Dice: model-ASHS {r['dice_whole_model_ashs']:.3f}, ASHS-GT {r['dice_whole_ashs_gt']:.3f}, model-GT {r['dice_whole_model_gt']:.3f}"
          f"   (GT volume {r['vol_gt']:.2f} ml; ASHS reaches crop edge: {r['ashs_touches_box']})")
    print(f"  edges vs GT (mm): inner edge  model {-sgn * r['inner_model_minus_gt']:+.1f}, ASHS {-sgn * r['inner_ashs_minus_gt']:+.1f}  (+ = further inwards than GT)")
    print(f"                    front edge  model {r['front_model_minus_gt']:+.1f}, ASHS {r['front_ashs_minus_gt']:+.1f}  (+ = further forward)")
    print(f"                    outer edge  model {sgn * r['outer_model_minus_gt']:+.1f}, ASHS {sgn * r['outer_ashs_minus_gt']:+.1f}  (+ = further outwards)")
    print(f"                    back edge   model {r['back_model_minus_gt']:+.1f}, ASHS {r['back_ashs_minus_gt']:+.1f}  (+ = further forward)")
json.dump(rows, open(f"{OUT}/ashs_compare.json", "w"), indent=1, default=float)

import numpy as np
print("\n== mean over the 4 hippocampi (model = largest piece)")
for k in ("dice_ant", "dice_post", "dice_whole_model_ashs", "dice_whole_ashs_gt", "dice_whole_model_gt", "dice_whole_model_gt_plain"):
    print(f"  {k:28} {np.mean([r[k] for r in rows]):.3f}")
print(f"  boundary model - ASHS (mm, + = model further forward): {[round(r['boundary_model'] - r['boundary_ashs'], 1) for r in rows]}")
for name in ("inner", "front", "outer", "back"):
    sg = lambda r: (-1 if r["case"].endswith("L") else 1) if name in ("inner", "outer") else 1
    sg2 = lambda r: -sg(r) if name == "inner" else sg(r)
    print(f"  {name:5} edge vs GT, model {[round(sg2(r) * r[name + '_model_minus_gt'], 1) for r in rows]}, ASHS {[round(sg2(r) * r[name + '_ashs_minus_gt'], 1) for r in rows]}")
