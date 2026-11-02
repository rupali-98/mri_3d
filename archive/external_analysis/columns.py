"""Top and bottom edge of prediction vs label, column by column, in the inner and outer half of each hippocampus."""
import json, sys
import numpy as np
sys.path.insert(0, "src")
from hippo import external

cases = json.load(open("data/external/cases.json"))
res = {"inner": {"top": [], "bottom": []}, "outer": {"top": [], "bottom": []}}
only = {"inner": [0, 0, 0], "outer": [0, 0, 0]}   # columns: label only, prediction only, both
for case, pred, batch in external.predictions("runs/baseline", cases):
    p = external.largest_piece(pred[0, 0].numpy())[0] > 0
    y = batch["label"][0, 0].numpy() > 0
    cx = np.argwhere(y)[:, 0].mean()
    left = case["case"].endswith("L")          # RAS: left hippocampus inner side = larger x; right = smaller x
    for i in range(y.shape[0]):
        half = "inner" if (i > cx) == left else "outer"
        for j in range(y.shape[1]):
            zy, zp = np.flatnonzero(y[i, j]), np.flatnonzero(p[i, j])
            if len(zy) and len(zp):
                res[half]["top"].append(zp.max() - zy.max())        # + = prediction reaches higher
                res[half]["bottom"].append(zp.min() - zy.min())     # - = prediction reaches lower
                only[half][2] += 1
            elif len(zy):
                only[half][0] += 1
            elif len(zp):
                only[half][1] += 1
for half in ("outer", "inner"):
    t, b = np.array(res[half]["top"]), np.array(res[half]["bottom"])
    lo, po, both = only[half]
    print(f"{half} half: {both} columns with both | top edge: mean {t.mean():+.2f} mm, median {np.median(t):+.0f}, "
          f"same height {np.mean(t == 0)*100:.0f}%, prediction lower {np.mean(t < 0)*100:.0f}% "
          f"| bottom edge: mean {b.mean():+.2f} mm, median {np.median(b):+.0f}, same {np.mean(b == 0)*100:.0f}%, "
          f"prediction lower {np.mean(b < 0)*100:.0f}% | columns with label only {lo}, prediction only {po}")
