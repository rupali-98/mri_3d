# Archive: true 3D viewers (not in use for now)

Scans and predictions are viewed with the 3D Slicer-style slice notebooks instead: `notebooks/view_slices.ipynb` and `notebooks/compare_slices.ipynb`.

| File | What it does |
|---|---|
| `notebooks/view3d_vedo.ipynb` | vedo builds the surfaces, plotly shows them rotatable in the notebook. Predictions solid, ground truth as a faint ghost. `VIEW = "window"` opens a native vedo window |
| `notebooks/view3d_file.ipynb` | File picker with 3D volume, 3D surface or slices. The volume mode is slow in VS Code |
| `render3d.py` | Native vedo window (needs a screen): translucent MRI + solid label surfaces, several predictions side by side with a linked camera. Was `python -m hippo.render3d` |
| `tests/test_render3d.py` | Its tests; still collected by `python -m pytest -q` from the project root |

All of them still run from here and need vedo (in `requirements.txt` and `requirements-viewer.txt`):

```bash
PYTHONPATH=src python archive/render3d.py --case hippocampus_135 --pred runs/figures/compare/hippocampus_135_pred_*.nii.gz
```

The notebooks add `archive/` to the import path themselves. To bring `render3d` back into the package, move it to `src/hippo/` and change its imports back to `from . import CLASS_NAMES, defaults`.
