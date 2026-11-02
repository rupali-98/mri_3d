# Archive: true 3D viewers (not in use for now)

Scans and predictions are viewed with the Streamlit page instead (`streamlit_dashboard/compare_slices.py`: slices and 3D surfaces). The two 3D notebooks that were here (`view3d_vedo.ipynb`, `view3d_file.ipynb`) were deleted on 2026-11-02; git history has them.

| File | What it does |
|---|---|
| `render3d.py` | Native vedo window (needs a screen): translucent MRI + solid label surfaces, several predictions side by side with a linked camera. Was `python -m hippo.render3d` |
| `tests/test_render3d.py` | Its tests; still collected by `python -m pytest -q` from the project root |

Both still run from here and need vedo (in `requirements.txt` and `requirements-viewer.txt`):

```bash
PYTHONPATH=src python archive/render3d.py --case hippocampus_135 --pred runs/figures/compare/hippocampus_135_pred_*.nii.gz
```

To bring `render3d` back into the package, move it to `src/hippo/` and change its imports back to `from . import CLASS_NAMES, defaults`.

# Archive: crop shift test and external analysis scripts (archived 2026-10-28)

The external analysis settled on the plain atlas box with **no shift** (see `docs/external_validation.md`): shifting predictions to match the external labels made results worse on held-out subjects. These scripts are kept for the planned full-image training branch, where crop position will matter again.

| File | What it does |
|---|---|
| `shift.py` | Was `hippo.shift`. Shifted copies of the MSD test crops (zero padding) or of the external atlas box (`--external`), scored per condition; `--axes`, `--conditions` (e.g. `x-4_z-10`), `--largest-piece`. Writes `data/shifted*/` and `<run>/shifted*/` |
| `tests/test_shift.py` | Its tests; still collected by `python -m pytest -q` |
| `external_analysis/centres.py` | Predicted vs label centre and edge offsets (world mm) per hippocampus, external and MSD test -> `runs/analysis/external/centres.csv` |
| `external_analysis/columns.py` | Top and bottom edges column by column, inner vs outer half (shape difference vs shift) |
| `external_analysis/calib_shift.py` | Mean offset on 20 subjects applied to the held-out 5 (and 5-fold): shows shifting predictions lowers Dice |
| `external_analysis/gtcheck.py` | Label (yellow) and prediction (cyan) outlines on 1 mm T1, 0.4 mm T1 and T2 -> `runs/analysis/external/gtcheck/` |
| `external_analysis/nosub.py` | Re-scores against the external label without the subiculum (part 2) |
| `external_analysis/ashs_compare.py` | Model vs ASHS-PMC-T1 vs manual labels on s03/s17 (Dice per part, front/back boundary, edges) on ASHS's 0.5 mm grid -> `runs/analysis/external/ashs/` |
| `external_analysis/ashs_labels.py` | ASHS anterior/posterior cut to each external crop box (1 = anterior, 2 = posterior) -> `data/ashs/labelsTr/`, shown by the Streamlit page as "Second reference" |
| `external_analysis/lr.py`, `compare_largest.py` | Per-hemisphere Dice/NSD for none / z-5 / z-10, plain vs largest piece (read `runs/<run>/shifted_external*/`) |

Run from the project root with the venv; the analysis scripts use `runs/baseline` and `data/external/`:

```bash
PYTHONPATH=src python archive/shift.py --external data/mri_dataset --run runs/baseline --conditions z-5 z-10
PYTHONPATH=src python archive/external_analysis/centres.py
```

`hippo.external` still has the options these use (`prepare(shift=...)`, `--largest-piece`, `--recentre`, `--msd-crop`). To bring the shift test back into the package, move `shift.py` to `src/hippo/`, change `from hippo import ...` back to `from . import ...`, and move its test to `tests/`.
