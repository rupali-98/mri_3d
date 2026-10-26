# CLAUDE.md

Hippocampus segmentation and volumetry on 3D T1 MRI: a MONAI 3D U-Net trained on MSD Task04 (labels 0 = background, 1 = anterior, 2 = posterior), evaluated with Dice and NSD (1 mm), plus an external-dataset check and notebook viewers. Research use only, not for diagnosis; keep that disclaimer in user-facing text.

## Working with this user

- **Give pipeline commands, don't run them.** The user runs `hippo.train`, `hippo.evaluate`, `hippo.compare`, `hippo.external`, and `hippo.visualize` themselves. Print the exact command instead. Running the tests (`python -m pytest -q`) and quick scratch checks is fine.
- Don't download datasets; give the link so the user can check the data first.
- Don't commit, restore or reset git unless asked. The working tree usually has uncommitted work.
- Keep searches inside the project folder.
- Explain results in plain words, with numbers (Dice, p-values, case names), and say what was and wasn't verified.

## Environment

- macOS on Apple Silicon. Use the native venv, not Docker: Docker on macOS can't use the Apple GPU (Metal/MPS). `defaults.pick_device()` picks cuda, then mps, then cpu.
- Venv: `.venv` (Python 3.11, created with uv). Run things as `.venv/bin/python -m ...` from the project root with `PYTHONPATH=src`, or `source .venv/bin/activate && export PYTHONPATH=src`.
- `requirements.txt`: pinned training stack plus viewers (plotly, vedo) and Jupyter (`jupyterlab`, `ipywidgets`). `requirements-viewer.txt`: a light install for the archived `render3d.py` (vedo, no torch). Keep the two vedo versions the same.
- `anywidget` and `kaleido` are **not** installed. Without them plotly `FigureWidget` can't update in place and static plotly image export doesn't work.
- Docker (`Dockerfile`, `docker-compose.yml`) is for Linux/NVIDIA machines and CI. CI (`.github/workflows/tests.yml`) runs pytest on CPU.

## Pipeline and commands

```bash
python -m hippo.data                                    # download MSD Task04, write data/splits.json
python -m hippo.explore                                 # QC report -> data/explore/
python -m hippo.clean                                   # cleaned labels -> data/clean/ (after explore)
python -m hippo.train --config configs/baseline.yaml    # -> runs/baseline/  (--epochs 5 quick check, --resume)
python -m hippo.evaluate --run runs/baseline            # -> metrics_test.csv, summary_test.json
python -m hippo.compare runs/baseline runs/clean_labels # paired per-case comparison, Wilcoxon p
python -m hippo.external --src data/mri_dataset --run runs/baseline [--subjects s03]
python -m hippo.metrics_plot --runs runs/baseline runs/clean_labels runs/no_norm   # after evaluate + external
python -m hippo.visualize --case hippocampus_135 --run runs/baseline runs/clean_labels runs/no_norm --save-nifti
mlflow ui                                               # runs tracked in runs/mlflow.db
python -m pytest -q                                     # all tests, ~10 s
```

The dataset location comes from `--data-dir`, then `HIPPO_DATA_DIR`, then `hippo.conf` / `~/.hippo.conf`, then `defaults.DATA_DIR` (`data`). `train` saves it into the run folder so `evaluate` finds the same data.

## Files

### `src/hippo/`

| File | What it does |
|---|---|
| `defaults.py` | Default paths, the shared `configargparse` parser (`make_parser`), `pick_device()`, `find_case(name, data_dir)` |
| `data.py` | Download, 70/15/15 split (`splits.json`), `use_clean_labels` (train split only), MONAI transforms: `base_transforms(normalize)` = load, channel-first, RAS, optional z-score (`nonzero=True`, per scan), pad to a multiple of 8; `train_transforms` adds augmentation; `make_loaders` |
| `model.py` | `build_model(cfg)`: MONAI 3D U-Net from the `model:` config section |
| `train.py` | Training loop, validation Dice, atomic checkpoints (`best.pt`, `last.pt`), disk-space check, MLflow logging (incl. `labels`, `normalize`) |
| `evaluate.py` | Scores a run on a split with the run's own preprocessing; writes the per-case CSV and summary JSON |
| `metrics.py` | `case_metrics`: Dice and NSD per class for one case (MSD definitions) |
| `compare.py` | Paired comparison of two evaluated runs: mean difference, better/worse counts, Wilcoxon p |
| `metrics_plot.py` | Figure + table of each run's Dice and NSD on MSD test (mean of anterior/posterior per case) and external (whole hippocampus): `runs/figures/metrics_internal_external.png/.csv`. Interactive version: `notebooks/metrics_plot.ipynb` |
| `explore.py` | Dataset QC: measurements per case, FAIL/WARN flags, intensity groups, figures |
| `clean.py` | Removes small stray label pieces and excluded cases into `data/clean/`; raw data is never changed |
| `external.py` | Crops each hippocampus of the external whole-brain MNI dataset around the SPAM atlas centre (48×56×56 mm) and scores the whole hippocampus (pred > 0 vs label > 0) |
| `visualize.py` | Slice figure + 3D surfaces + volumes for a case; `--run` takes several runs; `--html`, `--montage`, `--save-nifti` |
| `slices.py` | 3D Slicer-style 2D views for the slice notebooks: `Scan.render()` draws axial/coronal/sagittal PNGs with overlays; `on_grid()` aligns masks by world position; `error_map()`; `load_ras()` is the shared NIfTI loader (RAS array, spacing, origin) |

### Other

- `configs/`: `baseline.yaml` (all settings, commented), `clean_labels.yaml` (`labels: clean`), `no_norm.yaml` (`normalize: none`). An experiment is a copy of the baseline with a different `out_dir` and one changed setting.
- `tests/`: one `test_<module>.py` per module. `synthetic.py` and `conftest.py` build a small fake MSD dataset so the whole pipeline runs in seconds.
- `notebooks/`: the slice viewers (see below) and `metrics_plot.ipynb` (internal vs external metrics figure and table).
- `archive/`: the true 3D viewers, not in use for now (`archive/README.md`).

## Data and output formats

- **MSD** `data/Task04_Hippocampus/`: `imagesTr/`, `labelsTr/`, `imagesTs/` (no labels), NIfTI `hippocampus_NNN.nii.gz`. They are 1 mm crops of about 35×50×35 voxels. Intensities fall in three scale groups (99th percentile about 100, 1,000 and 400,000), so z-score normalisation is required: `no_norm` fails on the 400,000 group.
- **`data/splits.json`**: `{"train": [...], "val": [...], "test": [...]}` with image/label paths.
- **External** `data/mri_dataset/`:
  - folders `s01/`–`s25/`, each with whole-brain MNI T1/T2 (`sNN_t1w_standard_defaced_MNI.nii.gz`, 1 mm; `hres` versions at 0.4 mm)
  - left and right label files (`sNN_hippolabels_t1w_standard_{L,R}_MNI.nii.gz`)
  - the `SPAM_<subfield>_<side>_MNI.nii.gz` atlas maps
  - labels are subfields, not anterior/posterior: 1 ≈ CA1-3, 2 ≈ subiculum, 3 ≈ CA4-DG. These names were inferred from the atlas and aren't documented (3 is the weakest match).
  - crops go to `data/external/imagesTr|labelsTr/sNN_{L,R}.nii.gz` plus `cases.json`, which lists image, label and coverage.
- **Run folder** `runs/<name>/`:
  - `config.yaml`, `best.pt`, `last.pt`, `history.json`
  - `metrics_<split>.csv`: `case, dice_anterior, nsd_anterior, vol_pred_ml_anterior, vol_true_ml_anterior`, then the same four for posterior
  - `summary_<split>.json`: `n_cases`, plus mean/median/std for each metric
  - `compare_<split>_vs_<other>.json`
  - `external/metrics_external.csv` (`case, dice, nsd, vol_pred_ml, vol_true_ml, label_coverage`) and `external/summary_external.json`
- **Predictions** are NIfTI uint8 masks saved with the affine of the loaded (RAS, padded) image, so they line up with the MRI by world position:
  - `hippo.visualize --save-nifti` writes them next to its PNG: `runs/figures/<case>_pred.nii.gz` for one run, `<case>_pred_<run>.nii.gz` for several. The user's comparison figures are in `runs/figures/compare/`, made with `--out`.
  - The notebooks save theirs to `runs/figures/notebook/<case>_pred_<run>.nii.gz`.

## Viewing images: preferences

The user prefers **3D Slicer-style slice views** (three separate planes, each with its own slider) for looking at scans and comparing predictions. Use these first:

1. **`notebooks/view_slices.ipynb`**: pick any NIfTI under `data/` or `runs/`, with an optional mask. It shows axial (red), coronal (green) and sagittal (yellow) panels with sliders, crosshair, contrast %, mask opacity and size. Rendering takes milliseconds, even for whole brains.
2. **`notebooks/compare_slices.ipynb`**: pick an image and runs. It predicts with each run, then shows one row per mask (ground truth, then each run) with one shared set of sliders. **Colour** switches between labels and errors: green correct, yellow wrong part, blue missed, red extra. Row titles give Dice. Crops only (MSD or external crops), no whole brains.

**Conventions** for these views (kept in `slices.py`):
- Radiological orientation as in Slicer: in axial and coronal views the patient's right is on the screen's left. Sagittal has anterior on the left. Every view has edge letters.
- Pixels are square in mm.
- Colours:
  - anterior = orangered, posterior = steelblue
  - external subfields: CA1-3 = gold, subiculum = mediumseagreen, CA4-DG = orchid

**True 3D is archived** in `archive/`, because the user isn't using it for now: `view3d_vedo.ipynb`, `view3d_file.ipynb` and `render3d.py` (see `archive/README.md`). Don't suggest or extend them unless the user asks. They still run: `PYTHONPATH=src python archive/render3d.py ...` (give the command, the user runs it). In the VS Code notebook backend, vedo's own inline output ignores the camera, so in-notebook 3D used plotly.

When editing notebooks, use NotebookEdit (Read first). `Edit` does not work on `.ipynb`.

## Code style

- Every module starts with a docstring that says what it does, its outputs and `Usage:` lines. Keep it current when behaviour changes.
- Short functions with one-line docstrings. Comments explain *why*, not what.
- Names are plain words (`normalize`, `use_clean_labels`, `on_grid`). Comments and notebook text use British spelling ("colour").
- Settings live in `configs/*.yaml` with a comment per key. Reading a new key uses `.get(key, default)`, so old run configs still load.
- Each run must be evaluated, visualised and predicted with **its own** preprocessing: read `normalize` from `<run>/config.yaml`.
- Add or update a test in `tests/test_<module>.py` for every change, and run `python -m pytest -q`.
- Notebook text is written for the user: what each control does, what the colours mean, and the "Research use only" line.

## Known issues and open decisions

- **Split leak (pending the user's decision):** MSD files `2k-1` and `2k` very likely come from the same subject (left/right). With the current split, 22 of 39 test cases have their partner in train, so the README's "no subject appears in two sets" is wrong. Proposed fix: split by subject `(id + 1) // 2`, then retrain. Don't change the split without the user agreeing.
- **clean_labels vs baseline:** no significant difference (all p > 0.5; cleaning changed 4 voxels in 1 case). Differences in better/worse counts are noise between training runs (1 seed).
- **External s03_L whole-hippocampus Dice:** baseline 0.796, clean_labels 0.817, no_norm 0.669. Volumes there are in MNI space, not the subject's own head size.
