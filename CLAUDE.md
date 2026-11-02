# CLAUDE.md

Hippocampus segmentation and volumetry on 3D T1 MRI: a MONAI 3D U-Net trained on MSD Task04 (labels 0 = background, 1 = anterior, 2 = posterior), evaluated with Dice and NSD (1 mm), plus an external-dataset check and a Streamlit viewer page. Research use only, not for diagnosis; keep that disclaimer in user-facing text.

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
streamlit run streamlit_dashboard/compare_slices.py                    # compare_slices as a web page (PYTHONPATH=src)
python streamlit_dashboard/make_preview.py              # README preview image -> docs/figures/streamlit_preview.png
mlflow ui                                               # runs tracked in runs/mlflow.db
python -m pytest -q                                     # all tests, ~10 s
```

The dataset location comes from `--data-dir`, then `HIPPO_DATA_DIR`, then `hippo.conf` / `~/.hippo.conf`, then `defaults.DATA_DIR` (`data`). `train` saves it into the run folder so `evaluate` finds the same data.

## Files

### `src/hippo/`

| File | What it does |
|---|---|
| `defaults.py` | Default paths, the shared `configargparse` parser (`make_parser`), `pick_device()`, `find_case(name, data_dir)` |
| `data.py` | Download, 70/15/15 split (`splits.json`), `use_clean_labels` (train: cleaned labels, excluded cases dropped; val: excluded cases dropped, original labels; test untouched), MONAI transforms: `base_transforms(normalize)` = load, channel-first, RAS, optional z-score (`nonzero=True`, per scan), pad to a multiple of 8; `train_transforms` adds augmentation; `make_loaders` |
| `model.py` | `build_model(cfg)`: MONAI 3D U-Net from the `model:` config section |
| `train.py` | Training loop, validation Dice, atomic checkpoints (`best.pt`, `last.pt`), disk-space check, MLflow logging (incl. `labels`, `normalize`) |
| `evaluate.py` | Scores a run on a split with the run's own preprocessing; writes the per-case CSV and summary JSON. Every case is predicted; a case whose ground truth lacks a part (`metrics.missing_parts`) gets blank Dice/NSD and is listed under `dropped` instead of scored |
| `metrics.py` | `case_metrics`: Dice and NSD per class for one case (MSD definitions) |
| `compare.py` | Paired comparison of two evaluated runs: mean difference, better/worse counts, Wilcoxon p; leaves out (and counts) cases blank in either run |
| `metrics_plot.py` | Figure + table of each run's Dice and NSD on MSD test (mean of anterior/posterior per case), external (whole hippocampus) shifted MSD test (`archive/shift.py`; each case averaged over its 12 shifted copies, control left out; green squares) and shifted external (yellow triangles): `runs/figures/metrics_internal_external.png/.csv`. Only groups that exist are drawn; `--datasets internal external` keeps only those (the README figure `docs/figures/baseline_msd_vs_external.png`, made with `--runs runs/baseline --names "3D U-Net" --datasets internal external external_largest --out docs/figures/baseline_msd_vs_external`; group `external_largest` (dark orange triangles) reads `<run>/external_largest/` from `hippo.external --largest-piece`; `--names` labels runs in the figure/table; user-facing text says "3D U-Net", not "baseline"; `docs/` is tracked, `runs/` is git-ignored) |
| `explore.py` | Dataset QC: measurements per case, FAIL/WARN flags, intensity groups, figures |
| `clean.py` | Removes small stray label pieces and excluded cases into `data/clean/`; raw data is never changed |
| `external.py` | Crops each hippocampus of the external whole-brain MNI dataset around the SPAM atlas centre (48×56×56 mm) and scores the whole hippocampus (pred > 0 vs label > 0). Opt-in `--largest-piece` (`largest_piece()`, 26-neighbour) keeps each prediction's largest connected piece and writes to `<run>/external_largest/` (extra columns `pieces`, `removed_ml`); the default results in `<run>/external/` are unchanged. Opt-in `--recentre` (`recentre()`): test-time re-centring, no labels used: predict on the atlas box, keep the largest piece, centre a same-sized box on its world centroid (`prepare(centres=...)`, to the nearest voxel), predict again and score only that; crops in `data/external_recentred/{first_pass,recentred}/`, results in `<run>/external_recentred[_largest]/` with extra columns `recentred`, `moved_mm`, `moved_x/y/z_mm`. Cases are now also given `centre_mm` (box centre, world mm). Opt-in `--msd-crop` (`msd_crop()`): crop like MSD: the first prediction's bounding box (largest piece) plus `msd_margins()` = median margin of the run's MSD training crops per axis (baseline: 6.5, 5.5, 5.0 mm), via `prepare(bounds=...)` / `world_box()`; box size varies per case; crops in `data/external_msd_crop/{first_pass,msd_crop}/`, results in `<run>/external_msd_crop[_largest]/` (adds `crop_x/y/z_mm`) |
| `visualize.py` | Slice figure + 3D surfaces + volumes for a case; `--run` takes several runs; `--html`, `--montage`, `--save-nifti` |
| `slices.py` | 3D Slicer-style 2D views for the Streamlit page: `Scan.render()` draws axial/coronal/sagittal PNGs with overlays; `on_grid()` aligns masks by world position; `error_map()`; `load_ras()` is the shared NIfTI loader (RAS array, spacing, origin) |

### Other

- `configs/`: `baseline.yaml` (all settings, commented), `clean_labels.yaml` (`labels: clean`), `no_norm.yaml` (`normalize: none`). An experiment is a copy of the baseline with a different `out_dir` and one changed setting.
- `tests/`: one `test_<module>.py` per module. `synthetic.py` and `conftest.py` build a small fake MSD dataset so the whole pipeline runs in seconds.
- `streamlit_dashboard/`: `compare_slices.py` (the viewer page, see below) and `make_preview.py` (README preview image `docs/figures/streamlit_preview.png`, drawn with the page's own slice code, not a screenshot).
- **No Jupyter notebooks (deleted 2026-11-02, user's choice: covered by the Streamlit page).** `notebooks/` (`view_slices`, `compare_slices`, `metrics_plot`) and `archive/notebooks/` (`view3d_*`) are gone; git history has them. Don't add notebooks back unless asked. `jupyterlab`/`ipywidgets` are still in `requirements.txt` (unused).
- `archive/`: `render3d.py` (native vedo 3D window), not in use for now (`archive/README.md`). Also, archived 2026-10-28 for the planned full-image training branch (next branch): `archive/shift.py` (was `hippo.shift`: shifted MSD crops / shifted external atlas box, `--axes`, `--conditions x-4_z-10`, `--largest-piece`; run as `PYTHONPATH=src python archive/shift.py ...`), `archive/tests/test_shift.py` (still collected by pytest), and `archive/external_analysis/*.py` (one-off external analyses: centre/edge offsets, columns, held-out shift calibration, label overlays, subiculum check; outputs to `runs/analysis/external/`). `data/shifted/` and `data/shifted_external/` were deleted; `runs/<run>/shifted*/` metrics are kept. Don't bring the shift test back into `src/hippo/` unless the user asks.
- `docs/external_validation.md`: first-iteration external report (MNI-HISUB25): main result is the plain atlas box, no shift or calibration (Dice 0.811 / NSD 0.777 vs MSD test whole hippocampus 0.897 / 0.959); largest piece and box moves are secondary/diagnostic; limitations. Decided 2026-10-28: don't shift predictions (held-out test: Dice 0.840 → 0.794). Keep it in sync if external numbers change.

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
  - `metrics_<split>.csv`: `case, dice_anterior, nsd_anterior, vol_pred_ml_anterior, vol_true_ml_anterior`, then the same four for posterior. Dice/NSD are blank for a case that couldn't be scored (part missing from the ground truth); `compare` and `metrics_plot` skip those rows
  - `summary_<split>.json`: `n_cases` (scored), `n_dropped`, `dropped` (`{case: reason}`), plus mean/median/std for each metric over the scored cases
  - `compare_<split>_vs_<other>.json`
  - `external/metrics_external.csv` (`case, dice, nsd, vol_pred_ml, vol_true_ml, label_coverage`) and `external/summary_external.json`
- **Predictions** are NIfTI uint8 masks saved with the affine of the loaded (RAS, padded) image, so they line up with the MRI by world position:
  - `hippo.visualize --save-nifti` writes them next to its PNG: `runs/figures/<case>_pred.nii.gz` for one run, `<case>_pred_<run>.nii.gz` for several. The user's comparison figures are in `runs/figures/compare/`, made with `--out`.
  - The Streamlit page saves its predictions to `runs/figures/notebook/<case>_pred_<run>.nii.gz` (folder name kept from the old notebooks), and `<case>_shifted_<condition>_pred_<run>.nii.gz` for `data/shifted/` images (`<case>_shifted_external_<condition>_...` for `data/shifted_external/`) (it lists them alongside MSD and external crops).

## Viewing images: preferences

The user prefers **3D Slicer-style slice views** (three separate planes, each with its own slider) for looking at scans and comparing predictions, in the **Streamlit page** (the only viewer; the notebooks were deleted). Whole-brain scans aren't listed there: point the user to ITK-SNAP or 3D Slicer for those.

- **`streamlit_dashboard/compare_slices.py`** (axial red, coronal green, sagittal yellow panels; crosshair, contrast %, mask opacity and size; plane sliders in the sidebar; predictions cached; a "3D surfaces" checkbox adds one rotatable plotly view per mask below the slices, made with `slices.surfaces()` (marching cubes, world mm), ground truth as a grey ghost, Errors mode with "correct" see-through; hover shows world mm and the plane-slider values (0-based voxel index) of that point; added 2026-10-27 at the user's request, separate from the archived vedo viewers; a "Second reference" selector (2026-10-30) adds a row for another label file, picked automatically from `data/ashs/labelsTr/<case>` (ASHS anterior/posterior), scored vs the ground truth, and each run's title adds Dice vs it; Errors still compare with the ground truth; `PYTHONPATH=src .venv/bin/streamlit run streamlit_dashboard/compare_slices.py`, test in `tests/test_compare_slices_app.py`): pick an image and runs. It predicts with each run, then shows one row per mask (ground truth, then each run) with one shared set of sliders. **Colour** switches between labels and errors: green correct, yellow wrong part, blue missed, red extra. Row titles give Dice. Crops only (MSD or external crops), no whole brains.

**Conventions** for these views (kept in `slices.py`):
- Radiological orientation as in Slicer: in axial and coronal views the patient's right is on the screen's left. Sagittal has anterior on the left. Every view has edge letters.
- Pixels are square in mm.
- Colours:
  - anterior = orangered, posterior = steelblue
  - external subfields: CA1-3 = gold, subiculum = mediumseagreen, CA4-DG = orchid

**The native vedo 3D window is archived** in `archive/render3d.py` (see `archive/README.md`), because the user isn't using it for now. Don't suggest or extend it unless the user asks. It still runs: `PYTHONPATH=src python archive/render3d.py ...` (give the command, the user runs it).

## Code style

- Every module starts with a docstring that says what it does, its outputs and `Usage:` lines. Keep it current when behaviour changes.
- Short functions with one-line docstrings. Comments explain *why*, not what.
- Names are plain words (`normalize`, `use_clean_labels`, `on_grid`). Comments and page text use British spelling ("colour").
- Settings live in `configs/*.yaml` with a comment per key. Reading a new key uses `.get(key, default)`, so old run configs still load.
- Each run must be evaluated, visualised and predicted with **its own** preprocessing: read `normalize` from `<run>/config.yaml`.
- Add or update a test in `tests/test_<module>.py` for every change, and run `python -m pytest -q`.
- Streamlit page text is written for the user: what each control does, what the colours mean, and the "Research use only" line.

## Known issues and open decisions

- **Possible split leak (decided 2026-10-26: keep as a documented limitation):** MSD files `2k-1` and `2k` may be the same subject's left/right side, but MSD doesn't document this. With the current file-level split, 22 of 39 test cases have their possible partner in train. The README's Limitations section and the `make_splits` docstring say so. Keep the file-level split; don't switch to a subject split `(id + 1) // 2` or retrain unless the user asks.
- **clean_labels vs baseline:** no significant difference (all p > 0.5; cleaning changed 4 voxels in 1 case). Differences in better/worse counts are noise between training runs (1 seed).
- **Crop position test (`hippo.shift`, now `archive/shift.py`, 2026-10-27, 39 MSD test cases):** adding a 10-voxel zero border alone (control "none") lowers mean Dice from 0.874 to 0.833 (baseline; 35/39 worse, p < 0.001; clean_labels 0.875 → 0.836). Shifting inside the padded canvas changes it by -0.035 (y-5) to +0.015 (y+10); not monotonic with distance (y-5 worse than y-10), and moving the content anteriorly (less zero border in front) helps. So the model is more sensitive to unfamiliar surroundings (zero border, which is the mean grey after z-scoring) than to position as such. Training has random rotate/scale (`RandAffined` scale ±10%) but no translation. The test can't avoid an artificial border on MSD crops; a cleaner position test would move the external atlas box (real tissue around it).
- **Shifted data:** `data/shifted/` and `data/shifted_external/` are both deleted (no shift in external datasets, user's choice 2026-10-28). The metrics in `runs/<run>/shifted*/` were kept, so `metrics_plot`'s shifted groups still work; the crops can be rebuilt with `archive/shift.py`.
- **MSD has no metadata (2026-11-01):** `dataset.json` gives only name, "Left and right hippocampus segmentation", Vanderbilt, licence, labels and file lists; NIfTI origins are reset to 0/1 in all 390 files. No subject ID, side, age, sex, diagnosis or scanner. 390 = 195 x 2 fits one left + one right per person (195 from memory of the MSD paper, not checked). Listed in the README and report Limitations.
- **Wording (2026-11-01):** user-facing text says "3D U-Net", not "baseline" (paths like `runs/baseline` stay). The external data is "the MNI-based external dataset (MNI-HISUB25, downloaded from NITRC)". The README no longer mentions `archive/`.
- **Next steps (decided 2026-11-02, README "Next steps"):** ASHS check done (external labels correct; the gap comes from MSD's training labels, not the model's capacity). Second external dataset: candidates (EADC-ADNI HarP needs an ADNI application; Henry Ford on NITRC) still to be checked for access. Fine-tuning on external labels deferred (uses up the only external test set, builds on incomplete MSD labels); don't propose it again unless asked. Next branch: full-image training on a new dataset with whole-head scans and whole-hippocampus labels, patch-based for the GPU limit. MSD has only crops, so full images means a new dataset.
- **External box shift test (`hippo.shift --external`, now archived, 2026-10-27, 50 hippocampi, real tissue all round; all 600 boxes verified moved by the full shift):** Dice depends strongly on the box's vertical placement. Baseline (control 0.811): z-5 +0.031 (47/50 better), z-10 +0.037; z+5 -0.030 (47/50 worse), z+10 -0.086 (all 50 worse). x+10 -0.050, y-10 -0.020. clean_labels follows the same pattern; no_norm is noisier. z- puts the hippocampus lower in the box, i.e. less tissue below it, where the over-segmented islands were; the volume ratio drops 1.09 -> 1.05 at z-10. ±10 mm boxes cut off part of some hippocampi (coverage down to 0.918 at z+10, 43/50 below 0.99), which confounds those rows. Moving the default box up would be tuned on the same external data, so not done.
- **Crop placement differs between MSD and external (2026-10-28):** MSD crops are centred on the hippocampus's bounding box (bounding-box centre within 1 mm of the crop centre in 93% of cases along z; margins about 5-6 mm, equal on opposite sides), while the external atlas box is centred on the centre of mass, which lies 2-3 mm below the bounding-box centre (the tail rises). So the external box puts the hippocampus ~2.5-3 mm too high (margin below 12 mm vs 5-6 above) and is larger (48x56x56 vs ~35x50x36 mm); the x difference is mirrored per side. This explains why z-5 helped and why one x shift can't help both sides. `--msd-crop` reproduces MSD's cropping without external labels; `--recentre` centres on the centre of mass, so it does not fix this.
- **Prediction vs label offsets are labelling differences, not a shifted label (2026-10-28):** per hemisphere the prediction's centre is ~1 mm lateral, 1.2-1.6 mm anterior and 1.4-1.8 mm lower than the label's, the same in both halves of the subjects; but the edges don't move together (medial edge 3 mm short, front edge 4 mm long), MSD test shows no offset (< 0.05 mm), 1 mm and 0.4 mm labels agree within 0.2 mm, and the label follows the T1/T2 anatomy on s03 and s17 (visual check). The external labels (parts 1+2+3: one connected piece in all 50, small holes 0.6%) include subiculum, uncus and a medial tail strip that the MSD-trained model leaves out. Don't shift predictions to fit the labels.
- **ASHS-PMC-T1 check (2026-10-30, ITK-SNAP DSS cloud, s03/s17 whole-brain T1 only):** ASHS centre within 0.8 mm of the manual external label, model 1.8-3.3 mm off (always outwards, forwards, lower) -> the labels are right, the model draws MSD's narrower hippocampus. Model vs ASHS Dice anterior 0.80, posterior 0.74; boundary 2-6 mm forward. Whole vs manual: ASHS 0.840, model largest piece 0.841, plain 0.808. ASHS can't run on MSD crops (135 failed, 017 empty). Results in `data/ashs/`, scripts `archive/external_analysis/ashs_{compare,labels}.py`; Streamlit shows them as "Second reference". `itksnap-wt` must be the fixed copy `~/itksnap-tools/bin/itksnap-wt`. Ask before uploading any more data. README "Conclusion (first iteration)" states: MSD chosen for GPU limits, labels don't cover the whole hippocampus, next branch uses a different dataset (whole-head, whole-hippocampus labels, patch-based training).
- **External over-segmentation (2026-10-27):** predicted/true volume median 1.09 on external (44/50 too big) vs 1.02 on MSD test, for baseline. Two parts: islands away from the hippocampus (s03_L: 6 pieces, 47% of extra voxels > 3 mm away, mostly below), likely because the external crop is larger than MSD's; and a ~1 voxel wider outline (s03_R: 76% of extra within 1 mm), possibly a different labelling protocol. Voxel size is the same (1 mm, RAS) in both. All 50 external hippocampi, mean whole-hippocampus Dice: baseline 0.811, clean_labels 0.813 (vs baseline: 25 better, 25 worse, p 0.31), no_norm 0.702; only 1/50 baseline predictions is a single piece. Largest-piece post-processing was tried (baseline Dice → 0.840) and removed on 2026-10-27 (the user judged it not valid then), then re-added later on 2026-10-27 at the user's request as an **opt-in** option only: `hippo.external --largest-piece` (own output folder) and the Streamlit "Largest piece only" checkbox. Keep it off by default. Volumes there are in MNI space, not the subject's own head size.
