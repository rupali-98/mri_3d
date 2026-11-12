# Usage: set up, train, evaluate, track

Research use only, not for diagnosis. Back to the [README](../README.md).

## 1. Set up (once)

```bash
git clone https://github.com/rupali-98/mri_3d.git && cd mri_3d
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt                        # on NVIDIA GPUs, install the CUDA build of PyTorch first
export PYTHONPATH=src                                  # Windows PowerShell: $env:PYTHONPATH="src"
python -m hippo.data --folds 5                         # downloads MSD Task04 (~30 MB), writes data/splits.json and data/folds/
```

By default data goes in `./data`. To keep it elsewhere, use the first of: `--data-dir /path` on each command, the `HIPPO_DATA_DIR` environment variable, or `data-dir = /path` in `hippo.conf` (copy `hippo.conf.example`). The external dataset MNI-HISUB25 is downloaded by hand from [NITRC](https://www.nitrc.org/projects/mni-hisub25/) into `data/mri_dataset/`.

## 2. Get a trained model

The trained weights are **not in the repository** (`runs/` is git-ignored). A trained model is a run folder holding at least `config.yaml` (the settings it was trained with, including its preprocessing) and `best.pt` (the weights with the best validation Dice). Copy that folder from whoever trained it, or train it yourself (section 5).

## 3. Evaluate

```bash
python -m hippo.evaluate --run runs/baseline                                           # MSD test split
python -m hippo.external --src data/mri_dataset --run runs/baseline                   # main result -> runs/baseline/external/
python -m hippo.external --src data/mri_dataset --run runs/baseline --largest-piece   # secondary  -> runs/baseline/external_largest/
python -m hippo.external --src data/mri_dataset --run runs/baseline --msd-crop --largest-piece   # MSD-style box
python -m hippo.errors --run runs/baseline                                             # worst 10 per test set, by error type
```

`evaluate` writes `metrics_test.csv` (one row per case: Dice, NSD, HD95 and volumes per part and for the whole hippocampus) and `summary_test.json` (mean, median and SD per metric, plus volume agreement per label). A case whose label lacks a part can't be scored; it is listed under `dropped`. Each evaluation is also logged to MLflow (section 6).

**Figures:**

```bash
python -m hippo.metrics_plot --runs runs/baseline --names "3D U-Net" --datasets internal_anterior internal_posterior --out docs/figures/msd_test_parts
python -m hippo.metrics_plot --runs runs/baseline --names "3D U-Net" --datasets internal external external_largest --out docs/figures/baseline_msd_vs_external
python -m hippo.visualize --case hippocampus_135 --run runs/baseline --save-nifti     # slice figure + prediction as NIfTI
```

## 4. Use it on one scan from Python

```python
from pathlib import Path
import numpy as np, yaml
from hippo import visualize
from hippo.train import spacing_from_affine

run = Path("runs/baseline")
cfg = yaml.safe_load((run / "config.yaml").read_text())
image = Path("data/Task04_Hippocampus/imagesTr/hippocampus_135.nii.gz")      # any 1 mm T1 hippocampus crop

d = visualize.load(image, None, cfg["data"].get("normalize", "zscore"))     # same preprocessing as training
mask = visualize.predict(run, d["image"])                                    # 0 background, 1 anterior, 2 posterior
voxel_ml = np.prod(spacing_from_affine(d["image"].affine)) / 1000
print("anterior", (mask == 1).sum() * voxel_ml, "ml, posterior", (mask == 2).sum() * voxel_ml, "ml")
```

For hippocampus_135 this prints about 1.37 ml anterior and 1.35 ml posterior. Always read `normalize` from the run's own `config.yaml`, so each model gets the preprocessing it was trained with.

## 5. Train

```bash
python -m hippo.train --config configs/baseline.yaml --epochs 5   # optional quick check that everything runs
python -m hippo.train --config configs/baseline.yaml              # one full run (150 epochs) -> runs/baseline/
python -m hippo.train --config configs/baseline.yaml --resume     # continue an interrupted run
python -m hippo.train --config configs/baseline.yaml --fold 0     # one cross-validation fold -> runs/baseline/fold0/
```

**Cross-validation of one experiment** (all folds, then the evaluations and tables; resumable, Linux or macOS):

```bash
scripts/run_cv.sh                                          # configs/baseline.yaml, named "3D U-Net"
scripts/run_cv.sh configs/random_trim.yaml "Random position"
DRY_RUN=1 scripts/run_cv.sh                                # print the commands only
```

Then `python -m hippo.results --experiment runs/baseline --name "3D U-Net"` writes the experiment's table (mean ± SD over the folds of Dice, HD95, NSD, volume error and ICC per test set and label) and its Bland–Altman plot to `runs/baseline/results/`.

**What training saves** (in the config's `out_dir`, or `<out_dir>/fold<k>/`):

| File | What it is |
|---|---|
| `config.yaml` | The exact settings used, including the data location, fold and preprocessing |
| `best.pt` | Weights with the best mean validation Dice: **the model that `evaluate`, `external`, `visualize` and the viewer load** |
| `last.pt` | Weights after the final epoch |
| `history.json` | Training loss per epoch and validation Dice/NSD every 5 epochs |
| `mlflow_run_id.txt` | The run's MLflow id, so evaluations are added to the same MLflow run |
| `checkpoint.pt` | Full training state while running (for `--resume`); removed when the run finishes |

**A new experiment:** copy `configs/baseline.yaml` to e.g. `configs/my_run.yaml`, change `out_dir` (e.g. `runs/my_run`) and the one setting you are testing, then train with `--config configs/my_run.yaml` (or `scripts/run_cv.sh configs/my_run.yaml`). Compare two single runs case by case with `python -m hippo.compare runs/baseline runs/my_run` (paired Wilcoxon). Optional data checks and label cleaning: `python -m hippo.explore`, then `python -m hippo.clean` and `labels: clean` in the config.

## 6. MLflow

```bash
mlflow ui --backend-store-uri sqlite:///runs/mlflow.db --port 5001   # then open http://127.0.0.1:5001
python -m hippo.results --ablation                                   # one row per experiment -> runs/ablation.md
```

Each run folder (each fold) is one MLflow run in the experiment `hippo`, holding:
- the config as parameters, including `experiment` and `fold`;
- the training curves (loss and validation Dice per epoch) and the run folder as artifacts;
- every evaluation, named by where its results were written (`test.dice_whole`, `external.dice`, `external_largest.dice`, `external_msd_crop_largest.icc`, …) and tagged with its post-processing and box type.

Port 5000, MLflow's default, is used by macOS AirPlay; without `--backend-store-uri` the page is empty. `HIPPO_MLFLOW=0` turns logging off.

## Serve the model (API)

Package the model and start the FastAPI service locally; deployment to Google Cloud Run is in [deployment](deployment.md).

```bash
python -m hippo.package_model --experiment runs/baseline --src data/mri_dataset --single   # -> deploy/model/
PYTHONPATH=src uvicorn hippo.api:app --port 8000                                          # http://127.0.0.1:8000/docs
```

## 7. Tests

```bash
python -m pytest -q
```

The tests check the metrics on hand-made cases (ICC against the Shrout & Fleiss example, HD95 on a known shift) and run the whole pipeline (split, folds, train, evaluate, MLflow logging, the external crops, the results tables, the error analysis and the Streamlit page) on a small synthetic dataset with the same folder layout as MSD, in about 30 seconds. They also run on CPU in CI.

## Project structure

```
configs/                    experiment settings (baseline, clean_labels, no_norm, random_trim)
scripts/run_cv.sh           cross-validation of one experiment, end to end
src/hippo/defaults.py       default paths, shared command-line options, device choice
src/hippo/data.py           download, splits and folds, preprocessing and augmentation, loaders
src/hippo/model.py          3D U-Net (MONAI)
src/hippo/train.py          training loop, checkpoints, --fold
src/hippo/tracking.py       MLflow logging shared by training and evaluation
src/hippo/evaluate.py       MSD test metrics per case and summary
src/hippo/metrics.py        Dice, NSD, HD95, volumes, ICC, volume agreement
src/hippo/external.py       external dataset: atlas-centred and MSD-style crops, whole-hippocampus scores
src/hippo/results.py        table per experiment, Bland-Altman plot, ablation table from MLflow
src/hippo/errors.py         error analysis of the worst cases
src/hippo/api.py            FastAPI service: /segment/crop, /segment/mni
src/hippo/package_model.py  model package for the API (deploy/model/)
src/hippo/predict.py        predict MSD crops without labels (masks on the input grid, label-free checks)
compose.yaml, frontend/, backend/, model-service/   one Dockerfile per service, see docker.md
deploy/                     Cloud Build config and the packaged model
src/hippo/compare.py        paired comparison of two runs
src/hippo/metrics_plot.py   Dice/NSD figures per run and dataset, position curve
src/hippo/visualize.py      slice figure, 3D surfaces and volumes for one case
src/hippo/explore.py, clean.py   data quality checks and label cleaning
streamlit_dashboard/        the viewer (compare_slices.py) and its preview images (make_preview.py)
docs/                       methods, results, limitations, usage, viewer, risk, reports and README figures
tests/                      unit and end-to-end tests
```
