# Hippocampus Volumetry on 3D MRI

A research pipeline that segments the hippocampus (anterior and posterior parts) on 3D T1 MRI, measures its volume, and reports how much the model can be trusted on each scan.

> **Intended use:** research and education only. This software is not a medical device, has not been clinically validated, and must not be used for diagnosis or any clinical decision.

## Status

| Step | What | Status |
|---|---|---|
| 1 | Baseline 3D U-Net on MSD Task04, evaluated with Dice and NSD (1 mm), as in the [Medical Segmentation Decathlon](https://www.nature.com/articles/s41467-022-30695-9) | Done |
| 2 | Uncertainty: MC dropout and a small ensemble; flag low-confidence cases | Next |
| 3 | DICOM ingest: de-identify, convert to NIfTI, automatic image QC | Planned |
| 4 | Cross-site check on IXI (three hospitals): volume distributions per site and vs. age | Planned |
| 5 | FastAPI service in Docker, CI | Planned |
| 6 | Model card, risk table | Planned |

## Data

[MSD Task04 Hippocampus](http://medicaldecathlon.com/): 3D T1 MPRAGE crops around the hippocampus from Vanderbilt University Medical Center, labels 0 = background, 1 = anterior, 2 = posterior. License CC-BY-SA 4.0.

MSD's own test set has no public labels, so the labelled cases are split 70/15/15 into train/validation/test (one file per subject, so no subject appears in two sets).

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# Optional on NVIDIA GPUs: install the CUDA build of PyTorch from pytorch.org first.

export PYTHONPATH=src                                  # Windows PowerShell: $env:PYTHONPATH="src"
python -m hippo.data                                   # downloads ~30 MB and writes data/splits.json
python -m hippo.train --config configs/baseline.yaml   # add --epochs 5 for a quick check
python -m hippo.evaluate --run runs/baseline
mlflow ui                                              # browse runs at http://127.0.0.1:5000
```

### Dataset location

By default the data goes in `./data`. To keep it elsewhere, set the location once instead of passing it to every command. All scripts use the first of these that is set:

1. `--data-dir /path/to/data` on the command line
2. the `HIPPO_DATA_DIR` environment variable
3. `data-dir = /path/to/data` in `hippo.conf` (project root) or `~/.hippo.conf`; copy `hippo.conf.example` to start
4. the default in `src/hippo/defaults.py`

`train` saves the location it used into the run folder, so `evaluate` finds the same data without being told.

Outputs in `runs/baseline/`: `best.pt`, `history.json`, `metrics_test.csv` (one row per case: Dice, NSD, predicted and true volume in ml) and `summary_test.json` (mean, median, std).

Hardware: the crops are about 35×50×35 voxels, so training fits comfortably on a 4 GB GPU and also runs on CPU.

## Tests

```bash
python -m pytest -q
```

The tests check the metrics on hand-made cases and run the whole pipeline (split → train → evaluate) on a small synthetic dataset with the same folder layout as MSD.

## Design notes

- **Metrics match the challenge:** Dice and Normalized Surface Dice with the 1 mm tolerance clinicians chose for this task, so results are comparable with published MSD numbers.
- **Augmentation:** random affine, intensity changes and noise. Flips are left–right only, because flipping front-to-back would swap the anterior and posterior labels.
- **Dropout is kept in the model** so that step 2 can use it for MC-dropout uncertainty without retraining a different architecture.

## Structure

```
configs/baseline.yaml   experiment settings
src/hippo/defaults.py   default paths, shared command-line options
src/hippo/data.py       download, splits, transforms, loaders
src/hippo/model.py      3D U-Net (MONAI)
src/hippo/metrics.py    Dice, NSD, volumes
src/hippo/train.py      training loop, MLflow logging
src/hippo/evaluate.py   per-case test metrics and summary
tests/                  unit and end-to-end tests
```
