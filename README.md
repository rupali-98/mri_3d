# Hippocampus Segmentation and Volumetry on 3D MRI

A 3D U-Net that outlines the hippocampus on T1-weighted MRI, splits it into an anterior and a posterior part, and measures each part's volume. Trained on the Medical Segmentation Decathlon (MSD), tested on a held-out MSD split and on an independent external dataset, tracked in MLflow, and explorable in a Streamlit viewer.

> **Intended use:** research and education only. This software is not a medical device, has not been clinically validated, and must not be used for diagnosis or any clinical decision. See [intended use and risk](docs/intended_use_and_risk.md).

**Details:** [methods](docs/methods.md) · [results and experiments](docs/results.md) · [limitations](docs/limitations.md) · [usage](docs/usage.md) · [Streamlit viewer](docs/streamlit.md) · [external validation report](docs/external_validation.md) · [retraining experiment](docs/retraining_plan.md) · [intended use and risk](docs/intended_use_and_risk.md) · [deployment](docs/deployment.md)

## At a glance

- **Problem:** hippocampal volume, and the balance between its head and its body and tail, is studied in ageing and neurodegeneration. Outlining it by hand is slow and varies between raters.
- **Model:** MONAI 3D U-Net with a Dice + cross-entropy loss, on 1 mm T1 crops around one hippocampus.
- **Main result:** whole-hippocampus Dice **0.897** on 39 held-out MSD cases, **0.811** on 50 external hippocampi (MNI-HISUB25).
- **Key finding:** most of the external gap comes from what the training labels contain (MSD leaves out the inner edge of the hippocampus), not from the model or the cropping. An independent tool (ASHS) confirmed this, and a retraining experiment ruled out cropping as the cause.
- **Next:** a dataset with whole-hippocampus labels on whole-head scans, with patch-based training.

## Why this model

Hippocampal volumetry is a common measurement in research on memory, ageing, epilepsy and dementia, and the anterior and posterior parts are thought to change differently. The goal of this work is to build that measurement end to end the way a medical-imaging team would: a reproducible pipeline with fixed splits, metrics that match a public benchmark, an external test on data the model never saw, honest error analysis, experiment tracking, and a viewer for checking every prediction. The model outlines a structure; it does not diagnose anything.

## Data and split

| Dataset | Role | Contents |
|---|---|---|
| [MSD Task04 Hippocampus](http://medicaldecathlon.com/) (CC-BY-SA 4.0) | Train, validation, test | 260 labelled T1 crops (about 35 × 50 × 35 voxels at 1 mm) from Vanderbilt; labels anterior and posterior |
| [MNI-HISUB25](https://www.nitrc.org/projects/mni-hisub25/) (CC BY-NC-SA 3.0) | External test only | Whole-brain 3 T T1 of 25 healthy adults; manual subfield labels of both hippocampi (50 hippocampi) |
| ASHS-PMC-T1 | Independent second opinion | Automatic anterior/posterior labels for 2 external subjects (4 hippocampi) |

- **Why MSD:** its small crops train on a small GPU, and it has anterior/posterior labels. The cost: its labels leave out the thin inner part of the hippocampus.
- **Split:** by file, 182 train / 39 validation / 39 test (seed 42). The 39 test cases stay fixed and are never used for training or choosing weights.
- **Cross-validation:** the 221 train + validation cases are split into 5 fixed folds. Each experiment is reported as the mean ± SD of its 5 fold models.
- **External data is test-only:** each hippocampus is cut out with a box placed by the dataset's atlas, never by its own label. Only the whole hippocampus can be scored there, because its labels are subfields, not parts.

More: [methods](docs/methods.md#datasets).

## Training

- **Model:** 3D U-Net with 4 levels (16–128 channels), residual units and dropout 0.1. Loss Dice + cross-entropy; AdamW (learning rate 0.001) with cosine decay; 150 epochs, batch size 8, seed 42.
- **Preprocessing:** RAS orientation, z-score per scan (needed: MSD intensities come in three scales up to 400,000), and padding to a multiple of 8.
- **Augmentation:** random rotation (±11°) and scaling (±10%), left–right flip, brightness changes and noise. No front–back flip, because that would swap the anterior and posterior labels.
- **Tuning:** no hyperparameter search. Settings are fixed in `configs/baseline.yaml`, and each experiment changes one setting. The best epoch is picked by mean anterior/posterior validation Dice.
- **Hardware:** a 4 GB GPU, Apple Silicon (about 13 s per epoch) or CPU.

More: [methods](docs/methods.md#training).

## Results

> **The model performs best on in-distribution data.** Part of the external drop comes from the two datasets labelling the hippocampus differently, not only from model error.

| Test set | Scored as | Dice | NSD (1 mm) | Cases |
|---|---|---|---|---|
| MSD test | anterior / posterior | 0.880 / 0.868 | 0.931 / 0.942 | 39 |
| MSD test | whole hippocampus | **0.897** | **0.959** | 39 |
| MNI-HISUB25 (external) | whole hippocampus | **0.811** | **0.777** | 50 |
| MNI-HISUB25, largest piece kept (secondary) | whole hippocampus | 0.840 | 0.825 | 50 |

![3D U-Net Dice and NSD on MSD test and on the external dataset](docs/figures/baseline_msd_vs_external.png)

*One dot per case; the bar and the number are the mean. Compare the external groups with the MSD whole hippocampus (0.897): the figure's MSD group is the mean of the two parts.*

**Error analysis** (the worst 10 cases per test set, sorted by their biggest error type):

| Error type | External (worst 10) | MSD test (worst 10) |
|---|---:|---:|
| Outline too wide (over-segmentation of the main piece) | 6 | 7 |
| False-positive islands | 2 | 0 |
| Under-segmentation | 2 | 3 |
| Anterior/posterior split errors | 0 (not measurable) | 0 |
| Complete failures (Dice < 0.5) | 0 | 0 |

So the boundary of the outline matters most. No case fails outright (worst external Dice 0.742).

**All models compared** (whole hippocampus; cross-validation rows are mean ± SD over 5 fold models, each scored on the same test sets):

| Model | MSD test Dice | MSD test NSD | External Dice, atlas box | External, largest piece | External, MSD-style box + largest piece |
|---|---:|---:|---:|---:|---:|
| 3D U-Net, single run (main model) | 0.897 | 0.959 | 0.811 | 0.840 | 0.843 |
| 3D U-Net, 5-fold cross-validation | 0.897 ± 0.001 | 0.958 ± 0.001 | 0.831 ± 0.016 | 0.853 ± 0.004 | 0.858 ± 0.003 |
| Random position, 5-fold cross-validation | 0.896 ± 0.001 | 0.955 ± 0.002 | 0.826 ± 0.011 | 0.853 ± 0.001 | 0.859 ± 0.001 |
| **3D U-Net, 5-fold ensemble** (folds' probabilities averaged; one model) | 0.898 | 0.960 | **0.853** | 0.859 | 0.862 |
| Random position, 5-fold ensemble (folds' probabilities averaged; one model) | 0.898 | 0.958 | 0.851 | 0.858 | 0.861 |

In distribution all models are equal (MSD test, mean of anterior and posterior: 0.874 single run, 0.874 and 0.873 for the two ensembles). **Averaging the five fold models is what helps on external data**, not the random position training:

- **3D U-Net single run → 3D U-Net ensemble:** plain external Dice 0.811 → 0.853, 48 of 50 hippocampi better (Wilcoxon p < 0.001). The averaged models leave fewer stray islands (keeping only the largest piece adds just 0.006 Dice to the ensemble, against 0.029 to the single run) and draw a less wide outline (median predicted/true volume 1.05 vs 1.09).
- **3D U-Net ensemble vs random position ensemble:** equal, or slightly in favour of the 3D U-Net ensemble (plain box 0.853 vs 0.851, p = 0.04; with the largest piece p = 0.59). The random position model's advantage is its steadiness when the box moves (below), which ensembling already gives.
- **Single runs carry chance:** the fold models score higher externally than the single run (0.831 ± 0.016 vs 0.811), so one run's external number can be off by about that much.

NSD, volume errors, ICC and Bland–Altman plots per model, and the paired tests: [results](docs/results.md#cross-validation). NSD, volume errors, ICC and Bland–Altman plots per model: [results](docs/results.md#cross-validation).

Full results, the per-part figure and all tables: [results](docs/results.md).

## Experiments

| Experiment | Change | Result | Decision |
|---|---|---|---|
| Clean labels | Remove stray label pieces | No change (MSD 0.875, external 0.813, p > 0.3) | Keep the raw labels |
| No normalisation | Raw intensities | MSD 0.874 → 0.774, external 0.811 → 0.740 | Keep z-scoring |
| Largest piece | Keep the biggest connected piece (external only) | External 0.811 → 0.840, all 50 better | Opt-in, secondary |
| Shifted predictions | Correct the mask offset | 0.840 → 0.794 on held-out subjects | Rejected |
| ASHS check | Independent tool on 2 subjects | ASHS within 0.8 mm of the external labels; model 1.8–3.3 mm off | The gap comes from MSD's labels |
| Random position | Random sub-crop in z during training | Sensitivity to box position halved (0.052 → 0.024); islands 0.33 → 0.11 ml; external 0.849 vs 0.843; MSD 0.870 vs 0.874 | Failed one criterion set in advance; the 3D U-Net stays |

![Dice and NSD against the hippocampus's height in the crop box for both models](docs/figures/random_trim_position_curve.png)

*The random position model (orange) stays flatter as the crop box moves; the 3D U-Net (blue) is better when the hippocampus sits low in the box. The vertical axes don't start at 0.*

The random position criteria were written down before training. All experiments, in detail: [results](docs/results.md#experiments) and the [retraining plan](docs/retraining_plan.md).

## Experiment tracking with MLflow

Every training run and every evaluation is logged to MLflow (`runs/mlflow.db`):
- **One MLflow run per fold:** config, fold, training curves, and every test and external score, tagged with its post-processing (largest piece, box type).
- **Why it helps:** every number in this README traces back to the run and config that produced it, and runs can be compared without spreadsheets.
- **Ablation table:** `python -m hippo.results --ablation` builds it from MLflow, with one row per experiment.

```bash
mlflow ui --backend-store-uri sqlite:///runs/mlflow.db --port 5001   # http://127.0.0.1:5001
```

More: [usage](docs/usage.md#6-mlflow).

## Streamlit viewer

![Slice views of MSD test case hippocampus_135 on the Streamlit page](docs/figures/streamlit_preview_msd.png)

*MSD test case hippocampus_135: label, 3D U-Net prediction (Dice whole 0.913) and errors (green correct, yellow wrong part, blue missed, red extra).*

1. Start it with `PYTHONPATH=src streamlit run streamlit_dashboard/compare_slices.py` and open http://localhost:8501.
2. Pick an image (type `hippocampus_135` or `s03_L` in **Filter**), tick the runs, and click **Predict and show**.
3. Move the axial, coronal and sagittal sliders; every row (label, then each run) follows.
4. Switch **Colour** between **Labels** and **Errors**, and tick **3D surfaces** for rotatable 3D views.

All controls, plus external and 3D screenshots: [Streamlit viewer](docs/streamlit.md).

## Deployment: FastAPI on Google Cloud Run

- **API:** `hippo.api` (FastAPI) serves the 3D U-Net. `/segment/crop` takes a hippocampus crop; `/segment/mni` takes a whole-brain T1 in MNI space and returns both sides. Each returns volumes per part, quality flags and optional masks.
- **Safety checks from the risk table:** wrong inputs are refused (not 3D, not about 1 mm, too large, not MNI-shaped), unusual results are flagged for review, and every response gives the model version and the research-use notice.
- **Cloud:** a CPU Docker image built by Cloud Build and run on Cloud Run. It scales to zero when idle and is capped at one instance, so a demo fits the free tier. The same image runs anywhere.
- **Integration design:** the path from hospital DICOM to this API (de-identification, conversion, registration to MNI, human review).

- **Upload page:** a second Cloud Run service runs a Streamlit page where a visitor uploads one scan and sees the volumes, quality flags, slices, 3D view and mask download; it calls the API, so the model still comes from the MLflow bucket.

Architecture, commands and cost: [deployment](docs/deployment.md).

## Intended use and regulatory path

This is research software and meets no medical-device standard. As Software as a Medical Device (SaMD), real deployment would need:
- **Classification:** under EU MDR (Rule 11) or the FDA.
- **Quality and lifecycle:** quality management (ISO 13485), risk management (ISO 14971), software lifecycle (IEC 62304).
- **Good Machine Learning Practice:** including a plan for retraining.
- **EU AI Act:** compliance as a high-risk system.
- **Validation:** multi-site, in the intended population.

An intended-use statement and an ISO 14971-style risk table, based on this project's results: [intended use and risk](docs/intended_use_and_risk.md).

## Limitations

- **Incomplete training labels:** MSD's labels leave out the inner edge of the hippocampus, so the model misses 15–18% of the external hippocampus.
- **No external dataset with MSD's labels:** the anterior/posterior split can't be checked outside MSD.
- **Possible overlap between training and test:** the split is by file, and 22 of 39 test cases may have the other side of the same person in training.
- **Sensitive to the crop:** the model depends on how the scan is cropped.
- **Small evidence base:** one training seed so far, one external site, and 4 hippocampi in the ASHS check.

All limitations with numbers: [limitations](docs/limitations.md).

## Future work

1. **Whole-hippocampus labels:** train on a dataset with whole-hippocampus labels on whole-head scans, using patch-based training for the GPU limit.
2. **Serve the 3D U-Net ensemble:** it is the best-scoring model (external 0.853, MSD unchanged); the API still serves the single run as its default.
3. **Second external dataset:** a fresh one that stays untouched until the final model.
4. **Uncertainty and quality flags:** to flag poor cases automatically, using the dropout the model already has.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
export PYTHONPATH=src
python -m hippo.data --folds 5                     # MSD Task04 (~30 MB), splits and folds
scripts/run_cv.sh                                  # 5-fold training + evaluation + tables
python -m pytest -q                                # tests, about 30 s
```

All commands, the run folder contents and the project structure: [usage](docs/usage.md). Viewer, API and MLflow UI as containers (`docker compose up --build`): [Docker](docs/docker.md).

## Conclusion

1. **In distribution the model works well:** whole-hippocampus Dice 0.897 and NSD 0.959 on 39 held-out MSD cases.
2. **Externally it drops to 0.811:** mainly because MSD's labels leave out tissue that the external labels include. An independent tool confirmed this.
3. **Averaging five fold models recovers part of it:** a 5-fold ensemble reaches 0.853 externally (48 of 50 better) with in-distribution results unchanged; random position training adds nothing on top.
4. **Cropping fixes islands, not missing tissue:** cropping and position training remove islands, but the missed chunks stay the same (0.31 ml per hippocampus; not yet measured for the ensembles).
5. **Next:** training labels that cover the whole hippocampus.

Results on two small datasets; one seed per run, five folds per experiment.
