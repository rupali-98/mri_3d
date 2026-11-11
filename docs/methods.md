# Methods: data, split and training

Research use only, not for diagnosis. Back to the [README](../README.md).

## What the model predicts

- **Input:** one 3D T1-weighted MRI crop around **one** hippocampus (left or right), at 1 mm voxels, about 35 × 50 × 35 voxels.
- **Output:** a 3D mask on the same voxel grid: `0` background, `1` **anterior** hippocampus (the head), `2` **posterior** hippocampus (body and tail).
- **Derived measurements:** the volume of each part and of the whole hippocampus in ml (voxel count × voxel volume).

The model is a MONAI 3D U-Net (`src/hippo/model.py`). It does not diagnose anything: it only outlines the structure.

## What the model sees

This is an image model, so its "features" are the **voxel intensities** of the MRI. No other information goes in: no age, sex, side or scanner details.

| Input | Meaning | How it is prepared (`src/hippo/data.py`) |
|---|---|---|
| T1-weighted MRI crop | Brightness of each 1 mm voxel. On T1, grey matter (the hippocampus) is mid-grey, white matter brighter, fluid dark | Loaded from NIfTI, turned to **RAS orientation** (x = left→right, y = back→front, z = down→up), **z-scored per scan** (mean 0, standard deviation 1 over non-zero voxels), padded to a multiple of 8 voxels per side |
| Label (training and scoring only) | Manual outline: 0 background, 1 anterior, 2 posterior | Same orientation and padding; never z-scored |

**Why z-scoring matters:** MSD scans come in three very different intensity scales (99th percentile about 100, 1,000 and 400,000). A run trained without per-scan normalisation (`configs/no_norm.yaml`) scores MSD test Dice 0.774 instead of 0.874 and external Dice 0.740 instead of 0.811.

## Datasets

| Dataset | Used for | What it contains | Source and licence |
|---|---|---|---|
| **MSD Task04 Hippocampus** | Training, validation, in-distribution test | 260 labelled T1 MPRAGE crops (plus 130 unlabelled) from Vanderbilt University Medical Center; labels anterior/posterior | [Medical Segmentation Decathlon](http://medicaldecathlon.com/) ([paper](https://www.nature.com/articles/s41467-022-30695-9)); CC-BY-SA 4.0. Downloaded by `python -m hippo.data` (~30 MB) |
| **MNI-HISUB25** (MNI-based external dataset) | External test only, never trained on | Whole-brain 3 T T1 of 25 healthy adults in MNI space, manual labels of both hippocampi as three subfields (CA1-3, subicular complex, CA4-DG) | [NITRC](https://www.nitrc.org/projects/mni-hisub25/) (Kulaga-Yoskovitz et al., *Scientific Data* 2015); CC BY-NC-SA 3.0. Download it yourself into `data/mri_dataset/` |
| **ASHS-PMC-T1** | Independent second opinion on 2 external subjects | Automatic anterior/posterior hippocampus labels from a multi-atlas tool | [Cloud ASHS](https://sites.google.com/view/ashs-dox/cloud-ashs/cloud-ashs-for-t1-mri) via the ITK-SNAP Distributed Segmentation Service |

**Why MSD Task04:** it was chosen because of GPU limits: its small crops train comfortably on a small GPU, and its anterior/posterior labels allow volumetry of each part. The cost is that **its labels do not cover the whole hippocampus**: they include "the hippocampus proper and parts of the subiculum" and leave out the thin inner extensions (subicular tip, uncus, the strip along the tail). See [Limitations](limitations.md).

**Why MNI-HISUB25:** it is public, has manual hippocampal labels drawn by a different group on different scanners, and has whole-brain scans, so the model can be tested on images it has never seen. Its labels are subfields with a wider outline, so only the whole hippocampus can be compared.

## Split

- **Fixed test split:** MSD's own test files have no public labels, so the 260 labelled cases are split by file into **182 train / 39 validation / 39 test** (70/15/15, seed 42), written to `data/splits.json`. The 39 test cases are never trained on or used to pick weights.
- **Cross-validation:** `python -m hippo.data --folds 5` splits the 221 train + validation cases into 5 fixed folds by file (seed 42), written to `data/folds/fold0.json` … `fold4.json`. Each fold validates on one fold and trains on the other four; the test split is the same in every fold. Each experiment's results are the mean ± SD of the 5 fold models, each scored on the whole test split.
- **External data is test-only.** Each external hippocampus is cut out as a 48 × 56 × 56 mm box centred on the dataset's own probability atlas, never on the subject's label (`python -m hippo.external`), giving 50 crops (25 subjects × left/right).
- **Known risk:** MSD gives no subject IDs, so the split is by file. Files `2k-1` and `2k` may be one person's two sides; 22 of the 39 test cases may have their partner in training (see [Limitations](limitations.md)).

## Training

Settings are in `configs/baseline.yaml` (one comment per setting); the loop is `src/hippo/train.py`.

- **Model:** MONAI 3D U-Net, 1 input channel, 3 output classes, 4 levels with 16/32/64/128 channels, stride 2, 2 residual units per level, dropout 0.1.
- **Loss:** Dice + cross-entropy (`DiceCELoss`).
- **Optimiser:** AdamW, learning rate 0.001, weight decay 0.00001, cosine learning-rate decay.
- **Schedule:** 150 epochs, batch size 8, seed 42 (deterministic; an interrupted run resumes exactly with `--resume`). Mixed precision on NVIDIA GPUs.
- **Hardware:** the crops are small, so it trains on a 4 GB GPU, Apple Silicon (MPS) or CPU (about 13 s per epoch on Apple Silicon).
- **Model selection:** validation every 5 epochs; the weights with the best mean of the anterior and posterior validation Dice are saved as `best.pt` (single run: epoch 105, mean Dice 0.877).

## Augmentation

Training only (`train_transforms` in `src/hippo/data.py`):

- random rotation (up to about ±11°) and scaling (±10%), half the time;
- left–right flip, half the time. Never front–back, because that would swap the anterior and posterior labels;
- brightness scale and shift (±10%), half the time, and Gaussian noise (σ 0.05), a fifth of the time;
- optional random sub-crop (`random_trim`, off in the 3D U-Net): cuts voxels per axis at a random position without cutting into the label; used in the [random position experiment](retraining_plan.md).

## Tuning

No hyperparameter search was run: the settings above were fixed in advance and are the same for every experiment, so experiments differ in one setting only. The only data-driven choices are the best epoch (by validation Dice) and, on the external data, the largest-piece post-processing and the MSD-style box. Those two were chosen after looking at external errors, so their results are reported as secondary.

## Evaluation

- **Dice:** overlap between prediction and label (1 = identical, 0 = no overlap).
- **NSD (1 mm):** share of the two outlines within 1 mm of each other, the MSD challenge's tolerance for this task, so results compare with published MSD numbers.
- **HD95:** the 95th percentile of the distances between the two surfaces (mm).
- **Volume:** absolute error (ml), relative error (% of the true volume), ICC(A,1) (absolute agreement) and a Bland–Altman plot.
- **Labels:** per part (anterior, posterior) and for the whole hippocampus (labels merged). External data is scored as the whole hippocampus only, because its labels have no anterior/posterior parts.
- Tools: `src/hippo/evaluate.py` (MSD test), `src/hippo/external.py` (external), `src/hippo/results.py` (table per experiment), `src/hippo/errors.py` (error analysis). Full analysis of the external data: [external_validation.md](external_validation.md).
