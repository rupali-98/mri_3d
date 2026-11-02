# External validation on MNI-HISUB25: first iteration

Research use only, not for diagnosis.

This report checks how the 3D U-Net trained on MSD Task04 does on an  in-distribution **MSD** splitted test datasets and **MNI-based external dataset** ([MNI-HISUB25](https://www.nitrc.org/projects/mni-hisub25/), from NITRC; see [Data](#data)) that it has never seen, and where its errors come from.

**Main finding:** whole-hippocampus Dice drops from 0.897 (MSD test) to 0.811 (external), but the model still finds the hippocampus in the right place. The drop has two parts:
- **Stray islands** in the larger external box. 
- **Label differences.** The remaining disagreement sits mostly at the inner edge, where the external labels include tissue that MSD's labels leave out , and at the front edge, where the prediction reaches 4 mm further.

The placement is confirmed three ways: the outer and back edges match, the prediction's centre matches the label's on MSD (within 0.05 mm), and an independent tool (ASHS-PMC-T1, 2 subjects) lands within 0.8 mm of the external labels. So the main result is reported **without any shift or calibration**.

## Summary

| | Dice | NSD (1 mm) | Hippocampi |
|---|---|---|---|
| MSD test, anterior | 0.880 ± 0.035 | 0.931 ± 0.059 | 39 |
| MSD test, posterior | 0.868 ± 0.039 | 0.942 ± 0.047 | 39 |
| MSD test, whole hippocampus (reference) | 0.897 ± 0.027 | 0.959 ± 0.032 | 39 |
| **External, whole hippocampus (main result)** | **0.811 ± 0.030** | **0.777 ± 0.038** | 50 (25 subjects × 2) |
| External, left / right | 0.807 / 0.815 | 0.776 / 0.778 | 25 / 25 |
| External + largest connected piece (post-processing, secondary) | 0.840 ± 0.022 | 0.825 ± 0.034 | 50 |

- Model: 3D U-Net (run folder `runs/baseline`; z-score normalisation, raw labels, one training seed).
- MSD reference: the model's own MSD test split, scored the same way (prediction > 0 vs label > 0), so the two rows can be compared directly. The anterior and posterior rows are MSD's usual per-part scores; the external labels have no such parts.
- Left and right do not differ (paired by subject, p = 0.49).

## Data

**Source:** the MNI-based external dataset **[MNI-HISUB25](https://www.nitrc.org/projects/mni-hisub25/)**, downloaded from NITRC (Kulaga-Yoskovitz et al., *Scientific Data* 2015; licence CC BY-NC-SA 3.0): 25 healthy adults, 3 T, T1 and T2 scans in MNI space (1 mm and 0.4 mm), with manual hippocampal labels drawn as three parts: **subicular complex, CA1-3 and CA4-DG**, on each side.

We use the 1 mm T1 and its labels. The three parts together form the whole hippocampus: one connected piece in all 50 files, with small internal holes (0.6% of the volume).

## Method

1. **Crop.** Each hippocampus is cut out as a 48 × 56 × 56 mm box centred on the SPAM probability atlas that comes with the dataset. The subject's own label never places the box. Every hippocampus lies fully inside its box (coverage 1.0).
2. **Predict** with the run's own preprocessing (z-score per scan, pad to a multiple of 8).
3. **Score the whole hippocampus** (prediction > 0 vs label > 0). The labels are subfields, not anterior/posterior, so only the whole structure can be compared.

Nothing is tuned on the external labels for the main result.

## Where the errors come from

### 1. A different definition of the hippocampus (the main cause)

MSD's labels cover "the hippocampus proper and parts of the subiculum". MNI-HISUB25 includes the **whole subicular complex**, which lies on the inner (medial) side of the hippocampus. Its labels also include the uncus and a thin inner strip along the tail.

We compared, per hemisphere, where the prediction and the label start and end along each axis (prediction minus label, median mm):

| Axis | Edge on one side | Opposite edge |
|---|---|---|
| Inner–outer (x) | inner edge **3 mm short** | outer edge matches (0) |
| Back–front (y) | front edge **4 mm too far** | back edge matches (0 to +1) |
| Down–up (z) | top about 1.5 mm low | bottom about 1–2 mm low |

A displaced prediction would move both opposite edges the same way. Here only one edge differs on x and y. Column by column, the label has tissue where the prediction has none in **4,663 inner-half columns but only 386 outer-half columns**. These are the subicular and uncal parts the MSD-trained model never draws.

On MSD itself the prediction's centre matches the label's within 0.05 mm on every axis, so the model has no general placement bias.

**The model does segment the subiculum; the difference is only its thin inner extensions.** As a check, we re-scored the same predictions against the external label with the subiculum part (value 2) removed:
- Dice fell from 0.811 to **0.671**, and the prediction became 52% larger than the label (volume ratio 1.52).
- Removing only the inner half of the subiculum gave 0.753.
- All 50 cases got worse either way.

So most of the subiculum is in the model's prediction too. The subiculum part lies only about 0.9 mm inwards of CA1-3 on average, mostly beneath it. The disagreement is confined to the thin extensions at the inner edge (the subicular tip, the uncus, the strip along the tail). No label part can be dropped to "align" the protocols. A rule that cuts where the prediction stops would make the label agree with the model by construction.

### 2. Islands and a thin band below the hippocampus

The external box shows about 12 mm of tissue below the hippocampus. MSD crops show about 5 mm. The model spills into it: small separate islands, and a thin extra band (about 0.5 mm) along the bottom of the outer half. Before post-processing, the predicted volume is 9% larger than the label (median ratio 1.09; 44 of 50 too big). Only 1 of 50 predictions is a single piece.

Keeping the largest connected piece removes the islands. It improves every one of the 50 cases (0.811 → 0.840), but it was chosen after looking at these errors, so it is reported as a secondary result.

### 3. MNI space

The scans and labels are in MNI space, scaled to a template brain. Volumes (external median 4.67 ml vs MSD 3.33 ml) cannot be compared with MSD or with the subjects' real head size.

## Why the predictions are not shifted

The prediction's centre is offset from the label's centre in the same direction in every subject. Per side, the median offset is about 1 mm outwards, 1.2–1.6 mm forwards and 1.4–1.8 mm lower. Two random halves of the subjects give the same offset to within 0.2–0.8 mm. So shifting the predictions to match looked tempting.

It is wrong, for four reasons:

1. **It is a difference in shape, not in position.** The label has extra tissue on one side (subiculum, uncus, inner tail strip; mostly inner and upper). That pulls the label's centre inwards and upwards while the rest of the outline still matches the prediction.
2. **The label is in the right place.** On the 0.4 mm T1 and T2 of s03 and s17, the label follows the real boundaries of the hippocampus, including the fluid above it. The 1 mm and 0.4 mm labels agree within 0.2 mm, so the labels were not displaced by resampling.
3. **Shifting makes the results worse, tested on held-out subjects.** The mean offset was estimated on 20 subjects and applied to the other 5, rotating so every subject was tested once.
   - The centres lined up as intended.
   - But Dice fell from **0.840 to 0.794** and NSD from **0.825 to 0.740**, with **47 of 50** hippocampi worse (p < 10⁻¹³).
   - Shifting the other way was worse still.
4. **It would fit the labelling style, not improve the model.** Moving the outline cannot draw the subicular parts the model leaves out; it only damages the edges that already match.

## Box placement experiments (diagnostic only)

Moving the **box** (what the model sees) is different from moving the **prediction**. These runs explain the errors. They are not used for the main result.

All rows keep the largest piece, so the reference is the normal box at 0.840 / 0.825.

| Box | Dice | NSD | How it was chosen |
|---|---|---|---|
| Atlas-centred (normal) | 0.840 | 0.825 | Not tuned |
| Moved so the hippocampus sits 4 mm lower (z−4) | 0.849 | 0.842 | From MSD: where MSD crops place the hippocampus (0.43 of the box height) |
| z−5 | 0.851 | 0.847 | Close to the MSD placement |
| z−10 | 0.856 | 0.857 | Best on the external labels, so **tuned on test data**; cuts off part of 6 hippocampi | |

What these show:
- **MSD and the external data are cropped around different centres.** MSD crops are centred on the hippocampus's bounding box, while the external box is centred on its centre of mass, 2–3 mm lower. So the external box puts the hippocampus about 3 mm higher than the model is used to. That explains why a downward move (z−4/−5) helps a little.
- **Copying MSD's tight cropping rule did not help.** The extra context of the larger box seems to help this model on external scans, even though it also causes the islands.
- **No box move closes the gap to MSD.** The remaining error is the definition difference at the inner edge.

## Independent check with ASHS-PMC-T1 (pilot, 2 subjects)

[ASHS-PMC-T1](https://sites.google.com/view/ashs-dox/cloud-ashs/cloud-ashs-for-t1-mri) is a multi-atlas tool that labels anterior and posterior hippocampus (plus surrounding cortex) on whole-head T1 scans. It was run through the ITK-SNAP Distributed Segmentation Service on the 1 mm whole-brain T1 of **s03 and s17**. Only the T1 images were uploaded, no labels. It was compared on its own 0.5 × 1 × 0.5 mm grid, inside each external crop box. In the table, "model" means the largest piece, so islands do not distort the edges.

| | s03_L | s03_R | s17_L | s17_R | Mean |
|---|---|---|---|---|---|
| Whole Dice, ASHS vs manual label | | | | | **0.840** |
| Whole Dice, model vs manual label (plain: 0.808) | | | | | **0.841** |
| Anterior / posterior Dice, model vs ASHS | | | | | **0.80 / 0.74** |
| Front/back boundary, model minus ASHS (mm, + = forwards) | +2.0 | +2.5 | +6.0 | +2.0 | |
| Centre offset from the manual label, ASHS (mm) | 0.4 | 0.8 | 0.7 | 0.7 | |
| Centre offset from the manual label, model (mm) | 1.8 | 2.4 | 1.9 | 3.3 | |
| Inner edge short of the manual label, ASHS (mm) | 0.5 | 0.5 | 2.5 | 2 | |
| Inner edge short of the manual label, model (mm) | 2 | 1 | 7 | 6 | |

What this shows:
- **The manual labels are in the right place.** An independent tool that never saw them lands within 0.8 mm of their centre, with no common direction. The model's offset always points the same way: outwards, forwards and lower.
- **The model draws MSD's narrower hippocampus.** ASHS, like the manual labels, includes most of the inner tissue. The model stops further short. So the external gap is mostly a protocol difference.
- **The anterior/posterior split agrees reasonably with ASHS.** The model's boundary lies 2–6 mm further forwards. MSD's exact boundary rule is not published, so this is a difference between two methods, not an error of either.

**ASHS cannot check MSD itself.** It needs a whole head to register to its template. On two MSD crops (hippocampus_135, hippocampus_017), one run failed at registration and the other finished with an empty segmentation. Instead, MSD and external label shapes were compared directly, 260 vs 50 labels. At the same length (39 vs 38 mm), MSD labels are 3–5 mm narrower from inner to outer edge (most in the tail) and 2–3 mm lower. This supports the protocol difference, but different people are compared, and MSD does not record the side.

The ASHS labels can be viewed as the "Second reference" in `streamlit_dashboard/compare_slices.py`. Note that the page scores on the 1 mm crop grid, so its numbers differ slightly from the table.

## Limitations

- **Different labelling protocol.** MNI-HISUB25 includes the whole subicular complex, the uncus and the inner tail strip; MSD includes only part of the subiculum. External Dice and NSD therefore measure agreement with another protocol as well as model error. The largest disagreement is at the inner edge.
- **Whole hippocampus only.** The external labels are subfields, so the model's anterior/posterior split cannot be checked on this dataset.
- **Label names are inferred.** Which label value is which part (1 = CA1-3, 2 = subiculum, 3 = CA4-DG) was inferred from the atlas and part sizes, not read from a key.
- **MNI space.** Volumes are scaled to a template brain and cannot be compared with MSD or real head size.
- **Fixed atlas-based crop.** The box placement depends on the SPAM atlas and the MNI grid. The pipeline does not yet handle scans in native space.
- **Small and homogeneous.** 25 healthy adults from one site and scanner. No patients, no age or scanner variation.
- **One training seed.** Differences between runs (e.g. clean labels vs raw) are within training noise.
- **No subject or side information in MSD.** MSD records neither the subject nor the side of each crop, so the MSD/external shape comparison cannot tell inner from outer side, and MSD results cannot be split by side.
- **Optimistic MSD reference.** Neighbouring MSD files may be the left and right hippocampus of one person, so the MSD test split may be optimistic (see the README).
- **Post-processing and box moves were chosen after seeing these errors.** The largest-piece step and z−4/−5 are reported as secondary or diagnostic results, never as the main score.
- **The visual label check and the ASHS check covered 2 subjects** (s03, s17). The numerical checks cover all 50 hippocampi.
- **No dataset with MSD's anterior/posterior protocol was found**, so the model's split can only be compared with another method (ASHS), not with a matching manual label.

## Next steps

1. ASHS-PMC-T1 on all 25 subjects (the pilot covers 2), so the model-vs-ASHS comparison has 50 hippocampi and a p-value.
2. **Next branch:** train on a different dataset with whole-hippocampus labels on whole-head scans (MSD's labels do not cover the whole hippocampus; MSD was chosen for its small crops because of GPU limits). Patch-based training keeps it within the GPU budget.
3. A second external dataset with whole-hippocampus labels closer to MSD's protocol, e.g. the EADC-ADNI Harmonized Protocol labels (needs ADNI access) or the Henry Ford Hospital set on NITRC. Run the same edge checks before reading anything into Dice.
4. If the external protocol should be matched: fine-tune on some external subjects and test on the rest, keeping each subject's two hemispheres together.

## Reproduce

```bash
source .venv/bin/activate && export PYTHONPATH=src
python -m hippo.external --src data/mri_dataset --run runs/baseline                    # main result
python -m hippo.external --src data/mri_dataset --run runs/baseline --largest-piece    # secondary
PYTHONPATH=src python archive/shift.py --external data/mri_dataset --run runs/baseline --conditions z-4 z-5 z-10 --largest-piece   # archived
python -m hippo.external --src data/mri_dataset --run runs/baseline --recentre --largest-piece
python -m hippo.external --src data/mri_dataset --run runs/baseline --msd-crop --largest-piece
```

Results are written to `runs/baseline/external*/` and `runs/baseline/shifted_external*/`. The shift test (`archive/shift.py`) and the one-off analysis scripts (centre offsets, edges, columns, held-out shifts, label overlays, subiculum check) are archived in `archive/` for the planned full-image training branch; see `archive/README.md`. The shifted crops themselves were deleted (`data/shifted_external/`); the shift test rebuilds them.
