# Intended use and risk analysis

**Status: research software, not a medical device.** This is an illustrative risk analysis in the
style of ISO 14971, written to show which hazards this model has and what a real medical-device
project would have to control. It is not a risk-management file, it has not been reviewed by a
clinician or a regulatory expert, and the model complies with no medical-device standard. Research
use only, not for diagnosis.

## Intended use

**What it does.** The 3D U-Net outlines the hippocampus on a 3D T1-weighted MRI crop and splits it into
an anterior part (head) and a posterior part (body and tail). From the mask it reports the volume of
each part and of the whole hippocampus in ml.

| | |
|---|---|
| **Purpose** | Research and education: automatic hippocampal volumetry for studies of anatomy, ageing and neurodegeneration, and as a portfolio example of a medical-imaging pipeline. |
| **Intended users** | Researchers and students who can review the masks themselves (the Streamlit page shows every prediction against the scan). |
| **Input** | One crop around one hippocampus (left or right), T1-weighted, 1 mm voxels, about 35 × 50 × 35 voxels, cut out the way MSD Task04 crops were made. The model was trained only on such crops. |
| **Output** | A mask (0 background, 1 anterior, 2 posterior) on the input grid, and volumes in ml. |
| **Data it was developed on** | MSD Task04 Hippocampus (Vanderbilt University Medical Center; 260 labelled crops, no subject, age, sex, diagnosis or scanner information). Tested externally on MNI-HISUB25 (25 healthy adults, one site, whole-brain scans in MNI space). |
| **Performance it was shown to have** | MSD test (39 crops): whole-hippocampus Dice 0.897, NSD 0.959 (anterior 0.880, posterior 0.868). MNI-HISUB25 (50 hippocampi): whole Dice 0.811, or 0.840 with the largest piece kept. |

**Not intended for:**
- diagnosis, prognosis, treatment decisions or any other clinical decision about a patient;
- whole-brain scans fed in directly, or crops placed differently from MSD's (results depend on the crop);
- other MRI contrasts (T2, FLAIR), other voxel sizes, or 2D images;
- populations it was not tested on: children, people with lesions, tumours, surgery or severe atrophy,
  and scanners or sites other than the two above;
- the anterior/posterior split as a precise anatomical boundary: MSD doesn't publish its rule, and the
  model's boundary sits 2–6 mm further forward than ASHS's.

**If it were a medical device**, the intended use would also have to name the clinical condition, the
patient population, the user (for example a neuroradiologist), the clinical setting, and whether the
output informs or drives a decision. That is what decides its risk class.

## Risk table

**Severity** (if the output were used to inform a clinical judgement): *Minor* = no effect on care;
*Serious* = could lead to a wrong or delayed assessment; *Critical* = could lead to a wrong diagnosis or
treatment. **Probability** is a judgement from this project's results: *Low*, *Medium*, *High*.
**Acceptability:** Critical with Medium or High, or Serious with High, is not acceptable without further
control.

| # | Hazard (cause) | Hazardous situation | Possible harm | Severity | Probability (evidence) | Control in this project | Residual risk / what a device would need |
|---|---|---|---|---|---|---|---|
| 1 | **Under-segmentation:** the training labels leave out the inner (medial) strip, subiculum tip and uncus | Hippocampal volume reported too small | Atrophy over-estimated: a healthy person looks impaired | Serious | High: misses 15–18% of the external hippocampus; missed chunks unchanged by retraining (0.31 ml per hippocampus) | Documented as the main limitation; whole-hippocampus labels are the next step | **Not acceptable.** Train on labels that cover the whole hippocampus; validate against the protocol clinicians use |
| 2 | **Over-segmentation and islands:** unfamiliar tissue around the hippocampus | Volume reported too large | Real atrophy missed or under-estimated | Serious | High on external scans: volume ratio median 1.09, 44 of 50 too big; extra 1.12 ml per hippocampus, 0.33 ml of it in islands | Opt-in largest-piece post-processing (islands 0.33 → 0 ml), MSD-style cropping, random position training (islands 0.11 ml) | Outline still about 1 voxel too wide. Fix cropping in the pipeline; per-case QC; acceptance limits on volume error |
| 3 | **Crop placement:** the model depends on where the hippocampus sits in the box | Results change with how the scan was cropped | Inconsistent volumes between scans or visits | Serious | Medium: moving the box 10 mm changes Dice by up to 0.086; random position training halves this | Defined cropping (atlas or MSD-style box); position test reported | Lock one validated cropping step into the software, or train on whole images |
| 4 | **Anterior/posterior boundary:** MSD's rule is unpublished | Part volumes shifted between head and body | Wrong conclusions in studies of one part | Minor (research) to Serious | Medium: boundary 2–6 mm forward of ASHS; model vs ASHS Dice 0.80 anterior, 0.74 posterior (4 hippocampi only) | Whole volume reported as the main result; part volumes marked research-only | Define the boundary protocol and validate the parts against it |
| 5 | **Wrong input:** whole-brain scan, other contrast, other voxel size | Meaningless mask presented as a result | Wrong volumes trusted | Serious | Medium: only the Streamlit page rejects images larger than 128 voxels; the command-line tools check nothing | Intended use states the input; crops made by the pipeline itself | Input checks (size, spacing, contrast) that refuse instead of predicting |
| 6 | **Preprocessing mismatch:** a model run with the wrong intensity normalisation | Much worse masks with no warning | Wrong volumes | Serious | Low: each run reads its own `normalize` setting; without z-scoring, external Dice falls to 0.740 (MSD test 0.774) | Settings stored in each run folder and in MLflow; tests check this | Lock preprocessing into the released model version |
| 7 | **Domain shift:** one training site, unknown population, no metadata | Performance on a new scanner or population unknown | Unreliable volumes in exactly the patients it would be used for | Critical | Medium: Dice drops from 0.897 (MSD) to 0.811 (external); no patient groups tested | External test on a second dataset; limitation documented | Multi-site, multi-scanner validation in the intended population, with subgroup analysis |
| 8 | **Optimistic test results:** possible left/right partners across the file-level split | Users trust the model more than they should | Over-reliance on wrong outputs | Serious | Medium: 22 of 39 test cases may have their partner in training | Documented; external dataset is an independent check | Subject-level splits; a locked, independent test set |
| 9 | **Silent failure:** no confidence or quality score | A bad mask looks as trustworthy as a good one | Errors not caught | Serious | Medium: no complete failures seen (worst external Dice 0.742), but nothing flags a poor case | Every mask can be reviewed in the Streamlit page; dropout kept for later uncertainty estimates | Automatic QC flags and uncertainty, with a "needs review" output |
| 10 | **Misuse:** used for diagnosis despite the notice | Clinical decision based on research software | Wrong diagnosis or treatment | Critical | Low (portfolio project, not distributed to clinics) | "Research use only, not for diagnosis" in the README, the reports and the Streamlit page | Regulatory clearance before any clinical use; labelling and user training |

## What real deployment would add

A medical-device version would keep this analysis as a living risk-management file under ISO 14971,
linked to the software lifecycle (IEC 62304) and a quality system (ISO 13485). It would add risks this
research code doesn't cover: cybersecurity, data protection, integration with hospital systems, usability
(IEC 62366-1), and model changes after release, which would need a predetermined change control plan. Every
control would need verification evidence, and residual risks would be weighed against clinical benefit.
These standards are listed from general knowledge; their current versions should be checked before
relying on them.
