---
name: readme-writer
description: Plan, write or restructure the README of a medical-imaging deep-learning project (this hippocampus segmentation repo, or one like it), so that a reviewer, recruiter or interviewer understands why the model exists, which data trains, validates and tests it and why, how it was trained, tuned and improved, what the results are (with plots and screenshots), how MLflow and the Streamlit viewer are used, what SaMD standards real deployment would need, the limitations and the future work. Use it whenever the user asks to create, rewrite, reorganise, update or review the README, add results, experiments or figures to it, make it portfolio- or interview-ready, or asks what the README should contain, even if they don't say "skill" or name every section.
---

# README writer

A README for a medical-imaging model is read by people deciding whether to trust the work. They need
the purpose, the data, the method, the evidence and the honest limits, in that order, and every number
must hold up if they check it. This skill turns that into a procedure.

## Audience and shape: a compact README, details in docs

The default reader is a **hiring manager or interviewer** (the user's choice for this project). A hiring
manager reads the first screen and decides; an interviewer opens the links and asks about them. So:

- **The README is a compact landing page**, about 150–200 lines. The first screen is the title, a
  two-line summary, the intended-use notice, a row of links to the detail docs, and an **"At a glance"**
  block of four or five bullets (problem, model, main result, key finding, next step).
- **Each README section is short**: two to six bullets or one small table, plus at most one figure, and
  ends with a "More: [doc](…)" link. Prefer one table that summarises over several that repeat.
- **Detail moves into `docs/`**, one file per topic, each starting with the research-use line and a link
  back to the README:
  - `docs/methods.md`: what the model sees, datasets and why, split and folds, training, augmentation,
    tuning, evaluation metrics;
  - `docs/results.md`: all result tables and figures, cross-validation, every experiment in full, error
    analysis;
  - `docs/limitations.md`: every limitation with its numbers;
  - `docs/usage.md`: set-up, evaluation, Python use, training, cross-validation, MLflow, tests, project
    structure;
  - `docs/streamlit.md`: how to navigate the viewer, all its controls, all screenshots;
  - `docs/intended_use_and_risk.md`: intended-use statement and ISO 14971-style risk table;
  - topic reports that already exist (here `docs/external_validation.md`, `docs/retraining_plan.md`) stay
    and are linked, not copied.
- **Move text, don't lose it.** When shortening the README, every paragraph that is cut must land in a
  doc (moved, not rewritten), and image paths must be fixed for the doc's folder (`figures/…` from
  `docs/`).
- **Keep the README and the docs in step.** A number that appears in both must be the same in both;
  when a result changes, update both.

If the user names a different reader (a technical reviewer, a clinician), ask whether they want
everything in one file instead.

## Workflow: plan first, then write

1. **Gather the facts from the repo, not from memory.** Read the project notes (`CLAUDE.md`), the current
   README, `configs/*.yaml`, the docs (`docs/*.md`), the figures in `docs/figures/`, and the result files
   in the run folders (`runs/<run>/summary_*.json`, `metrics_*.csv`, `<run>/results/results.md`,
   `runs/ablation.md`, `<run>/error_analysis/summary.md`). Note which results exist and which are
   still pending (for example folds not trained yet).
2. **Show a plan before editing anything.** List the sections in order; for each, say what it will
   contain, where the content comes from (existing README text, a doc, a result file, a figure), and what
   is missing or would need a new run. Flag decisions the user has to make. Wait for approval: the README
   is the project's public face and the user knows things the files don't.
3. **Write in the approved order**, reusing good existing text instead of rewriting it for its own sake.
4. **Verify.** Re-read every number against its source file, and check every image path and every
   internal link and anchor in the README and all docs with a short script (find each `](...)` target,
   check the file exists, and for `#anchor` links check a matching heading exists). Confirm that no
   figure shows a training case where a test case was meant, and that no paragraph cut from the README
   was lost. Report what was checked and anything that could not be.

## Sections

Use this order in the README, each kept short with a link to its doc. Merge or rename where the project
calls for it, but don't drop a topic silently: if there is nothing to say (no tuning was done), say that
in one line, because a missing section reads as something hidden. The detail for each topic goes in the
doc named in the audience section above.

1. **Title, summary, intended use, links and "At a glance".** Two lines on what the model does and how
   it was tested. The intended-use notice: research and education only, not a medical device, not for
   diagnosis. One line of links to every detail doc. Then four or five "At a glance" bullets.
2. **Why this model: purpose and goal.** The clinical or research question (for hippocampus:
   volumetry of the anterior and posterior parts, relevant to ageing and neurodegeneration research),
   what the model outputs, and what this work set out to show. Say plainly what it does not do
   (it outlines a structure; it doesn't diagnose).
3. **Data: training, validation and test.**
   - Each dataset with its source, licence, size, image type and label definition.
   - **Why this dataset was chosen** and what that costs (for MSD Task04: small crops fit a small GPU;
     its labels don't cover the whole hippocampus).
   - **How the split was done:** by file or by subject, fractions, seed, cross-validation folds, what is
     test-only, and known risks (possible left/right partners across the split).
   - **External datasets:** which, why, what they can and cannot test (different label protocol, whole
     hippocampus only), and that they are never trained on.
4. **How the model is trained.** Architecture, loss, optimiser, schedule, epochs, batch size, seed,
   hardware, model selection rule (which validation metric picks the weights). Preprocessing.
5. **Augmentation.** Every transform with its range and probability, what is deliberately left out
   and why (for example no front-back flip, because it swaps the anterior and posterior labels).
6. **Tuning.** What was tuned, how, and on which data. If hyperparameters were fixed and not searched,
   say so. Name anything chosen after looking at test data (it makes that result secondary).
7. **Experiments for improvement.** One subsection or table row per experiment: the question, the one
   change, criteria set in advance where they exist, the result with numbers and p-values, and the
   decision. Include experiments that failed; they show judgement.
8. **Results.** Lead with the main result and a highlighted note on where the model works best.
   - Per label and for the whole structure: Dice, HD95, NSD (state the tolerance), volume error
     (absolute and relative), ICC; mean ± SD over folds when there are folds.
   - Figures: score plots, Bland-Altman plot, curves, Streamlit previews and 3D screenshots, each with a
     caption that says what to look at and how it was made. In-distribution first, then external.
   - Error analysis: how the worst cases split into error types, and what that says to work on next.
   - Say which results are main and which are secondary or diagnostic.
9. **Experiment tracking with MLflow.** What is logged (config, fold, training curves, every
   evaluation with its post-processing flags), where it lives, the command to open the page, and why it
   is useful: runs can be compared without spreadsheets, every number traces back to the run and config
   that made it, and the ablation table is generated from it.
10. **Streamlit viewer and how to navigate it.** How to start it, what each control does, what the
    colours mean, and one or two screenshots with a short walk-through (pick image and runs, click
    predict, move the plane sliders, switch Labels/Errors, open 3D).
11. **Regulatory path: SaMD.** See the section below. Never claim compliance.
12. **Limitations.** Data, labels, split, evidence base (seeds, number of cases, sites), what was
    chosen after seeing test data, what can't be checked. Specific and with numbers.
13. **Future work.** What is needed next and why, in priority order, tied to the limitations and the
    error analysis.
14. **Quick start.** Five or so commands (install, data, cross-validation script, tests) in the README;
    every command, the run folder contents and the project structure go in `docs/usage.md`.
15. **Conclusion:** three or four numbered points a reader can stop at, ending with the next step.

## Regulatory path (SaMD)

A segmentation model that measures brain structures for clinical decisions would be Software as a
Medical Device (SaMD). A research project complies with none of the below; the README should say so in
a few lines, list what real deployment would require, and link to `docs/intended_use_and_risk.md` for the
intended-use statement and the risk table. Phrase it as "would need", not "meets". Check names and
versions with the user before publishing, since standards and guidance are revised.

The risk table follows ISO 14971's shape: hazard (cause), hazardous situation, possible harm, severity,
probability, control in the project, residual risk and what a device would need. Base every probability
on a result from the project (a percentage, a count, a Dice change), define the severity and probability
levels above the table, and say the levels are judgements, not a clinical review.

- **Classification:** EU MDR 2017/745 (software is usually Class IIa or higher under Rule 11), the FDA
  pathway (510(k) or De Novo), and the IMDRF SaMD risk categories.
- **Quality and risk:** ISO 13485 quality management, ISO 14971 risk management.
- **Software lifecycle:** IEC 62304 (and IEC 82304-1 for health software products); cybersecurity
  (for example IEC 81001-5-1) and usability (IEC 62366-1).
- **Machine-learning practice:** the FDA, Health Canada and MHRA Good Machine Learning Practice guiding
  principles; a predetermined change control plan if the model will be retrained.
- **EU AI Act:** medical-device AI is treated as high-risk.
- **Data protection:** GDPR or HIPAA for patient data.
- **Evidence:** a locked model, prospective or independent multi-site validation on the intended
  population and scanners, subgroup analysis, a defined intended use and user, and post-market
  monitoring.

## Writing rules

- **Numbers come from files.** Copy them from the result files and keep the decimals consistent. If a
  number isn't in a file, don't write it; say it's pending.
- **Plain words with numbers.** "External Dice drops from 0.897 to 0.811" beats "performance degrades".
  Give case counts, p-values and what was compared.
- **Name things the reader understands.** Use the model's public name (here "3D U-Net", not the run
  folder "baseline"); explain any metric the first time.
- **Figures are evidence.** Use test cases, not training cases, give every image alt text and a caption,
  and only link figures that are in the repository (tracked folders such as `docs/figures/`).
- **Main versus secondary.** Anything chosen after looking at test results (post-processing, box
  placement) is reported as secondary, and the README says why.
- **Keep the disclaimer** ("research use only, not for diagnosis") in the notice, the results and the
  conclusion.
- **Match the project's spelling and style** (this repo uses British spelling).
- **Don't change results to fit a story.** If an experiment failed its criteria, say it failed.
