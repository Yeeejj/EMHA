# INSIDE-OUT / EMHA — Systems Guide

**Emotion Recognition through Handwriting Analysis using a CNN-HMM Hybrid Approach**

Author: Cyrel Jane A. Edano | Adviser: Christian V. Maderazo, M.Eng.
Institution: University of San Carlos, DCISM
Psychometricians: Caira Wynn Louise B. Yntig RPm., Hazel T. Suquib RPm.,
  Peter John Poreta RPm., Alden M. Arda RPm.

**Authoritative source: ProcessPipeline.txt (19 phases P0-P18, July 2026)**

---

## Assessment: FINALE Thesis Assessment

A 3-part instrument administered in a single sitting (no time limit, no erasing):

| Part | Pages | Content |
|------|-------|---------|
| Questionnaire | 1 | 24-item Likert scale (1-5) — mixed positive/negative items |
| Drawing Exercise | 1 | 4 tasks: Overlapping Circles, Connect the Dots, Person in the Rain, House and Tree |
| Writing Exercise | 2 | 5 emotion-laden words × 3 handedness variants + 5 cursive sentence rows |

**Participant requirements:** Age 18-25, right-handed, complete all sections in one sitting.

---

## Labeling Scheme (Section A — Non-negotiable)

```
happiness_items  = Q2 Q4 Q6 Q8 Q10 Q12 Q13 Q16 Q19 Q20 Q21 Q23  (kept raw)
sadness_items    = Q1 Q3 Q5 Q7 Q9  Q11 Q14 Q15 Q17 Q18 Q22 Q24  (kept raw)

happiness_sum   = sum of 12 happiness items          range 12-60
sadness_sum     = sum of 12 sadness items            range 12-60
adjusted_total  = happiness_sum + (72 - sadness_sum) range 24-120
label           = HAPPY if adjusted_total >= 72 else SAD
p_happy         = (adjusted_total/24 - 1) / 4        stored only

NO NEUTRAL class.  All 400 respondents receive a binary label.
Integer comparisons only at the >= 72 boundary.
```

---

## 19-Phase Pipeline

### Phase 0 — Environment Setup
Clone repo, create venv, `pip install -r requirements.txt`. Run
`python -m src.utils.config` to confirm config import. Confirm Colab can
pull repo and mount Drive.

### Phase 1 — Participant Registration
`python -m src.data.register_participants`
Walk `DATASET/raw`, map `respondent_NNN` → `PNNN`. Verify barcodes with pyzbar.
Creates: `DATA/METADATA/participants.csv`

### Phase 2 — Page Role Assignment
`python -m src.data.assign_pages`
Assign page_questionnaire / page_drawing / page_writing roles by alphabetical sort.
Creates: `DATA/METADATA/page_manifest.csv`

### Phase 3 — Questionnaire Scoring (Gate 2)
`python -m src.data.questionnaire_scorer`
Score the 24-item FINALE per Section A. No NEUTRAL.
Creates: `DATA/METADATA/labels.csv`

### Phase 4 — Content Extraction (Gate 3)
`python -m src.data.content_extractor`
`WritingCellExtractor` cuts 24 crops per respondent using fractional bounding
boxes. `skip_skew=True` for draw_*, `False` for word_* and cursive_*.
Creates: `DATA/CROPS/{P###}/`, `DATA/METADATA/extraction_report.csv`

### Phase 5 — Label Propagation
`python -m src.data.propagate_labels`
Join extraction report with labels.csv. All labelled respondents included.
Creates: `DATA/METADATA/crop_index.csv`
  (crop_path, participant_id, task_code, task_type, label, p_happy, excluded)

### Phase 5B — Data Tabulation Close-out
Freeze Google Sheets self-report tab once all forms encoded. Sync
`ERHA_HappySad_Tabulation.xlsx`. Lock a dated snapshot. Confirm
sheet/xlsx/labels.csv agree row-for-row.

### Phase 6 — Quality Control
`python -m src.data.qc`
Per-crop checks: missing_file, corrupt_image, low_resolution, blank_content.
Exclusion is per-crop; one bad crop does not exclude the respondent.
Creates: updated `crop_index.csv`, `qc_report.csv`, `exclusions.csv`

### Phase 7 — Preprocessing
`python -m src.preprocessing.run_preprocessing`
Grayscale → Otsu binarization → morphological denoising → conditional deskew
→ 224×224 resize → normalize [0,1]. All params from config.py.
Creates: `DATA/PROCESSED/{P###}/` (mirrors CROPS)

### Phase 8 — Participant-Aware Splitting
`python -m src.data.participant_aware_splitter`
Stratified train/val/test (70/15/15) at the RESPONDENT level, then 5-fold CV
assignments within train. Fixed seed = 42. No file copying.
Creates: `DATA/METADATA/splits.json`

### Phase 9 — Bug Fix + Model Training (Gate 1 first)
Colab: `EMHA_Colab_Pipeline.ipynb`
(a) hybrid.py Gate 1 fix already applied.
(b) `use_pretrained=True` (ResNet18) already set in config.
(c) `CNNFeatureExtractor` → 256-dim spatial sequence → per-class GaussianHMM.
(d) Train logistic regression baseline on handcrafted graphological features.
Creates: trained checkpoints in `models/`, `results/training_log.csv`

### Phase 10 — Evaluation
Colab: Respondent-level stratified 5-fold CV using splits.json.
Metrics: accuracy, precision, recall, F1, confusion matrix, ROC-AUC.
Report per-task-type breakdown (drawings/words/cursive) + hybrid vs baseline.
Frame against 70-85% target.
Creates: `results/` tables + figures

### Phase 11 — End-to-End Inference
`python -m src.predict <path>`
Raw scanned pages → HAPPY/SAD + confidence. No questionnaire at inference.
Defense demo artifact.

### Phase 12 — Artifact Generation
`python -m src.utils.artifact_generator`
Confusion matrices, ROC curves, fold bars, dataset composition, Grad-CAM.
All at 300 DPI in `FIGURES/`.

### Phase 13 — Psychometrics
`python -m src.analysis.psychometrics`
Cronbach's alpha (overall + per subscale), item-total correlations, score
distribution. Supervised by four licensed psychometricians.
Creates: `DATA/METADATA/psychometrics_report.csv`

### Phase 14 — Smoke Test and Pilot
`python smoke_test.py`
Stages A-G on 3-5 respondents. Must pass before any scale-up.
Then: 30-respondent pilot to measure real exclusion rate vs pre-registered floor.
Then: full run.

### Phases 15-18 — Writing, Defense, Revisions, Submission
See ProcessPipeline.txt Sections D-E for details.

---

## Tool Routing

| Tool | Purpose |
|------|---------|
| Cursor + Claude Code | Local implementation (E:\EMHA_Thesis), git push |
| GitHub (Yeeejj/EMHA) | Version control, Colab pulls from here |
| Google Colab | Heavy work: CNN training, HMM fitting, 5-fold CV, Grad-CAM |
| Google Sheets | Self-report encoding via Google Form + Apps Script bridge |
| Local Excel | ERHA_HappySad_Tabulation.xlsx — mirrors the sheet |
| Claude.ai Project | Planning, documents, prompts, decisions |

---

## Architecture Summary

```
Scanned page
    ↓
WritingCellExtractor (24 crops, fractional bounding boxes)
    ↓
PreprocessingPipeline (224×224, binarize, denoise, conditional deskew)
    ↓
PretrainedCNNExtractor (ResNet18, 256-dim features)
    ↓
HMMClassifier (per-class GaussianHMM, 4 states, diag covariance)
    ↓
HAPPY / SAD + confidence
```

Baseline: Logistic regression on handcrafted features (mean intensity,
pixel density, slant angle) — thesis comparison only.

---

## Critical Path (do these next, in order)

1. `python smoke_test.py` — validates Gates 1-3 without data
2. `python -m src.data.questionnaire_scorer` — after encoding responses
3. `python -m src.data.content_extractor` — tune DRAWING/WORD/CURSIVE_CROPS
4. Pilot run (30 respondents) — measure exclusion rate vs floor
5. Full run, Colab training, evaluation, predict.py demo
6. Chapters 5-6, defense, revisions, binding

*Last updated: July 2026*
