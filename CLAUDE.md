# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

INSIDE-OUT — Emotion recognition (HAPPY vs SAD) from offline scanned handwriting and drawing samples using a hybrid CNN-HMM architecture. Thesis project by Cyrel Jane A. Edano at University of San Carlos, DCISM. Target: 70-85% F1 macro on ~400 right-handed participants aged 18-25. Primary metric is F1 macro, NOT accuracy (class imbalance expected after NEUTRAL exclusions).

## Critical Rules

1. READ-ONLY RAW DATA: The original raw dataset is at E:\EMHA_Thesis\DATASET\raw. NEVER write to, rename, move, or modify anything inside it. All pipeline work happens on copies in DATA\STAGING\.

2. TEST SPLIT LOCK: Never read or evaluate on DATA/SPLITS/test/ until final evaluation (Phase 10). Use only train/ and val/ during development.

3. KNOWN BUG: src/models/hybrid.py calls HMMClassifier with an unsupported n_features argument. HMMClassifier.__init__ in hmm.py accepts only n_states, n_iter, covariance_type. This must be fixed before any training run.

4. MODEL PRIORITY: Use ResNet18 backbone (use_pretrained=True in CNNConfig), not the custom 4-block CNN, given dataset size.

5. Before making changes, check the latest documentation for any library API you are unsure about. Only implement when confident it will work.

6. PyTorch only — no TensorFlow.

## Raw Dataset Structure

E:\EMHA_Thesis\DATASET\raw\        (READ-ONLY, never touched)
  respondent_001\
    PNG\    (exactly 4 PNG files)
    PDF\    (4 PDF files, archival backup only)
  respondent_002\
  ...

File naming: EMHA{YYYYMMDD}_{HHMMSSff}.ext
Alphabetical sort = chronological page order.
Page positions: index 0 = cover, 1 = selfreport, 2 = drawing, 3 = writing.
Barcode rule: respondent_001 becomes P001 (strip prefix, zero-pad to 3 digits, prepend P).

## Assessment Structure (FINALE)

Page 1 (cover): instructions, age and gender fields, signatures — not a training sample.
Page 2 (selfreport): 24-item Likert grid (1-5) — ground truth label source only.
Page 3 (drawing): 2x2 grid — circles (top-left), dots (top-right, has pre-printed dot stimuli), person in rain (bottom-left), house and tree (bottom-right).
Page 4 (writing): 5x3 word table (Content, Melancholic, Optimistic, Disconnected, Vibrant in rows; left hand, right hand, uppercase in columns) plus 5 cursive sentence rows, each with a pre-printed prompt.

24 crops per participant total: 4 drawings + 15 word cells + 5 cursive cells.

## Task Codes

draw_circles, draw_dots, draw_person, draw_house
word_content_left, word_content_right, word_content_upper
word_melancholic_left, word_melancholic_right, word_melancholic_upper
word_optimistic_left, word_optimistic_right, word_optimistic_upper
word_disconnected_left, word_disconnected_right, word_disconnected_upper
word_vibrant_left, word_vibrant_right, word_vibrant_upper
cursive_01, cursive_02, cursive_03, cursive_04, cursive_05

Output filename format: {participant_id}_{task_code}.png
skip_skew = True for all draw_* task codes (drawings have no text baseline).
skip_skew = False for all word_* and cursive_* task codes.

## Data Pipeline

DATASET\raw (read-only)
  -> copy -> DATA/STAGING/
  -> validate_intake -> validation_report.csv
  -> register_participants -> participants.csv
  -> assign_pages -> page_manifest.csv
  -> score_questionnaire -> labels.csv
  -> extract_content -> DATA/EXTRACTED/{P###}/ (24 crops each)
  -> propagate_labels -> samples_manifest.csv
  -> validate_quality -> quality_report.csv
  -> run_preprocessing -> DATA/PROCESSED/{HAPPY|SAD}/
  -> split_dataset (PARTICIPANT-LEVEL split) -> DATA/SPLITS/{train|val|test}/
  -> trainer -> models/ and results/training_log.csv
  -> evaluator -> results/ (test split, first and only touch)

Participant-level splitting is a hard constraint: all 24 files from one participant must land in the same split. File-level splitting causes data leakage.

## Commands

pip install -r requirements.txt

Pipeline stages in order (run as modules from project root):

python -m src.data.validate_intake             (Phase 0: copy raw to staging, validate)
python -m src.data.register_participants       (Phase 1: barcode ID registry)
python -m src.data.assign_pages                (Phase 2: page role manifest)
python -m src.data.score_questionnaire         (Phase 3: scoring and labels)
python -m src.data.extract_content             (Phase 4: 24 crops per participant)
python -m src.data.propagate_labels            (Phase 5: samples manifest)
python -m src.data.validate_quality            (Phase 6: quality checks)
python -m src.preprocessing.run_preprocessing  (Phase 7: preprocess images)
python -m src.data.split_dataset               (Phase 8: participant-level splits)
python -m src.training.trainer                 (Phase 9: train CNN-HMM and LR baseline)
python -m src.training.evaluator               (Phase 10: final evaluation)

Tests and linting:

pytest
black .
flake8

The C files at the root (pharmacy_queue.c, queue_sorting.c, stack_sorting.c, warehouse_inventory.c) are standalone data structure exercises, not part of the ML pipeline.

## Architecture

Two-stage hybrid model in src/:

CNN (src/models/cnn.py): ResNet18 backbone via PretrainedCNNExtractor extracts 256-dim spatial features from 224x224 grayscale images. The custom 4-block CNN exists but is secondary.

HMM (src/models/hmm.py): 4-state diagonal-covariance GaussianHMMs, one per emotion class. Classification by log-likelihood comparison.

Hybrid (src/models/hybrid.py): HybridCNNHMM orchestrates CNN -> spatial sequence (batch, 28, 256) -> HMM -> prediction with confidence.

Baseline: logistic regression on handcrafted features (mean intensity, pixel density, slant angle) for thesis comparison.

## Labeling (FINALE 24-item self-report)

happiness_items = [2, 4, 6, 8, 10, 12, 13, 16, 19, 20, 21, 23]
sadness_items   = [1, 3, 5, 7, 9, 11, 14, 15, 17, 18, 22, 24]

Reverse-score sadness items: new = 6 - original.
Both composites range 12-60.
Thresholds live in config.py LabelingConfig (confirm with psychometrician panel before finalizing).

HAPPY:   happiness_score >= happiness_high AND sadness_score < sadness_threshold
SAD:     sadness_score >= sadness_threshold OR happiness_score < happiness_low
NEUTRAL: otherwise — excluded from training, files retained in EXTRACTED/

## Configuration

All hyperparameters in src/utils/config.py via dataclasses. Import with: from src.utils.config import config

Key entries:
raw_data_dir = "E:\\EMHA_Thesis\\DATASET\\raw"   (READ-ONLY)
staging_dir = "DATA/STAGING"
Crop coordinate dicts: DRAWING_CROPS, WORD_CROPS, CURSIVE_CROPS
skip_skew mapping per task code
224x224 images, batch_size=32, lr=0.001, 100 epochs, 5-fold CV, random_state=42, splits 70/15/15

## Key Files

DATA/METADATA/ — all pipeline CSVs (validation_report, participants, page_manifest, questionnaire_scores, labels, extraction_log, samples_manifest, quality_report, preprocessing_log, split_manifest)
SYSTEMS_GUIDE.md — detailed documentation of all 23 systems
notebooks/EMHA_Colab_Pipeline.ipynb — Google Colab training pipeline

## Git Rules

Never commit image data. .gitignore must exclude: DATASET/, DATA/STAGING/, DATA/EXTRACTED/, DATA/PROCESSED/, DATA/SPLITS/.
Only source code and DATA/METADATA/*.csv go to GitHub.

## Dependencies

Core: PyTorch, hmmlearn, OpenCV, scikit-learn, pandas, numpy, matplotlib, PIL. See requirements.txt.