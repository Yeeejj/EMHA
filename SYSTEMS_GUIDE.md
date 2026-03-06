# INSIDE-OUT: Complete Systems Guide

## Emotion Recognition through Handwriting & Drawing Analysis using CNN-HMM Hybrid Approach

**Author:** Cyrel Jane A. Edano | **Adviser:** Christian V. Maderazo, M.Eng.
**Institution:** University of San Carlos, DCISM
**Psychometricians:** Caira Wynn Louise B. Yntig RPm., Hazel T. Suquib RPm., Peter John Poreta RPm., Alden M. Arda RPm.
**Systems Count:** 23 | **Phases:** 10
**Target:** 80-85% accuracy classifying happiness vs. sadness from handwriting and drawing samples

---

## Assessment Overview

The INSIDE-OUT FINALE Thesis Assessment is a **3-part instrument** administered in a single sitting:

| Part | Type | Content |
|------|------|---------|
| **Part 1** | Self-Report | 24-item Likert scale (1-5) measuring emotional states — mix of positive items ("I feel cheerful most of the time") and negative items ("I feel emotionally drained") |
| **Part 2** | Drawing Exercise | 4 tasks: (1) Draw Three Overlapping Circles, (2) Connect the Dots, (3) Draw a Person in the Rain, (4) Draw a House and a Tree |
| **Part 3** | Writing Exercise | Write 5 emotion-laden words (Content, Melancholic, Optimistic, Disconnected, Vibrant) with LEFT hand, RIGHT hand, and UPPERCASE; Write 5 sentences in CURSIVE |

**Participant Requirements:** Age 18-25, right-handed, complete all sections in one sitting, no time limit, no erasing.

---

## Table of Contents

- [Phase 1: Data Infrastructure](#phase-1-data-infrastructure)
- [Phase 2: Data Management](#phase-2-data-management)
- [Phase 3: File Organization](#phase-3-file-organization)
- [Phase 4: Image Processing](#phase-4-image-processing)
- [Phase 5: Feature Engineering](#phase-5-feature-engineering)
- [Phase 6: Labeling & Annotation](#phase-6-labeling--annotation)
- [Phase 7: Model Development](#phase-7-model-development)
- [Phase 8: Validation & Monitoring](#phase-8-validation--monitoring)
- [Phase 9: Research & Reporting](#phase-9-research--reporting)
- [Phase 10: Deployment & Visualization](#phase-10-deployment--visualization)
- [Complete System Flow Diagram](#complete-system-flow-diagram)
- [Quick Reference Table](#quick-reference-table)

---

## Phase 1: Data Infrastructure

### System 1: Barcode Registry & Auto-Update

**Summary:**
Generates and manages unique participant identifiers (P001, P002, ..., P300+) that link every assessment page — self-report, drawing exercise, and writing exercise — back to a single individual. The registry auto-increments IDs and prevents duplicates, ensuring traceability across all three assessment parts. Implemented via `DataCollector.generate_participant_id()` in `src/data/collector.py`.

**Problems & Errors:**

| # | Problem | Description |
|---|---------|-------------|
| 1 | ID collision | Two participants assigned the same ID if the counter file is corrupted or read concurrently |
| 2 | Gap in sequence | Deleted or abandoned entries leave holes (P001, P003) that confuse downstream counts |
| 3 | Lost mapping | If `participants.csv` is accidentally overwritten, the link between barcode and participant's 3-part assessment is severed |
| 4 | Format inconsistency | Manual entries may use different padding (P1 vs P001) breaking filename patterns |
| 5 | Multi-page tracking | Each participant produces at least 4 pages (self-report, drawing, writing words, writing sentences); pages can get separated from their ID |

**Solutions:**

| # | Solution |
|---|----------|
| 1 | Use file-lock or atomic write when updating the counter; validate uniqueness against `participants.csv` before committing |
| 2 | Never reuse IDs — treat gaps as intentional; add a `status` column (active/withdrawn) instead of deleting rows |
| 3 | Keep automatic backups of `DATA/METADATA/participants.csv` with timestamps; version-control the metadata folder |
| 4 | Enforce zero-padded 3-digit format in `generate_participant_id()` with strict regex validation `^P\d{3}$` |
| 5 | Print participant ID on every page of the assessment form; use page numbering (e.g., P001_page1, P001_page2) |

**Positive Outcome:**
Every self-report score, drawing sample, and writing sample is permanently traceable to a verified participant, satisfying both research reproducibility and ethical audit requirements across all three assessment parts.

---

### System 2: Data Quality Control

**Summary:**
Validates incoming data from all three assessment parts before they enter the pipeline. For Part 1 (Self-Report): checks completeness of all 24 items and score ranges (1-5). For Part 2 (Drawing): checks image resolution and that all 4 drawings are present. For Part 3 (Writing): verifies all 5 words are written in all 3 styles (left, right, uppercase) and all 5 cursive sentences are present. Prevents garbage-in-garbage-out by rejecting or flagging substandard inputs.

**Problems & Errors:**

| # | Problem | Description |
|---|---------|-------------|
| 1 | Low-resolution scans | Images below usable DPI produce unreadable features after preprocessing |
| 2 | Incomplete self-report | Missing items from the 24-item questionnaire make scoring unreliable |
| 3 | Missing drawing tasks | Participant skips one of the 4 drawing exercises (e.g., Person in the Rain) |
| 4 | Missing writing variants | Participant omits the left-hand or uppercase version of words |
| 5 | Corrupted files | Truncated PNGs or zero-byte files crash the preprocessing pipeline |
| 6 | Class imbalance | Uneven HAPPY/SAD distribution degrades model generalization |

**Solutions:**

| # | Solution |
|---|----------|
| 1 | Set minimum resolution threshold (300 DPI or 1000x1000 px); reject and re-scan if below |
| 2 | Check all 24 items have a response in range 1-5 before accepting; flag partial self-reports for follow-up |
| 3 | Require all 4 drawing tasks complete before processing; mark incomplete sets for re-collection |
| 4 | Validate that each of the 5 words (Content, Melancholic, Optimistic, Disconnected, Vibrant) has left-hand, right-hand, and uppercase entries |
| 5 | Validate file integrity (check PNG header bytes, file size > 0) on ingestion |
| 6 | Monitor class distribution regularly; recruit targeted participants or apply class-weighted loss |

**Positive Outcome:**
Only clean, complete, and valid data from all three assessment parts enters the pipeline, producing reliable emotion labels and preventing downstream failures during preprocessing, training, and evaluation.

---

## Phase 2: Data Management

### System 3: Participant Management

**Summary:**
Handles participant registration, consent tracking, demographic recording (age, gender), and assessment association. Stores participant profiles in `DATA/METADATA/participants.csv`. Only right-handed participants aged 18-25 are eligible per the study protocol. Tracks completion status of all three assessment parts per participant. The assessment is administered under the supervision of 4 licensed psychometricians (RPm).

**Problems & Errors:**

| # | Problem | Description |
|---|---------|-------------|
| 1 | Ineligible participants | Left-handed or out-of-age-range participants slip through screening |
| 2 | Missing consent | Samples collected without documented consent (signed by researcher + adviser) violate ethics approval |
| 3 | Duplicate enrollment | Same person registers twice under different IDs |
| 4 | Incomplete assessment | Participant completes Part 1 and Part 2 but not Part 3; partial data is hard to use |
| 5 | CSV encoding issues | Special characters in names corrupt the CSV when opened across platforms |

**Solutions:**

| # | Solution |
|---|----------|
| 1 | Add eligibility checks in `create_participant_folder()`: assert `handedness == 'R'` and `18 <= age <= 25` |
| 2 | Make `consent_date` a required non-null field; block folder creation until consent form is signed by researcher (Cyrel Jane A. Edano) and adviser (Christian V. Maderazo) |
| 3 | Add a check against existing demographics (age + gender + consent_date) to flag potential duplicates |
| 4 | Track per-part completion: `part1_complete`, `part2_complete`, `part3_complete` columns; only process fully complete assessments |
| 5 | Enforce UTF-8 encoding on all CSV reads/writes; avoid special characters in the notes field |

**Positive Outcome:**
A clean, ethics-compliant participant registry that ensures every sample comes from a consented, eligible individual, with all three assessment parts accounted for, satisfying IRB requirements and strengthening research validity.

---

### System 4: Database System

**Summary:**
The structured CSV-based metadata system in `DATA/METADATA/` acts as the project's database layer. Core tables — `participants.csv`, `questionnaire_scores.csv`, and `labels.csv` — are joined by `participant_id` as the primary key. Extended to track data from all three assessment parts: 24-item self-report scores, 4 drawing exercise files, and writing exercise files (3 handedness variants + cursive sentences). Managed via pandas DataFrames.

**Problems & Errors:**

| # | Problem | Description |
|---|---------|-------------|
| 1 | Referential integrity | A label references a participant_id that doesn't exist in participants.csv |
| 2 | Concurrent writes | Two processes writing to the same CSV simultaneously cause data loss |
| 3 | Schema drift | New columns added inconsistently across CSVs break downstream readers |
| 4 | No transaction support | A crash mid-write leaves the CSV in a partial/corrupted state |
| 5 | Multi-part complexity | Each participant has self-report data, drawing data, and writing data spread across multiple records |

**Solutions:**

| # | Solution |
|---|----------|
| 1 | Validate foreign keys on every write: confirm participant_id exists in `participants.csv` before inserting into any other table |
| 2 | Use file-locking (e.g., `fcntl.flock`) or single-process write access to metadata files |
| 3 | Define schemas in `config.py` with expected column names and types; validate on load |
| 4 | Write to a temporary file first, then atomically rename to the target path |
| 5 | Use a consistent naming scheme linking sample type: `{P###}_{part}_{task}_{variant}.png` |

**Positive Outcome:**
A lightweight, portable, version-controllable data store that keeps participant records, scores, and labels consistent and queryable without requiring a database server, while cleanly organizing data from the 3-part assessment.

---

## Phase 3: File Organization

### System 5: File Sorter

**Summary:**
Organizes raw assessment scans from `DATA/RAW/` into the appropriate directory structure. Separates the three assessment parts: self-report pages go to scoring, drawing exercises go to `DATA/LABELED/{HAPPY|SAD}/drawings/`, and writing exercises go to `DATA/LABELED/{HAPPY|SAD}/writing/`. Manages the flow through `DATA/PROCESSED/` and into `DATA/SPLITS/` (train/val/test). Uses the naming convention `{participant_id}_sample_{number}_{date}.png`.

**Problems & Errors:**

| # | Problem | Description |
|---|---------|-------------|
| 1 | Misrouted files | A drawing exercise is classified as writing, or vice versa, corrupting the training data |
| 2 | Filename collision | Two samples with identical names overwrite each other during sorting |
| 3 | Orphaned files | Samples in RAW/ with no matching label record are never sorted |
| 4 | Part misidentification | Self-report page sorted into drawing folder due to similar visual layout |

**Solutions:**

| # | Solution |
|---|----------|
| 1 | Tag files with part identifiers during scanning: `P001_part2_circles.png`, `P001_part3_cursive.png` |
| 2 | Append a unique suffix or timestamp if filename already exists in destination |
| 3 | Run an orphan-detection scan: compare RAW/ filenames against labels.csv and report unmatched files |
| 4 | Use page headers ("Part 2: The Drawing Exercise", "Part 3: The Writing Exercise") for template-based classification |

**Positive Outcome:**
A clean, deterministic directory structure where every self-report, drawing, and writing sample is in exactly the right place, enabling the dataloader and preprocessing pipeline to operate without path errors or label mismatches.

---

### System 6: Page Classifier & Auto-Renamer

**Summary:**
Identifies and classifies scanned assessment pages by part and task type. Distinguishes between the cover/instructions page, Part 1 (self-report grid), Part 2 (drawing boxes — circles, dots, person, house/tree), and Part 3 (writing grid for words + cursive lines). Automatically renames files to follow the project naming convention. Handles multi-page scans where a single scan may contain multiple pages.

**Problems & Errors:**

| # | Problem | Description |
|---|---------|-------------|
| 1 | Misclassification | A Part 2 drawing page is classified as Part 3 writing and enters the wrong analysis pipeline |
| 2 | Page order errors | Multi-page scans are split but reassembled in wrong order |
| 3 | Drawing quadrant confusion | The 4 drawing tasks share one page; splitting into individual tasks fails at the quadrant boundaries |
| 4 | Unsupported formats | TIFF, BMP, or HEIC scans are not handled by the renamer |

**Solutions:**

| # | Solution |
|---|----------|
| 1 | Use template matching on section headers ("Part 2: The Drawing Exercise") to distinguish pages; require human verification for low-confidence classifications |
| 2 | Embed page sequence numbers in filenames; sort by scan timestamp before splitting |
| 3 | Use fixed grid coordinates to split the drawing page into 4 quadrants (top-left: circles, top-right: dots, bottom-left: person, bottom-right: house/tree) |
| 4 | Convert all inputs to PNG on ingestion using Pillow; support common formats via `Image.open()` |

**Positive Outcome:**
Scanned assessment pages are automatically organized by part and task, correctly classified, and consistently named, eliminating manual file management and reducing human error in multi-page assessment processing.

---

## Phase 4: Image Processing

### System 7: Data Extraction & Content Analyzer

**Summary:**
Extracts meaningful content regions from raw assessment scans. For Part 2 drawings: isolates each of the 4 drawing tasks from their quadrant boxes. For Part 3 writing: isolates each word (Content, Melancholic, Optimistic, Disconnected, Vibrant) in its 3 variants (left-hand, right-hand, uppercase) and each of the 5 cursive sentences. Identifies content bounding boxes and crops to regions of interest.

**Problems & Errors:**

| # | Problem | Description |
|---|---------|-------------|
| 1 | Over-cropping | Part of a drawing (e.g., tree branches extending to edge) is cut off |
| 2 | Under-cropping | Grid lines and printed headers remain, adding noise to feature extraction |
| 3 | Empty crops | Blank drawing boxes or very faint writing produce empty content regions |
| 4 | Word-level segmentation | The 3-column writing grid (left/right/uppercase) is misaligned, mixing variants |
| 5 | Cursive line bleed | Long cursive sentences overlap with adjacent sentence areas |

**Solutions:**

| # | Solution |
|---|----------|
| 1 | Add padding (10-20 px) around detected content bounding boxes before cropping |
| 2 | Use contour detection with area thresholds to filter out printed grid lines and headers |
| 3 | Set a minimum ink-pixel ratio threshold; flag boxes below it for re-collection |
| 4 | Use the printed column headers ("Left Hand", "Right Hand", "Uppercase") as alignment anchors for column detection |
| 5 | Detect horizontal ruled lines between sentence areas; use them as splitting boundaries |

**Positive Outcome:**
Clean, focused content regions from each assessment task — individual drawings and individual writing samples — giving the CNN feature extractor consistent and informative inputs that directly represent the participant's expressive output.

---

### System 8: Image Preprocessor

**Summary:**
The 5-stage preprocessing pipeline in `src/preprocessing/pipeline.py` transforms raw scans into model-ready images:
1. **Grayscale conversion** — RGB to single channel
2. **Binarization** — Otsu's thresholding separates ink from background
3. **Denoising** — Morphological opening/closing with 3x3 kernel removes speckles
4. **Skew correction** — Rotates to straighten tilted content
5. **Normalization** — Resizes to 224x224, scales pixel values to [0, 1]

Applied to both drawing exercises (circles, dots, person, house/tree) and writing exercises (left-hand, right-hand, uppercase, cursive).

**Problems & Errors:**

| # | Problem | Description |
|---|---------|-------------|
| 1 | Otsu failure | Uniform images (very light or very dark) cause Otsu's method to pick a poor threshold |
| 2 | Over-denoising | Aggressive morphological operations erode thin strokes, destroying pressure information in drawings |
| 3 | Skew over-correction | The algorithm detects the wrong dominant angle in non-text content like drawings |
| 4 | Aspect ratio distortion | Resizing non-square drawing quadrants to 224x224 stretches the artwork |
| 5 | Drawing vs. writing differences | Drawings (circles, person, house) have fundamentally different visual characteristics than cursive text |

**Solutions:**

| # | Solution |
|---|----------|
| 1 | Fall back to adaptive thresholding (Gaussian) when Otsu variance is below a minimum; configurable via `PreprocessingConfig.binarize_threshold` |
| 2 | Use a small kernel (3x3) and limit morphological iterations; use gentler settings for drawings than for text |
| 3 | Skip skew correction for drawing tasks (circles, person, house/tree); apply only to writing samples |
| 4 | Pad to square with white background before resizing to preserve aspect ratio |
| 5 | Use separate preprocessing profiles: one for drawings (gentler denoising, no skew correction) and one for writing (full pipeline) |

**Positive Outcome:**
Standardized 224x224 grayscale images with clean presentation for both drawings and writing, enabling the CNN to learn features without being distracted by scan artifacts, skew, or resolution differences.

---

### System 9: Data Augmentation

**Summary:**
Expands the training dataset by applying controlled transformations to existing samples, addressing class imbalance and improving model generalization. Applied only during training (not validation/test). For writing samples: random rotation (+-10 degrees), slight scaling, brightness/contrast jitter, and elastic distortion. For drawing samples: more conservative augmentation to preserve spatial relationships (slight rotation, minor scaling only).

**Problems & Errors:**

| # | Problem | Description |
|---|---------|-------------|
| 1 | Unrealistic augmentations | Flipping a "Draw a Person in the Rain" image produces an unrealistic inverted drawing |
| 2 | Label leakage | Augmented samples from validation participants leak into training set |
| 3 | Over-augmentation | Too many synthetic samples drown out real data patterns |
| 4 | Cross-task contamination | Augmentation parameters tuned for writing samples distort drawing samples |

**Solutions:**

| # | Solution |
|---|----------|
| 1 | Disable horizontal/vertical flips for drawing tasks; limit rotation to +-5 degrees for drawings vs. +-10 degrees for writing |
| 2 | Split by participant_id first, then augment only the training split — all samples from one participant stay together |
| 3 | Cap augmented-to-real ratio at 2:1 or 3:1; monitor validation metrics for overfitting |
| 4 | Define separate augmentation pipelines: `drawing_transforms` and `writing_transforms` |

**Positive Outcome:**
A larger, more diverse training set that helps the CNN-HMM model generalize to unseen handwriting and drawing styles, reducing overfitting while respecting the different characteristics of drawing vs. writing tasks.

---

## Phase 5: Feature Engineering

### System 10: Feature Extractor

**Summary:**
The `CNNFeatureExtractor` in `src/models/cnn.py` transforms preprocessed 224x224 images into compact 256-dimensional feature vectors. Architecture: 4 convolutional blocks (Conv2d -> BatchNorm -> ReLU -> MaxPool) progressing from 1 -> 32 -> 64 -> 128 -> 256 channels, followed by AdaptiveAvgPool and a fully connected layer. Processes both drawing and writing samples into a unified feature space. Optionally uses pretrained ResNet18/VGG backbones via `PretrainedCNNExtractor`.

**Problems & Errors:**

| # | Problem | Description |
|---|---------|-------------|
| 1 | Vanishing gradients | Deep CNN layers stop learning due to gradient decay |
| 2 | Overfitting | 256-dimensional features memorize training data with small dataset (300 participants, multiple samples each) |
| 3 | Feature collapse | All samples map to similar feature vectors, losing discriminative power between HAPPY and SAD |
| 4 | Pretrained mismatch | ImageNet-pretrained weights expect 3-channel RGB; assessment scans are 1-channel grayscale |
| 5 | Multi-task features | Drawing features (spatial/structural) differ fundamentally from writing features (stroke-based) |

**Solutions:**

| # | Solution |
|---|----------|
| 1 | Use batch normalization (already included) and skip connections; monitor gradient norms during training |
| 2 | Apply dropout (0.5 rate from `CNNConfig`), weight decay, and data augmentation to regularize |
| 3 | Visualize feature distributions with t-SNE/UMAP; ensure HAPPY and SAD clusters are separable |
| 4 | Replicate grayscale to 3 channels for pretrained models; or fine-tune only later layers |
| 5 | Consider separate CNN branches for drawing and writing, merging at the feature level before HMM classification |

**Positive Outcome:**
Rich, compact 256-dimensional feature vectors that capture both drawing characteristics (circle symmetry, figure completeness, spatial organization) and writing characteristics (stroke width, slant, spacing, pressure) for HMM emotion classification.

---

### System 11: Micro-Level Letter & Drawing Analysis

**Summary:**
Analyzes fine-grained characteristics at the individual element level across all assessment tasks:

**Writing Analysis (Part 3):** Measures 5 key features linked to emotional states:

| Feature | Happy Indicators | Sad Indicators |
|---------|------------------|----------------|
| Baseline | Rising upward | Descending downward |
| Slant | Right-leaning (positive) | Left-leaning or vertical |
| Pressure | Heavy, consistent | Light, inconsistent |
| Size | Large letters | Small letters |
| Spacing | Wide, open | Narrow, cramped |

**Drawing Analysis (Part 2):** Analyzes structural and spatial features:
- **Overlapping Circles:** Symmetry, size consistency, overlap precision
- **Connect the Dots:** Line straightness, connection accuracy, stroke pressure
- **Person in the Rain:** Figure size, umbrella presence, rain density, figure placement
- **House and Tree:** Proportions, detail level, spatial organization, groundline presence

**Left-Hand vs. Right-Hand Comparison (Part 3):** Measures motor control differences that may correlate with emotional state.

**Problems & Errors:**

| # | Problem | Description |
|---|---------|-------------|
| 1 | Segmentation failure | Individual letters cannot be reliably isolated from cursive sentences |
| 2 | Pressure estimation | Scanned images lack true pressure data — only intensity approximation is available |
| 3 | Drawing interpretation subjectivity | "Person in the Rain" analysis requires subjective feature identification (umbrella, rain, etc.) |
| 4 | Left-hand baseline | No established baseline for left-hand writing features in emotion research |
| 5 | Cultural variation | Drawing conventions differ across cultural backgrounds |

**Solutions:**

| # | Solution |
|---|----------|
| 1 | For cursive: measure features at the word/sentence level; for individual words: use the isolated word cells from Part 3 grid |
| 2 | Use grayscale intensity as a pressure proxy; document this limitation in the thesis |
| 3 | Define objective, measurable features (figure height ratio, umbrella area, rain stroke count) rather than subjective interpretations |
| 4 | Use left-hand writing as a comparative control rather than a primary feature source; focus analysis on right-hand and cursive |
| 5 | Control by limiting participants to similar educational background at University of San Carlos |

**Positive Outcome:**
Quantitative measurements of handwriting and drawing features across all assessment tasks, providing interpretable evidence that complements the CNN's black-box feature extraction and supports multi-modal emotion classification.

---

## Phase 6: Labeling & Annotation

### System 12: Emotion Labeler

**Summary:**
The `EmotionLabeler` in `src/data/labeler.py` assigns ground-truth emotion labels to all of a participant's samples (drawings + writing) based on Part 1 Self-Report scores. The 24-item self-report uses a 5-point Likert scale (1 = Strongly Disagree to 5 = Strongly Agree) with reverse-scored negative items. Combined with DASS-21 depression subscale thresholds, the labeling rules from `LabelingConfig` are:
- **HAPPY**: happiness_score >= 40.0 AND dass_depression < 14.0
- **SAD**: happiness_score < 30.0 OR dass_depression >= 14.0
- **NEUTRAL**: all other cases (excluded from training)

The 24 self-report items include:
- Positive: "I feel enthusiastic about my activities", "I laugh easily", "I feel cheerful most of the time", "I feel grateful for what I have"
- Negative: "I feel emotionally drained", "I feel a sense of emptiness", "I feel like crying", "I feel disconnected from others"

**Problems & Errors:**

| # | Problem | Description |
|---|---------|-------------|
| 1 | Threshold sensitivity | Small changes in threshold values (e.g., 40 vs 38) significantly change class assignments |
| 2 | Neutral exclusion | Too many participants fall in the neutral zone, reducing usable dataset size |
| 3 | Self-report bias | Participants may not honestly report their emotional state on the 24-item questionnaire |
| 4 | Reverse scoring errors | Negative items ("My thoughts tend to be more negative than positive") scored in wrong direction |
| 5 | Mixed signals | Self-report indicates HAPPY but drawing/writing features suggest SAD |
| 6 | Temporal mismatch | Emotional state during Part 1 may shift by Part 3 if assessment takes too long |

**Solutions:**

| # | Solution |
|---|----------|
| 1 | Validate thresholds against published norms; perform sensitivity analysis across threshold ranges |
| 2 | Adjust thresholds to maximize usable samples while maintaining clear class separation; report neutral exclusion rate |
| 3 | The assessment instructions emphasize honesty ("Be honest — there are no right or wrong answers"); administered under psychometrician supervision |
| 4 | Clearly document which items are reverse-scored; automate reverse scoring in code to prevent manual errors |
| 5 | Use self-report as the ground truth label; note discrepancies as potential research findings rather than errors |
| 6 | Assessment instructions require completion "in one sitting"; psychometricians monitor pacing |

**Positive Outcome:**
Psychologically grounded, reproducible emotion labels derived from a validated 24-item self-report instrument, providing reliable ground truth for supervised learning across all drawing and writing samples, administered under professional psychometric supervision.

---

## Phase 7: Model Development

### System 13: CNN-HMM Model Trainer

**Summary:**
The `Trainer` class in `src/training/trainer.py` and `HybridCNNHMM` in `src/models/hybrid.py` implement the two-stage training pipeline:
1. **Stage 1 — CNN Training**: Train the CNN feature extractor using cross-entropy loss, Adam optimizer (lr=0.001), batch size 32, up to 50 epochs with early stopping (patience=10). Processes both drawing and writing samples.
2. **Stage 2 — HMM Training**: Extract 256-dim features from trained CNN, fit separate Gaussian HMMs (4 states, diagonal covariance) for HAPPY and SAD classes.

Integrated in the Colab pipeline (`notebooks/EMHA_Colab_Pipeline.ipynb`) with 5-fold stratified cross-validation.

**Problems & Errors:**

| # | Problem | Description |
|---|---------|-------------|
| 1 | CNN overfitting | Even with 300 participants, multiple samples per person share writing style, causing memorization |
| 2 | HMM convergence failure | EM algorithm fails to converge with insufficient or poorly distributed features |
| 3 | Stage mismatch | CNN features optimized for classification may not be optimal for HMM modeling |
| 4 | GPU memory overflow | Large batch sizes or high-resolution images exhaust Colab GPU memory |
| 5 | Training instability | Loss spikes or NaN values during CNN training |
| 6 | Multi-modal input | Drawings and writing have different feature characteristics that confuse a single model |

**Solutions:**

| # | Solution |
|---|----------|
| 1 | Apply dropout (0.5), weight decay, data augmentation, and early stopping; split by participant not by sample |
| 2 | Increase HMM iterations (`n_iter=100`), try different covariance types (full, tied), ensure sufficient samples per class |
| 3 | Fine-tune: after HMM training, optionally adjust CNN features based on HMM classification performance |
| 4 | Reduce batch size to 16 or 8; use mixed-precision training (torch.cuda.amp) |
| 5 | Use gradient clipping (max_norm=1.0), learning rate warmup, and robust optimizer settings |
| 6 | Consider task-specific feature weighting or separate CNN branches for drawings vs. writing |

**Positive Outcome:**
A trained hybrid model that combines the CNN's spatial feature learning with the HMM's sequential pattern modeling, achieving the target 80-85% accuracy on emotion classification from the multi-modal assessment data.

---

### System 14: Model Validator & Tester

**Summary:**
The `Evaluator` in `src/training/evaluator.py` and `CrossValidator` in `src/training/cross_validate.py` rigorously assess model performance. Implements 5-fold stratified cross-validation ensuring balanced class distribution per fold. Computes accuracy, precision, recall, F1-score (macro-averaged), and confusion matrix. Final evaluation on held-out test set (15% of data) that was never seen during training or validation.

**Problems & Errors:**

| # | Problem | Description |
|---|---------|-------------|
| 1 | Data leakage | Same participant's samples (drawings + writing) appear in both train and test folds |
| 2 | Fold variance | Large performance variation across folds indicates instability |
| 3 | Metric misinterpretation | High accuracy with imbalanced classes can be misleading |
| 4 | Task-specific bias | Model may perform well on writing but poorly on drawings, masked by aggregate metrics |

**Solutions:**

| # | Solution |
|---|----------|
| 1 | Split by participant_id, not by sample — ALL samples (drawings + writing) from one participant stay in the same fold |
| 2 | Report mean +/- std across folds; investigate folds with outlier performance |
| 3 | Prioritize F1-score and per-class precision/recall alongside accuracy; report confusion matrix |
| 4 | Report per-task metrics (drawing accuracy vs. writing accuracy) in addition to aggregate metrics |

**Positive Outcome:**
Statistically rigorous evaluation that demonstrates the model's true generalization ability across both drawing and writing tasks, with confidence intervals and per-class metrics that satisfy thesis committee scrutiny.

---

### System 15: Model Performance Monitor

**Summary:**
Tracks training dynamics in real time: training loss, validation loss, training accuracy, and validation accuracy across epochs. Detects early stopping conditions (validation loss not improving for 10 consecutive epochs). Logs metrics for each fold of cross-validation and alerts when performance degrades.

**Problems & Errors:**

| # | Problem | Description |
|---|---------|-------------|
| 1 | Overfitting signal missed | Training continues too long without detecting the train-val divergence |
| 2 | Metric logging gaps | Incomplete logs make it impossible to reconstruct the training history |
| 3 | False early stopping | Temporary validation fluctuations trigger premature training termination |
| 4 | No checkpoint recovery | A Colab session timeout loses all training progress |

**Solutions:**

| # | Solution |
|---|----------|
| 1 | Set early stopping patience to 10 epochs (already configured); also monitor train-val accuracy gap |
| 2 | Log all metrics every epoch to a JSON/CSV file; include timestamps and fold numbers |
| 3 | Use patience > 5 to tolerate normal fluctuations; consider smoothed validation loss for stopping decision |
| 4 | Save checkpoints every epoch via `save_checkpoint()` in `Trainer`; save to Google Drive for persistence across Colab sessions |

**Positive Outcome:**
Full visibility into training behavior, enabling informed decisions about hyperparameter adjustments, early stopping, and model selection, while protecting against lost work from Colab interruptions.

---

### System 16: Model Versioning

**Summary:**
Manages saved model artifacts in the `models/` directory: CNN weights (`.pth`), HMM parameters (`.pkl`), training configuration, and cross-validation results (`.json`). Each model version is associated with its hyperparameters, training data split, and evaluation metrics. Saved to Google Drive via the Colab pipeline for persistence.

**Problems & Errors:**

| # | Problem | Description |
|---|---------|-------------|
| 1 | Overwritten models | A new training run overwrites the previous best model without backup |
| 2 | Version confusion | Multiple model files with unclear naming make it hard to identify the best version |
| 3 | Missing metadata | Saved weights without associated hyperparameters make reproduction impossible |
| 4 | Drive sync issues | Google Drive sync delays cause checkpoint corruption in Colab |

**Solutions:**

| # | Solution |
|---|----------|
| 1 | Include timestamp and fold number in checkpoint filenames: `cnn_fold3_20250415_best.pth` |
| 2 | Maintain a `models/model_registry.json` listing all versions with their metrics and paths |
| 3 | Save config alongside weights: `torch.save({'model_state': state, 'config': config, 'metrics': metrics})` |
| 4 | Save checkpoints locally first, then copy to Drive; verify file sizes after transfer |

**Positive Outcome:**
Full experiment reproducibility — any result in the thesis can be regenerated from the saved model version, its configuration, and the corresponding data split.

---

## Phase 8: Validation & Monitoring

### System 17: Master Data Compiler

**Summary:**
Aggregates all data streams — participant demographics, 24-item self-report scores, emotion labels, drawing analysis features, writing analysis features, CNN feature vectors, and model predictions — into unified datasets for analysis and reporting. Produces the final compiled tables used in the thesis results chapter.

**Problems & Errors:**

| # | Problem | Description |
|---|---------|-------------|
| 1 | Join mismatches | Records fail to merge due to inconsistent participant_id formatting across CSVs |
| 2 | Missing data | Some participants have self-report scores but missing drawing scans, or vice versa |
| 3 | Stale data | Compiled tables become outdated when upstream data changes |
| 4 | Multi-part complexity | Each participant has data from 3 assessment parts, 4 drawing tasks, and multiple writing tasks |

**Solutions:**

| # | Solution |
|---|----------|
| 1 | Standardize participant_id format (P###) across all files; validate before merging |
| 2 | Use outer joins to identify incomplete records; generate a completeness report per assessment part |
| 3 | Re-run compilation after any upstream change; add timestamps to compiled outputs |
| 4 | Use hierarchical table structure: participant -> assessment_part -> task -> sample |

**Positive Outcome:**
A single, comprehensive dataset linking every participant to their self-report responses, drawing features, writing features, and model predictions — ready for statistical analysis and thesis reporting.

---

### System 18: Citations Researcher

**Summary:**
Manages the project's 100+ academic references documented in `THESIS_CITATIONS_SUMMARY.md`. Organized across 8 categories: EMOTHAW database research, Ekman's emotion theory, graphology and handwriting analysis studies, CNN-HMM hybrid models, deep learning emotion recognition, image processing, handwriting OCR/feature extraction, and general emotion recognition. Each citation includes full reference, summary, and relevance to the thesis.

**Problems & Errors:**

| # | Problem | Description |
|---|---------|-------------|
| 1 | Missing citations | Key claims (e.g., drawing-based emotion indicators) lack supporting references |
| 2 | Outdated references | Cited papers have been superseded by newer, more relevant work |
| 3 | Citation format inconsistency | Mix of APA, IEEE, and informal formats across the document |
| 4 | Drawing analysis gap | Limited published research on "Person in the Rain" or "House-Tree" drawing analysis with ML |

**Solutions:**

| # | Solution |
|---|----------|
| 1 | Map every thesis claim to at least one citation; specifically find references for projective drawing tests |
| 2 | Search for papers published 2022-2025 in each category; prioritize recent survey papers |
| 3 | Standardize all citations to APA 7th edition format as required by the university |
| 4 | Reference classic projective test literature (House-Tree-Person test, Draw-a-Person-in-the-Rain) alongside ML papers |

**Positive Outcome:**
A well-organized, comprehensive bibliography that supports every claim in the thesis — including the novel drawing exercise analysis — with credible academic sources.

---

## Phase 9: Research & Reporting

### System 19: Thesis Progress Reporter

**Summary:**
Tracks the completion status of each thesis chapter, section, and system implementation across all 10 phases and 23 systems. Provides an overview of what has been completed, what is in progress, and what remains, along with milestone dates and blockers.

**Problems & Errors:**

| # | Problem | Description |
|---|---------|-------------|
| 1 | Stale status | Progress tracker not updated as work advances |
| 2 | Undefined milestones | Vague completion criteria make it unclear when a section is "done" |
| 3 | Scope creep | Adding new drawing/writing analysis features without updating the plan |
| 4 | Missing dependencies | Work on model training starts before all assessment data is collected |

**Solutions:**

| # | Solution |
|---|----------|
| 1 | Update progress after every significant work session; automate where possible |
| 2 | Define clear "done" criteria: data collected (300+ participants), model trained (80%+ accuracy), thesis written and reviewed |
| 3 | Freeze scope for each phase before starting; document any changes as formal amendments |
| 4 | Enforce phase dependencies: data collection must complete before model training begins |

**Positive Outcome:**
Clear visibility into thesis completion status, enabling timely course corrections, productive adviser meetings with Christian V. Maderazo, and on-time submission.

---

### System 20: Results Auto-Generator

**Summary:**
Automatically generates thesis-ready result artifacts from model evaluation outputs: formatted accuracy tables, precision/recall reports, confusion matrix heatmaps, training loss curves, cross-validation summary statistics (mean +/- std), per-class performance breakdowns, and per-task performance comparisons (drawing vs. writing accuracy). Outputs are saved to `results/` in formats ready for thesis insertion.

**Problems & Errors:**

| # | Problem | Description |
|---|---------|-------------|
| 1 | Figure formatting | Generated plots don't match thesis formatting requirements (font size, DPI, margins) |
| 2 | Stale results | Auto-generated tables reflect an older model version, not the final one |
| 3 | Statistical errors | Incorrect aggregation of cross-validation metrics |
| 4 | Missing task breakdown | Aggregate metrics hide that the model excels at writing but struggles with drawings |

**Solutions:**

| # | Solution |
|---|----------|
| 1 | Configure matplotlib with thesis-compliant settings: 300 DPI, Times New Roman font, proper axis labels |
| 2 | Regenerate all results from the final model version before thesis submission; timestamp all outputs |
| 3 | Use `CrossValidator._aggregate_results()` for correct mean/std computation; verify against manual calculation |
| 4 | Generate separate result tables for: (a) overall, (b) drawing tasks only, (c) writing tasks only, (d) per individual task |

**Positive Outcome:**
Publication-quality tables, figures, and statistical summaries generated directly from model outputs, with granular task-level breakdowns that demonstrate the model's strengths across the full 3-part assessment.

---

### System 21: Research Dashboard UI

**Summary:**
A visual interface (notebook-based via Colab or Jupyter) that provides a unified view of the project's status: dataset statistics per assessment part, class distribution charts, training progress curves, current model performance metrics, and per-task accuracy breakdowns. Enables the researcher to monitor the entire pipeline at a glance.

**Problems & Errors:**

| # | Problem | Description |
|---|---------|-------------|
| 1 | Data refresh lag | Dashboard shows cached data instead of current values |
| 2 | Visualization errors | Charts render incorrectly with missing data points or wrong scales |
| 3 | Colab session limits | Dashboard state lost when Colab runtime disconnects |
| 4 | Overhead | Dashboard development time detracts from core research work |

**Solutions:**

| # | Solution |
|---|----------|
| 1 | Add a refresh button or auto-refresh interval; read directly from source CSVs and result files |
| 2 | Handle missing data gracefully with fallback values; use fixed axis ranges for consistent comparisons |
| 3 | Save dashboard state to Google Drive; reload on session restart |
| 4 | Keep it minimal: focus on key metrics per assessment part rather than building a full web application |

**Positive Outcome:**
At-a-glance project monitoring across all three assessment parts, catching issues early (e.g., class imbalance, task-specific performance drops) and providing compelling visuals for thesis presentations and adviser meetings.

---

## Phase 10: Deployment & Visualization

### System 22: Visual Explainability & Emotion Report

**Summary:**
Generates interpretable visual explanations of model predictions across all assessment tasks. For writing samples: Grad-CAM heatmaps showing which handwriting regions the CNN focused on. For drawing samples: attention maps highlighting which drawing elements (figure size, rain density, tree detail) influenced the prediction. Produces per-sample emotion reports with confidence scores, making the black-box CNN-HMM model transparent for the thesis defense.

**Problems & Errors:**

| # | Problem | Description |
|---|---------|-------------|
| 1 | Misleading heatmaps | Grad-CAM highlights irrelevant regions (grid lines, page edges) instead of meaningful features |
| 2 | Low confidence predictions | Model predictions with near-50% confidence are unreliable but still presented |
| 3 | Drawing interpretation | Grad-CAM on drawings is harder to interpret than on text — what does "attending to the umbrella" mean? |
| 4 | Report generation failure | Missing dependencies or file path errors crash report generation |

**Solutions:**

| # | Solution |
|---|----------|
| 1 | Apply Grad-CAM to the final convolutional layer; mask out non-content regions before visualization |
| 2 | Display confidence scores alongside predictions; flag predictions below 60% confidence as "uncertain" |
| 3 | Map drawing Grad-CAM regions to named features: "attended to figure size", "attended to rain density", "attended to tree detail" |
| 4 | Include all visualization dependencies in `requirements.txt`; use absolute paths from `config.py` |

**Positive Outcome:**
Transparent, interpretable model explanations demonstrating that the CNN-HMM system makes predictions based on legitimate assessment features — handwriting characteristics AND drawing elements — strengthening thesis credibility and novelty.

---

### System 23: Emotion Detection System

**Summary:**
The complete end-to-end pipeline that brings all 22 preceding systems together into a unified emotion detection workflow. Given a participant's complete FINALE Thesis Assessment (all 3 parts), it:
1. Validates all assessment parts are present and complete (System 2)
2. Classifies and separates pages into Part 1/Part 2/Part 3 (System 6)
3. Scores Part 1 Self-Report (24 items) and assigns ground-truth label (System 12)
4. Extracts individual drawings and writing samples from Parts 2 & 3 (System 7)
5. Preprocesses all images through the 5-stage pipeline (System 8)
6. Extracts 256-dimensional features via trained CNN (System 10)
7. Classifies emotion via HMM likelihood comparison (System 13)
8. Generates visual explanations and confidence reports (System 22)
9. Returns: **HAPPY** or **SAD** with confidence score, Grad-CAM visualizations, and per-task breakdown

This is the capstone system — the deliverable that the thesis demonstrates and defends.

**Problems & Errors:**

| # | Problem | Description |
|---|---------|-------------|
| 1 | Pipeline cascade failure | An error in any upstream system (page classification, extraction, preprocessing) crashes the entire prediction |
| 2 | Out-of-distribution inputs | Assessments from outside the target demographic (non-USC students, different age, left-handed) produce unreliable predictions |
| 3 | Incomplete assessments | Participant completed Part 1 and Part 3 but skipped Part 2 drawings |
| 4 | Multi-task disagreement | Drawing analysis predicts HAPPY but writing analysis predicts SAD for the same participant |
| 5 | Confidence calibration | Reported confidence scores don't accurately reflect true prediction reliability |
| 6 | Ethical misuse | System could be misused for unauthorized psychological assessment or profiling |

**Solutions:**

| # | Solution |
|---|----------|
| 1 | Wrap each stage in try/except with informative error messages; validate intermediate outputs between stages |
| 2 | Document the model's valid input domain clearly; this system is designed for right-handed USC students aged 18-25 |
| 3 | Allow partial prediction with reduced confidence; report which parts were available and which were missing |
| 4 | Use weighted ensemble: combine drawing and writing predictions with learned weights; report per-task predictions alongside the final prediction |
| 5 | Calibrate confidence using Platt scaling or temperature scaling on the validation set |
| 6 | Include clear disclaimers: this is a research tool, not a clinical diagnostic instrument; require informed consent for any use; assessment administered only by licensed psychometricians |

**Positive Outcome:**
A working, end-to-end emotion detection system that processes the complete 3-part FINALE Thesis Assessment — self-report questionnaire, drawing exercises, and writing exercises — and returns an interpretable emotion classification with confidence. This is the core contribution of the INSIDE-OUT thesis: demonstrating that CNN-HMM hybrid models can recognize emotional states from both handwriting and drawings, validated under psychometric supervision.

---

## Complete System Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          INSIDE-OUT SYSTEM FLOW                                 │
│            Emotion Detection from Handwriting & Drawing Analysis                │
│                                                                                 │
│  Assessment: Part 1 (Self-Report) + Part 2 (Drawing) + Part 3 (Writing)        │
└─────────────────────────────────────────────────────────────────────────────────┘

                    ┌─────────────────────────────────────┐
                    │     FINALE THESIS ASSESSMENT         │
                    │                                     │
                    │  Part 1: 24-item Self-Report (1-5)  │
                    │  Part 2: 4 Drawing Exercises         │
                    │  Part 3: Words (L/R/Upper) + Cursive│
                    │                                     │
                    │  Signed: Researcher + Adviser        │
                    │  Supervised by: 4 Psychometricians   │
                    └──────────────┬──────────────────────┘
                                   │
PHASE 1: DATA INFRASTRUCTURE       │
┌──────────────────┐    ┌──────────▼───────────┐
│  [1] Barcode      │───>│  [2] Data Quality     │
│  Registry &       │    │  Control              │
│  Auto-Update      │    │  (validate all 3      │
│  (participant IDs)│    │   parts complete)     │
└──────────────────┘    └──────────┬───────────┘
                                   │
PHASE 2: DATA MANAGEMENT           │
┌──────────────────┐    ┌──────────▼───────────┐
│  [3] Participant  │───>│  [4] Database         │
│  Management       │    │  System               │
│  (demographics,   │    │  (CSV metadata for    │
│   consent, age)   │    │   all 3 parts)        │
└──────────────────┘    └──────────┬───────────┘
                                   │
PHASE 3: FILE ORGANIZATION         │
┌──────────────────┐    ┌──────────▼───────────┐
│  [5] File Sorter  │<───│  [6] Page Classifier  │
│  (RAW -> LABELED  │    │  & Auto-Renamer       │
│   by part & task) │    │  (Part1/Part2/Part3)  │
└────────┬─────────┘    └──────────────────────┘
         │
         ├──────────────────────────────────────────────────────┐
         │                                                      │
         ▼                                                      ▼
┌─────────────────────┐                              ┌─────────────────────┐
│  PART 2: DRAWINGS    │                              │  PART 3: WRITING     │
│  - 3 Overlapping     │                              │  - 5 Words x 3 styles│
│    Circles           │                              │    (L/R/Uppercase)   │
│  - Connect the Dots  │                              │  - 5 Cursive         │
│  - Person in Rain    │                              │    Sentences         │
│  - House and Tree    │                              │                     │
└─────────┬───────────┘                              └──────────┬──────────┘
          │                                                     │
PHASE 4: IMAGE PROCESSING                                       │
┌─────────▼───────────┐    ┌──────────────────┐    ┌───────────▼──────────┐
│  [7] Data Extraction │    │  [8] Image        │    │  [7] Data Extraction  │
│  & Content Analyzer  │───>│  Preprocessor     │<───│  & Content Analyzer   │
│  (split quadrants)   │    │  (5-stage pipe)   │    │  (split grid cells)   │
└──────────────────────┘    └────────┬─────────┘    └──────────────────────┘
                                     │
                            ┌────────▼─────────┐
                            │  [9] Data         │
                            │  Augmentation     │
                            │  (training only)  │
                            └────────┬─────────┘
                                     │
PHASE 5: FEATURE ENGINEERING         │
┌──────────────────┐    ┌────────────▼─────────────────────────┐
│  [11] Micro-Level │───>│  [10] Feature Extractor              │
│  Letter & Drawing │    │  (CNN: 224x224 -> 256-dim vector)    │
│  Analysis         │    │  Drawings + Writing -> unified space │
└──────────────────┘    └────────────┬─────────────────────────┘
                                     │
PHASE 6: LABELING                    │
┌──────────────────┐                 │
│  [12] Emotion     │── ground ──────┤
│  Labeler          │   truth        │
│  (24-item self-   │   labels       │
│   report scoring) │                │
└──────────────────┘                 │
                                     │
PHASE 7: MODEL DEVELOPMENT           │
                        ┌────────────▼─────────────────────────┐
                        │  [13] CNN-HMM Model Trainer           │
                        │  Stage 1: CNN training (cross-entropy) │
                        │  Stage 2: HMM fitting (EM algorithm)  │
                        └────────────┬─────────────────────────┘
                                     │
PHASE 8: VALIDATION & MONITORING     │
┌──────────────────┐    ┌────────────▼─────────┐    ┌─────────────────┐
│  [15] Performance │<───│  [14] Model Validator │───>│  [16] Model      │
│  Monitor          │    │  & Tester (5-fold CV) │    │  Versioning      │
└──────────────────┘    └──────────────────────┘    └─────────────────┘
                                     │
         ┌───────────────────────────┤
         │                           │
┌────────▼─────────┐      ┌──────────▼─────────┐
│  [17] Master Data │      │  [18] Citations     │
│  Compiler         │      │  Researcher         │
└────────┬─────────┘      └──────────┬─────────┘
         │                           │
PHASE 9: RESEARCH & REPORTING        │
┌────────▼─────────┐    ┌────────────▼─────────────┐
│  [19] Thesis      │<───│  [20] Results            │
│  Progress         │    │  Auto-Generator          │
│  Reporter         │    │  (per-task breakdowns)   │
└──────────────────┘    └──────────────────────────┘
                                     │
PHASE 10: DEPLOYMENT & VISUALIZATION │
┌──────────────────┐    ┌────────────▼─────────────┐
│  [21] Research    │<───│  [22] Visual             │
│  Dashboard UI     │    │  Explainability &        │
│  (monitoring)     │    │  Emotion Report          │
└──────────────────┘    │  (Grad-CAM + drawings)   │
                        └────────────┬─────────────┘
                                     │
                        ┌────────────▼─────────────────────┐
                        │                                   │
                        │  ⭐ [23] EMOTION DETECTION SYSTEM  │
                        │                                   │
                        │  Input:  Complete 3-Part           │
                        │          FINALE Assessment         │
                        │                                   │
                        │  Analyzes:                         │
                        │    Part 1: 24-item Self-Report     │
                        │    Part 2: 4 Drawing Exercises     │
                        │    Part 3: Words + Cursive Writing │
                        │                                   │
                        │  Output: HAPPY / SAD               │
                        │          + Confidence Score        │
                        │          + Grad-CAM Visuals        │
                        │          + Per-Task Breakdown      │
                        │                                   │
                        └───────────────────────────────────┘
```

---

## Quick Reference Table

| No. | System Name | Phase | Input | Output | Key Library |
|-----|-------------|-------|-------|--------|-------------|
| 1 | Barcode Registry & Auto-Update | 1 - Data Infrastructure | Participant info | Unique ID (P###) | pandas |
| 2 | Data Quality Control | 1 - Data Infrastructure | All 3 assessment parts | Validated/rejected flags | opencv-python, pandas |
| 3 | Participant Management | 2 - Data Management | Demographics, consent | participants.csv | pandas |
| 4 | Database System | 2 - Data Management | All CSV tables | Joined metadata | pandas |
| 5 | File Sorter | 3 - File Organization | Labeled files by part | Organized directories | os, shutil |
| 6 | Page Classifier & Auto-Renamer | 3 - File Organization | Raw assessment scans | Part1/Part2/Part3 classified files | Pillow, opencv-python |
| 7 | Data Extraction & Content Analyzer | 4 - Image Processing | Assessment pages | Cropped drawings + writing samples | opencv-python |
| 8 | Image Preprocessor | 4 - Image Processing | Cropped content | 224x224 normalized image | opencv-python, scikit-image |
| 9 | Data Augmentation | 4 - Image Processing | Preprocessed images | Augmented training set | torchvision.transforms |
| 10 | Feature Extractor | 5 - Feature Engineering | 224x224 image | 256-dim feature vector | torch, torchvision |
| 11 | Micro-Level Letter & Drawing Analysis | 5 - Feature Engineering | Preprocessed drawings + writing | Handwriting + drawing measurements | opencv-python, numpy |
| 12 | Emotion Labeler | 6 - Labeling & Annotation | 24-item self-report scores | HAPPY/SAD/NEUTRAL label | pandas |
| 13 | CNN-HMM Model Trainer | 7 - Model Development | Features + labels | Trained hybrid model | torch, hmmlearn |
| 14 | Model Validator & Tester | 8 - Validation & Monitoring | Trained model + test data | Metrics (acc, F1, etc.) | scikit-learn |
| 15 | Model Performance Monitor | 8 - Validation & Monitoring | Training logs | Loss/accuracy curves | matplotlib |
| 16 | Model Versioning | 8 - Validation & Monitoring | Model weights + config | Versioned checkpoints | torch, pickle |
| 17 | Master Data Compiler | 8 - Validation & Monitoring | All metadata + results | Unified dataset | pandas |
| 18 | Citations Researcher | 9 - Research & Reporting | Research papers | Annotated bibliography | manual / Zotero |
| 19 | Thesis Progress Reporter | 9 - Research & Reporting | Task completion data | Progress summary | markdown |
| 20 | Results Auto-Generator | 9 - Research & Reporting | Model eval outputs | Thesis-ready figures/tables | matplotlib, seaborn |
| 21 | Research Dashboard UI | 10 - Deployment & Visualization | All system data | Visual dashboard | matplotlib, jupyter |
| 22 | Visual Explainability & Emotion Report | 10 - Deployment & Visualization | Model + input images | Grad-CAM + reports | torch, matplotlib |
| 23 | Emotion Detection System | 10 - Deployment & Visualization | Complete 3-part assessment | Emotion + confidence + visuals | torch, hmmlearn, opencv-python |

---

## Assessment Tasks Summary

### Part 2: Drawing Exercises

| Task | What Participant Does | What System Analyzes |
|------|----------------------|---------------------|
| Three Overlapping Circles | Draw 3 circles that overlap | Symmetry, size consistency, overlap precision, stroke pressure |
| Connect the Dots | Connect two columns of 8 dots | Line straightness, connection accuracy, stroke confidence |
| Person in the Rain | Draw a person standing in rain | Figure size, umbrella presence, rain density, figure placement, detail level |
| House and a Tree | Draw a house and a tree | Proportions, detail level, spatial organization, groundline, completeness |

### Part 3: Writing Exercises

| Task | Words/Sentences | Variants |
|------|----------------|----------|
| Emotion Words | Content, Melancholic, Optimistic, Disconnected, Vibrant | Left hand, Right hand, Uppercase |
| Cursive Sentences | "Crazy people are seeking for purple flowers..." | Cursive only |
| | "Happiness is the meaning and the purpose of life..." | Cursive only |
| | "Life is like a box of chocolates." | Cursive only |
| | "What does not kill me makes me stronger." | Cursive only |
| | "Happiness is not doing what you want, but wanting what you do." | Cursive only |

---

*Generated for the INSIDE-OUT thesis project — University of San Carlos, DCISM*
*Researcher: Cyrel Jane A. Edano | Adviser: Christian V. Maderazo, M.Eng.*
