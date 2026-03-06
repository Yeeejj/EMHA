# INSIDE-OUT: Complete Systems Guide

## Emotion Recognition through Handwriting Analysis using CNN-HMM Hybrid Approach

**Author:** Cyrel Jane A. Edano | **Institution:** University of San Carlos, DCISM
**Systems Count:** 23 | **Phases:** 10
**Target:** 80-85% accuracy classifying happiness vs. sadness from handwriting

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
Generates and manages unique participant identifiers (P001, P002, ..., P300+) that link every handwriting sample, questionnaire score, and metadata record back to a single individual. The registry auto-increments IDs and prevents duplicates, ensuring traceability across the entire pipeline. Implemented via `DataCollector.generate_participant_id()` in `src/data/collector.py`.

**Problems & Errors:**

| # | Problem | Description |
|---|---------|-------------|
| 1 | ID collision | Two participants assigned the same ID if the counter file is corrupted or read concurrently |
| 2 | Gap in sequence | Deleted or abandoned entries leave holes (P001, P003) that confuse downstream counts |
| 3 | Lost mapping | If `participants.csv` is accidentally overwritten, the link between barcode and participant is severed |
| 4 | Format inconsistency | Manual entries may use different padding (P1 vs P001) breaking filename patterns |

**Solutions:**

| # | Solution |
|---|----------|
| 1 | Use file-lock or atomic write when updating the counter; validate uniqueness against `participants.csv` before committing |
| 2 | Never reuse IDs — treat gaps as intentional; add a `status` column (active/withdrawn) instead of deleting rows |
| 3 | Keep automatic backups of `DATA/METADATA/participants.csv` with timestamps; version-control the metadata folder |
| 4 | Enforce zero-padded 3-digit format in `generate_participant_id()` with strict regex validation `^P\d{3}$` |

**Positive Outcome:**
Every handwriting sample, questionnaire response, and model prediction is permanently traceable to a verified participant, satisfying both research reproducibility and ethical audit requirements.

---

### System 2: Data Quality Control

**Summary:**
Validates incoming handwriting samples and questionnaire data before they enter the pipeline. Checks image resolution, file format, completeness of metadata fields, and questionnaire score ranges. Prevents garbage-in-garbage-out by rejecting or flagging substandard inputs at the gate.

**Problems & Errors:**

| # | Problem | Description |
|---|---------|-------------|
| 1 | Low-resolution scans | Images below usable DPI produce unreadable features after preprocessing |
| 2 | Incomplete questionnaires | Missing DASS-21 or happiness scores make emotion labeling impossible |
| 3 | Corrupted files | Truncated PNGs or zero-byte files crash the preprocessing pipeline |
| 4 | Outlier scores | Questionnaire values outside valid ranges (e.g., DASS depression > 42) indicate data entry errors |
| 5 | Class imbalance | Uneven HAPPY/SAD distribution degrades model generalization |

**Solutions:**

| # | Solution |
|---|----------|
| 1 | Set minimum resolution threshold (300 DPI or 1000x1000 px); reject and re-collect if below |
| 2 | Require all questionnaire fields before saving to `questionnaire_scores.csv`; flag partial records for follow-up |
| 3 | Validate file integrity (check PNG header bytes, file size > 0) on ingestion |
| 4 | Enforce score ranges in `EmotionLabeler`: DASS depression 0-42, happiness 0-100; reject out-of-range values |
| 5 | Monitor `get_class_distribution()` regularly; recruit targeted participants or apply class-weighted loss |

**Positive Outcome:**
Only clean, complete, and valid data enters the pipeline, producing reliable emotion labels and preventing downstream failures during preprocessing, training, and evaluation.

---

## Phase 2: Data Management

### System 3: Participant Management

**Summary:**
Handles participant registration, consent tracking, demographic recording, and sample association. Stores participant profiles in `DATA/METADATA/participants.csv` with fields: participant_id, age, gender, handedness, consent_date, and notes. Only right-handed participants aged 18-25 are eligible per the study protocol.

**Problems & Errors:**

| # | Problem | Description |
|---|---------|-------------|
| 1 | Ineligible participants | Left-handed or out-of-age-range participants slip through screening |
| 2 | Missing consent | Samples collected without documented consent violate ethics approval |
| 3 | Duplicate enrollment | Same person registers twice under different IDs |
| 4 | CSV encoding issues | Special characters in names or notes corrupt the CSV when opened across platforms |

**Solutions:**

| # | Solution |
|---|----------|
| 1 | Add eligibility checks in `create_participant_folder()`: assert `handedness == 'R'` and `18 <= age <= 25` |
| 2 | Make `consent_date` a required non-null field; block folder creation until consent is recorded |
| 3 | Add a check against existing demographics (age + gender + consent_date) to flag potential duplicates |
| 4 | Enforce UTF-8 encoding on all CSV reads/writes; avoid special characters in the notes field |

**Positive Outcome:**
A clean, ethics-compliant participant registry that ensures every sample comes from a consented, eligible individual, satisfying IRB requirements and strengthening research validity.

---

### System 4: Database System

**Summary:**
The structured CSV-based metadata system in `DATA/METADATA/` acts as the project's database layer. Three core tables — `participants.csv`, `questionnaire_scores.csv`, and `labels.csv` — are joined by `participant_id` as the primary key. Managed via pandas DataFrames throughout the codebase.

**Problems & Errors:**

| # | Problem | Description |
|---|---------|-------------|
| 1 | Referential integrity | A label references a participant_id that doesn't exist in participants.csv |
| 2 | Concurrent writes | Two processes writing to the same CSV simultaneously cause data loss |
| 3 | Schema drift | New columns added inconsistently across CSVs break downstream readers |
| 4 | No transaction support | A crash mid-write leaves the CSV in a partial/corrupted state |
| 5 | Scalability | CSV performance degrades beyond thousands of rows for repeated reads |

**Solutions:**

| # | Solution |
|---|----------|
| 1 | Validate foreign keys on every write: confirm participant_id exists in `participants.csv` before inserting into `labels.csv` |
| 2 | Use file-locking (e.g., `fcntl.flock`) or single-process write access to metadata files |
| 3 | Define schemas in `config.py` with expected column names and types; validate on load |
| 4 | Write to a temporary file first, then atomically rename to the target path |
| 5 | For larger datasets, migrate to SQLite while keeping CSV export for portability |

**Positive Outcome:**
A lightweight, portable, version-controllable data store that keeps participant records, scores, and labels consistent and queryable without requiring a database server.

---

## Phase 3: File Organization

### System 5: File Sorter

**Summary:**
Organizes raw handwriting samples from `DATA/RAW/` into the appropriate directory structure: `DATA/LABELED/HAPPY/` and `DATA/LABELED/SAD/` based on assigned emotion labels. Also manages the flow from labeled to `DATA/PROCESSED/` and finally to `DATA/SPLITS/` (train/val/test). Uses the naming convention `{participant_id}_sample_{number}_{date}.png`.

**Problems & Errors:**

| # | Problem | Description |
|---|---------|-------------|
| 1 | Misrouted files | A HAPPY sample lands in the SAD directory due to label lookup error |
| 2 | Filename collision | Two samples with identical names overwrite each other during sorting |
| 3 | Orphaned files | Samples in RAW/ with no matching label record are never sorted |
| 4 | Broken symlinks | If source files are moved after creating symbolic links, paths break |

**Solutions:**

| # | Solution |
|---|----------|
| 1 | Cross-reference `labels.csv` during sorting; log every move operation for audit |
| 2 | Append a unique suffix or timestamp if filename already exists in destination |
| 3 | Run an orphan-detection scan: compare RAW/ filenames against labels.csv and report unmatched files |
| 4 | Use file copies instead of symlinks for the labeled directory; keep RAW/ as immutable archive |

**Positive Outcome:**
A clean, deterministic directory structure where every file is in exactly the right place, enabling the dataloader and preprocessing pipeline to operate without path errors or label mismatches.

---

### System 6: Page Classifier & Auto-Renamer

**Summary:**
Identifies and classifies scanned pages by content type (handwriting sample page 1, page 2, questionnaire, consent form) and automatically renames files to follow the project naming convention. Handles multi-page scans where a single scan may contain multiple document types.

**Problems & Errors:**

| # | Problem | Description |
|---|---------|-------------|
| 1 | Misclassification | A questionnaire page is classified as a handwriting sample and enters the analysis pipeline |
| 2 | Page order errors | Multi-page scans are split but reassembled in wrong order |
| 3 | Naming conflicts | Auto-generated names collide with manually named files |
| 4 | Unsupported formats | TIFF, BMP, or HEIC scans are not handled by the renamer |

**Solutions:**

| # | Solution |
|---|----------|
| 1 | Use template matching or header detection to distinguish document types; require human verification for low-confidence classifications |
| 2 | Embed page sequence numbers in filenames; sort by scan timestamp before splitting |
| 3 | Check for existing files before renaming; use the participant registry to generate guaranteed-unique names |
| 4 | Convert all inputs to PNG on ingestion using Pillow; support common formats via `Image.open()` |

**Positive Outcome:**
Scanned documents are automatically organized, correctly classified, and consistently named, eliminating manual file management and reducing human error in data preparation.

---

## Phase 4: Image Processing

### System 7: Data Extraction & Content Analyzer

**Summary:**
Extracts meaningful content regions from raw handwriting images — isolating the actual handwriting from form borders, headers, printed text, and blank margins. Identifies the writing area bounding box and crops to the region of interest before preprocessing begins.

**Problems & Errors:**

| # | Problem | Description |
|---|---------|-------------|
| 1 | Over-cropping | Part of the handwriting is cut off, losing features at the edges |
| 2 | Under-cropping | Form borders and printed headers remain, adding noise to feature extraction |
| 3 | Empty crops | Blank pages or very faint writing produce empty or near-empty content regions |
| 4 | Multi-region content | Writing appears in multiple disconnected areas of the page |

**Solutions:**

| # | Solution |
|---|----------|
| 1 | Add padding (10-20 px) around detected content bounding boxes before cropping |
| 2 | Use contour detection with area thresholds to filter out printed form elements |
| 3 | Set a minimum ink-pixel ratio threshold; flag pages below it for re-collection |
| 4 | Detect all contiguous writing regions, merge nearby regions, and crop to the combined bounding box |

**Positive Outcome:**
Clean, focused handwriting regions with minimal noise, giving the CNN feature extractor a consistent and informative input that directly represents the participant's writing.

---

### System 8: Image Preprocessor

**Summary:**
The 5-stage preprocessing pipeline in `src/preprocessing/pipeline.py` transforms raw scans into model-ready images:
1. **Grayscale conversion** — RGB to single channel
2. **Binarization** — Otsu's thresholding separates ink from background
3. **Denoising** — Morphological opening/closing with 3x3 kernel removes speckles
4. **Skew correction** — Rotates to straighten tilted handwriting
5. **Normalization** — Resizes to 224x224, scales pixel values to [0, 1]

**Problems & Errors:**

| # | Problem | Description |
|---|---------|-------------|
| 1 | Otsu failure | Uniform images (very light or very dark) cause Otsu's method to pick a poor threshold |
| 2 | Over-denoising | Aggressive morphological operations erode thin strokes and fine details |
| 3 | Skew over-correction | The algorithm detects the wrong dominant angle and rotates incorrectly |
| 4 | Aspect ratio distortion | Resizing non-square images to 224x224 stretches handwriting features |
| 5 | Information loss | Binarization discards pressure-related grayscale intensity information |

**Solutions:**

| # | Solution |
|---|----------|
| 1 | Fall back to adaptive thresholding (Gaussian) when Otsu variance is below a minimum; configurable via `PreprocessingConfig.binarize_threshold` |
| 2 | Use a small kernel (3x3) and limit morphological iterations; visually inspect samples from each batch |
| 3 | Constrain rotation to +/- 15 degrees; use Hough line detection for more robust angle estimation |
| 4 | Pad to square with white background before resizing to preserve aspect ratio |
| 5 | Keep a parallel grayscale-normalized version for pressure-sensitive features if needed |

**Positive Outcome:**
Standardized 224x224 grayscale images with clean ink-on-white presentation, enabling the CNN to learn handwriting features without being distracted by scan artifacts, skew, or resolution differences.

---

### System 9: Data Augmentation

**Summary:**
Expands the training dataset by applying controlled transformations to existing samples, addressing class imbalance and improving model generalization. Applied only during training (not validation/test). Transformations include random rotation (+-10 degrees), horizontal flip, slight scaling, brightness/contrast jitter, and elastic distortion — all within ranges that preserve handwriting characteristics.

**Problems & Errors:**

| # | Problem | Description |
|---|---------|-------------|
| 1 | Unrealistic augmentations | Excessive rotation or flipping produces samples that don't resemble real handwriting |
| 2 | Label leakage | Augmented samples from validation participants leak into training set |
| 3 | Over-augmentation | Too many synthetic samples drown out real data patterns |
| 4 | Inconsistent application | Augmentation applied at inference time changes predictions between runs |

**Solutions:**

| # | Solution |
|---|----------|
| 1 | Limit rotation to +-10 degrees; avoid vertical flips; keep elastic distortion mild |
| 2 | Split by participant_id first, then augment only the training split |
| 3 | Cap augmented-to-real ratio at 2:1 or 3:1; monitor validation metrics for overfitting |
| 4 | Apply augmentation only when `split='train'` in `HandwritingDataset`; use deterministic transforms for val/test |

**Positive Outcome:**
A larger, more diverse training set that helps the CNN-HMM model generalize to unseen handwriting styles, reducing overfitting and improving accuracy on the held-out test set.

---

## Phase 5: Feature Engineering

### System 10: Feature Extractor

**Summary:**
The `CNNFeatureExtractor` in `src/models/cnn.py` transforms preprocessed 224x224 images into compact 256-dimensional feature vectors. Architecture: 4 convolutional blocks (Conv2d -> BatchNorm -> ReLU -> MaxPool) progressing from 1 -> 32 -> 64 -> 128 -> 256 channels, followed by AdaptiveAvgPool and a fully connected layer. Optionally uses pretrained ResNet18/VGG backbones via `PretrainedCNNExtractor`.

**Problems & Errors:**

| # | Problem | Description |
|---|---------|-------------|
| 1 | Vanishing gradients | Deep CNN layers stop learning due to gradient decay |
| 2 | Overfitting | 256-dimensional features memorize training data with small dataset (300 samples) |
| 3 | Feature collapse | All samples map to similar feature vectors, losing discriminative power |
| 4 | Pretrained mismatch | ImageNet-pretrained weights expect 3-channel RGB; handwriting is 1-channel grayscale |
| 5 | Computational cost | Full CNN forward pass is slow on CPU-only machines |

**Solutions:**

| # | Solution |
|---|----------|
| 1 | Use batch normalization (already included) and skip connections; monitor gradient norms during training |
| 2 | Apply dropout (0.5 rate from `CNNConfig`), weight decay, and data augmentation to regularize |
| 3 | Visualize feature distributions with t-SNE/UMAP; ensure HAPPY and SAD clusters are separable |
| 4 | Replicate grayscale to 3 channels for pretrained models; or fine-tune only later layers |
| 5 | Use Google Colab GPU runtime (T4/V100) as implemented in the notebook pipeline |

**Positive Outcome:**
Rich, compact 256-dimensional feature vectors that capture handwriting characteristics (stroke width, slant, spacing, pressure) in a form that the HMM classifier can effectively model for emotion prediction.

---

### System 11: Micro-Level Letter & Drawing Analysis

**Summary:**
Analyzes fine-grained handwriting characteristics at the individual letter and stroke level. Measures five key features that research links to emotional states: baseline direction (rising = happy, falling = sad), letter slant (right-leaning = happy), stroke pressure (heavy = happy), letter size (large = happy), and inter-word spacing (wide = happy). These micro-features complement the CNN's learned representations.

**Problems & Errors:**

| # | Problem | Description |
|---|---------|-------------|
| 1 | Segmentation failure | Individual letters cannot be reliably isolated from cursive or connected handwriting |
| 2 | Pressure estimation | Scanned images lack true pressure data — only intensity approximation is available |
| 3 | Baseline ambiguity | Multi-line text has multiple baselines with different directions |
| 4 | Cultural variation | Handwriting norms differ across writing systems and educational backgrounds |

**Solutions:**

| # | Solution |
|---|----------|
| 1 | Use connected component analysis for print; for cursive, measure features at the word level instead |
| 2 | Use grayscale intensity as a pressure proxy; document this limitation in the thesis |
| 3 | Detect individual text lines first (horizontal projection), then compute baseline per line and average |
| 4 | Control for this by limiting participants to a single language/script (English) and similar educational background |

**Positive Outcome:**
Quantitative measurements of handwriting features that graphology research has linked to emotional states, providing interpretable evidence that complements the CNN's black-box feature extraction.

---

## Phase 6: Labeling & Annotation

### System 12: Emotion Labeler

**Summary:**
The `EmotionLabeler` in `src/data/labeler.py` assigns ground-truth emotion labels to handwriting samples based on validated psychological questionnaires. Labeling rules from `LabelingConfig`:
- **HAPPY**: happiness_score >= 40.0 AND dass_depression < 14.0
- **SAD**: happiness_score < 30.0 OR dass_depression >= 14.0
- **NEUTRAL**: all other cases (excluded from training)

Uses DASS-21 depression subscale thresholds and Oxford/SDHS happiness scores.

**Problems & Errors:**

| # | Problem | Description |
|---|---------|-------------|
| 1 | Threshold sensitivity | Small changes in threshold values (e.g., 40 vs 38) significantly change class assignments |
| 2 | Neutral exclusion | Too many participants fall in the neutral zone, reducing usable dataset size |
| 3 | Self-report bias | Participants may not honestly report their emotional state on questionnaires |
| 4 | Temporal mismatch | Emotional state during questionnaire may differ from state during handwriting |
| 5 | Mixed emotions | A participant may feel both happy and stressed simultaneously |

**Solutions:**

| # | Solution |
|---|----------|
| 1 | Validate thresholds against published norms for DASS-21; perform sensitivity analysis across threshold ranges |
| 2 | Adjust thresholds to maximize usable samples while maintaining clear class separation; report neutral exclusion rate |
| 3 | Use validated, well-established instruments (DASS-21 is clinically validated); collect samples in controlled settings |
| 4 | Administer questionnaire immediately before handwriting collection in a single session |
| 5 | Use the dual-criterion approach (both happiness AND depression scores) to ensure clear emotional classification |

**Positive Outcome:**
Psychologically grounded, reproducible emotion labels that provide reliable ground truth for supervised learning, backed by clinically validated assessment instruments.

---

## Phase 7: Model Development

### System 13: CNN-HMM Model Trainer

**Summary:**
The `Trainer` class in `src/training/trainer.py` and `HybridCNNHMM` in `src/models/hybrid.py` implement the two-stage training pipeline:
1. **Stage 1 — CNN Training**: Train the CNN feature extractor using cross-entropy loss, Adam optimizer (lr=0.001), batch size 32, up to 50 epochs with early stopping (patience=10)
2. **Stage 2 — HMM Training**: Extract features from trained CNN, fit separate Gaussian HMMs (4 states, diagonal covariance) for HAPPY and SAD classes

Integrated in the Colab pipeline with 5-fold stratified cross-validation.

**Problems & Errors:**

| # | Problem | Description |
|---|---------|-------------|
| 1 | CNN overfitting | Small dataset (300 samples) causes memorization; training accuracy >> validation accuracy |
| 2 | HMM convergence failure | EM algorithm fails to converge with insufficient or poorly distributed features |
| 3 | Stage mismatch | CNN features optimized for classification may not be optimal for HMM modeling |
| 4 | GPU memory overflow | Large batch sizes or high-resolution images exhaust Colab GPU memory |
| 5 | Training instability | Loss spikes or NaN values during CNN training |

**Solutions:**

| # | Solution |
|---|----------|
| 1 | Apply dropout (0.5), weight decay, data augmentation, and early stopping; monitor train-val gap |
| 2 | Increase HMM iterations (`n_iter=100`), try different covariance types (full, tied), ensure sufficient samples per class |
| 3 | Fine-tune: after HMM training, optionally adjust CNN features based on HMM classification performance |
| 4 | Reduce batch size to 16 or 8; use mixed-precision training (torch.cuda.amp); resize to 128x128 if needed |
| 5 | Use gradient clipping (max_norm=1.0), learning rate warmup, and robust optimizer settings |

**Positive Outcome:**
A trained hybrid model that combines the CNN's spatial feature learning with the HMM's sequential pattern modeling, achieving the target 80-85% accuracy on emotion classification from handwriting.

---

### System 14: Model Validator & Tester

**Summary:**
The `Evaluator` in `src/training/evaluator.py` and `CrossValidator` in `src/training/cross_validate.py` rigorously assess model performance. Implements 5-fold stratified cross-validation ensuring balanced class distribution per fold. Computes accuracy, precision, recall, F1-score (macro-averaged), and confusion matrix. Final evaluation on held-out test set (15% of data) that was never seen during training or validation.

**Problems & Errors:**

| # | Problem | Description |
|---|---------|-------------|
| 1 | Data leakage | Same participant's samples appear in both train and test folds |
| 2 | Fold variance | Large performance variation across folds indicates instability |
| 3 | Metric misinterpretation | High accuracy with imbalanced classes can be misleading |
| 4 | Overfitting to validation | Hyperparameters tuned to maximize validation performance don't generalize |

**Solutions:**

| # | Solution |
|---|----------|
| 1 | Split by participant_id, not by sample — all samples from one participant stay in the same fold |
| 2 | Report mean +/- std across folds; investigate folds with outlier performance |
| 3 | Prioritize F1-score and per-class precision/recall alongside accuracy; report confusion matrix |
| 4 | Keep the test set (15%) completely untouched until final evaluation; never tune on test results |

**Positive Outcome:**
Statistically rigorous evaluation that demonstrates the model's true generalization ability, with confidence intervals and per-class metrics that satisfy thesis committee scrutiny.

---

### System 15: Model Performance Monitor

**Summary:**
Tracks training dynamics in real time: training loss, validation loss, training accuracy, and validation accuracy across epochs. Detects early stopping conditions (validation loss not improving for 10 consecutive epochs). Logs metrics for each fold of cross-validation and alerts when performance degrades or anomalies appear.

**Problems & Errors:**

| # | Problem | Description |
|---|---------|-------------|
| 1 | Overfitting signal missed | Training continues too long without detecting the train-val divergence |
| 2 | Metric logging gaps | Incomplete logs make it impossible to reconstruct the training history |
| 3 | False early stopping | Temporary validation fluctuations trigger premature training termination |
| 4 | No checkpoint recovery | A crash loses all training progress |

**Solutions:**

| # | Solution |
|---|----------|
| 1 | Set early stopping patience to 10 epochs (already configured); also monitor train-val accuracy gap |
| 2 | Log all metrics every epoch to a JSON/CSV file; include timestamps and fold numbers |
| 3 | Use patience > 5 to tolerate normal fluctuations; consider smoothed validation loss for stopping decision |
| 4 | Save checkpoints every epoch via `save_checkpoint()` in `Trainer`; resume from best checkpoint on restart |

**Positive Outcome:**
Full visibility into training behavior, enabling informed decisions about hyperparameter adjustments, early stopping, and model selection, while protecting against lost work from interruptions.

---

### System 16: Model Versioning

**Summary:**
Manages saved model artifacts in the `models/` directory: CNN weights (`.pth`), HMM parameters (`.pkl`), training configuration, and cross-validation results (`.json`). Each model version is associated with its hyperparameters, training data split, and evaluation metrics, enabling reproducibility and comparison across experiments.

**Problems & Errors:**

| # | Problem | Description |
|---|---------|-------------|
| 1 | Overwritten models | A new training run overwrites the previous best model without backup |
| 2 | Version confusion | Multiple model files with unclear naming make it hard to identify the best version |
| 3 | Missing metadata | Saved weights without associated hyperparameters make reproduction impossible |
| 4 | Large file sizes | Multiple CNN checkpoints fill up storage (especially on Google Drive) |

**Solutions:**

| # | Solution |
|---|----------|
| 1 | Include timestamp and fold number in checkpoint filenames: `cnn_fold3_20250415_best.pth` |
| 2 | Maintain a `models/model_registry.json` listing all versions with their metrics and paths |
| 3 | Save config alongside weights: `torch.save({'model_state': state, 'config': config, 'metrics': metrics})` |
| 4 | Keep only the best model per fold plus the final model; delete intermediate checkpoints after training completes |

**Positive Outcome:**
Full experiment reproducibility — any result in the thesis can be regenerated from the saved model version, its configuration, and the corresponding data split.

---

## Phase 8: Validation & Monitoring

### System 17: Master Data Compiler

**Summary:**
Aggregates all data streams — participant demographics, questionnaire scores, emotion labels, preprocessing metadata, feature vectors, and model predictions — into unified datasets for analysis and reporting. Produces the final compiled tables used in the thesis results chapter.

**Problems & Errors:**

| # | Problem | Description |
|---|---------|-------------|
| 1 | Join mismatches | Records fail to merge due to inconsistent participant_id formatting across CSVs |
| 2 | Missing data | Some participants have scores but no handwriting samples, or vice versa |
| 3 | Stale data | Compiled tables become outdated when upstream data changes |
| 4 | Column name conflicts | Same column name (e.g., "score") means different things in different source tables |

**Solutions:**

| # | Solution |
|---|----------|
| 1 | Standardize participant_id format (P###) across all files; validate before merging |
| 2 | Use outer joins to identify incomplete records; generate a completeness report |
| 3 | Re-run compilation after any upstream change; add timestamps to compiled outputs |
| 4 | Prefix columns with source name during merge (e.g., `dass_depression`, `happiness_score`) |

**Positive Outcome:**
A single, comprehensive dataset linking every participant to their demographics, psychological assessments, handwriting features, and model predictions — ready for statistical analysis and thesis reporting.

---

### System 18: Citations Researcher

**Summary:**
Manages the project's academic references documented in `THESIS_CITATIONS_SUMMARY.md`. Organizes 100+ research papers across 8 categories: EMOTHAW database research, Ekman's emotion theory, graphology studies, CNN-HMM hybrid models, deep learning emotion recognition, image processing, handwriting OCR, and general emotion recognition. Each citation includes full reference, summary, and relevance to the thesis.

**Problems & Errors:**

| # | Problem | Description |
|---|---------|-------------|
| 1 | Missing citations | Key claims in the thesis lack supporting references |
| 2 | Outdated references | Cited papers have been superseded by newer, more relevant work |
| 3 | Citation format inconsistency | Mix of APA, IEEE, and informal formats across the document |
| 4 | Broken links | DOI links or URLs to papers become inaccessible |

**Solutions:**

| # | Solution |
|---|----------|
| 1 | Map every thesis claim to at least one citation; run a coverage check against thesis sections |
| 2 | Search for papers published 2022-2025 in each category; prioritize recent survey papers |
| 3 | Standardize all citations to APA 7th edition format as required by the university |
| 4 | Use DOIs (persistent identifiers) instead of URLs; maintain local PDF copies in `CITATIONS/` |

**Positive Outcome:**
A well-organized, comprehensive bibliography that supports every claim in the thesis with credible academic sources, strengthening the research's scholarly foundation.

---

## Phase 9: Research & Reporting

### System 19: Thesis Progress Reporter

**Summary:**
Tracks the completion status of each thesis chapter, section, and system implementation. Provides an overview of what has been completed, what is in progress, and what remains, along with milestone dates and blockers. Used to coordinate work and report progress to the thesis adviser.

**Problems & Errors:**

| # | Problem | Description |
|---|---------|-------------|
| 1 | Stale status | Progress tracker not updated as work advances, giving misleading status |
| 2 | Undefined milestones | Vague completion criteria make it unclear when a section is "done" |
| 3 | Scope creep | New requirements are added without updating the plan or timeline |
| 4 | Missing dependencies | Work on a later phase starts before its prerequisite phases are complete |

**Solutions:**

| # | Solution |
|---|----------|
| 1 | Update progress after every significant work session; automate where possible (e.g., detect committed code changes) |
| 2 | Define clear "done" criteria for each section: draft written, reviewed, figures complete, citations added |
| 3 | Freeze scope for each phase before starting; document any changes as formal amendments |
| 4 | Enforce phase dependencies: Phase N cannot start until Phase N-1 deliverables are verified |

**Positive Outcome:**
Clear visibility into thesis completion status, enabling timely course corrections, productive adviser meetings, and on-time submission.

---

### System 20: Results Auto-Generator

**Summary:**
Automatically generates thesis-ready result artifacts from model evaluation outputs: formatted accuracy tables, precision/recall reports, confusion matrix heatmaps, training loss curves, cross-validation summary statistics (mean +/- std), and per-class performance breakdowns. Outputs are saved to `results/` in formats ready for thesis insertion.

**Problems & Errors:**

| # | Problem | Description |
|---|---------|-------------|
| 1 | Figure formatting | Generated plots don't match thesis formatting requirements (font size, DPI, margins) |
| 2 | Stale results | Auto-generated tables reflect an older model version, not the final one |
| 3 | Statistical errors | Incorrect aggregation of cross-validation metrics (e.g., averaging accuracies wrong) |
| 4 | Missing context | Tables lack captions, units, or explanatory notes needed for the thesis |

**Solutions:**

| # | Solution |
|---|----------|
| 1 | Configure matplotlib with thesis-compliant settings: 300 DPI, Times New Roman font, proper axis labels |
| 2 | Regenerate all results from the final model version before thesis submission; timestamp all outputs |
| 3 | Use `CrossValidator._aggregate_results()` for correct mean/std computation; verify against manual calculation |
| 4 | Include auto-generated captions with model version, date, and dataset size in every output |

**Positive Outcome:**
Publication-quality tables, figures, and statistical summaries generated directly from model outputs, eliminating manual transcription errors and ensuring results exactly match the trained model.

---

### System 21: Research Dashboard UI

**Summary:**
A visual interface (web-based or notebook-based) that provides a unified view of the project's status: dataset statistics, class distribution charts, training progress curves, current model performance metrics, and system health indicators. Enables the researcher to monitor the entire pipeline at a glance.

**Problems & Errors:**

| # | Problem | Description |
|---|---------|-------------|
| 1 | Data refresh lag | Dashboard shows cached data instead of current values |
| 2 | Visualization errors | Charts render incorrectly with missing data points or wrong scales |
| 3 | Browser compatibility | Dashboard layout breaks in certain browsers or screen sizes |
| 4 | Overhead | Dashboard development time detracts from core research work |

**Solutions:**

| # | Solution |
|---|----------|
| 1 | Add a refresh button or auto-refresh interval; read directly from source CSVs and result files |
| 2 | Handle missing data gracefully with fallback values; use fixed axis ranges for consistent comparisons |
| 3 | Use Jupyter notebook widgets or simple matplotlib dashboards that work reliably in Colab |
| 4 | Keep it minimal: focus on 5-6 key metrics rather than building a full web application |

**Positive Outcome:**
At-a-glance project monitoring that saves time, catches issues early (e.g., class imbalance, training stalls), and provides compelling visuals for thesis presentations and adviser meetings.

---

## Phase 10: Deployment & Visualization

### System 22: Visual Explainability & Emotion Report

**Summary:**
Generates interpretable visual explanations of model predictions: Grad-CAM heatmaps overlaid on handwriting images showing which regions the CNN attended to, feature importance charts, and per-sample emotion reports with confidence scores. Makes the black-box CNN-HMM model transparent and explainable for the thesis defense.

**Problems & Errors:**

| # | Problem | Description |
|---|---------|-------------|
| 1 | Misleading heatmaps | Grad-CAM highlights irrelevant regions (background, form edges) instead of meaningful handwriting features |
| 2 | Low confidence predictions | Model predictions with near-50% confidence are unreliable but still presented as definitive |
| 3 | Feature attribution noise | Gradient-based explanations are noisy and inconsistent across similar inputs |
| 4 | Report generation failure | Missing dependencies (seaborn, matplotlib) or file path errors crash report generation |

**Solutions:**

| # | Solution |
|---|----------|
| 1 | Apply Grad-CAM to the final convolutional layer; mask out non-handwriting regions before visualization |
| 2 | Display confidence scores alongside predictions; flag predictions below 60% confidence as "uncertain" |
| 3 | Use SmoothGrad (average over multiple noisy inputs) for more stable gradient attributions |
| 4 | Include all visualization dependencies in `requirements.txt`; use absolute paths from `config.py` |

**Positive Outcome:**
Transparent, interpretable model explanations that demonstrate the CNN-HMM system is making predictions based on legitimate handwriting features (slant, pressure, spacing) rather than artifacts, strengthening thesis credibility.

---

### System 23: Emotion Detection System

**Summary:**
The complete end-to-end pipeline that brings all 22 preceding systems together into a unified emotion detection workflow. Given a new handwriting image, it:
1. Validates input quality (System 2)
2. Preprocesses the image through the 5-stage pipeline (System 8)
3. Extracts 256-dimensional features via trained CNN (System 10)
4. Classifies emotion via HMM likelihood comparison (System 13)
5. Generates visual explanation and confidence report (System 22)
6. Returns: **HAPPY** or **SAD** with confidence score and Grad-CAM visualization

This is the capstone system — the deliverable that the thesis demonstrates and defends.

**Problems & Errors:**

| # | Problem | Description |
|---|---------|-------------|
| 1 | Pipeline cascade failure | An error in any upstream system (preprocessing, feature extraction) crashes the entire prediction |
| 2 | Out-of-distribution inputs | Handwriting from outside the target demographic (non-English, left-handed, different age) produces unreliable predictions |
| 3 | Latency | Full pipeline takes too long for real-time or interactive use |
| 4 | Model drift | Performance degrades when deployed on data that differs from the training distribution |
| 5 | Confidence calibration | Reported confidence scores don't accurately reflect true prediction reliability |
| 6 | Ethical misuse | System could be misused for unauthorized psychological assessment or profiling |

**Solutions:**

| # | Solution |
|---|----------|
| 1 | Wrap each stage in try/except with informative error messages; validate intermediate outputs between stages |
| 2 | Document the model's valid input domain clearly; add input validation for image properties and warn on out-of-domain inputs |
| 3 | Optimize preprocessing with vectorized operations; use GPU inference; cache CNN features for repeated predictions |
| 4 | Retrain periodically with new data; monitor prediction confidence distribution for drift indicators |
| 5 | Calibrate confidence using Platt scaling or temperature scaling on the validation set |
| 6 | Include clear disclaimers: this is a research tool, not a clinical diagnostic instrument; require informed consent for any use |

**Positive Outcome:**
A working, end-to-end emotion detection system that accepts a handwriting image and returns an interpretable emotion classification with confidence — the core contribution of the INSIDE-OUT thesis, demonstrating that CNN-HMM hybrid models can recognize emotional states from static handwriting images.

---

## Complete System Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        INSIDE-OUT SYSTEM FLOW                               │
│                   Emotion Detection from Handwriting                        │
└─────────────────────────────────────────────────────────────────────────────┘

PHASE 1: DATA INFRASTRUCTURE
┌──────────────────┐    ┌──────────────────┐
│  [1] Barcode      │───>│  [2] Data Quality │
│  Registry &       │    │  Control          │
│  Auto-Update      │    │                   │
│  (participant IDs)│    │  (validation)     │
└──────────────────┘    └────────┬─────────┘
                                 │
PHASE 2: DATA MANAGEMENT         │
┌──────────────────┐    ┌────────▼─────────┐
│  [3] Participant  │───>│  [4] Database     │
│  Management       │    │  System           │
│  (demographics,   │    │  (CSV metadata    │
│   consent)        │    │   storage)        │
└──────────────────┘    └────────┬─────────┘
                                 │
PHASE 3: FILE ORGANIZATION       │
┌──────────────────┐    ┌────────▼─────────┐
│  [5] File Sorter  │<───│  [6] Page         │
│  (RAW -> LABELED  │    │  Classifier &     │
│   -> PROCESSED)   │    │  Auto-Renamer     │
└────────┬─────────┘    └──────────────────┘
         │
PHASE 4: IMAGE PROCESSING
┌────────▼─────────┐    ┌──────────────────┐    ┌──────────────────┐
│  [7] Data         │───>│  [8] Image        │───>│  [9] Data         │
│  Extraction &     │    │  Preprocessor     │    │  Augmentation     │
│  Content Analyzer │    │  (5-stage pipe)   │    │  (training only)  │
└──────────────────┘    └──────────────────┘    └────────┬─────────┘
                                                         │
PHASE 5: FEATURE ENGINEERING                             │
┌──────────────────┐    ┌────────────────────────────────▼─────────┐
│  [11] Micro-Level │───>│  [10] Feature Extractor                  │
│  Letter & Drawing │    │  (CNN: 224x224 -> 256-dim vector)        │
│  Analysis         │    │  src/models/cnn.py                       │
└──────────────────┘    └────────────────────┬─────────────────────┘
                                             │
PHASE 6: LABELING                            │
┌──────────────────┐                         │
│  [12] Emotion     │─── ground truth ───────┤
│  Labeler          │    labels               │
│  (DASS-21 +       │                         │
│   happiness score)│                         │
└──────────────────┘                         │
                                             │
PHASE 7: MODEL DEVELOPMENT                  │
                        ┌────────────────────▼─────────────────────┐
                        │  [13] CNN-HMM Model Trainer              │
                        │  Stage 1: CNN training (cross-entropy)   │
                        │  Stage 2: HMM fitting (EM algorithm)     │
                        │  src/models/hybrid.py + src/training/    │
                        └────────────────────┬─────────────────────┘
                                             │
PHASE 8: VALIDATION & MONITORING             │
┌──────────────────┐    ┌────────────────────▼──────┐    ┌─────────────────┐
│  [15] Model       │<───│  [14] Model Validator     │───>│  [16] Model      │
│  Performance      │    │  & Tester                 │    │  Versioning      │
│  Monitor          │    │  (5-fold CV + test set)   │    │  (checkpoints)   │
└──────────────────┘    └───────────────────────────┘    └─────────────────┘
                                             │
         ┌───────────────────────────────────┤
         │                                   │
┌────────▼─────────┐              ┌──────────▼─────────┐
│  [17] Master Data │              │  [18] Citations     │
│  Compiler         │              │  Researcher         │
│  (unified dataset)│              │  (100+ papers)      │
└────────┬─────────┘              └──────────┬─────────┘
         │                                   │
PHASE 9: RESEARCH & REPORTING               │
┌────────▼─────────┐    ┌────────────────────▼──────┐
│  [19] Thesis      │<───│  [20] Results             │
│  Progress         │    │  Auto-Generator           │
│  Reporter         │    │  (tables, figures, stats) │
└──────────────────┘    └───────────────────────────┘
                                             │
PHASE 10: DEPLOYMENT & VISUALIZATION         │
┌──────────────────┐    ┌────────────────────▼──────┐
│  [21] Research    │<───│  [22] Visual              │
│  Dashboard UI     │    │  Explainability &         │
│  (monitoring)     │    │  Emotion Report           │
└──────────────────┘    │  (Grad-CAM, reports)      │
                        └───────────┬───────────────┘
                                    │
                        ┌───────────▼───────────────┐
                        │                           │
                        │  ⭐ [23] EMOTION           │
                        │  DETECTION SYSTEM          │
                        │                           │
                        │  Input:  Handwriting Image │
                        │  Output: HAPPY / SAD       │
                        │          + Confidence      │
                        │          + Grad-CAM        │
                        │                           │
                        │  The Complete Pipeline     │
                        └───────────────────────────┘
```

---

## Quick Reference Table

| No. | System Name | Phase | Input | Output | Key Library |
|-----|-------------|-------|-------|--------|-------------|
| 1 | Barcode Registry & Auto-Update | 1 - Data Infrastructure | Participant info | Unique ID (P###) | pandas |
| 2 | Data Quality Control | 1 - Data Infrastructure | Raw images + scores | Validated/rejected flags | opencv-python, pandas |
| 3 | Participant Management | 2 - Data Management | Demographics, consent | participants.csv | pandas |
| 4 | Database System | 2 - Data Management | All CSV tables | Joined metadata | pandas |
| 5 | File Sorter | 3 - File Organization | Labeled files | Organized directories | os, shutil |
| 6 | Page Classifier & Auto-Renamer | 3 - File Organization | Raw scans | Classified + renamed files | Pillow, opencv-python |
| 7 | Data Extraction & Content Analyzer | 4 - Image Processing | Raw image | Cropped writing region | opencv-python |
| 8 | Image Preprocessor | 4 - Image Processing | Cropped image | 224x224 normalized image | opencv-python, scikit-image |
| 9 | Data Augmentation | 4 - Image Processing | Preprocessed images | Augmented training set | torchvision.transforms |
| 10 | Feature Extractor | 5 - Feature Engineering | 224x224 image | 256-dim feature vector | torch, torchvision |
| 11 | Micro-Level Letter & Drawing Analysis | 5 - Feature Engineering | Preprocessed image | Handwriting measurements | opencv-python, numpy |
| 12 | Emotion Labeler | 6 - Labeling & Annotation | Questionnaire scores | HAPPY/SAD/NEUTRAL label | pandas |
| 13 | CNN-HMM Model Trainer | 7 - Model Development | Features + labels | Trained hybrid model | torch, hmmlearn |
| 14 | Model Validator & Tester | 8 - Validation & Monitoring | Trained model + test data | Metrics (acc, F1, etc.) | scikit-learn |
| 15 | Model Performance Monitor | 8 - Validation & Monitoring | Training logs | Loss/accuracy curves | matplotlib |
| 16 | Model Versioning | 8 - Validation & Monitoring | Model weights + config | Versioned checkpoints | torch, pickle |
| 17 | Master Data Compiler | 8 - Validation & Monitoring | All metadata + results | Unified dataset | pandas |
| 18 | Citations Researcher | 9 - Research & Reporting | Research papers | Annotated bibliography | manual / Zotero |
| 19 | Thesis Progress Reporter | 9 - Research & Reporting | Task completion data | Progress summary | markdown |
| 20 | Results Auto-Generator | 9 - Research & Reporting | Model eval outputs | Thesis-ready figures/tables | matplotlib, seaborn |
| 21 | Research Dashboard UI | 10 - Deployment & Visualization | All system data | Visual dashboard | matplotlib, jupyter |
| 22 | Visual Explainability & Emotion Report | 10 - Deployment & Visualization | Model + input image | Grad-CAM + report | torch, matplotlib |
| 23 | Emotion Detection System | 10 - Deployment & Visualization | Handwriting image | Emotion + confidence | torch, hmmlearn, opencv-python |

---

*Generated for the INSIDE-OUT thesis project — University of San Carlos, DCISM*
