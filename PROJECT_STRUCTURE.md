# INSIDE-OUT Project Structure

This document describes the folder structure and organization of the INSIDE-OUT emotion recognition system.

---

## Directory Overview

```
EMHA/
├── src/                    # Source code modules
├── DATA/                   # All data files
├── notebooks/              # Jupyter notebooks for exploration
├── models/                 # Saved model weights
├── results/                # Evaluation outputs and reports
├── CITATIONS/              # Research papers and references
├── DOCS/                   # Thesis documents and proposals
├── requirements.txt        # Python dependencies
├── README.md               # Project overview
└── PROJECT_STRUCTURE.md    # This file
```

---

## Source Code (`src/`)

### `src/data/` - Data Management

| File | Description |
|------|-------------|
| `collector.py` | Handles image capture, participant registration, and sample organization |
| `labeler.py` | Assigns emotion labels (HAPPY/SAD) based on questionnaire scores |
| `dataloader.py` | PyTorch dataset class for loading preprocessed images |

### `src/preprocessing/` - Image Processing

| File | Description |
|------|-------------|
| `pipeline.py` | Complete preprocessing pipeline: grayscale → binarization → denoising → skew correction → normalization |

### `src/models/` - Neural Network Models

| File | Description |
|------|-------------|
| `cnn.py` | CNN feature extractor for spatial pattern recognition |
| `hmm.py` | HMM classifier for temporal sequence classification |
| `hybrid.py` | Combined CNN-HMM model (main classification system) |

### `src/training/` - Training & Evaluation

| File | Description |
|------|-------------|
| `trainer.py` | Training loop with validation, early stopping, checkpointing |
| `evaluator.py` | Computes metrics: accuracy, precision, recall, F1-score, confusion matrix |
| `cross_validate.py` | Stratified 5-fold cross-validation |

### `src/utils/` - Utilities

| File | Description |
|------|-------------|
| `config.py` | Centralized configuration for hyperparameters and paths |

---

## Data Directory (`DATA/`)

```
DATA/
├── RAW/                    # Original unprocessed samples
│   ├── participant_001/
│   │   ├── handwriting_01.png
│   │   ├── handwriting_02.png
│   │   └── questionnaire.pdf
│   ├── participant_002/
│   └── ...
│
├── LABELED/                # After emotion assignment
│   ├── HAPPY/
│   │   ├── P001_sample_01.png
│   │   └── ...
│   └── SAD/
│       ├── P002_sample_01.png
│       └── ...
│
├── PROCESSED/              # After preprocessing (ready for training)
│   ├── HAPPY/
│   └── SAD/
│
├── SPLITS/                 # Train/validation/test splits
│   ├── train/
│   │   ├── HAPPY/
│   │   └── SAD/
│   ├── val/
│   │   ├── HAPPY/
│   │   └── SAD/
│   └── test/
│       ├── HAPPY/
│       └── SAD/
│
└── METADATA/               # CSV files for tracking
    ├── participants.csv
    ├── questionnaire_scores.csv
    └── labels.csv
```

### Data Flow

```
1. COLLECT          2. SCORE             3. LABEL            4. PREPROCESS        5. SPLIT
┌─────────┐      ┌──────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  RAW/   │  →   │ questionnaire │  →  │  LABELED/   │  →  │ PROCESSED/  │  →  │  SPLITS/    │
│(scanned)│      │  _scores.csv  │     │ HAPPY/ SAD/ │     │ HAPPY/ SAD/ │     │train/val/test│
└─────────┘      └──────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
```

---

## Metadata Files

### `participants.csv`

Tracks participant demographics and consent.

```csv
participant_id,age,gender,handedness,consent_date,notes
P001,22,F,R,2025-03-01,
P002,19,M,R,2025-03-01,
P003,24,F,R,2025-03-02,glasses
```

| Column | Description |
|--------|-------------|
| `participant_id` | Unique ID (P001, P002, ...) |
| `age` | Participant age (18-25) |
| `gender` | M/F/Other |
| `handedness` | R (right-handed only for this study) |
| `consent_date` | Date consent was given |
| `notes` | Any additional notes |

### `questionnaire_scores.csv`

Stores DASS-21 and happiness scale scores.

```csv
participant_id,dass_depression,dass_anxiety,dass_stress,happiness_score,assigned_emotion
P001,5,8,10,48,HAPPY
P002,22,18,25,18,SAD
P003,3,5,7,52,HAPPY
```

| Column | Description |
|--------|-------------|
| `dass_depression` | DASS-21 depression subscale (0-42) |
| `dass_anxiety` | DASS-21 anxiety subscale (0-42) |
| `dass_stress` | DASS-21 stress subscale (0-42) |
| `happiness_score` | Oxford/SDHS happiness score |
| `assigned_emotion` | Final label: HAPPY, SAD, or NEUTRAL |

### `labels.csv`

Maps image files to emotion labels.

```csv
filename,participant_id,emotion,collection_date,validated
P001_sample_01.png,P001,HAPPY,2025-03-01,TRUE
P001_sample_02.png,P001,HAPPY,2025-03-01,TRUE
P002_sample_01.png,P002,SAD,2025-03-01,FALSE
```

---

## Labeling Thresholds

Emotions are assigned based on questionnaire scores:

| Condition | Label |
|-----------|-------|
| `happiness_score >= 40` AND `dass_depression < 14` | **HAPPY** |
| `happiness_score < 30` OR `dass_depression >= 14` | **SAD** |
| Otherwise | NEUTRAL (excluded from training) |

---

## Model Outputs (`models/`)

```
models/
├── best_model.pth          # Best CNN weights (based on val loss)
├── cnn_epoch_50.pth        # Checkpoint at epoch 50
├── hmm_happy.pkl           # Trained HMM for HAPPY class
├── hmm_sad.pkl             # Trained HMM for SAD class
└── hybrid_final.pkl        # Complete hybrid model
```

---

## Results (`results/`)

```
results/
├── training_history.json    # Loss and accuracy per epoch
├── confusion_matrix.png     # Visualization
├── classification_report.txt
├── cross_validation_results.csv
└── predictions/
    └── test_predictions.csv
```

---

## Notebooks (`notebooks/`)

| Notebook | Purpose |
|----------|---------|
| `01_data_exploration.ipynb` | Explore raw data, check class balance |
| `02_preprocessing_test.ipynb` | Test preprocessing pipeline |
| `03_model_training.ipynb` | Train and tune the hybrid model |
| `04_evaluation.ipynb` | Generate final evaluation metrics |

---

## File Naming Conventions

### Handwriting Samples

```
{participant_id}_sample_{number}_{date}.png

Examples:
P001_sample_01_20250301.png
P001_sample_02_20250301.png
P042_sample_01_20250315.png
```

### Model Checkpoints

```
{model_type}_{description}.{ext}

Examples:
cnn_epoch_50.pth
hmm_happy.pkl
hybrid_best_f1_0.85.pkl
```

---

## Configuration (`src/utils/config.py`)

Key hyperparameters defined in `config.py`:

```python
# Image settings
image_size = (224, 224)

# CNN settings
num_features = 256
dropout_rate = 0.5

# HMM settings
n_states = 4
covariance_type = "diag"

# Training settings
batch_size = 32
epochs = 100
learning_rate = 0.001
patience = 10  # Early stopping

# Cross-validation
n_folds = 5
```

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Collect Data

Place scanned handwriting samples in `DATA/RAW/participant_XXX/`

### 3. Score Questionnaires

Fill in `DATA/METADATA/questionnaire_scores.csv` with DASS and happiness scores

### 4. Label Samples

```bash
python -m src.data.labeler
```

### 5. Preprocess Images

```bash
python -m src.preprocessing.pipeline
```

### 6. Train Model

```bash
python -m src.training.trainer
```

### 7. Evaluate

```bash
python -m src.training.evaluator
```

---

## Dependencies

Core libraries used in this project:

| Library | Purpose |
|---------|---------|
| `torch` | Deep learning (CNN) |
| `hmmlearn` | Hidden Markov Models |
| `opencv-python` | Image preprocessing |
| `scikit-learn` | Metrics, cross-validation |
| `pandas` | Data management |
| `numpy` | Numerical operations |
| `matplotlib` | Visualization |

---

*Last updated: March 2026*
