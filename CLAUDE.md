# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**INSIDE-OUT** — Emotion recognition (happiness vs. sadness) from handwriting/drawing samples using a hybrid CNN-HMM architecture. Thesis project by Cyrel Jane A. Edaño at University of San Carlos, DCISM. Target: 80-85% classification accuracy on 300+ right-handed participants aged 18-25.

## Rules
Always before making any change. Search on the web for the newst documentation.
And only implement if you are 100% sure it will work.

## Commands

```bash
pip install -r requirements.txt

# Pipeline stages (run as modules from project root)
python -m src.data.labeler             # Label samples from questionnaire scores
python -m src.preprocessing.pipeline   # Preprocess images
python -m src.training.trainer         # Train hybrid model
python -m src.training.evaluator       # Evaluate and generate metrics

# Tests and linting
pytest
black .
flake8
```

The C files at the root (`pharmacy_queue.c`, `queue_sorting.c`, `stack_sorting.c`, `warehouse_inventory.c`) are standalone data structure exercises, not part of the ML pipeline. Compile with `cc -o <name> <name>.c`.

## Architecture

Two-stage hybrid model in `src/`:

- **CNN** (`src/models/cnn.py`): Extracts 256-dim spatial features from 224×224 grayscale images. Optional pretrained ResNet18 backbone.
- **HMM** (`src/models/hmm.py`): Classifies sequential patterns using 4-state diagonal-covariance HMMs (one per emotion class).
- **Hybrid** (`src/models/hybrid.py`): `HybridCNNHMM` orchestrates CNN → HMM → prediction with confidence.

### Data Pipeline

```
DATA/RAW/ → labeler → DATA/LABELED/{HAPPY,SAD}/ → pipeline → DATA/PROCESSED/ → DATA/SPLITS/{train,val,test}/
```

- `src/data/collector.py` — Participant registration, barcode IDs (zero-padded `P001` format)
- `src/data/labeler.py` — Assigns HAPPY/SAD from DASS-21 + happiness scores
- `src/data/dataloader.py` — PyTorch Dataset for preprocessed images
- `src/preprocessing/pipeline.py` — Grayscale → binarization → denoise → skew correction → normalize
- `src/training/trainer.py` — Training with early stopping (patience=10), checkpointing
- `src/training/cross_validate.py` — Stratified 5-fold cross-validation
- `src/training/evaluator.py` — Accuracy, precision, recall, F1, confusion matrix

### Configuration

All hyperparameters in `src/utils/config.py` via dataclasses. Import with `from src.utils.config import config`. Key defaults: 224×224 images, batch_size=32, lr=0.001, 100 epochs, 5-fold CV.

### Labeling Thresholds

| Label | Condition |
|-------|-----------|
| HAPPY | happiness_score ≥ 40 AND dass_depression < 14 |
| SAD | happiness_score < 30 OR dass_depression ≥ 14 |
| NEUTRAL | Otherwise (excluded from training) |

## Key Files

- `DATA/METADATA/` — `participants.csv`, `questionnaire_scores.csv`, `labels.csv`
- `SYSTEMS_GUIDE.md` — Detailed documentation of all 23 systems across 10 phases
- `notebooks/EMHA_Colab_Pipeline.ipynb` — Google Colab training pipeline

## Dependencies

Core: PyTorch, hmmlearn, OpenCV, scikit-learn, pandas, numpy, matplotlib. See `requirements.txt`.
