# INSIDE-OUT / EMHA — Project Structure

Authoritative as of July 2026. See `ProcessPipeline.txt` for the full
19-phase pipeline (P0-P18). See `CLAUDE.md` for coding rules.

---

## Repository Layout

```
EMHA/
├── src/                        Source code
│   ├── data/                   Pipeline data modules (Phases 1-8)
│   ├── models/                 CNN, HMM, Hybrid model
│   ├── preprocessing/          Image preprocessing pipeline
│   ├── training/               Trainer, evaluator, cross-validator
│   ├── analysis/               Psychometrics (Phase 13)
│   ├── utils/                  Config, artifact generator
│   └── predict.py              End-to-end inference (Phase 11)
├── DATA/
│   ├── METADATA/               All pipeline CSVs (tracked in git)
│   ├── CROPS/                  Extracted crops — GITIGNORED
│   └── PROCESSED/              Preprocessed images — GITIGNORED
├── DATASET/
│   └── raw/                    Original scans — READ-ONLY, gitignored
├── FIGURES/                    Thesis figures 300 DPI — GITIGNORED
├── models/                     Checkpoints (*.pth, *.pkl) — GITIGNORED
├── results/                    CV results, evaluation CSVs
├── notebooks/
│   └── EMHA_Colab_Pipeline.ipynb  Colab GPU training notebook
├── DOCS/                       Thesis docs and instrument scans
├── CITATIONS/                  Research papers
├── ProcessPipeline.txt         AUTHORITATIVE 19-phase pipeline
├── CLAUDE.md                   Claude Code rules and commands
├── SYSTEMS_GUIDE.md            Detailed system documentation
├── smoke_test.py               Phase 14 smoke test (run before full run)
├── requirements.txt            Python dependencies
├── setup.cfg                   Black / pytest config
└── .flake8                     Flake8 linting config
```

---

## Source Code (`src/`)

### `src/data/` — Pipeline Phases 1-8

| File | Phase | Creates |
|------|-------|---------|
| `register_participants.py` | P1 | `participants.csv` |
| `assign_pages.py` | P2 | `page_manifest.csv` |
| `questionnaire_scorer.py` | P3 | `labels.csv` |
| `content_extractor.py` | P4 | `DATA/CROPS/{P###}/`, `extraction_report.csv` |
| `propagate_labels.py` | P5 | `crop_index.csv` |
| `qc.py` | P6 | `qc_report.csv`, `exclusions.csv` |
| `participant_aware_splitter.py` | P8 | `splits.json` |
| `dataloader.py` | P9 | PyTorch Dataset class |
| `preview_crops.py` | util | Crop tuning visualisation |

### `src/preprocessing/`

| File | Phase | Description |
|------|-------|-------------|
| `pipeline.py` | P7 | Grayscale→binarize→denoise→deskew→224×224→norm |
| `run_preprocessing.py` | P7 | Batch runner → `DATA/PROCESSED/{P###}/` |

### `src/models/`

| File | Description |
|------|-------------|
| `cnn.py` | `PretrainedCNNExtractor` (ResNet18, primary) + `CNNFeatureExtractor` (4-block, secondary) |
| `hmm.py` | `HMMClassifier` — per-class GaussianHMM, 4 states, diagonal covariance |
| `hybrid.py` | `HybridCNNHMM` — CNN→sequence→HMM→prediction |

### `src/training/`

| File | Phase | Description |
|------|-------|-------------|
| `trainer.py` | P9 | Training loop, early stopping, CNN + baseline |
| `evaluator.py` | P10 | Metrics: accuracy, precision, recall, F1, confusion matrix, ROC-AUC |
| `cross_validate.py` | P10 | Respondent-level 5-fold CV using splits.json |

### `src/utils/`

| File | Description |
|------|-------------|
| `config.py` | All hyperparameters in dataclasses. Import: `from src.utils.config import config` |
| `artifact_generator.py` | Phase 12 — confusion matrix, ROC, fold bars, dataset charts @ 300 DPI |

### `src/analysis/`

| File | Phase | Description |
|------|-------|-------------|
| `psychometrics.py` | P13 | Cronbach's alpha, item-total correlations, score distribution |

### `src/predict.py` — Phase 11

End-to-end inference: raw scanned folder (or single crop) → HAPPY/SAD + confidence.
No questionnaire at inference. Defense demo artifact.

---

## Data Directory (`DATA/`)

```
DATA/
├── METADATA/               Tracked in git (pipeline CSVs)
│   ├── participants.csv    P###, folder, barcode status
│   ├── page_manifest.csv   P###, page_index, file, page_role
│   ├── labels.csv          P###, age, gender, scores, p_happy, label
│   ├── extraction_report.csv  per-crop success/failure
│   ├── crop_index.csv      crop_path, P###, task_code, task_type, label, p_happy
│   ├── qc_report.csv       quality gate summary
│   ├── exclusions.csv      excluded crops with reasons
│   ├── splits.json         train/val/test + 5-fold assignments
│   ├── preprocessing_log.csv
│   └── psychometrics_report.csv
│
├── CROPS/                  GITIGNORED — Phase 4 output
│   └── P001/
│       ├── P001_draw_circles.png
│       ├── P001_word_content_left.png
│       └── ...  (24 files per respondent)
│
└── PROCESSED/              GITIGNORED — Phase 7 output (mirrors CROPS)
    └── P001/
        └── ...  (224×224 preprocessed PNG)
```

---

## Labeling — FINALE 24-item (Section A, non-negotiable)

```
happiness_items  = (2,4,6,8,10,12,13,16,19,20,21,23)   kept raw
sadness_items    = (1,3,5,7,9,11,14,15,17,18,22,24)    kept raw
adjusted_total   = happiness_sum + (72 - sadness_sum)   range 24-120
label            = HAPPY if adjusted_total >= 72 else SAD
p_happy          = (adjusted_total/24 - 1) / 4          stored only
```

**NO NEUTRAL class.** All 400 respondents are labelled and included.

---

## Configuration (`src/utils/config.py`)

```python
# Key defaults
config.data.raw_data_dir      = "E:\\EMHA_Thesis\\DATASET\\raw"  # READ-ONLY
config.data.crops_dir         = "DATA/CROPS"
config.data.processed_dir     = "DATA/PROCESSED"
config.cnn.use_pretrained     = True          # ResNet18
config.cnn.num_features       = 256
config.hmm.n_states           = 4
config.hmm.covariance_type    = "diag"
config.training.batch_size    = 32
config.training.epochs        = 100
config.training.learning_rate = 0.001
config.training.n_folds       = 5
config.training.random_state  = 42
config.labeling.adjusted_total_threshold = 72
```

Crop bounding boxes (fractional 0.0-1.0): `DRAWING_CROPS`, `WORD_CROPS`, `CURSIVE_CROPS`.
Tune against real scans using `python -m src.data.preview_crops P001`.

---

## Sequencing Rule (Phase 14)

smoke test (3-5 respondents) → 30-respondent pilot → full run.
Never scale before the previous stage passes.

*Last updated: July 2026*
