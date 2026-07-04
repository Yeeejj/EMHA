# CLAUDE.md

This file provides guidance to Claude Code when working in this repository.
Supersedes all prior versions. Authoritative source: ProcessPipeline.txt.

## Project

INSIDE-OUT — Emotion recognition (HAPPY or SAD) from offline scanned handwriting
and drawing samples using a hybrid CNN-HMM architecture.
Author: Cyrel Jane A. Edano | University of San Carlos, DCISM
Adviser: Christian V. Maderazo, M.Eng.
Target: 70-85% accuracy. Primary contribution = FINALE dataset + pipeline.

## Non-Negotiable Rules (Section A)

1. **Binary labels only: HAPPY or SAD.** No NEUTRAL class anywhere — not in
   code, docs, or comments. All 400 respondents contribute to training.
2. **Scoring scheme (FINALE 24-item, Likert 1-5):**
   ```
   happiness_items = (2,4,6,8,10,12,13,16,19,20,21,23)  kept raw
   sadness_items   = (1,3,5,7,9,11,14,15,17,18,22,24)   kept raw
   adjusted_total  = happiness_sum + (72 - sadness_sum)   range 24-120
   label           = HAPPY if adjusted_total >= 72 else SAD
   p_happy         = (adjusted_total/24 - 1) / 4          stored only
   ```
   Integer comparisons only at the >= 72 boundary.
3. **Participant-level splitting only.** All 24 crops of one respondent must
   stay in the same split. File-level splitting = data leakage.
4. `skip_skew = True` for all `draw_*` task codes.
   `skip_skew = False` for all `word_*` and `cursive_*` task codes.
5. **PyTorch only.** No TensorFlow anywhere.
6. **ResNet18 pretrained backbone.** `use_pretrained = True` in CNNConfig.
7. All hyperparameters live in dataclasses in `src/utils/config.py`.
   Modules run as `python -m src.*` from the project root.
8. **`DATASET/raw` is READ-ONLY.** Never write, rename, or modify anything inside it.
9. Self-report circularity is acknowledged openly in Chapter 6, not hidden.

## Raw Dataset Structure

```
E:\EMHA_Thesis\DATASET\raw\        (READ-ONLY)
  respondent_001\
    PNG\    (exactly 4 PNG files, alphabetical = chronological order)
    PDF\    (4 PDF files, archival backup only)
  respondent_002\
  ...
```

File naming: `EMHA{YYYYMMDD}_{HHMMSSff}.ext`  
Page roles (0-indexed): 0=questionnaire, 1=drawing, 2-3=writing  
Barcode rule: `respondent_001` → `P001`

## Assessment Structure (FINALE)

- **Page 0** (questionnaire): 24-item Likert grid — ground-truth label source only.
- **Page 1** (drawing): 2x2 grid — Overlapping Circles (top-left), Connect the
  Dots (top-right), Person in the Rain (bottom-left), House and Tree (bottom-right).
- **Pages 2-3** (writing): 5x3 word table (Content, Melancholic, Optimistic,
  Disconnected, Vibrant × left/right/uppercase) + 5 cursive sentence rows.

24 crops per participant: 4 drawings + 15 word cells + 5 cursive cells.

## Task Codes

```
draw_circles, draw_dots, draw_person, draw_house
word_content_left, word_content_right, word_content_upper
word_melancholic_left, word_melancholic_right, word_melancholic_upper
word_optimistic_left, word_optimistic_right, word_optimistic_upper
word_disconnected_left, word_disconnected_right, word_disconnected_upper
word_vibrant_left, word_vibrant_right, word_vibrant_upper
cursive_01, cursive_02, cursive_03, cursive_04, cursive_05
```

Output filename: `{participant_id}_{task_code}.png`

## Data Pipeline (19 phases — P0-P18)

```
DATASET/raw (READ-ONLY)
  P1: register_participants    -> DATA/METADATA/participants.csv
  P2: assign_pages             -> DATA/METADATA/page_manifest.csv
  P3: questionnaire_scorer     -> DATA/METADATA/labels.csv
  P4: content_extractor        -> DATA/CROPS/{P###}/  (24 crops each)
                                  DATA/METADATA/extraction_report.csv
  P5: propagate_labels         -> DATA/METADATA/crop_index.csv
  P6: qc                       -> DATA/METADATA/qc_report.csv
                                  DATA/METADATA/exclusions.csv
  P7: run_preprocessing        -> DATA/PROCESSED/{P###}/  (mirrors CROPS)
                                  DATA/METADATA/preprocessing_log.csv
  P8: participant_aware_splitter -> DATA/METADATA/splits.json
  P9: trainer (Colab)          -> models/ + results/training_log.csv
  P10: evaluator (Colab)       -> results/ (first and only test-split touch)
  P11: predict                 -> HAPPY/SAD + confidence (demo)
  P12: artifact_generator      -> FIGURES/ (300 DPI)
  P13: psychometrics           -> DATA/METADATA/psychometrics_report.csv
  P14: smoke_test.py           -> smoke test before any full run
```

## Commands

```bash
pip install -r requirements.txt

python -m src.data.register_participants       # Phase 1
python -m src.data.assign_pages               # Phase 2
python -m src.data.questionnaire_scorer       # Phase 3
python -m src.data.content_extractor          # Phase 4
python -m src.data.propagate_labels           # Phase 5
python -m src.data.qc                         # Phase 6
python -m src.preprocessing.run_preprocessing # Phase 7
python -m src.data.participant_aware_splitter # Phase 8
python -m src.training.trainer                # Phase 9 (Colab)
python -m src.training.evaluator              # Phase 10 (Colab)
python -m src.predict <path>                  # Phase 11
python -m src.utils.artifact_generator        # Phase 12
python -m src.analysis.psychometrics          # Phase 13
python smoke_test.py                          # Phase 14 (run first)

black .
flake8
pytest
```

## Architecture

**CNN** (`src/models/cnn.py`): `PretrainedCNNExtractor` — ResNet18 backbone,
1-channel grayscale input (conv1 weights averaged from RGB), 256-dim spatial
features. Custom 4-block `CNNFeatureExtractor` kept for reference only.

**HMM** (`src/models/hmm.py`): One `GaussianHMM` per class (4 states, diagonal
covariance). Classification by log-likelihood comparison.

**Hybrid** (`src/models/hybrid.py`): `HybridCNNHMM` — CNN → spatial sequence
(batch, seq_len, 256) → per-class HMM → HAPPY/SAD + confidence.

**Baseline**: Logistic regression on handcrafted graphological features for
thesis comparison (mean intensity, pixel density, slant angle).

## Configuration

All hyperparameters in `src/utils/config.py`. Import with:
```python
from src.utils.config import config
```

Key entries:
- `config.data.raw_data_dir = "E:\\EMHA_Thesis\\DATASET\\raw"` (READ-ONLY)
- `config.data.crops_dir = "DATA/CROPS"` (Phase 4 output)
- `config.data.processed_dir = "DATA/PROCESSED"` (Phase 7 output)
- `config.cnn.use_pretrained = True` (ResNet18)
- `config.labeling.adjusted_total_threshold = 72`
- Crop coords: `DRAWING_CROPS`, `WORD_CROPS`, `CURSIVE_CROPS` (fractional 0.0-1.0)

## Key Files

```
DATA/METADATA/          - all pipeline CSVs
FIGURES/                - thesis figures (300 DPI, Phase 12)
notebooks/EMHA_Colab_Pipeline.ipynb  - Colab training pipeline
ProcessPipeline.txt     - authoritative 19-phase pipeline (supersedes all)
smoke_test.py           - run before any full data processing
```

## Git Rules

Never commit image data. `.gitignore` excludes:
`DATASET/`, `DATA/CROPS/*`, `DATA/PROCESSED/*`, `DATA/STAGING/`, `DATA/SPLITS/`,
`FIGURES/*`, `models/*.pth`, `models/*.pkl`.
Only source code and `DATA/METADATA/*.csv` go to GitHub.

## Dependencies

PyTorch, hmmlearn, OpenCV, scikit-learn, pandas, numpy, matplotlib, seaborn,
Pillow, pyzbar, openpyxl. See `requirements.txt`.
