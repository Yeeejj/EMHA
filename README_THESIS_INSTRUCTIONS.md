# INSIDE-OUT / EMHA — Thesis Implementation Guide

**Emotion Recognition through Handwriting Analysis using a CNN-HMM Hybrid Approach**
Cyrel Jane A. Edano | University of San Carlos, DCISM
Adviser: Christian V. Maderazo, M.Eng.

> **Authoritative reference: `ProcessPipeline.txt`** (19-phase P0-P18, July 2026).
> This document summarises the implementation order and key decisions.

---

## Quick Start

```bash
git clone https://github.com/Yeeejj/EMHA.git
cd EMHA
python -m venv venv && venv\Scripts\activate
pip install -r requirements.txt
python smoke_test.py          # must pass before any full run
```

---

## Thesis Goal

Predict a person's **concurrent self-reported emotional state (HAPPY or SAD)**
from offline scanned handwriting and drawing images alone.

- Questionnaire used **only** to create ground-truth training labels.
- At inference: raw scanned pages in → `HAPPY` or `SAD` out (no questionnaire).
- Primary contribution: the **FINALE dataset** and the extraction pipeline.
- Accuracy target: **70-85%** (secondary metric; contribution is the dataset).

---

## Non-Negotiable Decisions

| Rule | Decision |
|------|----------|
| Labels | **Binary only: HAPPY or SAD. NO NEUTRAL.** All 400 respondents labelled. |
| Scoring | `adjusted_total = happiness_sum + (72 - sadness_sum)` → HAPPY if ≥ 72 |
| Split | **Participant-level only.** All 24 crops of one respondent in the same split. |
| Deskew | `skip_skew=True` for `draw_*`; `False` for `word_*` and `cursive_*` |
| Framework | **PyTorch only.** No TensorFlow. |
| CNN | **ResNet18 pretrained** (`use_pretrained=True`). |
| Config | All hyperparams in dataclasses in `src/utils/config.py`. |
| Raw data | `DATASET/raw` is **READ-ONLY**. Never write or modify it. |

---

## 19-Phase Implementation Order

### Gates (do these first)

| Gate | Action |
|------|--------|
| Gate 1 | ~~Fix `hybrid.py` n_features bug~~ — **already fixed** |
| Gate 2 | Run `python -m src.data.questionnaire_scorer` after encoding responses |
| Gate 3 | Run `python -m src.data.content_extractor` after tuning crop boxes |
| Gate 4 | ~~Rewrite docs for NEUTRAL/DASS-21~~ — **already done** (this file) |

### Full Pipeline

```
P0  environment setup         pip install -r requirements.txt
P1  register participants     python -m src.data.register_participants
P2  assign page roles         python -m src.data.assign_pages
P3  score questionnaire       python -m src.data.questionnaire_scorer
P4  extract content           python -m src.data.content_extractor
P5  propagate labels          python -m src.data.propagate_labels
P6  quality control           python -m src.data.qc
P7  preprocess images         python -m src.preprocessing.run_preprocessing
P8  participant-aware split   python -m src.data.participant_aware_splitter
P9  train model (Colab)       EMHA_Colab_Pipeline.ipynb
P10 evaluate (Colab)          EMHA_Colab_Pipeline.ipynb
P11 end-to-end inference      python -m src.predict <path>
P12 generate figures          python -m src.utils.artifact_generator
P13 psychometrics             python -m src.analysis.psychometrics
P14 smoke test + pilot        python smoke_test.py → 30-respondent pilot
P15 thesis chapters 5-6       Claude.ai drafting support
P16 defense preparation       deck + predict.py demo rehearsal
P17 revisions                 incorporate panel feedback
P18 submission + binding      tag GitHub release
```

---

## Labeling Detail (Section A)

```python
happiness_items = (2, 4, 6, 8, 10, 12, 13, 16, 19, 20, 21, 23)  # kept raw
sadness_items   = (1, 3, 5, 7, 9, 11, 14, 15, 17, 18, 22, 24)   # kept raw

happiness_sum   = sum(q[i] for i in happiness_items)   # range 12-60
sadness_sum     = sum(q[i] for i in sadness_items)     # range 12-60
adjusted_total  = happiness_sum + (72 - sadness_sum)   # range 24-120
label           = "HAPPY" if adjusted_total >= 72 else "SAD"
p_happy         = (adjusted_total / 24 - 1) / 4        # stored only
```

Integer comparisons only. `p_happy` stored for calibration analysis but
**not used** for the binary decision.

---

## 24 Crops per Respondent

| Task type | Count | Task codes |
|-----------|-------|------------|
| Drawing | 4 | `draw_circles`, `draw_dots`, `draw_person`, `draw_house` |
| Word (5×3 table) | 15 | `word_{word}_{hand}` |
| Cursive | 5 | `cursive_01` … `cursive_05` |

Filename format: `{participant_id}_{task_code}.png` (e.g. `P001_draw_circles.png`)

---

## Crop Bounding Boxes

Stored as **fractional (0.0-1.0)** coordinates in `config.py` (`DRAWING_CROPS`,
`WORD_CROPS`, `CURSIVE_CROPS`). Resolution-independent — valid for any scan DPI.

Tune visually: `python -m src.data.preview_crops P001`

---

## Data Flow

```
DATASET/raw (READ-ONLY)
  ↓ Phase 4
DATA/CROPS/{P###}/      (24 crops, gitignored)
  ↓ Phase 7
DATA/PROCESSED/{P###}/  (224×224 preprocessed, gitignored)
  ↓ Phase 8
DATA/METADATA/splits.json  (train/val/test + 5-fold — tracked in git)
  ↓ Phase 9-10 (Colab)
models/, results/
```

---

## Circularity Acknowledgement

Labels are derived from self-report questionnaires and used to train a model
that predicts those same self-reports from handwriting. This concurrent
self-report design is a **scoped limitation**, not a hidden flaw:
- Chapter 6 discusses it explicitly.
- The thesis claim is bounded: "predicting concurrent self-reported emotional
  state from offline scanned handwriting," not trait emotion or ground-truth
  emotion.
- The panel should expect this question. Lead with it in the defense.

---

## Ethical Compliance

- Informed consent obtained from all respondents.
- Data stored locally; raw scans not committed to GitHub.
- Psychometrician supervision documented in SYSTEMS_GUIDE.md.
- Self-report circularity disclosed in manuscript.

---

*Last updated: July 2026*
