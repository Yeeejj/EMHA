# EMHA Evaluation Protocol

Status: **DRAFT — approved text pending freeze.** The protocol is frozen by the
git tag `protocol-frozen` on the commit that adds this file. That commit is made
only after labeling is complete and `folds.csv` has been regenerated on the
final labels (section 3). No model may be run on real data before the tag.
Fields marked `[AT FREEZE]` are filled in that commit.

| Item | Value |
|---|---|
| Approved by thesis author | 2026-10-05 (text); revision 1 approved 2026-10-05; revision 2 pending re-approval |
| Frozen (tag `protocol-frozen`) | `[AT FREEZE]` date and commit |
| Code base at approval | `3e222ff` |

## 1. Question and analysis sets

Binary classification of a participant's emotional state (HAPPY vs SAD) from
offline scanned handwriting and drawings. Labels come verbatim from
`DATASET/metadata/questionnaire_export.csv` via `labels.csv` (never
recomputed).

**Analysis design (B3 decision).** Extreme-groups design, decided from
`labels.csv` score data only, before any model result, in commit `2bb867a`
(2026-10-04, `DOCS/decisions/analysis_design.txt`):

- Middle-band fraction: **0.20** of participants closest to the cutoff (72)
  are set aside (`in_middle_band = True`).
- At decision time: primary set 228 participants (HAPPY 126, SAD 102),
  middle band 57. Final counts on the complete sample: `[AT FREEZE]`.

| Set | Definition | Role |
|---|---|---|
| **Primary** | `in_primary_analysis` and QC status `qc_passed` | All headline results |
| **Full sample** | every `qc_passed` participant | Secondary. Baselines required (majority, LR handcrafted, LR embeddings); CNN models optional if compute-limited, and if run, reported in full |

Membership comes from `src.training.splits.analysis_ids`. Participants are
never added or removed based on predictions.

## 2. Unit of analysis and metrics

The unit is the **participant**. Crop-level models (fine-tuned ResNet18,
CNN-HMM) produce crop-level `prob_sad`, aggregated per participant by
`src.training.aggregate.aggregate_crops` with **`mean_prob`** (mean of all of
the participant's crop probabilities). The LR baselines are fit on
participant-level features and output participant probabilities directly.
`pred = SAD` iff `prob_sad >= 0.5`. The threshold is fixed and never tuned.

- **Primary metric:** participant-level **macro-F1**.
- **Also reported:** accuracy, balanced accuracy, precision and recall (per
  class, SAD as positive for ROC), ROC-AUC, confusion matrix, and per-family
  macro-F1 (drawing, word, cursive — each aggregating only that family's
  crops with `mean_prob`).
- **Accuracy is always shown next to the majority baseline.** No accuracy is
  reported without the majority baseline and macro-F1.
- **Majority baseline:** in each outer fold, predict the majority label of
  that fold's training participants (inner train + inner validation).
- **Pooling:** metrics are computed once on the pooled out-of-fold
  participant predictions (each participant predicted exactly once). Per-fold
  values are reported as well, all folds, never only the best.
- `mean_logit` and `majority_vote` aggregation are secondary sensitivity
  analyses only.

Target (not a hypothesis test): primary-set macro-F1 and accuracy >= 0.70.

## 3. Cross-validation

- **Outer CV:** 5 folds from `DATASET/metadata/folds.csv`, one row per
  participant, made by `python -m src.training.splits`: sklearn
  `StratifiedKFold` on the participant table, strata label x `in_middle_band`,
  seed 42 (`TrainingConfig.seed`), over **all labeled participants**. Runners
  filter it to the analysis set at load time.
  - `folds.csv` SHA-256: `[AT FREEZE]` (regenerated with `--force` once
    labeling is complete; the provisional 285-participant file is
    `fdf5a63e03ab22690908e52a4664508a16630daef8bbeb83f00d0471baea7106`).
  - `labels.csv` SHA-256: `[AT FREEZE]`.
- **Inner validation, two fixed mechanisms**, both on outer-train
  participants only:
  - *LR baselines:* C is chosen by stratified 5-fold inner CV (macro-F1) over
    the outer-train participants, then refit on all of them.
  - *CNN models:* `splits.inner_split` holds out a stratified 15% of the
    outer-train participants (seed 42 + fold) for early stopping and Platt
    calibration only.
  - *CNN-HMM:* the same inner-validation participants also select the HMM
    setting (see section 5) before Platt calibration on that set. Selecting
    and calibrating on one set makes the calibration mildly optimistic;
    this is reported as a limitation.

  Nothing is ever fit, selected, or calibrated on outer test participants.
- **Leakage:** `assert_no_leakage(inner_train, inner_val, test)` is called in
  every fold (`CrossValidator.splits`), and each fold's predictions must cover
  exactly its test participants. Scalers and PCA (64 components for
  embeddings) are fit on inner-train participants only.

## 4. Uncertainty and significance

- **Bootstrap CIs:** 2,000 participant-level resamples (with replacement) of
  the pooled out-of-fold predictions; 95% percentile intervals for every
  metric in section 2. Seed 42.
- **Paired comparison vs LR-handcrafted:** for every other model, the
  difference in macro-F1 on the same 2,000 bootstrap resamples (paired by
  participant); report the mean difference and its 95% CI.
- **Permutation tests** (null: no association between handwriting and label).
  In each permutation, within every outer fold, the labels of that fold's
  outer-train participants (for CNN models: inner train + inner validation)
  are shuffled among them (a participant's crops keep one shared label).
  The whole pipeline is retrained, and test participants are scored against
  their **true** labels.
  p = (1 + #{null macro-F1 >= observed}) / (1 + N).
  - LR models: **N = 1,000**, only for the two feature sets that enter the
    ensemble, `lr_handcrafted/all` and `lr_embeddings/concat`. The other LR
    feature sets (per-family, exploratory `lr_handwriting`) get no
    permutation test; they are reported with bootstrap CIs only.
  - CNN models (fine-tuned ResNet18, CNN-HMM) and the ensemble (which
    contains CNN-HMM): **N = 100** full retrains, primary set only (smallest
    p 0.0099). This is the reduced test.
- Significance level 0.05, two-sided intervals. The primary claim (section 5)
  is the single confirmatory test; all others are reported without
  multiplicity correction and labeled as secondary.

## 5. Models

All hyperparameters live in `src/utils/config.py` at the tagged commit.
Selection uses inner-validation participant macro-F1 only.

| Model | Specification |
|---|---|
| Majority | Training-fold majority label (section 2); prob_sad = training-fold SAD share |
| LR handcrafted | `StandardScaler -> LogisticRegression` on participant-level handcrafted features (`results/features/handcrafted_participant.csv`; graphological, drawing, stroke-level). C from {0.001, 0.01, 0.1, 1, 10} by stratified 5-fold inner CV (macro-F1) on outer-train participants, refit on all of them; `class_weight="balanced"`, `max_iter=5000`. Feature sets: **all** (used in the ensemble), plus drawing / word / cursive only (secondary). |
| LR embeddings | Frozen ImageNet ResNet18 embeddings averaged per participant within each family; **concat** = the three 512-d family means side by side (used in the ensemble), plus each family alone (secondary). `StandardScaler -> PCA -> LogisticRegression`; PCA keeps min(64, smallest inner-train size - 1) components, exact SVD, fit inside every inner and outer training split; same C grid, inner CV and class weights as above. Participant-level model: outputs participant probabilities directly (no crop aggregation). |
| Fine-tuned ResNet18 | 1-channel conv1 from summed pretrained weights; layer3, layer4 and the head trained (earlier layers frozen); AdamW, lr head 1e-3, lr backbone 1e-4, cosine schedule; class-balanced cross-entropy (inverse frequency of the inner-train crops' labels); early stopping on inner-val participant macro-F1, best epoch restored. Head: layer4 -> GAP -> Linear(512, 256) -> ReLU -> Dropout -> Linear(256, 2). One model per task family; crops pooled with `mean_prob`; the three families fused with equal weights (`cnn_head_fused`). |
| CNN-HMM | Word and cursive crops only: layer3 column sequences from the fine-tuned CNN checkpoint of the same fold (no retraining; split IDs asserted equal). Scaler + PCA fit on inner-train frames, per-class GaussianHMM. Setting chosen from n_states {2, 3, 4, 6} x PCA {8, 16, 32} x topology {ergodic, left-right} by inner-val participant macro-F1 (2 restarts per candidate), then refit on inner train with 5 restarts and Platt-calibrated on inner val. Crop probabilities pooled with `mean_prob`; word and cursive participant probabilities fused with **equal weights** (`fuse_task_families`). Drawing (`--include-drawing`) is secondary and never fused. Limitation: the HMMs see features of images the CNN trained on; inner-val selection and calibration mitigate it. |
| **Ensemble (primary)** | Fixed equal-weight mean of participant `prob_sad` from LR handcrafted, LR embeddings, and CNN-HMM fused. No weights learned. |

**Primary claim (confirmatory):** the ensemble's participant-level macro-F1
on the primary set, with bootstrap 95% CI, shown against the majority
baseline, with its paired comparison vs LR-handcrafted and its permutation
p-value (N = 100). Every other model and set is reported in full as
secondary. No model is promoted to the headline after seeing results.

**Exploratory (never part of the ensemble or any confirmatory claim):**
handwriting-encoder embeddings (`microsoft/trocr-base-handwritten`,
`src/features/embeddings_handwriting.py`), evaluated as `lr_handwriting`
(word + cursive participant means, concatenated; same LR pipeline). The
processor resizes crops to 384x384, discarding absolute size.

## 6. Secondary analyses

1. **The other analysis set:** everything above on the full sample (baselines
   required; CNN models optional).
2. **Middle band:** models trained on primary-set training participants
   predict the set-aside middle-band participants of each fold's test set;
   report accuracy and macro-F1 against their export labels, next to the
   majority baseline.
3. **Aggregation sensitivity:** `mean_logit` and `majority_vote` on the
   primary set.

## 7. Rules

1. Nothing is tuned, selected, calibrated, or thresholded on outer test
   folds.
2. Folds, seed, middle band, threshold, aggregation rule, model list, grids,
   and the primary claim are fixed by this protocol.
3. **Pilot:** participants with codes 001-100 (`CVConfig.pilot_id_range`),
   intersected with the analysis set, run on Kaggle as a technical check of
   the pipeline. Pilot metrics are not reported as results. Any change after
   the pilot is a Deviation.
4. **Deviations:** any change after the tag is recorded below, dated, with
   reason and whether any real result had been seen. Original and deviated
   analyses are both reported when both can be run.
5. Report all folds, seeds, models, and sets; never only the best.
6. Never drop participants based on predictions.
7. **Enforced in code:** every runner calls
   `src.utils.protocol_guard.require_frozen_or_synthetic` and refuses to run
   on a real data root until the `protocol-frozen` tag exists. Synthetic
   roots (`src/utils/synthetic.py`, marker file) are always allowed.

## 8. Pre-freeze artifacts

`results/` already holds four Stage C data-quality images made from real
crops on 2026-10-04 (no labels, no model output). They are not results:

| File | SHA-256 |
|---|---|
| `results/qc/contact_sheet_cursive.png` | `e422edb318a46edaf48de63a1d2670a977fc4827eac28adf281a0bc0de5fc562` |
| `results/qc/contact_sheet_drawing.png` | `3153eae464dc6d25cbd2fb50b55e4ec4cade7b1d4459f6a736faab0083a7827f` |
| `results/qc/contact_sheet_word.png` | `db327792f7ad93fb8542623f6f37953d0ab5494eee5447059572b947d30a15fd` |
| `results/qc/preprocess_before_after.png` | `cea1f46daccafc010f9230380224a49ecf85e2c2cab9103b097e440a53893c33` |

A **real result** is any model prediction, metric, or figure derived from real
participants' data. Every such file must post-date the `protocol-frozen` tag.

## 9. Deviations

None.
