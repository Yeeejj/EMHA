# EMHA Evaluation Protocol

Status: **FROZEN 2026-10-16.** The protocol is frozen by the
git tag `protocol-frozen` on the commit that adds this file. That commit is made
only after labeling is complete and `folds.csv` has been regenerated on the
final labels (section 3). No model may be run on real data before the tag.
Fields marked `[AT FREEZE]` are filled in that commit.

| Item | Value |
|---|---|
| Approved by thesis author | 2026-10-05 (text); revision 1 approved 2026-10-05; revision 2 approved 2026-10-16; revision 3 (pre-freeze, section 4: CNN permutations N = 100 -> 20) approved 2026-10-16 |
| Frozen (tag `protocol-frozen`) | 2026-10-16; the commit carrying the tag `protocol-frozen` (`git rev-list -n 1 protocol-frozen`) |
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
  middle band 57. Final counts on the complete sample (400 labeled
  participants, all `qc_passed`; HAPPY 224, SAD 176): **primary set 320
  (HAPPY 174, SAD 146), middle band 80** (scores 70-74; ties at the band
  edges are split by the labeler's stable sort on participant order).
  Primary-set score ranges: HAPPY [74, 119], SAD [37, 70].

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
  - `folds.csv` SHA-256:
    `40403812420cde0e1f7d1980ce02ad0f77c531fdf59aab62b7987ca4b90751bf`
    (400 participants, regenerated with `--force` after labeling was
    complete; the superseded provisional 285-participant file was
    `fdf5a63e03ab22690908e52a4664508a16630daef8bbeb83f00d0471baea7106`).
  - `labels.csv` SHA-256:
    `694d360091e28dc331422cf7171b8e3afcf5dda16545e7ad5b2a9291204d3006`.
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
    contains CNN-HMM): **N = 20** full retrains, primary set only (smallest
    p 1/21 = 0.0476). This is the reduced test. Revision 3 (2026-10-16,
    before the tag and before any real result): N reduced from 100 because
    100 full retrains x 3 models (~150-250 GPU-hours) exceeded the available
    GPU budget (Kaggle, ~30 GPU-hours/week). With N = 20, p < 0.05 is
    attainable only if the observed macro-F1 exceeds every null draw.
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
p-value (N = 20). Every other model and set is reported in full as
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

1. 2026-10-18 — Success criterion raised from primary-set macro-F1 and
   accuracy >= 0.70 to >= 0.75 (both), with 0.80 as a stated stretch goal.
   Reason: the thesis author's target, set with the adviser. No analysis,
   model, fold, band, seed or threshold changes. Real results seen: none
   (results/ held only the section 8 QC images; verified 2026-10-18).
   Clarification, same date: on the pilot (a technical check, never
   reported), permutation tests run for the LR models only
   (`PermutationConfig.pilot_models`); CNN-path tests run on the full run
   only, as in section 4.

2. 2026-10-21 — Implementation only: the CNN-HMM candidate fits
   (`run_hybrid.select_hmm`) and the HMM restarts (`HMMClassifier`) run in
   parallel across CPU cores (joblib, `RunnerConfig.hmm_n_jobs`), and every
   reported run pins BLAS/OpenMP to one thread (`OMP_NUM_THREADS`,
   `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS` = 1, set before Python starts;
   `RunnerConfig.hmm_threads_per_fit` = 1). No change to the grid, restarts,
   seeds, selection rule, calibration, folds, labels or any analysis
   setting: each restart's random_state is seed + restart, and the best
   restart is the first with the highest training log-likelihood (strict >,
   in restart order), in both the serial and the parallel code. Evidence
   (synthetic 40-participant image root, real HMM grid and restarts; commit
   0c29636, re-verified by tests/test_parallel_hmm.py at commit 9a5c0ae):
   with threads pinned, n_jobs = 1 and n_jobs = -1 both reproduce
   the pre-change code (main 19d8d48) bit for bit: run_hybrid
   predictions.csv SHA-256
   5a6c5951b3693cb34db2e48dee28374060d93aa8fde27fd2086aa6a1502a9bfc for all
   three (tests/test_parallel_hmm.py). Without the pin, the pre-change
   code's output depends on the thread count: default threads vs one thread
   differ by at most 5.0e-16 in crop prob_sad, with 0/120 participant
   predictions and 0/15 fold-level macro-F1 values changed. Without the
   environment pin, the branch also differs from the pre-change code, by at
   most 1.1e-16 in crop prob_sad with 0/120 participant predictions and 0/15
   fold-level macro-F1 values changed, which is consistent with it limiting
   each HMM fit to one BLAS/OpenMP thread internally
   (`RunnerConfig.hmm_threads_per_fit` = 1) while the pre-change code does
   not; with the pin, both match bit for bit. This was checked on the
   synthetic root only, not on real data. Reported runs therefore pin
   threads to 1, so results do not depend on the machine's core count.
   Section 7.4 (original and deviated analyses both reported) does not
   apply: under pinned threads the parallel path is bit-identical to the
   serial path, and the pin removes only the thread-count effect described
   above, so no unpinned second run is made. Reason: serial CNN-HMM fitting
   made the full run about 59 compute-hours
   (DOCS/plans/R4.5_full_run_plan.md at commit 0352fb0 (branch
   docs/full-run-plan)). Real results seen: pilot outputs on real data
   exist (section 7.3, technical check, never reported), but no
   pilot metric was read before this change. Because the change leaves
   outputs bit-identical under pinned threads, it cannot have been informed
   by any result.

3. 2026-10-22 — Implementation only: the n_cnn = 20 CNN-path permutation
   draws of the primary set may be computed on more than one machine.
   `python -m src.analysis.permutation --draws START-END` (notebooks:
   `DRAWS`) computes and caches only those draw indices and runs no test;
   `python -m scripts.merge_draws SRC DST` copies one machine's cached
   draws into the other's runs folder and refuses to merge when the two
   `run_config.json` files differ (analysis, seed, folds SHA-256, CNN,
   training, HMM, hybrid, baseline and aggregation settings). The test
   itself (observed value, p, outputs) runs once, after all 20 draws are
   cached, exactly as before. No change to n_cnn, seeds, shuffling, folds,
   labels, models or any analysis setting: each draw is seeded only by
   `permutation_seed(seed, draw, fold)` and cached as its own
   `perm_<draw>.csv`, so which machine computes a draw does not change its
   shuffled labels. As already stated for GPU training, a draw's retrained
   models are not bit-identical across GPU types; the machine and GPU of
   each draw range are recorded in REPRODUCE.md. A machine that computes
   its own embeddings and handcrafted features for its draws (rather than
   reading the main run's) feeds those to the draws' LR and ensemble
   refits; those features differ from the main run's at most by GPU
   floating-point differences (embeddings) and not at all for the CPU
   handcrafted features under the same code. Reason: the 20 draws are
   about 80% of the full run's compute (DOCS/plans/R4.5_full_run_plan.md),
   and running them in turn on one machine did not fit the thesis
   schedule. Real results seen: none from the full run (not started); no
   pilot metric was read before this change.

4. 2026-10-23 — Reporting change: the headline (primary) claim moves from
   the ensemble to the fused CNN-HMM (`cnn_hmm_fused/fused`), the method
   named in the thesis title and in the approved proposal; the ensemble is
   reported as a supporting analysis. This was decided AFTER the full
   run's results were read (results/final/RESULTS.txt, git HEAD 996340d),
   so the rule in section 5 that no model is promoted to the headline
   after results are seen is knowingly broken; the change is disclosed
   here and in the paper. Nothing is recomputed: no model, fold, band,
   seed, threshold, grid, aggregation rule or metric changes, and every
   number is reported exactly as RESULTS.txt prints it. The primary claim
   is now the CNN-HMM's participant-level macro-F1 on the primary set with
   its bootstrap 95% CI, shown against the majority baseline, with its
   paired comparison vs `lr_handcrafted/all` and its permutation p-value
   from the N = 20 retrains already run (sections 4 and 5). The ensemble
   keeps all of its outputs (macro-F1 and CI, paired comparison,
   permutation p) and is reported in full beside the CNN-HMM, as is every
   other model. The change does not favour the headline: on the primary
   set the CNN-HMM's macro-F1 is 0.434 [0.384, 0.486], accuracy 0.534,
   permutation p = 0.7143, against the ensemble's 0.516 [0.462, 0.572],
   0.522, p = 0.4286. Section 7.4 (original and deviated analyses both
   reported) is met: the paper reports both claims. Reason: the thesis
   author designates the CNN-HMM as the proposed system; the ensemble was
   added by this protocol as a combined reference. Real results seen: all
   (the full run is complete).

5. 2026-10-23 — Exploratory analysis added after all results were seen: CNN-HMM v2, a
   task-order hidden Markov model. Each respondent is one sequence of 24 frozen ImageNet
   ResNet18 embeddings (512 values per task image, the cached results/embeddings/
   resnet18_<family>.npz used by lr_embeddings), ordered as on the FINALE form
   (src/data/crop_manifest.py ALL_CELLS: D1-D4, W1_LH ... W5_UC, CS1-CS5). Per outer fold
   (folds.csv, primary set): StandardScaler and PCA fit on the training participants' frames;
   one GaussianHMM per class (diagonal covariance); the setting is chosen from n_states
   {2, 3, 4} x PCA components {8, 16, 32} x topology {ergodic, left_right} by participant
   macro-F1 on the same inner validation split and selection rule as run_hybrid, refit with
   HMMConfig.n_restarts, Platt-calibrated on the inner validation participants; a participant
   is SAD when P(SAD) >= 0.5. v1 (tag cnn-hmm-v1, results/hybrid/) is kept unchanged and
   remains the primary model; v2 is written only to separate v2 paths. Reported: macro-F1, accuracy, balanced accuracy, ROC-AUC with
   2,000-resample bootstrap CIs (seed 42), majority baseline, every fold, the confusion
   matrix, paired macro-F1 differences vs lr_handcrafted/all, lr_embeddings/concat and
   cnn_hmm_fused/fused, and a permutation test with N = 100 label shuffles (full v2 refit per
   draw, permutation_seed(seed, draw, fold) as in src/analysis/permutation.py). v2 is never
   part of the primary claim, is labeled EXPLORATORY everywhere, and is reported whatever its
   result. Reason: v1's left-to-right column sequences carried no information (AUC 0.485),
   while the same frozen features did (lr_embeddings p = 0.008); task order is the only real
   order in offline scans. Real results seen: all (the full run is complete).
