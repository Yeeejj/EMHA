# GAP_REPORT.md

Evaluated against the current `CLAUDE.md` ("EMHA THESIS — MASTER BUILD BRIEF").
No files were changed to produce this report.

**Context the reader needs up front:** `CLAUDE.md` was just rewritten and
describes a different project shape than the code in this repo implements.
The repo's `src/` tree is the old `ProcessPipeline.txt` 19-phase (P0–P18)
design: data root `DATA/METADATA|CROPS|PROCESSED`, raw scans under
`DATASET/raw/respondent_NNN/{PNG,PDF}` with barcode IDs, and labels
*recomputed* from a 24-item Likert formula. The new `CLAUDE.md` moves the
data root to `DATASET/metadata|processed`, raw scans to
`DATASET/raw-3Page`/`raw-4Page` with a bare 3-digit code, and — most
importantly — says labels must be *read, never recomputed*, from
`DATASET/metadata/questionnaire_export.csv`. That one change invalidates
the current labeling module, not just its file name. Keep this in mind
below: many "missing" module-map entries have an old-pipeline analog that
is close in spirit but wrong in a load-bearing way, not just absent.

---

## 1. Module Map status

### src/utils
| Module | Status | Note |
|---|---|---|
| `config.py` | **partial** | Exists, but holds the old P0-P18 schema (`raw_data_dir`, `crops_dir`, `processed_dir`, `labeling.*`, `DRAWING_CROPS`/`WORD_CROPS`/`CURSIVE_CROPS`). No `middle_band_fraction`, no `DATASET/raw-3Page`/`raw-4Page` paths, no fixed-scale+padding size. Several *other* modules read `config.data.splits_dir`, `.labeled_dir`, `.staging_dir`, `.extracted_dir` — none of these fields exist on `DataConfig` (`src/utils/config.py:12-24`), so those modules currently crash (see §2). |
| `seed.py` | **missing** | No single seed-everything entrypoint; `config.training.random_state` is passed ad hoc into individual sklearn calls only. |
| `synthetic.py` | **missing** | No synthetic fixture generator anywhere, and no `tests/` directory exists at all. |
| `run_log.py` | **missing** | `trainer.py` and `evaluator.py` each hand-roll their own `csv.DictWriter` logging inline; no shared run-log module. |
| `package_dataset.py` | **missing** | No reproducibility-bundle/export script. |

### src/data
| Module | Status | Note |
|---|---|---|
| `labeler.py` | **missing — and the closest analog breaks a non-negotiable** | `questionnaire_scorer.py` *recomputes* `adjusted_total`/label from raw 24-item Likert values (happiness/sadness sums, `>=72` cutoff). New Rule 3 requires labels to be read verbatim from `DATASET/metadata/questionnaire_export.csv` and never recomputed. |
| `ingest.py` | **missing** | Closest analogs are `register_participants.py` + `assign_pages.py`, built around barcode-based `respondent_NNN/PNG,PDF` folders under the old `DATASET/raw` — not the new `raw-3Page`/`raw-4Page` + bare-code layout. |
| `crop_manifest.py` | **missing** | `content_extractor.py` exists but it *extracts* crops itself via independent fractional boxes against full-page scans in `DATASET/raw`, duplicating `src/cropping/p3/crop_pictures.py` and `src/cropping/p4/crop_p4.py` — the scripts CLAUDE.md says must stay untouched and not be reimplemented. A real `crop_manifest.py` should index what those two scripts already produce, not re-crop. |
| `dataloader.py` | **exists, violates non-negotiables** | See §2 (folder-based HAPPY/SAD split, disallowed augmentations, stretch-resize). |
| `transforms.py` | **missing** | Augmentation lives inline inside `dataloader.py` instead of its own module, and includes transforms the new brief forbids. |
| `collector.py` | **missing** | |

### src/preprocessing
| Module | Status | Note |
|---|---|---|
| `pipeline.py` | **exists, directly contradicts the Accuracy Strategy** | Otsu binarization, skew correction, and stretch-`cv2.resize` are exactly the three operations the new brief forbids. See §2. |

### src/features
| Module | Status | Note |
|---|---|---|
| `handcrafted.py` | **missing** | Only a 3-feature inline helper exists (`trainer.py:60-80`, `_image_features`: mean intensity, pixel density, slant angle). No drawing-specific or stroke-level (skeleton stroke width/darkness, tremor) features anywhere. |
| `embeddings.py` | **missing** | No standalone frozen-ResNet18-embeddings-to-feature-vector module for the LR-on-embeddings model; embedding extraction only exists inline inside `EmotionCNN`. |
| `embeddings_handwriting.py` | **missing** | (exploratory/optional per module map) |

### src/models
| Module | Status | Note |
|---|---|---|
| `cnn.py` | **exists, reasonably aligned** | `EmotionCNN` + `CNNFeatureExtractor` present (plus an extra `PretrainedCNNExtractor`, fine). The specified fine-tune recipe (unfreeze layer3+layer4 only, AdamW with split lr_head/lr_backbone, cosine schedule) is **not** implemented — current code freezes the whole backbone except conv1 and is trained with plain Adam in `trainer.py`. |
| `hmm.py` | **exists, partial** | `HMMClassifier` with per-class `GaussianHMM` is present; missing the restarts, scaler+PCA, and Platt calibration the Accuracy Strategy calls for. |
| `hybrid.py` | **exists** | `HybridCNNHMM` wraps CNN+HMM; functionally overlaps with logic already duplicated in `trainer.py`/`cross_validate.py`. |

### src/training
| Module | Status | Note |
|---|---|---|
| `splits.py` | **missing** | `participant_aware_splitter.py` exists and is participant-correct, but emits `splits.json` from `crop_index.csv` for one fixed train/val/test cut, not a `folds.csv` over *all* labeled participants scoped to the extreme-groups primary set. |
| `aggregate.py` | **missing** | No per-participant mean-of-crop-probabilities aggregation exists anywhere, and no shared prediction schema (`analysis, model, feature_set, fold, participant_id, label, prob_sad, pred, in_middle_band`). |
| `trainer.py` | **exists, wrong protocol** | Trains on crop-level folder splits, not participant-level folds; no extreme-groups band. See §2. |
| `run_baselines.py` | **missing** | No standalone majority-class / LR-handcrafted / LR-embeddings runners (`trainer.py` bundles one ad hoc 3-feature LR only). |
| `run_cnn.py` | **missing** | No script implementing the specified fine-tune recipe. |
| `run_hybrid.py` | **missing** | No CNN-HMM-on-layer3-sequences-with-PCA-and-Platt-calibration script restricted to word/cursive crops. |
| `run_ensemble.py` | **missing** | No equal-weight ensemble of LR-handcrafted + LR-embeddings + CNN-HMM-fused exists. |
| `evaluator.py` | **exists, wrong protocol** | Reports accuracy/F1/ROC-AUC point estimates only — no bootstrap CIs, no balanced accuracy, no majority baseline comparison, no paired test, no permutation test. Also invokes a crop-level `CrossValidator` (§2 item 1). |

### src/analysis
| Module | Status | Note |
|---|---|---|
| `questionnaire_report.py` | **missing, partial analog** | `psychometrics.py` computes Cronbach's alpha and item-total correlations (good overlap) but recomputes `adjusted_total`/label from raw items (`psychometrics.py:75`) and never reports HAPPY/SAD balance counts. |
| `permutation.py` | **missing** | No permutation test implemented anywhere, though required by the Accuracy Strategy. |
| `secondary.py` | **missing** | No full-sample (outside extreme-groups) secondary analysis. |
| `gradcam.py` | **missing as a module** | Grad-CAM (`_GradCAM` class) is implemented inline inside `training/evaluator.py:223-266`, not as its own module. |
| `report.py` | **missing** | No `RESULTS.txt` generator. |

### Root
| Item | Status | Note |
|---|---|---|
| `smoke_test.py` | **exists, wrong pipeline** | Exercises the old P0-P18 phases (stages A-G: config → register → manifest → score → extract → propagate → preprocess), not the new A-H build order, and uses no synthetic fixture. |
| `EVALUATION_PROTOCOL.md` | **missing** | |
| `REPRODUCE.md` | **missing** | |
| `notebooks/` | **exists** | `EMHA_Colab_Pipeline.ipynb` (381 lines, ~8 def/class blocks) looks like a thin bootstrap already, but should be re-audited once the pipeline is rewritten to confirm no model code/hyperparameters leak into it. |

**Orphaned (in the repo, not in the new module map):** `src/data/register_participants.py`, `assign_pages.py`, `content_extractor.py`, `propagate_labels.py`, `questionnaire_scorer.py`, `qc.py`, `preview_crops.py`; `src/preprocessing/run_preprocessing.py`; `src/training/cross_validate.py`; `src/utils/artifact_generator.py`; `src/predict.py`; `src/analysis/psychometrics.py`; `ProcessPipeline.txt`, `PROJECT_STRUCTURE.md`, `SYSTEMS_GUIDE.md`, `README_THESIS_INSTRUCTIONS.md`, `THESIS_CITATIONS_SUMMARY.md`. None of these map 1:1 onto a module-map entry; each needs an explicit decision (retire, or repurpose as the named module) rather than silent reuse.

---

## 2. Non-negotiable / Accuracy-Strategy violations

**1. Participant-level splitting only (Rule 1)**
- `src/training/cross_validate.py:11,76-77,79-83,99` — `StratifiedKFold` is built directly over `dataset.get_labels()` / `np.arange(len(labels))`, i.e. individual crop-image indices from `HandwritingDataset` (one row per PNG file). A participant's 24 crops can land on both sides of a fold. No `assert_no_leakage` call anywhere in this file (or anywhere in the repo — confirmed by search).
- `src/training/evaluator.py:416-441` (`_run_cross_validation`) feeds that same crop-level `CrossValidator` a combined train+val `HandwritingDataset` pool — same leakage, on every evaluator run.
- By contrast, `src/data/participant_aware_splitter.py` *is* correct (splits/stratifies at participant-ID level, has its own `verify_no_leakage` check, lines 108-122) — but nothing downstream actually consumes its `splits.json`. `dataloader.py`/`trainer.py`/`cross_validate.py` instead read ad hoc `DATA/SPLITS/{train,val,test}/HAPPY|SAD/*.png` folders, so the one correct splitter is disconnected from the real training path.

**2. Folder-based HAPPY/SAD splits**
- `src/data/dataloader.py:16-56` — `HandwritingDataset` expects `data_dir/HAPPY/*.png` and `data_dir/SAD/*.png` and labels every image purely by which folder it's in. No `participant_id` is tracked per sample, so grouping crops back to a participant for the final decision is structurally impossible from this class as written.
- `src/preprocessing/pipeline.py:108-123,135-143` and the `DATA/SPLITS/{train,val,test}/HAPPY|SAD/...` paths assumed in `trainer.py:365-379` / `evaluator.py:418-490` all rely on this same folder convention — which nothing in phases P1-P8 actually populates (no script writes `DATA/SPLITS/*/HAPPY|SAD/`).

**3. Otsu binarization (forbidden — "grayscale … No binarization")**
- `src/preprocessing/pipeline.py:18,42-52`, called from `process()` at line 88.
- `src/training/trainer.py:69` — `_image_features()` also Otsu-thresholds each crop for the handcrafted-LR features.
- Both destroy the grayscale intensity information the new brief requires preserved for stroke-level features.

**4. Deskew (forbidden — "No … deskew")**
- `src/preprocessing/pipeline.py:61-77`, called from `process()` at line 90 (`skip_skew` gate).
- `src/data/content_extractor.py:13-14,90-103` and `src/preprocessing/run_preprocessing.py:68-83,118` — the whole `skip_skew_prefixes` plumbing actively rotates word/cursive crops. The new brief wants `baseline_angle_deg` *measured*, not corrected.

**5. `cv2.resize` to a fixed square / stretch-resize (forbidden — "No stretch-resize: one fixed scale factor plus white padding")**
- `src/preprocessing/pipeline.py:79-83` — `normalize()` calls `cv2.resize(image, self.target_size)` (224×224 fixed square), no aspect-ratio preservation or padding.
- `src/data/dataloader.py:89,105` — `transforms.Resize(image_size)` in both train/val transforms, same stretch-to-square behavior.

**6. RandomRotation / RandomAffine (forbidden — "Never rotation, scaling, shear, or flips")**
- `src/data/dataloader.py:90-95` — `get_train_transform()` applies `transforms.RandomRotation(degrees=5)` and `transforms.RandomAffine(degrees=0, translate=(0.05,0.05), scale=(0.95,1.05))`. Both the rotation and the scale component are explicitly disallowed now; only translation, brightness/contrast, and light blur are permitted.

**7. `use_pretrained` not wired in**
- Mostly fine: `src/models/cnn.py`, `src/training/trainer.py:400-407`, and `src/training/evaluator.py:70-79,429-439` all correctly thread `config.cnn.use_pretrained` through to `EmotionCNN`.
- Gap: `src/training/cross_validate.py:47` defaults `use_pretrained: bool = False` on `CrossValidator.__init__`, and the module's own `__main__` demo (line 231) never sets it — any direct use of `CrossValidator` outside `evaluator.py` silently falls back to the un-pretrained custom CNN.

**8. DASS / NEUTRAL labeling**
- No DASS references found anywhere — not applicable.
- No literal NEUTRAL class exists; `config.py:86,193`, `questionnaire_scorer.py:17,145`, `propagate_labels.py:9,131` all explicitly assert/print "NO NEUTRAL class." Good on this specific point.
- But this is moot for a different reason: `questionnaire_scorer.py:73-90` recomputes the label from raw Likert items via a happiness/sadness-sum formula that appears nowhere in the current `CLAUDE.md`. The new brief requires labels to be read verbatim from `questionnaire_export.csv` and never recomputed — `questionnaire_scorer.py`, its consumer `propagate_labels.py`, and `psychometrics.py:75` (a third independent recomputation) all violate this.

**9. TensorFlow references**
- None found (`grep -ri tensorflow|keras` across all `.py` files returned no matches). `requirements.txt:2,20-22` pins PyTorch-only explicitly. No violation.

**10. Other breaks found during review (adjacent to your checklist):**
- `cfg.data.splits_dir` is read in `trainer.py:365`, `evaluator.py:418,482` but is **not defined** on `DataConfig` (`config.py:12-24`) — these modules raise `AttributeError` immediately on use. Same for `config.data.labeled_dir` (`preprocessing/pipeline.py:136`), `config.data.staging_dir` (`data/assign_pages.py:154`), `config.data.extracted_dir` (`data/preview_crops.py:104`). Training, evaluation, and three data-pipeline modules are currently non-functional, independent of the spec pivot.
- `config.data.raw_data_dir` (`config.py:16`) points at `E:\EMHA_Thesis\DATASET\raw` — wrong drive and wrong shape (`respondent_NNN/`) versus the new `DATASET/raw-3Page`/`raw-4Page` under the repo root `D:\EMHA_Thesis\EMHA-1`.
- Two independent, conflicting crop-extraction paths exist: `src/cropping/p3/crop_pictures.py` + `src/cropping/p4/crop_p4.py` (fixed pixel boxes, the ones CLAUDE.md says must stay untouched) versus `src/data/content_extractor.py` (fractional boxes against full scans, separate output tree). Any new `crop_manifest.py` must index the former, not reimplement the latter.

---

## 3. .gitignore audit

Current `.gitignore` (repo root):

- **DATASET/ excluded:** Yes — line 6, blanket rule. ✅ matches Rule 2 (read-only raw data never committed).
- **Weights excluded:** Yes — `models/*.pth`, `models/*.pkl` (lines 20-21). ✅
- **Results excluded:** **Partial.** Only `results/*.png` and `results/gradcam/` are ignored (lines 25-26). CSV/JSON result artifacts that the pipeline actually writes — `training_log.csv`, `evaluation_report.csv`, `cv_results.json`, `model_comparison.csv` — are **not** excluded and would be committed as-is. Needs an explicit decision (track them intentionally for thesis reproducibility, or ignore them too) rather than the current accidental partial coverage.
- **Secrets excluded:** **Partial.** Only `.env` (line 45) is covered. There's no pattern for `credentials*.json`, `service-account*.json`, `*.key`, etc. This is newly relevant: the new labeling flow reads an external Google-Sheets export, so if any API credential file is ever added to authenticate that export, nothing currently catches it.
- **New gap introduced by the CLAUDE.md pivot:** the new layout puts the *authoritative* label file at `DATASET/metadata/questionnaire_export.csv`, but `DATASET/` is fully blanket-ignored (line 6) with no carve-out — unlike the old `.gitignore`, which explicitly un-ignored `DATA/CROPS/.gitkeep` style paths. If `DATASET/metadata/` is meant to be versioned under the new layout (the way `DATA/METADATA/*.csv` was under the old one), the `.gitignore` needs a `!DATASET/metadata/` exception; right now nothing under `DATASET/` can ever be committed.

---

## 4. Proposed task order (mapped to Build Order A-H)

**A — Foundation**
- Resolve the `.gitignore` gaps in §3 (results policy, secrets patterns, `DATASET/metadata` carve-out decision).
- Rewrite `src/utils/config.py`: replace old-pipeline fields (`raw_data_dir=E:\...`, `crops_dir`, `processed_dir`, `labeling.*`, `DRAWING_CROPS` etc.) with the new layout (`DATASET/raw-3Page`, `raw-4Page`, `DATASET/metadata`, `DATASET/processed`), plus `middle_band_fraction`, the restricted augmentation set, and fixed-scale+padding sizing. Eliminate the dangling `splits_dir`/`labeled_dir`/`staging_dir`/`extracted_dir` references across the codebase in the same pass.
- Add `src/utils/seed.py` and `src/utils/synthetic.py` (mirroring the real `raw-3Page`/`raw-4Page`/`metadata` layout).
- Stand up `tests/` wired to the synthetic fixture.
- Quarantine or retire `ProcessPipeline.txt`, `PROJECT_STRUCTURE.md`, `SYSTEMS_GUIDE.md` so they stop documenting the superseded P0-P18 design as current.

**B — Labels**
- Add `src/data/labeler.py`: read `DATASET/metadata/questionnaire_export.csv` verbatim, no recomputation; validate counts match the sheet exactly.
- Add `src/analysis/questionnaire_report.py` (alpha + HAPPY/SAD balance) — reuse the Cronbach's-alpha/item-total-correlation math already in `psychometrics.py`, but stop recomputing labels there.
- Write and freeze the analysis-design decision file (extreme-groups band, `middle_band_fraction` default 0.30).
- Retire `questionnaire_scorer.py` and `propagate_labels.py` from the label path (Rule 3 violation, §2 item 8).

**C — Data indexing**
- Add `src/data/ingest.py`: scan validator + checksums for `raw-3Page`/`raw-4Page`, matching `participant_id` to the tabulation export ID column.
- Add `src/data/crop_manifest.py` to index crops already produced by `src/cropping/p3` and `p4` — do not reimplement extraction; retire `content_extractor.py`'s competing fractional-box pipeline (§2 item 10). Add contact sheets.
- Promote `participant_aware_splitter.py`'s leakage check into a reusable `assert_no_leakage` helper for use in every later fold.

**D — Preprocessing and features**
- Rewrite `src/preprocessing/pipeline.py`: drop `binarize()` and `correct_skew()`; replace the stretch-`cv2.resize` in `normalize()` with fixed-scale + white padding; add `baseline_angle_deg` measurement (no rotation) for word/cursive crops.
- Add `src/data/transforms.py` (translation + brightness/contrast + light blur only); delete `RandomRotation`/scaling `RandomAffine` from `dataloader.py`.
- Rewrite `src/data/dataloader.py` to drop the `HAPPY`/`SAD` folder convention and load by participant_id + crop manifest instead, so participant grouping survives into training.
- Add `src/features/handcrafted.py` (expand the 3-feature inline helper into full handcrafted + drawing-specific + stroke-level/tremor features) and `src/features/embeddings.py` (frozen ResNet18 embeddings).

**E — Evaluation setup**
- Add `src/training/splits.py`: `folds.csv` over *all* labeled participants, scoped to the extreme-groups primary set (full sample kept secondary) — replacing `participant_aware_splitter.py`'s single train/val/test `splits.json`.
- Add `src/training/aggregate.py`: mean-of-crop-probabilities per participant, plus the shared prediction schema (`analysis, model, feature_set, fold, participant_id, label, prob_sad, pred, in_middle_band`).
- Write `EVALUATION_PROTOCOL.md`, freeze it, tag `protocol-frozen`.

**F — Models**
- Fix the sample-level `StratifiedKFold` in `cross_validate.py`/`evaluator.py` (§2 item 1) to fold over participant IDs from `splits.py` instead of crop indices.
- Add `src/training/run_baselines.py` (majority class, LR-handcrafted, LR-embeddings).
- Add `src/training/run_cnn.py` with the specified fine-tune recipe (unfreeze layer3+layer4, AdamW split lr_head/lr_backbone, cosine schedule, early stop on participant macro-F1) — replace, don't patch, `trainer.py`'s current frozen-backbone/Adam recipe.
- Add `src/training/run_hybrid.py` (CNN-HMM on layer3 sequences, word/cursive only, scaler+PCA, HMM restarts, Platt calibration).
- Add `src/training/run_ensemble.py` (equal-weight LR-handcrafted + LR-embeddings + CNN-HMM-fused).
- Fix the `use_pretrained` default in `CrossValidator.__init__` (§2 item 7) if that class survives the rewrite.

**G — Runs**
- Rewrite `smoke_test.py` to exercise the new A-H pipeline against the synthetic fixture, then local → Colab smoke test → pilot on the first ~100 participants (Kaggle) → full run.

**H — Results**
- Rewrite `evaluator.py` to report accuracy, macro-F1, balanced accuracy, and ROC-AUC with bootstrap CIs, the majority baseline, a paired comparison against LR-handcrafted, and a permutation test — none of which exist today.
- Add `src/analysis/permutation.py` and `src/analysis/secondary.py`.
- Extract `_GradCAM` out of `evaluator.py` into `src/analysis/gradcam.py`.
- Add `src/analysis/report.py` → `RESULTS.txt`, plus `src/utils/package_dataset.py` (stage A) for the reproducibility bundle.
