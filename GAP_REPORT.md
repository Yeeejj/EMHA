# GAP_REPORT.md

Evaluated against the current `CLAUDE.md` ("EMHA THESIS — MASTER BUILD BRIEF").
No files were changed to produce this report.

**Context the reader needs up front.** `CLAUDE.md` in the working tree (the
version you're reading) is **not yet committed** — `git diff CLAUDE.md` shows
it as modified; HEAD still holds an older `ProcessPipeline.txt`-era version.
`git log` shows a prior `docs: gap report` commit (787c9da) already added a
`GAP_REPORT.md`, but `CLAUDE.md` itself was never committed alongside it. This
report re-derives everything from scratch by reading the actual source files
rather than trusting that prior report, and supersedes it.

The repo's `src/` tree (config, dataloader, preprocessing, training, models)
still implements the **old** `ProcessPipeline.txt` 19-phase (P0–P18) design:
data root `DATA/METADATA|CROPS|PROCESSED`, raw scans under
`DATASET/raw/respondent_NNN/{PNG,PDF}` with barcode IDs, and labels
*recomputed* from the 24-item Likert formula. The current `CLAUDE.md` moves
the data root to `DATASET/metadata|processed`, raw scans to
`DATASET/raw-3Page`/`raw-4Page` with a bare 3-digit code, and says labels must
be *read, never recomputed*, from `DATASET/metadata/questionnaire_export.csv`.
That invalidates the current labeling module, not just its file name.

**Two pieces of good news the old report didn't have, found by checking disk
state (DATASET/ is gitignored, so `git status` is silent about it):**
- `src/cropping/p3/crop_pictures.py` and `src/cropping/p4/crop_p4.py` (both
  currently **untracked**, `git status` `??`) already implement the *new*
  layout exactly: they read `DATASET/raw-3Page` / `raw-4Page` and write
  `EMHA-P3_DrawingExercise_<code>_D<k>.png` / `EMHA-P4_WritingExercise_<code>_<cell>.png`
  into `D1..D4` / `W1..W5_{LH,RH,UC}` / `CS1..CS5` subfolders — matching
  CLAUDE.md's DATA LAYOUT section verbatim. These are the scripts CLAUDE.md
  says must stay untouched.
- They have already been run for real: `DATASET/raw-3Page/D1/` and
  `DATASET/raw-4Page/CS1/` each contain **285** output PNGs (codes 001–285+).

---

## 1. Module Map status

### src/utils
| Module | Status | Note |
|---|---|---|
| `config.py` | **partial** | Exists (`src/utils/config.py`) but holds the old P0-P18 schema: `raw_data_dir="E:\\EMHA_Thesis\\DATASET\\raw"` (wrong drive, wrong shape), `metadata_dir="DATA/METADATA"`, `crops_dir`, `processed_dir`, `labeling.*` (recompute fields), `DRAWING_CROPS`/`WORD_CROPS`/`CURSIVE_CROPS`. No `middle_band_fraction`, no `DATASET/raw-3Page`/`raw-4Page`/`metadata` paths, no fixed-scale+white-padding size, no restricted augmentation config. Several *other* modules read `config.data.splits_dir` / `.labeled_dir` / `.staging_dir` / `.extracted_dir` — none exist on `DataConfig` (`config.py:12-24`), so those modules crash on import-time use (§2.10). |
| `seed.py` | **missing** | No seed-everything entrypoint; `config.training.random_state` is passed ad hoc into individual sklearn calls only. |
| `synthetic.py` | **missing** | No synthetic fixture generator; no `tests/` directory exists at all in the repo. |
| `run_log.py` | **missing** | `trainer.py` and `evaluator.py` each hand-roll their own `csv.DictWriter` logging inline. |
| `package_dataset.py` | **missing** | No reproducibility-bundle/export script. |

### src/data
| Module | Status | Note |
|---|---|---|
| `labeler.py` | **missing — closest analog breaks Non-Negotiable 3** | `questionnaire_scorer.py` *recomputes* `adjusted_total`/label from raw 24-item Likert values (`score()` at lines 73-84, `assign_label()` at 87-89). Rule 3 requires labels read verbatim from `DATASET/metadata/questionnaire_export.csv`, never recomputed. `propagate_labels.py` and `psychometrics.py` (`:73-75`) repeat the same recomputation independently — three places compute the same thing three times, none of them reading the authoritative export. |
| `ingest.py` | **missing** | Closest analogs are `register_participants.py` + `assign_pages.py`, built around barcode `respondent_NNN/PNG,PDF` folders under the old `DATASET/raw` (confirmed still present on disk, 286 folders) — not the new `raw-3Page`/`raw-4Page` + bare-code layout. |
| `crop_manifest.py` | **missing** | `content_extractor.py` exists but *re-extracts* crops itself via independent fractional boxes against full-page scans in `DATASET/raw` — a third, competing crop pipeline alongside the two CLAUDE.md says to leave untouched (`src/cropping/p3`, `p4`, confirmed already producing real output; see above). A real `crop_manifest.py` should index what those two scripts already produced in `raw-3Page`/`raw-4Page`, not re-crop from `DATASET/raw`. |
| `dataloader.py` | **exists, violates non-negotiables** | Folder-based HAPPY/SAD split with no participant tracking; forbidden augmentations; stretch-resize. See §2.2, §2.5, §2.6. |
| `transforms.py` | **missing** | Augmentation lives inline inside `dataloader.py` (`get_train_transform`, lines 86-99) instead of its own module, and includes transforms the brief forbids. |
| `collector.py` | **missing** | |

### src/preprocessing
| Module | Status | Note |
|---|---|---|
| `pipeline.py` | **exists, directly contradicts the Accuracy Strategy** | Otsu binarization, skew correction, and stretch-`cv2.resize` are exactly the three operations the brief forbids. See §2.3, §2.4, §2.5. |

### src/features
| Module | Status | Note |
|---|---|---|
| *(entire package)* | **missing** | `src/features/` does not exist on disk at all — confirmed by directory listing. No `handcrafted.py`, `embeddings.py`, or `embeddings_handwriting.py`. The only handcrafted-feature code anywhere is a 3-feature inline helper (`trainer.py:60-80`, `_image_features`: mean intensity, pixel density, Otsu-derived slant angle). No drawing-specific or stroke-level (skeleton stroke width/darkness, tremor) features exist. No standalone frozen-ResNet18-embeddings module exists either — embedding extraction is only available inline via `EmotionCNN.extract_features`. |

### src/models
| Module | Status | Note |
|---|---|---|
| `cnn.py` | **exists, partially aligned** | `EmotionCNN` + `CNNFeatureExtractor` + `PretrainedCNNExtractor` (ResNet18) all present, matching the module-map class names reasonably well. But the fine-tune recipe is wrong: `PretrainedCNNExtractor.__init__` (lines 124-128) freezes the *entire* backbone and leaves only the re-initialised `conv1` trainable — CLAUDE.md requires training `layer3` and `layer4`, not `conv1` only. |
| `hmm.py` | **exists, partial** | `HMMClassifier` with per-class `GaussianHMM` is present (`hmm.py:38-46`). Missing: multiple-restart fitting (only one `random_state=42` fit per class), scaler+PCA preprocessing, and Platt calibration on inner validation — all required by the Accuracy Strategy. |
| `hybrid.py` | **exists** | `HybridCNNHMM` wraps CNN+HMM correctly at the API level; no `n_features`-to-`HMMClassifier` bug found (that bug, referenced in the old `ProcessPipeline.txt` GATE 1, is not present in the current file — already fixed or never existed in this version). Functionally duplicates logic already present in `trainer.py`/`cross_validate.py`. |

### src/training
| Module | Status | Note |
|---|---|---|
| `splits.py` | **missing** | `participant_aware_splitter.py` exists and is the one genuinely correct module in the repo (participant-level stratified split + `verify_no_leakage`, lines 108-122) — but it emits a single fixed train/val/test `splits.json` from `crop_index.csv`, not a `folds.csv` over *all* labeled participants scoped to the extreme-groups primary set. **Nothing downstream consumes it**: `dataloader.py`/`trainer.py`/`cross_validate.py`/`evaluator.py` all read ad hoc `DATA/SPLITS/{train,val,test}/HAPPY|SAD/*.png` folders instead, so the one correct splitter is disconnected from the real training path. |
| `aggregate.py` | **missing** | No per-participant mean-of-crop-probabilities aggregation exists anywhere, and no shared prediction schema (`analysis, model, feature_set, fold, participant_id, label, prob_sad, pred, in_middle_band`). |
| `trainer.py` | **exists, wrong protocol** | Trains on crop-level folder splits, not participant-level folds; no extreme-groups band; plain `Adam` (not AdamW with split `lr_head`/`lr_backbone`) and `ReduceLROnPlateau` (not cosine schedule). See §2.1, §2.2, §2.7. |
| `run_baselines.py` | **missing** | No standalone majority-class / LR-handcrafted / LR-embeddings runners; `trainer.py` bundles one ad hoc 3-feature LR only. |
| `run_cnn.py` | **missing** | No script implementing the specified fine-tune recipe. |
| `run_hybrid.py` | **missing** | No CNN-HMM-on-layer3-sequences-with-PCA-and-Platt-calibration script restricted to word/cursive crops. |
| `run_ensemble.py` | **missing** | No equal-weight ensemble of LR-handcrafted + LR-embeddings + CNN-HMM-fused exists. |
| `evaluator.py` | **exists, wrong protocol** | Reports accuracy/F1/ROC-AUC point estimates only — no bootstrap CIs, no balanced accuracy, no majority-baseline comparison, no paired test, no permutation test. Invokes the crop-level `CrossValidator` internally (§2.1). |

### src/analysis
| Module | Status | Note |
|---|---|---|
| `questionnaire_report.py` | **missing, partial analog** | `psychometrics.py` computes Cronbach's alpha (`cronbach_alpha`) and item-total correlations (good overlap) but recomputes `adjusted_total`/label from raw items (`psychometrics.py:73-75`) instead of reading them, and never reports HAPPY/SAD balance counts. |
| `permutation.py` | **missing** | No permutation test implemented anywhere, though required by the Accuracy Strategy. |
| `secondary.py` | **missing** | No full-sample (outside extreme-groups) secondary analysis. |
| `gradcam.py` | **missing as a module** | Grad-CAM (`_GradCAM` class) is implemented inline inside `training/evaluator.py:223-266`, not as its own module. |
| `report.py` | **missing** | No `RESULTS.txt` generator. |

### Root
| Item | Status | Note |
|---|---|---|
| `smoke_test.py` | **exists, wrong pipeline** | Exercises the old P0-P18 phases (stages A-G: config → register → manifest → score → extract → propagate → preprocess) against real `DATASET/raw` data (via `config.data.raw_data_dir`, itself pointed at the wrong drive/shape), not the new A-H build order, and uses no synthetic fixture. |
| `EVALUATION_PROTOCOL.md` | **missing** | |
| `REPRODUCE.md` | **missing** | |
| `notebooks/` | **exists** | `EMHA_Colab_Pipeline.ipynb` present; not re-audited line-by-line in this pass — flag for re-check once the pipeline is rewritten, to confirm no model code/hyperparameters leak into it (CLAUDE.md "NEVER" list, last line). |

**Orphaned / needs an explicit decision (in the repo, not in the module map):**
`src/data/register_participants.py`, `assign_pages.py`, `content_extractor.py`,
`propagate_labels.py`, `questionnaire_scorer.py`, `qc.py`, `preview_crops.py`;
`src/preprocessing/run_preprocessing.py`; `src/training/cross_validate.py`;
`src/utils/artifact_generator.py`; `src/predict.py`; `src/analysis/psychometrics.py`;
`ProcessPipeline.txt`, `PROJECT_STRUCTURE.md`, `SYSTEMS_GUIDE.md`,
`README_THESIS_INSTRUCTIONS.md`, `THESIS_CITATIONS_SUMMARY.md`,
`.cursorrules` (the last five still reference DASS/NEUTRAL/old-scheme
material per a grep for those terms). None of these map 1:1 onto a
module-map entry; each needs an explicit retire-or-repurpose decision rather
than silent reuse.

**New, not yet in any module-map slot, untracked in git:**
- `src/cropping/p3/crop_pictures.py`, `src/cropping/p4/crop_p4.py` — the
  "existing crop script" CLAUDE.md protects. Already correct and already run
  for real (285 participants cropped). Currently **uncommitted** (`??` in
  `git status`) — at risk of being lost; should be committed as-is early in
  Stage A, unmodified.
- `src/tabulation/ERHA_HappySad_Tabulation_FINALE - self-report data.csv` —
  a 400-row self-report export (399 data rows), already carrying
  `happiness_sum, sadness_sum, adjusted_total, mean_adjusted, p_happy, label`
  per participant. This is functionally the authoritative label source
  CLAUDE.md calls `DATASET/metadata/questionnaire_export.csv`, but it:
  (a) lives in `src/` (a code directory) instead of `DATASET/metadata/`;
  (b) is untracked in git; (c) its ID column header is literally `re` (not
  `participant_id`), and its ID values are `P001`-style, whereas the crop
  filenames the cropping scripts already produced use the bare code `001`
  (e.g. `EMHA-P3_DrawingExercise_001_D1.png`). **The ID-format mismatch
  (`P001` vs `001`) is exactly the fact CLAUDE.md's DATA LAYOUT section asks
  to confirm** before any labeler module is written — flagging per Rule 2
  ("Ask me for any missing fact... instead of guessing").

---

## 2. Non-negotiable / Accuracy-Strategy violations

**1. Participant-level splitting only (Non-Negotiable 1)**
- `src/training/cross_validate.py:11,76-83,99` — `StratifiedKFold` is built directly over `dataset.get_labels()` / `np.arange(len(labels))`, i.e. individual crop-image indices from `HandwritingDataset` (one row per PNG file). A participant's 24 crops can land on both sides of a fold. No `assert_no_leakage` call anywhere in this file, or anywhere in the repo (confirmed by search).
- `src/training/evaluator.py:416-441` (`_run_cross_validation`) feeds that same crop-level `CrossValidator` a combined train+val `HandwritingDataset` pool — same leakage, on every evaluator run.
- By contrast, `src/data/participant_aware_splitter.py` *is* correct (splits/stratifies at participant-ID level, has its own leakage check, `verify_no_leakage` lines 108-122) — but nothing downstream consumes its `splits.json`. `dataloader.py`/`trainer.py`/`cross_validate.py` instead read ad hoc `DATA/SPLITS/{train,val,test}/HAPPY|SAD/*.png` folders, so the one correct splitter is disconnected from the real training path.

**2. Folder-based HAPPY/SAD splits**
- `src/data/dataloader.py:16-56` — `HandwritingDataset` expects `data_dir/HAPPY/*.png` and `data_dir/SAD/*.png` and labels every image purely by which folder it's in. No `participant_id` is tracked per sample, so grouping crops back to a participant for the final decision is structurally impossible from this class as written.
- `src/preprocessing/pipeline.py:108-123,135-143` and the `DATA/SPLITS/{train,val,test}/HAPPY|SAD/...` paths assumed in `trainer.py:365-379` / `evaluator.py:418-490` all rely on this same folder convention — which nothing in phases P1-P8 actually populates (no script writes `DATA/SPLITS/*/HAPPY|SAD/`).

**3. Otsu binarization (forbidden — "grayscale … No binarization")**
- `src/preprocessing/pipeline.py:18,42-52`, called from `process()` at line 88.
- `src/training/trainer.py:69` — `_image_features()` also Otsu-thresholds each crop for the handcrafted-LR features.
- Both destroy the grayscale intensity information the brief requires preserved for stroke-level features.

**4. Deskew (forbidden — "No … deskew")**
- `src/preprocessing/pipeline.py:61-77`, called from `process()` at line 90 (`skip_skew` gate).
- `src/data/content_extractor.py:13-14,90-103` and `src/preprocessing/run_preprocessing.py:68-83,118` — the `skip_skew_prefixes` plumbing actively rotates word/cursive crops via `correct_skew()`. The brief wants `baseline_angle_deg` *measured*, not corrected.

**5. `cv2.resize` to a fixed square / stretch-resize (forbidden — "No stretch-resize: one fixed scale factor plus white padding")**
- `src/preprocessing/pipeline.py:79-83` — `normalize()` calls `cv2.resize(image, self.target_size)` (224×224 fixed square), no aspect-ratio preservation or padding.
- `src/data/dataloader.py:89,105` — `transforms.Resize(image_size)` in both train/val transforms, same stretch-to-square behavior.

**6. RandomRotation / RandomAffine (forbidden — "Never rotation, scaling, shear, or flips")**
- `src/data/dataloader.py:90-95` — `get_train_transform()` applies `transforms.RandomRotation(degrees=5)` and `transforms.RandomAffine(degrees=0, translate=(0.05,0.05), scale=(0.95,1.05))`. Both the rotation and the scale component are disallowed; only translation, brightness/contrast, and light blur are permitted (brightness/contrast via `ColorJitter` at line 96 is fine; there is no blur anywhere).

**7. Fine-tune recipe not implemented**
- `src/models/cnn.py:124-128` — `PretrainedCNNExtractor` freezes the entire ResNet18 backbone except the re-initialised `conv1`; CLAUDE.md specifies training `layer3` and `layer4`, not `conv1`.
- `src/training/trainer.py:165-171` — plain `optim.Adam` with one learning rate and `ReduceLROnPlateau`, not AdamW with split `lr_head=1e-3`/`lr_backbone=1e-4` and a cosine schedule.
- `src/training/cross_validate.py:47` — `CrossValidator.__init__` defaults `use_pretrained: bool = False`, inconsistent with `config.cnn.use_pretrained = True` (config.py:48); the module's own `__main__` demo never overrides it, so any standalone use of `CrossValidator` silently falls back to the un-pretrained custom CNN. (`evaluator.py` does pass the config value explicitly at line 439, so the main path is currently unaffected — but the inconsistent default is a live footgun.)

**8. DASS / NEUTRAL labeling**
- No DASS references found in `src/` code (only in `README_THESIS_INSTRUCTIONS.md`, `SYSTEMS_GUIDE.md`, `THESIS_CITATIONS_SUMMARY.md`, `.cursorrules` — docs, not code).
- No literal NEUTRAL class exists in code; `config.py:86,193`, `questionnaire_scorer.py:17,145`, `propagate_labels.py:9,131` all explicitly assert/print "NO NEUTRAL class." Good on this specific point.
- This is moot for a different reason: `questionnaire_scorer.py:73-90` recomputes the label from raw Likert items via a happiness/sadness-sum formula that appears nowhere in the current `CLAUDE.md`. The brief requires labels read verbatim from `questionnaire_export.csv` and never recomputed — `questionnaire_scorer.py`, its consumer `propagate_labels.py`, and `psychometrics.py:73-75` (a third independent recomputation) all violate this (Non-Negotiable 3).

**9. TensorFlow references**
- None found (`grep -i tensorflow|keras` across all `.py` files in `src/` returned no matches). `requirements.txt` pins PyTorch-only explicitly (header comment line 2, `torch`/`torchvision` lines 21-22). No violation.

**10. Other breaks found during review (adjacent to the checklist)**
- `config.data.splits_dir` is read in `trainer.py:365`, `evaluator.py:418,482` but is **not defined** on `DataConfig` (`config.py:12-24`) — these modules raise `AttributeError` immediately on use. Same for `config.data.labeled_dir` (`preprocessing/pipeline.py:136`), `config.data.staging_dir` (`data/assign_pages.py:154`), `config.data.extracted_dir` (`data/preview_crops.py:104`). Training, evaluation, and three data-pipeline modules are currently non-functional, independent of the spec pivot.
- `config.data.raw_data_dir` (`config.py:16`) points at `E:\EMHA_Thesis\DATASET\raw` — wrong drive (repo root is `D:\EMHA_Thesis\EMHA-1`) and wrong shape (`respondent_NNN/`) versus the new `DATASET/raw-3Page`/`raw-4Page`.
- Three independent, conflicting crop-extraction code paths exist: `src/cropping/p3/crop_pictures.py` + `src/cropping/p4/crop_p4.py` (fixed pixel boxes against `raw-3Page`/`raw-4Page`, the ones CLAUDE.md protects — already correct and already run) versus `src/data/content_extractor.py` (fractional boxes against the *old* `DATASET/raw` full scans, separate output tree `DATA/CROPS`). Any new `crop_manifest.py` must index the former, not reimplement the latter.
- Two conflicting flake8 configs exist simultaneously: `.flake8` (`max-line-length=100`, excludes `FIGURES`/`models`/`results`/`DOCS`/`CITATIONS`, has `per-file-ignores`) and `setup.cfg` (`max-line-length=88`, excludes `BACKUP`, no `results`/`models`/`DOCS`/`CITATIONS` exclusion). Flake8 reads `.flake8` preferentially when both exist in the same directory, so `setup.cfg`'s `[flake8]` section is silently dead configuration — CLAUDE.md Rule 7 ("black and flake8 must pass") can't be satisfied predictably while two configs disagree.
- The raw-data picture on disk is more fragmented than CLAUDE.md's DATA LAYOUT section describes: besides `raw-3Page`/`raw-4Page` (the ones named in CLAUDE.md, confirmed populated with 285 participants' worth of crops already), `DATASET/` also contains `raw/` (286 `respondent_NNN/{PNG,PDF}` folders, old scheme), `raw_scans/` (includes a `Missings/` subfolder and at least one `respondent_301` folder with raw timestamped files), and `raw-1and2Pages/` (per-page instruction/questionnaire scans, e.g. `EMHA-P1_Instruction_001.png`). None of the latter three are mentioned in CLAUDE.md's DATA LAYOUT — worth confirming with you whether they're superseded inputs to archive, or still-needed source material (e.g. `raw-1and2Pages` for the questionnaire page, `raw_scans/Missings` for exclusions tracking) before any ingest module is written.

---

## 3. .gitignore audit

Current `.gitignore` (repo root):

- **DATASET/ excluded:** Yes — line 6, blanket rule. This is working correctly in practice: `git status` shows nothing for the 285-participant `raw-3Page`/`raw-4Page` data actually sitting on disk.
- **Weights excluded:** Yes — `models/*.pth`, `models/*.pkl` (lines 20-21).
- **Results excluded:** **Partial.** Only `results/*.png` and `results/gradcam/` are ignored (lines 25-26). CSV/JSON result artifacts the pipeline actually writes — `training_log.csv`, `evaluation_report.csv`, `cv_results.json`, `model_comparison.csv` — are **not** excluded and would be committed as-is. Needs an explicit decision (track intentionally for thesis reproducibility, or ignore) rather than the current accidental partial coverage.
- **Secrets excluded:** **Partial.** Only `.env` (line 45) is covered. No pattern for `credentials*.json`, `service-account*.json`, `*.key`, etc. Newly relevant: the labeling flow is meant to read an external Google-Sheets export: if an API credential file is ever added to authenticate that export, nothing currently catches it.
- **`DATASET/metadata` carve-out:** The new CLAUDE.md layout puts the *authoritative* label file at `DATASET/metadata/questionnaire_export.csv`, but `DATASET/` is fully blanket-ignored (line 6) with no carve-out — unlike the old `.gitignore`'s pattern of explicitly un-ignoring `DATA/CROPS/.gitkeep`-style paths. If `DATASET/metadata/*.csv` is meant to be versioned the way `DATA/METADATA/*.csv` was under the old layout, `.gitignore` needs a `!DATASET/metadata/` exception; right now nothing under `DATASET/` can ever be committed. This is doubly relevant now that `src/tabulation/*.csv` (the de facto label source today) sits *outside* `DATASET/` entirely and so isn't gitignored at all — it would be committed by a bare `git add -A`, which may or may not be the intended handling for participant self-report data.
- **Untracked, not ignored, and arguably should be addressed before Stage A closes:** `src/cropping/p3/`, `src/cropping/p4/`, `src/tabulation/*.csv` — all real, load-bearing, already-run artifacts/code that `git status` reports as `??`. None are currently excluded by `.gitignore`, so they're just sitting uncommitted.

---

## 4. Proposed task order (mapped to Build Order A-H)

**A — Foundation**
- Commit `CLAUDE.md` itself (currently uncommitted working-tree-only).
- Commit `src/cropping/p3/`, `src/cropping/p4/` as-is (already correct, already run — don't touch the logic, just get them into git before they're at risk of being lost).
- Resolve the `.gitignore` gaps in §3 (results policy, secrets patterns, `DATASET/metadata` carve-out decision, decide where `src/tabulation/*.csv` belongs).
- Remove the duplicate flake8 config (`setup.cfg`'s `[flake8]` section is the one to drop; consolidate into `.flake8`).
- Rewrite `src/utils/config.py`: replace old-pipeline fields (`raw_data_dir=E:\...`, `crops_dir`, `processed_dir`, `labeling.*`, `DRAWING_CROPS` etc.) with the new layout (`DATASET/raw-3Page`, `raw-4Page`, `DATASET/metadata`, `DATASET/processed`), plus `middle_band_fraction`, the restricted augmentation set, and fixed-scale+padding sizing. Eliminate the dangling `splits_dir`/`labeled_dir`/`staging_dir`/`extracted_dir` references across the codebase in the same pass.
- Add `src/utils/seed.py` and `src/utils/synthetic.py` (mirroring `raw-3Page`/`raw-4Page`/`metadata`).
- Stand up `tests/` wired to the synthetic fixture.
- Get a decision from you on the orphaned `DATASET/raw`, `raw_scans`, `raw-1and2Pages` folders (§2.10) and on `ProcessPipeline.txt`/`PROJECT_STRUCTURE.md`/`SYSTEMS_GUIDE.md`/`README_THESIS_INSTRUCTIONS.md` (retire or keep as historical record).

**B — Labels**
- Get the `P001`-vs-`001` ID-format confirmation from you (flagged above) before writing `labeler.py`.
- Add `src/data/labeler.py`: read `DATASET/metadata/questionnaire_export.csv` verbatim, no recomputation; validate counts match `src/tabulation/...csv` (or wherever the canonical export ends up living) exactly.
- Add `src/analysis/questionnaire_report.py` (alpha + HAPPY/SAD balance) — reuse the Cronbach's-alpha/item-total-correlation math already in `psychometrics.py`, but stop recomputing labels there.
- Write and freeze the analysis-design decision file (extreme-groups band, `middle_band_fraction` default 0.30).
- Retire `questionnaire_scorer.py` and `propagate_labels.py` from the label path (Rule 3 violation, §2.8).

**C — Data indexing**
- Add `src/data/ingest.py`: scan validator + checksums for `raw-3Page`/`raw-4Page`, matching `participant_id` to the confirmed tabulation ID column.
- Add `src/data/crop_manifest.py` to index the 285-participant crops already produced by `src/cropping/p3`/`p4` — do not reimplement extraction; retire `content_extractor.py`'s competing fractional-box pipeline against the old `DATASET/raw` (§2.10).
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
- Fix the sample-level `StratifiedKFold` in `cross_validate.py`/`evaluator.py` (§2.1) to fold over participant IDs from `splits.py` instead of crop indices.
- Add `src/training/run_baselines.py` (majority class, LR-handcrafted, LR-embeddings).
- Add `src/training/run_cnn.py` with the specified fine-tune recipe (unfreeze `layer3`+`layer4`, AdamW split `lr_head`/`lr_backbone`, cosine schedule, early stop on participant macro-F1) — replace, don't patch, the frozen-backbone/plain-Adam recipe in `trainer.py` and `cnn.py:124-128`.
- Add `src/training/run_hybrid.py` (CNN-HMM on layer3 sequences, word/cursive only, scaler+PCA, HMM restarts, Platt calibration).
- Add `src/training/run_ensemble.py` (equal-weight LR-handcrafted + LR-embeddings + CNN-HMM-fused).
- Fix the `use_pretrained` default in `CrossValidator.__init__` (§2.7) if that class survives the rewrite.

**G — Runs**
- Rewrite `smoke_test.py` to exercise the new A-H pipeline against the synthetic fixture (not real `DATASET/raw` data), then local → Colab smoke test → pilot on the first ~100 participants (Kaggle, the 285 already-cropped participants more than cover this) → full run.

**H — Results**
- Rewrite `evaluator.py` to report accuracy, macro-F1, balanced accuracy, and ROC-AUC with bootstrap CIs, the majority baseline, a paired comparison against LR-handcrafted, and a permutation test — none of which exist today.
- Add `src/analysis/permutation.py` and `src/analysis/secondary.py`.
- Extract `_GradCAM` out of `evaluator.py` into `src/analysis/gradcam.py`.
- Add `src/analysis/report.py` → `RESULTS.txt`, plus `src/utils/package_dataset.py` (stage A) for the reproducibility bundle.
