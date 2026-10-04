EMHA THESIS — MASTER BUILD BRIEF

PROJECT
Binary emotion classification (HAPPY vs SAD) from offline scanned handwriting and drawings. ~400 right-handed USC students aged 18–25 completed the FINALE assessment, designed with 4 registered psychometricians. Each participant has one label from a 24-item Likert self-report, scored in Google Sheets. Every one of a participant's 24 crops inherits that label. Models: logistic-regression baselines, frozen ResNet18 embeddings, fine-tuned ResNet18, a CNN-HMM hybrid (the titled method), and a fixed equal-weight ensemble. Evaluation is participant-level 5-fold cross-validation.

DATA LAYOUT (repo root D:\EMHA_Thesis\EMHA-1; data root DATASET/)
- Drawing scans: DATASET/raw-3Page/EMHA-P3_DrawingExercise_<code>.png
- Drawing crops: DATASET/raw-3Page/D1..D4/EMHA-P3_DrawingExercise_<code>_D<k>.png
- Writing scans: DATASET/raw-4Page/ (ask me to confirm scan names)
- Word crops: DATASET/raw-4Page/<cell>/EMHA-P4_WritingExercise_<code>_<cell>.png, cell in W1..W5 x {LH, RH, UC}
- Cursive crops: DATASET/raw-4Page/CS1..CS5/EMHA-P4_WritingExercise_<code>_CS<k>.png
- 24 crops per participant, one fixed pixel size per exercise, all boxes answered.
- participant_id = the 3-digit <code> (e.g. 001); it must match the tabulation export ID column (ask me to confirm).
- Labels: DATASET/metadata/questionnaire_export.csv (authoritative).
- Derived outputs: DATASET/metadata/, DATASET/processed/, results/, models/.
- Existing crop script src/cropping/p3 stays untouched. Write no new cropping code.

NON-NEGOTIABLES
1. Participant-level splitting only. A participant's crops are never split across train and validation/test. Call assert_no_leakage in every fold.
2. raw-3Page and raw-4Page (scans and crops) are read-only. Never modify, move, or rename anything in them.
3. Labels are read from the export, never recomputed. Counts must match the sheet exactly.
4. PyTorch only. No TensorFlow or Keras anywhere.
5. All hyperparameters live in dataclasses in src/utils/config.py. No magic numbers in modules.
6. Run modules from the repo root as python -m src.<package>.<module>.
7. black and flake8 must pass. Tests are in tests/ and use the synthetic fixture.
8. The analysis design (extreme-groups band) and EVALUATION_PROTOCOL.md are fixed before any real result. Nothing is tuned on outer test folds. Any change after the pilot is a logged Deviation.
9. Ask me before deleting, moving, or overwriting any existing file, and before any git rm, force push, or history rewrite.

ACCURACY STRATEGY (target: >= 70% participant-level accuracy and macro-F1 on the primary set, measured honestly)
- Extreme-groups primary set: exclude the middle_band_fraction (default 0.30) of participants closest to the label cutoff. The full sample is secondary.
- Preserve the signal: grayscale with background flattening only. No binarization, no deskew, no stretch-resize: one fixed scale factor plus white padding. Augment with translation, brightness/contrast, and light blur only. Never rotation, scaling, shear, or flips.
- Measure baseline_angle_deg per word and cursive crop without rotating it.
- Aggregate per participant: mean of crop probabilities is the primary rule.
- Features: handcrafted graphological, drawing-specific, and stroke-level (stroke width and darkness variation along the skeleton, tremor).
- Models: majority class; LR on handcrafted features; LR on frozen ResNet18 embeddings; fine-tuned ResNet18 (1-channel conv1 from summed pretrained weights; train layer3 and layer4; AdamW, lr_head 1e-3, lr_backbone 1e-4, cosine schedule, early stopping on inner-validation participant macro-F1); CNN-HMM on layer3 column sequences for word and cursive crops (scaler + PCA, per-class GaussianHMM with restarts, Platt calibration on inner validation); equal-weight ensemble of LR-handcrafted, LR-embeddings, and CNN-HMM-fused.
- Report accuracy, macro-F1, balanced accuracy, and ROC-AUC with bootstrap CIs, the majority baseline, a paired comparison against LR-handcrafted, and a permutation test.

MODULE MAP
src/utils: config.py, seed.py, synthetic.py, run_log.py, package_dataset.py
src/data: labeler.py, ingest.py, crop_manifest.py, dataloader.py, transforms.py, collector.py
src/preprocessing: pipeline.py
src/features: handcrafted.py, embeddings.py, embeddings_handwriting.py (exploratory)
src/models: cnn.py (EmotionCNN, CNNFeatureExtractor), hmm.py (HMMClassifier), hybrid.py (HybridCNNHMM)
src/training: splits.py, aggregate.py, trainer.py, run_baselines.py, run_cnn.py, run_hybrid.py, run_ensemble.py, evaluator.py
src/analysis: questionnaire_report.py, permutation.py, secondary.py, gradcam.py, report.py
Root: smoke_test.py, EVALUATION_PROTOCOL.md, REPRODUCE.md, notebooks/ (thin bootstrap only)
Prediction schema for every model: analysis, model, feature_set, fold, participant_id, label, prob_sad, pred, in_middle_band.

BUILD ORDER (stop at each gate and wait for my confirmation)
A Foundation: env, .gitignore (DATASET/ excluded), config + data root, seed, synthetic fixture mirroring the real layout, docs cleanup.
B Labels: labeler, questionnaire report (alpha, balance), analysis-design decision file.
C Data indexing: scan validator and checksums, crop manifest and verifier, contact sheets.
D Preprocessing and features: pipeline, transforms, handcrafted features, ResNet18 embeddings.
E Evaluation setup: dataset, folds.csv for all labeled participants, aggregation, protocol frozen and tagged protocol-frozen.
F Models: baselines, CNN, CNN-HMM, ensemble (optional exploratory handwriting backbone).
G Runs: smoke test (local, then Colab) -> pilot on the first ~100 participants (Kaggle) -> full run.
H Results: evaluator, permutation test, secondary analyses, Grad-CAM, RESULTS.txt, reproducibility bundle.

HOW TO WORK ON EVERY TASK
1. Restate the task in two lines and list the files you will create or change.
2. Ask me for any missing fact (column names, sizes, IDs) instead of guessing.
3. Search the web for the current docs of any library API you use (torchvision weights, hmmlearn, scikit-learn, Kaggle CLI).
4. Write or update tests with the code.
5. Run pytest -q, black --check ., and flake8, and show me the output.
6. Show the acceptance-criteria results, then stop.

NEVER
- Split a participant's crops across train and test, or fit any scaler, PCA, or model on test-fold participants.
- Change thresholds, the middle band, seeds, or model choices after seeing test-fold results.
- Report only the best fold, seed, or model.
- Drop participants based on their predictions.
- Report accuracy without the majority baseline and macro-F1.
- Put pipeline logic, model code, or hyperparameters inside notebooks.