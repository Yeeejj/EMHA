# Reproducing the EMHA results

Every reported number comes from `results/final/RESULTS.txt`, which is built
from the files below by the commands below, in this order. The scripts
`scripts/reproduce_full.sh` (bash) and `scripts/reproduce_full.ps1`
(PowerShell) run exactly these commands and stop at the first error;
`--dry-run` / `-DryRun` prints them without running.

## 1. Code, environment, and data

```bash
git clone <repo-url> EMHA-1
cd EMHA-1
git checkout thesis-final-results        # tag created after RESULTS.txt is confirmed
python -m pip install -r requirements.txt
python -m pip install -r requirements-optional.txt   # transformers, for step 3d only
```

`requirements.txt` gives minimum versions only. Record the environment of
the final run with `python -m pip freeze > results/final/environment.txt`
and install from that file to match it exactly. [AT FINAL RUN: Python
version, `pip freeze` file, GPU model.]

Place the dataset (not in git; `.gitignore` excludes `DATASET/`) at
`DATASET/`, or point `EMHA_DATA_ROOT` at it:

| Path | Role |
|---|---|
| `raw-3Page/`, `raw-4Page/` | scans and crops, read-only |
| `metadata/questionnaire_export.csv` | authoritative labels (scoring sheet export) |
| `metadata/labels.csv` | frozen labels (checked byte-identical after step 1) |
| `metadata/participants.csv` | participant registry with final QC status |
| `metadata/folds.csv` | frozen outer folds |
| `metadata/crop_manifest.csv`, `metadata/qc_log.csv` | frozen crop index and QC decisions |

`crop_manifest.csv` and `qc_log.csv` are inputs, not regenerated:
`python -m src.data.crop_manifest` rewrites `qc_log.csv` with every crop
undropped, which would erase the QC decisions.

### Frozen input hashes (SHA-256)

These must equal the values recorded in `EVALUATION_PROTOCOL.md` section 3 at
the `protocol-frozen` tag; the scripts stop if they differ.

| File | SHA-256 |
|---|---|
| `metadata/folds.csv` | `[AT FREEZE]` |
| `metadata/labels.csv` | `[AT FREEZE]` |

Print them with `python -c "import hashlib,sys;print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" DATASET/metadata/folds.csv`.

## 2. Commands and expected outputs

Run from the repo root. `--device` is `auto`, `cpu`, or `cuda` (GPU strongly
recommended for steps 5c-5e and 6b). Paths are under `DATASET/` (data root)
or the repo root (output root).

| # | Command | Expected output |
|---|---|---|
| 0 | verify `folds.csv`, `labels.csv` hashes | match `EVALUATION_PROTOCOL.md` |
| 1a | `python -m src.data.labeler` | `metadata/labels.csv`, byte-identical to the frozen file |
| 1b | `python -m src.analysis.questionnaire_report` | `results/questionnaire/report.txt`, `score_distribution.png` |
| 2 | `python -m src.data.ingest` | `metadata/raw_manifest.csv` (scan checks and checksums; exit 1 on any problem) |
| 3a | `python -m src.preprocessing.pipeline --overwrite` | `processed/<id>/<cell>.png`, `metadata/processed_manifest.csv` |
| 3b | `python -m src.features.handcrafted` | `results/features/handcrafted_crop.csv`, `handcrafted_participant.csv`, `handcrafted_imputation_log.csv` |
| 3c | `python -m src.features.embeddings --device <d>` | `results/embeddings/resnet18_{drawing,word,cursive}.npz` |
| 3d | `python -m src.features.embeddings_handwriting --device <d>` (exploratory; skip with `--skip-exploratory`) | `results/embeddings/handwriting_{word,cursive}.npz` |
| 4 | `python -m src.training.splits` | prints `folds.csv unchanged`; hash re-checked |
| 5a | `python -m src.training.run_baselines --analysis primary --predict-middle` | `results/baselines/primary/predictions.csv`, `summary.json`, `lr_coefficients.csv` |
| 5b | `python -m src.training.run_baselines --analysis full` | `results/baselines/full/...` |
| 5c | `python -m src.training.run_cnn --analysis primary --predict-middle --device <d>` | `models/cnn/primary/<family>/fold_<k>.pth`, `results/cnn/primary/predictions.csv`, `crop_predictions.csv` |
| 5d | `python -m src.training.run_cnn --analysis primary --backbone simple --device <d>` | `results/cnn/primary_simple/...` (ablation) |
| 5e | `python -m src.training.run_hybrid --analysis primary --predict-middle --device <d>` | `models/hmm/primary/...`, `results/hybrid/primary/predictions.csv`, `selection.json` |
| 5f | `python -m src.training.run_ensemble --analysis primary` | `results/ensemble/primary/predictions.csv` |
| 6a | `python -m src.training.evaluator` | `results/final/model_comparison.{csv,txt}`, `per_fold_metrics.csv`, `confusion_matrices*`, `roc_curves_*.png`, `reliability_ensemble_primary.png` |
| 6b | `python -m src.analysis.permutation --model <m> --analysis <a>` for `lr_handcrafted`, `lr_embeddings` x `primary`, `full`; then `cnn_head_fused`, `cnn_hmm_fused`, `ensemble` x `primary` (`--device <d>`) | `results/final/permutation_<m>_<a>{.json,.png,_null.npy}`; p-values appended to `model_comparison.txt`; CNN draws cached in `results/final/permutation_runs/` (resumable) |
| 6c | `python -m src.analysis.secondary` | `results/final/secondary.txt`, `secondary_metrics.csv`, `secondary_*.png` |
| 6d | `python -m src.analysis.gradcam` | `results/final/gradcam/gradcam_<family>.png`, `<family>/*.png` |
| 6e | `python -m src.analysis.report` | `results/final/RESULTS.txt` |

Every model and evaluator run also appends rows to `results/RUN_LOG.csv`.
Model runners refuse real data unless the `protocol-frozen` git tag exists.
Steps 3a and 5c-5e resume or overwrite their own derived outputs only;
`raw-3Page/` and `raw-4Page/` are never written.

## 3. Checking a reproduction

Compare the new `results/final/RESULTS.txt` sections 1-4 (and
`model_comparison.csv`, `secondary_metrics.csv`) with the archived ones.
Sections 5-7 hold paths, hashes, and the git commit and are expected to differ
only in the data-root path. GPU training is not bit-deterministic across
hardware; on the same environment and device the numbers are expected to
match. [AT FINAL RUN: tolerance observed between two runs, if any.]

`tests/test_reproduce.py` runs both scripts end to end on a synthetic
fixture (smoke-sized config via `tests/reproduce_launcher.py`) and checks
that two fresh runs give byte-identical `model_comparison.csv`,
`per_fold_metrics.csv`, `secondary_metrics.csv`, and RESULTS.txt sections
1-4.

## 4. Tagging

After the thesis author confirms `results/final/RESULTS.txt` from the full
run:

```bash
git tag -a thesis-final-results -m "Final EMHA results (RESULTS.txt confirmed)"
```

The tag marks the commit whose code produced the confirmed RESULTS.txt.
