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

**Threads.** Every reported run pins BLAS/OpenMP to one thread, set before
Python starts (EVALUATION_PROTOCOL.md section 9, Deviation 2):
`export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1` (bash) or
`$env:OMP_NUM_THREADS = "1"` (and the same for the other two; PowerShell).
`scripts/reproduce_full.*` and both bootstrap notebooks set them, and the
`pipeline_runner` header prints what the main process and a worker actually
see. Without the pin, last-bit rounding depends on the machine's core count.
The CNN-HMM fits run in `RunnerConfig.hmm_n_jobs` processes (default -1, all
cores); results do not depend on it.

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
| `metadata/folds.csv` | `40403812420cde0e1f7d1980ce02ad0f77c531fdf59aab62b7987ca4b90751bf` |
| `metadata/labels.csv` | `694d360091e28dc331422cf7171b8e3afcf5dda16545e7ad5b2a9291204d3006` |

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

### Full run on Google Colab

The same packaged dataset as on Kaggle (`python -m src.utils.package_dataset`,
then `dist/emha-crops.zip` with `metadata/`, the crop folders and
`MANIFEST.txt` at the top level of the zip):

1. Upload the zip to Google Drive as `MyDrive/EMHA/emha-crops.zip`.
2. In Colab, open `notebooks/colab_bootstrap.ipynb` from GitHub (main), add
   the secret `GITHUB_TOKEN` (key icon, notebook access on), and choose
   Runtime -> Change runtime type -> A100 GPU.
3. Cell 1: `MODE = "run"`, `SUBSET = None`, `ANALYSIS = "primary,full"`,
   `STEPS = None`, `REQUIRE_GPU = "A100"`, `AUTO_DISCONNECT = True`. Run all.
   The notebook stops before any work on a different GPU, unzips the dataset
   to the local disk, runs `python -m src.pipeline_runner`, and releases the
   runtime when the run cell ends.
4. Pay As You Go has no background execution: keep the browser tab open and
   the PC awake. After a disconnect, Run all again with the same parameters;
   the step markers on Drive skip everything already finished.
5. Results, models, step markers and logs land on Drive under
   `MyDrive/EMHA/run_outputs/` (`results/`, `models/`,
   `logs/run_<UTC time>.log`). Download `results/` and `models/` from Drive
   (right-click -> Download) when the run has finished.

### Split permutation draws (Deviation 3)

The full run used the session order in `DOCS/plans/R4.6_split_run_sheet.md`:
the 20 CNN-path permutation draws were split between Colab and Kaggle
(`DRAWS`), and the Kaggle draws were copied into the Drive runs folder with
`python -m scripts.merge_draws`. Hardware per part:

| Part | Machine | GPU | vCPUs (`nproc`) |
|---|---|---|---|
| handcrafted (S1) | Colab (`logs/run_20261021T160932Z.log`: `--device cpu --steps handcrafted`) | none (CPU) | [CHECK] |
| embeddings → ensemble (S2) | Colab (`logs/run_20261021T170946Z.log`: `--device cuda`) | [CHECK: NVIDIA L4] | [CHECK: 12] |
| full LR permutation (S3) | [CHECK] | none | [CHECK] |
| draws 0–11 | Colab [CHECK: same session as S2] | [CHECK: NVIDIA L4] | [CHECK: 12] |
| draws 12–19 | Kaggle (`perm_0012`–`perm_0017` are in Drive `kaggle_draws/seed_42/`; 18–19 [CHECK]) | [CHECK: Tesla T4 x2] | [CHECK] |
| final runner session (S5) | Colab (`logs/run_20261022T151958Z.log`: every step `[SKIP]`, exit 0) | | |

Final RESULTS.txt PROVENANCE: git HEAD `996340d`, protocol tag `918990a`,
data root `/content/emha-crops`. Machine names and `--device` come from the
Drive run logs; the runner header records neither the GPU model nor `nproc`,
so those cells stay [CHECK] until confirmed from the notebook output.

### CNN-HMM v2 (exploratory, Deviation 5)

Post hoc and never part of the primary claim (EVALUATION_PROTOCOL.md
section 9, Deviation 5). It needs no GPU and no torch at run time: it reads
cached files only.

Inputs (read-only):

- `results/embeddings/resnet18_drawing.npz`, `resnet18_word.npz`,
  `resnet18_cursive.npz` (the full run's caches; `cache_key` must match the
  processed crops)
- `DATASET/metadata/folds.csv`, `labels.csv`, `participants.csv`
- for the paired comparisons: `results/baselines/primary/predictions.csv`
  and `results/hybrid/primary/predictions.csv`

Commands (PowerShell, repo root, tag `cnn-hmm-v1` or later):

```powershell
$env:EMHA_DATA_ROOT = "D:\EMHA_Thesis\EMHA-1\DATASET"
$env:EMHA_OUTPUT_ROOT = "D:\EMHA_Thesis\run_outputs"
$env:OMP_NUM_THREADS = "1"; $env:MKL_NUM_THREADS = "1"; $env:OPENBLAS_NUM_THREADS = "1"
python -m src.training.run_hybrid_v2                  # 5 outer folds, then report
python -m src.training.run_hybrid_v2 --permutation    # N = 100 full refits, then report
```

Both commands resume from their caches (`parts/fold_<k>.*`, `perm_<k>.json`)
and refuse to run if `run_config.json` / `settings.json` record different
settings. Settings: `HybridV2Config` in `src/utils/config.py`.

Run of record (2026-10-23, CPU only, 4 cores (`nproc`), one BLAS thread per
process; `logs/v2_run.log`): outer folds 44.0, 33.0, 35.6, 39.7 and 36.5 s
(main run 16:15:25 to 16:18:51 including evaluation); permutation
100/100 draws in 327.5 min (16:18:51 to 21:46:51).

Outputs (under `EMHA_OUTPUT_ROOT`; none are written to v1 paths):

- `results/hybrid_v2/primary/`: `predictions.csv`, `parts/`,
  `selected_settings.csv`, `selection.json`, `run_config.json`,
  `metrics.csv`, `per_fold.csv`, `confusion.csv`, `paired.csv`,
  `permutation.json`, `permutation_null_v2.png`
- `models/hmm_v2/primary/fold_<k>.joblib`
- `results/final/permutation_runs/primary_v2/seed_42/`
- `results/final/EXPLORATORY_v2.txt`

Checksums of the run-of-record outputs: `DOCS/plans/v2_result.txt`; backup
archive: `DOCS/plans/v2_outputs_backup.sha256`.

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
