#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

DEVICE=auto
SKIP_EXPLORATORY=0
DRY_RUN=0
while [ $# -gt 0 ]; do
  case "$1" in
    --device) DEVICE="$2"; shift 2 ;;
    --skip-exploratory) SKIP_EXPLORATORY=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

PY="${EMHA_PYTHON:-python}"
read -r -a PREFIX <<< "${EMHA_PYTHON_PREFIX:-}"
DATA_ROOT="${EMHA_DATA_ROOT:-DATASET}"
META="$DATA_ROOT/metadata"

step() {
  echo "==> python -m $*"
  if [ "$DRY_RUN" = 1 ]; then return 0; fi
  "$PY" ${PREFIX[@]+"${PREFIX[@]}"} -m "$@"
}

sha() {
  "$PY" -c 'import hashlib,sys;print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$1"
}

check_hash() {
  echo "==> verify $1 SHA-256"
  if [ "$DRY_RUN" = 1 ]; then return 0; fi
  local h
  h="$(sha "$META/$1")"
  echo "    $h"
  if [ -n "${2:-}" ] && [ "$h" != "$2" ]; then
    echo "ERROR: $1 changed during the run (was $2)" >&2
    exit 1
  fi
  if [ -f "$DATA_ROOT/.synthetic_root" ]; then
    echo "    synthetic root: not checked against EVALUATION_PROTOCOL.md"
  elif ! grep -q "$h" EVALUATION_PROTOCOL.md; then
    echo "ERROR: $1 SHA-256 $h is not recorded in EVALUATION_PROTOCOL.md" >&2
    exit 1
  fi
}

check_hash folds.csv
check_hash labels.csv
LABELS_SHA="$([ "$DRY_RUN" = 1 ] || sha "$META/labels.csv")"
FOLDS_SHA="$([ "$DRY_RUN" = 1 ] || sha "$META/folds.csv")"

step src.data.labeler
check_hash labels.csv "$LABELS_SHA"
step src.analysis.questionnaire_report

step src.data.ingest

step src.preprocessing.pipeline --overwrite
step src.features.handcrafted
step src.features.embeddings --device "$DEVICE"
if [ "$SKIP_EXPLORATORY" = 0 ]; then
  step src.features.embeddings_handwriting --device "$DEVICE"
fi

step src.training.splits
check_hash folds.csv "$FOLDS_SHA"

step src.training.run_baselines --analysis primary --predict-middle
step src.training.run_baselines --analysis full
step src.training.run_cnn --analysis primary --predict-middle --device "$DEVICE"
step src.training.run_cnn --analysis primary --backbone simple --device "$DEVICE"
step src.training.run_hybrid --analysis primary --predict-middle --device "$DEVICE"
step src.training.run_ensemble --analysis primary

step src.training.evaluator
for analysis in primary full; do
  for model in lr_handcrafted lr_embeddings; do
    step src.analysis.permutation --model "$model" --analysis "$analysis"
  done
done
for model in cnn_head_fused cnn_hmm_fused ensemble; do
  step src.analysis.permutation --model "$model" --analysis primary --device "$DEVICE"
done
step src.analysis.secondary
step src.analysis.gradcam
step src.analysis.report

echo "Done: results/final/RESULTS.txt"
