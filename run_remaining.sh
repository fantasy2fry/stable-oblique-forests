#!/usr/bin/env bash
# Resume the final suite after correlation, statistical and regularization completed.
# Compatible with the default macOS Bash 3.2.
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"
OUT="$ROOT_DIR/outputs/final"
mkdir -p "$OUT"
LOG="$OUT/log.txt"

# Append every subsequent stdout/stderr message to the existing final log.
exec > >(tee -a "$LOG") 2>&1

on_exit() {
    local status=$?
    if [ "$status" -eq 0 ]; then
        printf '\n[%s] Remaining experiments completed successfully.\n' "$(date '+%Y-%m-%d %H:%M:%S')"
    else
        printf '\n[%s] Remaining experiments FAILED (exit code %s).\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$status"
    fi
}
trap on_exit EXIT

export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

section() {
    printf '\n============================================================\n'
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$1"
    printf '============================================================\n'
}

section "RESUMING: only visual demos and OpenML (prior results are preserved)"
printf 'Output root: %s\nLog file:    %s\n' "$OUT" "$LOG"

# No --n-jobs option in the current visual_demo.py. This is deliberate.
for dataset in mixed diagonal moons w; do
    section "Visual demo: $dataset"
    uv run python examples/visual_demo.py \
        --dataset "$dataset" \
        --n-samples 2400 \
        --n-estimators 60 \
        --n-bootstraps 10 \
        --seed 42 \
        --grid-size 180 \
        --sparsity-penalty 0.001 \
        --stability-penalty 0.001 \
        --output-dir "$OUT/visual_demo"
done

section "OpenML: 15 datasets; nested CV; 12 outer-fold workers"
uv run python experiments/exp_openml.py \
    --outer-folds 5 \
    --inner-folds 3 \
    --n-repeats 1 \
    --n-iter 25 \
    --n-estimators 100 \
    --search-n-estimators 30 \
    --n-bootstraps 10 \
    --max-samples 3000 \
    --data-home "$OUT/openml/data_cache" \
    --strict \
    --seed 2026 \
    --n-jobs 12 \
    --output-dir "$OUT/openml"

section "Results produced by remaining steps"
find "$OUT/visual_demo" "$OUT/openml" -type f -print | sort
