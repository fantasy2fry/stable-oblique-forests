
#!/usr/bin/env bash
# Complete research experiment suite.
# Compatible with macOS Bash 3.2+.

set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

OUT="${ROOT_DIR}/outputs/final"
mkdir -p "$OUT"

# Preserve the exact script used for this run.
cp "${BASH_SOURCE[0]}" "$OUT/run_final.sh"

LOG="$OUT/log.txt"

# Save all output to log.txt and display it in the terminal.
if [[ "${RUN_FINAL_LOGGED:-0}" != "1" ]]; then
    export RUN_FINAL_LOGGED=1
    set +e

    bash "$0" "$@" 2>&1 | tee -a "$LOG"
    statuses=("${PIPESTATUS[@]}")

    if [[ "${statuses[0]}" -ne 0 ]]; then
        exit "${statuses[0]}"
    fi

    exit "${statuses[1]}"
fi

trap 'status=$?; printf "\n[%s] FINISHED with exit code: %s\n" "$(date "+%Y-%m-%d %H:%M:%S")" "$status"' EXIT

JOBS=12

# Avoid excessive BLAS/OpenMP threads in parallel processes.
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

section "START: complete experiment suite"

printf 'Repository: %s\n' "$ROOT_DIR"
printf 'Outputs:    %s\n' "$OUT"
printf 'Log file:   %s\n' "$LOG"
printf 'Workers:    %s\n' "$JOBS"

if command -v git >/dev/null 2>&1; then
    git rev-parse HEAD || true
    git status --short || true
fi

# ---------------------------------------------------------
# 0. Dependencies and tests
# ---------------------------------------------------------

section "0. Dependencies and unit tests"

uv sync --extra dev --extra experiments
uv run python -m pytest -q


# ---------------------------------------------------------
# 1. Synthetic correlation experiment
# ---------------------------------------------------------

section "1. Synthetic correlation experiment"

uv run python experiments/exp_correlation.py \
    --functions axis oblique oblique2 mixed \
    --rho 0.0 0.25 0.5 0.75 0.9 \
    --n-features 20 \
    --n-train 1000 \
    --n-test 10000 \
    --signal 2.0 \
    --n-replicates 30 \
    --n-estimators 50 \
    --n-bootstraps 10 \
    --sparsity-penalty 0.001 \
    --stability-penalty 0.001 \
    --include-controls \
    --seed 314159 \
    --n-jobs "$JOBS" \
    --output-dir "$OUT/correlation"


# ---------------------------------------------------------
# 2. Paired statistical analyses
# ---------------------------------------------------------

section "2. Paired Wilcoxon tests"

for function in axis oblique oblique2 mixed; do
    for rho in 0.0 0.25 0.5 0.75 0.9; do

        section "Statistical analysis: $function rho=$rho"

        uv run python experiments/exp_statistical_analysis.py \
            --results "$OUT/correlation/results_raw.csv" \
            --function "$function" \
            --rho "$rho" \
            --output-dir "$OUT/statistical/$function/rho_$rho"

    done
done


# ---------------------------------------------------------
# 3. Regularization paths
# ---------------------------------------------------------

section "3. Regularization paths"

for function in axis oblique2; do

    section "Regularization path: $function"

    uv run python experiments/exp_regularization.py \
        --function "$function" \
        --rho 0.9 \
        --n-features 20 \
        --n-train 1000 \
        --n-test 10000 \
        --signal 2.0 \
        --n-replicates 12 \
        --n-estimators 40 \
        --n-bootstraps 10 \
        --lambda-values 0.0 0.0005 0.001 0.002 0.004 \
        --gamma-values 0.0005 0.001 0.002 \
        --seed 271828 \
        --n-jobs "$JOBS" \
        --output-dir "$OUT/regularization/$function"

done


# ---------------------------------------------------------
# 4. Visual demos
# ---------------------------------------------------------

section "4. Visual demos"

# Current visual_demo.py has no --n-jobs parameter.
# If you add it, this runner will automatically use 12.
${DEMO_JOBS_ARGS[@]+"${DEMO_JOBS_ARGS[@]}"} \

if uv run python examples/visual_demo.py --help | grep -q -- '--n-jobs'; then
    DEMO_JOBS_ARGS=(--n-jobs "$JOBS")
else
    printf 'Note: visual_demo.py uses internal n_jobs=1.\n'
fi

for dataset in mixed diagonal moons w; do

    section "Visual demo: $dataset"

    uv run python examples/visual_demo.py \
        --dataset "$dataset" \
        --n-samples 2400 \
        --n-estimators 60 \
        --n-bootstraps 10 \
        "${DEMO_JOBS_ARGS[@]}" \
        --seed 42 \
        --grid-size 180 \
        --sparsity-penalty 0.001 \
        --stability-penalty 0.001 \
        --output-dir "$OUT/visual_demo"

done


# ---------------------------------------------------------
# 5. OpenML benchmark
# ---------------------------------------------------------

section "5. OpenML benchmark"

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
    --n-jobs "$JOBS" \
    --output-dir "$OUT/openml"


# ---------------------------------------------------------
# 6. Final report
# ---------------------------------------------------------

section "6. Output inventory"

find "$OUT" -type f -print | sort

section "SUCCESS: all experiments finished"
