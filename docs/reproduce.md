# Reproduction Guide

Commands for reproducing the experiments, figures and demo. Run everything from the repository root.

## 0. Final results (as reported)

All results in the final report (`report/report_final.pdf`) come from one script:

```bash
bash run_final.sh
```

It runs the test suite, the synthetic correlation experiment (with controls), the regularization paths, the paired statistical analysis, the OpenML benchmark and the visual demos, and writes everything to `outputs/final/`. Run it with Bash 4 or newer: on Bash 3.2 (the macOS default) the script can fail on an empty array under `set -u`.

The tables and figures of the report are rebuilt from `outputs/final/` with the scripts in `report_build/` (they only read CSV files and do not fit models).

The sections below list individual commands for separate or exploratory runs. Their settings (e.g. penalty values and output folders) are not necessarily those used in the final report.

## 1. Setup

```bash
uv sync --extra dev --extra experiments
```

Alternative, in an existing environment:

```bash
uv pip install -e ".[dev,experiments]"
```

Optionally verify the installation (expected: `33 passed`):

```bash
uv run python -m pytest -q
```

## 2. Synthetic correlation experiment

```bash
uv run python experiments/exp_correlation.py \
  --sparsity-penalty 0.001 \
  --stability-penalty 0.002 \
  --include-controls \
  --n-jobs 4 \
  --output-dir outputs/exp_correlation_small_penalty
```

Results are written to `outputs/exp_correlation_small_penalty`. `--include-controls` adds the uniform- and shuffled-stability controls alongside the main variants.

## 3. OpenML benchmark

```bash
uv run python experiments/exp_openml.py --n-jobs 4
```

Datasets may be fetched from OpenML on first use, so network access can be required.

Both experiment scripts write their tables and figures during the same run. No
separate plotting command is required. The correlation experiment writes
`brier_vs_rho.png` and `selection_vs_rho.png`; the OpenML benchmark writes
`plots/openml_benchmark.png`.

## 4. Visual demo

```bash
uv run python examples/visual_demo.py
```

## Notes

- `--n-jobs 4` parallelizes trees with joblib threads. Adjust it to your machine.
- Results depend on the random seeds and parameters set in each script. Keep the exact commands above to match reported settings.