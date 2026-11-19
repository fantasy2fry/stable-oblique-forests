
"""Exploratory regularization paths for sparse oblique forests.

Run from the repository root:
    uv run python experiments/exp_regularization.py --n-jobs 4

The same synthetic dataset and estimator seed are reused across all configurations
within each replicate. Do not treat selected configurations as confirmatory tests.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from stable_oblique_forests import binary_classification_metrics, make_ablation_models
from exp_correlation import FUNCTIONS, make_dataset, nodes_per_tree, rho_seed

VARIANTS = {
    "L0": "sparsity_only",
    "Stability": "sparsity_and_stability",
    "Uniform": "uniform_stability_control",
    "Shuffled": "shuffled_stability_control",
}


def run_replicate(replicate: int, args: argparse.Namespace) -> list[dict]:
    rho_index = 0  # Retained for compatibility; make_dataset now seeds by rho_seed(rho).
    X_train, y_train, X_test, y_test, p_test = make_dataset(
        args.rho, args.function, replicate, rho_index, args
    )
    seed = int(np.random.default_rng(
        [args.seed, rho_seed(args.rho), replicate, 1]
    ).integers(2**31 - 1))
    oracle = float(np.mean((y_test - p_test) ** 2))
    forest_params = dict(
        n_estimators=args.n_estimators,
        n_bootstraps=args.n_bootstraps,
        random_state=seed,
        n_jobs=1,
    )
    records = []

    for lam in args.lambda_values:
        # L0 is independent of gamma: fit it once for each lambda.
        settings = [(0.0, "L0", "sparsity_only")]
        settings.extend(
            (gam, label, VARIANTS[label])
            for gam in args.gamma_values
            for label in ("Stability", "Uniform", "Shuffled")
        )
        for gam, label, ablation_key in settings:
            model = make_ablation_models(
                sparsity_penalty=lam,
                stability_penalty=gam,
                **forest_params,
            )[ablation_key]
            started = time.perf_counter()
            model.fit(X_train, y_train)
            seconds = time.perf_counter() - started
            probabilities = model.predict_proba(X_test)[:, 1]
            brier = float(np.mean((y_test - probabilities) ** 2))
            support_sizes = [
                len(d["support"])
                for tree in model.estimators_
                for d in tree.node_diagnostics_
            ]
            records.append(dict(
                replicate=replicate,
                model=label,
                sparsity_penalty=lam,
                stability_penalty=gam,
                brier=brier,
                excess_brier=brier - oracle,
                oracle_brier=oracle,
                nodes_per_tree=nodes_per_tree(model),
                mean_support_size=(float(np.mean(support_sizes))
                                   if support_sizes else np.nan),
                fit_seconds=seconds,
            ))
    return records


def plot_paths(summary: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    for (label, gamma), group in summary.groupby(
        ["model", "stability_penalty"], sort=False
    ):
        group = group.sort_values("nodes_per_tree_mean")
        legend = label if label == "L0" else f"{label}, gamma={gamma:g}"
        ax.errorbar(
            group["nodes_per_tree_mean"],
            group["brier_mean"],
            yerr=group["brier_se"],
            marker="o",
            capsize=2,
            label=legend,
        )
    ax.set_xlabel("Mean nodes per tree")
    ax.set_ylabel("Mean held-out Brier loss (lower is better)")
    ax.set_title("Regularization paths: predictive risk vs tree complexity")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=250)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--function", choices=list(FUNCTIONS), default="axis")
    parser.add_argument("--rho", type=float, default=0.9)
    parser.add_argument("--n-features", type=int, default=20)
    parser.add_argument("--n-train", type=int, default=600)
    parser.add_argument("--n-test", type=int, default=5000)
    parser.add_argument("--signal", type=float, default=2.0)
    parser.add_argument("--n-replicates", type=int, default=5)
    parser.add_argument("--n-estimators", type=int, default=20)
    parser.add_argument("--n-bootstraps", type=int, default=5)
    parser.add_argument("--lambda-values", type=float, nargs="+",
                        default=[0.0, 0.001, 0.002])
    parser.add_argument("--gamma-values", type=float, nargs="+",
                        default=[0.0005, 0.001])
    parser.add_argument("--seed", type=int, default=271828)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/regularization"))
    args = parser.parse_args()
    if not 0 <= args.rho < 1:
        parser.error("--rho must be in [0, 1).")
    if args.n_features < 4 or args.n_train <= 0 or args.n_test <= 0:
        parser.error("n-features must be >= 4 and train/test sizes positive.")
    if args.n_replicates < 1 or args.n_estimators < 1 or args.n_bootstraps < 1:
        parser.error("replicates, trees and bootstraps must be >= 1.")
    if any(v < 0 for v in args.lambda_values):
        parser.error("All lambda values must be non-negative.")
    if any(v <= 0 for v in args.gamma_values):
        parser.error("All gamma values must be positive (L0 uses gamma=0).")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config.json").write_text(
        json.dumps({k: str(v) if isinstance(v, Path) else v
                    for k, v in vars(args).items()}, indent=2),
        encoding="utf-8",
    )
    chunks = Parallel(n_jobs=args.n_jobs)(
        delayed(run_replicate)(i, args) for i in range(args.n_replicates)
    )
    raw = pd.DataFrame([row for chunk in chunks for row in chunk])
    raw.to_csv(args.output_dir / "results_raw.csv", index=False)

    keys = ["model", "sparsity_penalty", "stability_penalty"]
    summary = raw.groupby(keys, as_index=False).agg(
        n_replicates=("brier", "size"),
        brier_mean=("brier", "mean"),
        brier_sd=("brier", "std"),
        nodes_per_tree_mean=("nodes_per_tree", "mean"),
        support_mean=("mean_support_size", "mean"),
        fit_seconds_mean=("fit_seconds", "mean"),
    )
    summary["brier_se"] = (
        summary["brier_sd"] / np.sqrt(summary["n_replicates"])
    ).fillna(0.0)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    plot_paths(summary, args.output_dir / "brier_vs_nodes.png")
    print(summary.sort_values(["model", "sparsity_penalty", "stability_penalty"])
          .round(6).to_string(index=False))
    print(f"Saved results to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
