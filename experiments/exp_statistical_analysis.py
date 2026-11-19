
"""Paired analysis of the synthetic oblique-forest experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


FULL = "Oblique + L0 + stability"

PRIMARY_COMPARATORS = [
    "Oblique + L0",
    "Oblique + L0 + uniform stability",
    "Oblique + L0 + shuffled stability",
]

OBLIQUE_MODELS = [
    "Oblique",
    "Oblique + L0",
    "Oblique + stability",
    FULL,
    "Oblique + L0 + uniform stability",
    "Oblique + L0 + shuffled stability",
    "Oblique + training selection",
]


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    """Holm correction for family-wise error control."""
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    corrected = np.empty(len(p_values), dtype=float)

    largest = 0.0
    for rank, index in enumerate(order):
        largest = max(
            largest,
            (len(p_values) - rank) * p_values[index],
        )
        corrected[index] = min(1.0, largest)

    return corrected


def bootstrap_mean_ci(
    values: np.ndarray,
    rng: np.random.Generator,
):
    """95% percentile bootstrap CI for the mean paired difference."""
    draws = rng.choice(
        values,
        size=(10_000, len(values)),
        replace=True,
    ).mean(axis=1)

    return tuple(np.quantile(draws, [0.025, 0.975]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--function", default="axis")
    parser.add_argument("--rho", type=float, default=0.9)
    parser.add_argument("--output-dir", type=Path, required=True)

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Read raw results and select one experiment setting.
    df = pd.read_csv(args.results)

    df = df[
        (df["function"] == args.function)
        & np.isclose(df["rho"], args.rho)
    ]

    if df.empty:
        raise ValueError("No matching (function, rho) rows.")

    if df.duplicated(["replicate", "model"]).any():
        raise ValueError(
            "Each (replicate, model) must appear exactly once."
        )

    # 2. Align models by replicate: this makes comparisons paired.
    wide = df.pivot(
        index="replicate",
        columns="model",
        values="brier",
    ).sort_index()

    needed = [FULL, *PRIMARY_COMPARATORS]
    missing = [m for m in needed if m not in wide.columns]

    if missing:
        raise ValueError(
            f"Missing models: {missing}. "
            "Run exp_correlation.py with --include-controls."
        )

    paired = wide[needed].dropna()

    if len(paired) < 10:
        raise ValueError("Need at least 10 complete replicates.")

    print(f"Complete paired replicates: {len(paired)}")

    if len(paired) < 20:
        print("Warning: small exploratory sample.")

    # 3. Boxplot for all oblique variants.
    models = [
        m for m in OBLIQUE_MODELS
        if m in wide.columns
    ]

    fig, ax = plt.subplots(
        figsize=(10, max(5, 0.7 * len(models) + 1.5))
    )

    ax.boxplot(
        [wide[m].dropna().to_numpy() for m in models],
        vert=False,
        tick_labels=models,
        showmeans=True,
    )

    ax.set_xlabel("Test Brier loss (lower is better)")
    ax.set_title(
        f"Oblique models: {args.function}, "
        f"rho={args.rho}, {len(wide)} replicates"
    )
    ax.grid(axis="x", alpha=0.25)

    fig.tight_layout()
    fig.savefig(
        args.output_dir / "brier_boxplots.png",
        dpi=250,
    )
    plt.close(fig)

    # 4. Paired Wilcoxon signed-rank tests.
    rng = np.random.default_rng(2026)
    tests = []
    deltas = []

    for competitor in PRIMARY_COMPARATORS:
        d = (
            paired[FULL] - paired[competitor]
        ).to_numpy(dtype=float)

        deltas.append(d)

        if np.all(d == 0):
            statistic, p_value = 0.0, 1.0
        else:
            result = wilcoxon(
                d,
                zero_method="wilcox",
                alternative="two-sided",
                method="auto",
            )

            statistic = float(result.statistic)
            p_value = float(result.pvalue)

        ci_low, ci_high = bootstrap_mean_ci(d, rng)

        tests.append({
            "competitor": competitor,
            "n_pairs": len(d),
            "mean_delta_brier": float(d.mean()),
            "median_delta_brier": float(np.median(d)),
            "mean_ci_low": ci_low,
            "mean_ci_high": ci_high,
            "full_better_count": int((d < 0).sum()),
            "ties_count": int((d == 0).sum()),
            "wilcoxon_statistic": statistic,
            "p_raw": p_value,
        })

    # 5. Holm adjustment for the three primary comparisons.
    table = pd.DataFrame(tests)
    table["p_holm"] = holm_adjust(
        table["p_raw"].to_numpy()
    )

    table.to_csv(
        args.output_dir / "paired_wilcoxon.csv",
        index=False,
    )

    print(
        table.to_string(
            index=False,
            float_format=lambda x: f"{x:.5f}",
        )
    )

    # 6. Boxplot of paired Brier differences.
    fig, ax = plt.subplots(figsize=(9, 4.5))

    ax.boxplot(
        deltas,
        vert=False,
        tick_labels=[
            "vs L0",
            "vs Uniform",
            "vs Shuffled",
        ],
        showmeans=True,
    )

    ax.axvline(
        0.0,
        color="black",
        linestyle="--",
        linewidth=1,
    )

    ax.set_xlabel(
        "Brier(Stability) - Brier(comparator); "
        "negative favors Stability"
    )
    ax.set_title(
        f"Paired differences: {args.function}, "
        f"rho={args.rho}"
    )
    ax.grid(axis="x", alpha=0.25)

    fig.tight_layout()
    fig.savefig(
        args.output_dir / "paired_differences.png",
        dpi=250,
    )
    plt.close(fig)

    print(f"\nSaved: {args.output_dir.resolve()}")

    # 7. Mean paired effects with bootstrap confidence intervals.
    fig, ax = plt.subplots(figsize=(9, 4.5))

    effects = table["mean_delta_brier"].to_numpy()
    lower = table["mean_ci_low"].to_numpy()
    upper = table["mean_ci_high"].to_numpy()

    xerr = np.vstack([
        effects - lower,
        upper - effects,
    ])

    y_positions = np.arange(len(table))

    ax.errorbar(
        effects,
        y_positions,
        xerr=xerr,
        fmt="o",
        capsize=4,
    )

    ax.axvline(
        0.0,
        color="black",
        linestyle="--",
        linewidth=1,
    )

    ax.set_yticks(
        y_positions,
        ["vs L0", "vs Uniform", "vs Shuffled"],
    )

    ax.set_xlabel(
        "Mean paired Brier difference "
        "(negative favors Stability)"
    )
    ax.set_title(
        "Mean effects with 95% bootstrap confidence intervals"
    )
    ax.grid(axis="x", alpha=0.25)

    fig.tight_layout()
    fig.savefig(
        args.output_dir / "paired_effects_ci.png",
        dpi=250,
    )
    plt.close(fig)



if __name__ == "__main__":
    main()
