"""Publication-ready figures for the stable oblique forest experiments.

Reads the per-run CSV files written by ``exp_correlation.py`` and ``exp_openml.py`` and saves
300-dpi PNGs to ``outputs/plots/``:

    correlation_vs_brier.png   Brier loss vs correlation rho, one panel per boundary function,
                               including the uniform and shuffled stability controls.
    correlation_metrics.png    Accuracy, F1, mAP and nodes per tree vs rho (mean over the three
                               boundary functions, error bars = standard error over replicates).
    openml_benchmark.png       Brier loss, F1 and mAP per OpenML dataset (error bars = fold sd).

Default inputs (override with the options below):

    outputs/exp_correlation_small_penalty/results_raw.csv   (config.json next to it is optional)
    outputs/openml_raw.csv

Run from the repository root:

    uv pip install matplotlib seaborn pandas        # once; seaborn >= 0.13
    uv run python experiments/generate_plots.py

A figure whose input file is missing is skipped with a message; the others are still produced.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend: must be selected before importing pyplot

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

# key in results_raw.csv -> (legend label, marker, line style)
CORR_MODELS = {
    "Random forest": ("Random forest", "s", "-"),
    "Oblique": ("Oblique (unpenalized)", "o", "-"),
    "Oblique + L0": ("Oblique + L0", "^", "-"),
    "Oblique + L0 + stability": ("Oblique + L0 + stability (proposed)", "D", "-"),
    "Oblique + L0 + uniform stability": ("Uniform-stability control", "v", "--"),
    "Oblique + L0 + shuffled stability": ("Shuffled-stability control", "P", ":"),
}
FUNCTION_TITLES = {
    "axis": r"$f_{\mathrm{axis}}(x)=x_1$",
    "oblique": r"$f_{\mathrm{oblique}}(x)=(x_1+x_2+x_3)/\sqrt{3}$",
    "mixed": r"$f_{\mathrm{mixed}}(x)$ (piecewise)",
}
OPENML_MODELS = {  # key in openml_raw.csv -> legend label
    "Random forest (sklearn default)": "Random forest (default)",
    "Random forest (size-matched)": "Random forest (size-matched)",
    "Oblique (lambda=0, gamma=0)": r"Oblique ($\lambda=\gamma=0$)",
    "Oblique tuned (nested CV)": "Oblique tuned (nested CV)",
}


def set_style():
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    plt.rcParams.update({"axes.titleweight": "bold", "legend.frameon": False})


def require_columns(df, columns, name):
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise SystemExit(f"{name} is missing columns {missing}; re-run the experiment script.")


def config_note(directory):
    """One-line description of the run from config.json (empty string if unavailable)."""
    try:
        cfg = json.loads((Path(directory) / "config.json").read_text(encoding="utf-8"))
        return (rf"$\lambda={cfg['sparsity_penalty']:g}$, $\gamma={cfg['stability_penalty']:g}$, "
                f"{cfg['n_estimators']} trees, n={cfg['n_train']}, {cfg['n_replicates']} replicates")
    except (OSError, KeyError, ValueError):
        return ""


def draw_lines(ax, data, value, models):
    """Mean +/- standard error over replicates for each model; x is nudged to avoid overlap."""
    colors = dict(zip(CORR_MODELS, sns.color_palette("colorblind", len(CORR_MODELS))))
    for i, model in enumerate(models):
        label, marker, style = CORR_MODELS[model]
        agg = data[data.model == model].groupby("rho")[value].agg(["mean", "sem"]).reset_index()
        offset = (i - (len(models) - 1) / 2) * 0.012
        ax.errorbar(agg.rho + offset, agg["mean"], yerr=agg["sem"].fillna(0), color=colors[model],
                    marker=marker, ls=style, lw=1.7, ms=6, capsize=2.5, label=label)


def style_rho_axis(ax, rhos):
    ax.set_xticks(rhos)
    ax.set_xlabel(r"Feature correlation $\rho$ (within $x_1,\dots,x_4$)")


def plot_correlation_brier(raw, out_path, note, dpi):
    require_columns(raw, ["function", "rho", "model", "brier", "oracle_brier"], "results_raw.csv")
    functions = [f for f in FUNCTION_TITLES if f in set(raw.function)]
    models = [m for m in CORR_MODELS if m in set(raw.model)]
    rhos = sorted(raw.rho.unique())
    fig, axes = plt.subplots(1, len(functions), figsize=(5.6 * len(functions), 5.0), squeeze=False)
    for ax, function in zip(axes[0], functions):
        sub = raw[raw.function == function]
        draw_lines(ax, sub, "brier", models)
        oracle = sub.groupby("rho").oracle_brier.mean()
        ax.plot(oracle.index, oracle.values, color="black", lw=1.2, ls="-.",
                label="Oracle (true probability)")
        ax.set_title(FUNCTION_TITLES[function])
        ax.set_ylabel("Test Brier loss (lower is better)")
        style_rho_axis(ax, rhos)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.0))
    title = "Brier loss vs feature correlation (mean $\\pm$ s.e. over replicates)"
    fig.suptitle(title + (f"\n{note}" if note else ""), y=1.02)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_correlation_metrics(raw, out_path, note, dpi):
    panels = [("accuracy", "Accuracy (higher is better)"), ("f1", "F1 score (higher is better)"),
              ("average_precision", "mAP / average precision (higher is better)"),
              ("nodes_per_tree", "Average nodes per tree")]
    require_columns(raw, ["rho", "model", "replicate"] + [p[0] for p in panels], "results_raw.csv")
    models = [m for m in CORR_MODELS if m in set(raw.model)]
    rhos = sorted(raw.rho.unique())
    # Average over the boundary functions within each replicate, then summarize over replicates.
    pooled = raw.groupby(["model", "rho", "replicate"], as_index=False)[[p[0] for p in panels]].mean()
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, (value, ylabel) in zip(axes.ravel(), panels):
        draw_lines(ax, pooled, value, models)
        ax.set_ylabel(ylabel)
        style_rho_axis(ax, rhos)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0))
    title = "Predictive metrics and tree size vs feature correlation\n(mean over boundary functions; error bars: s.e. over replicates)"
    fig.suptitle(title + (f"\n{note}" if note else ""), y=1.01)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_openml(raw, out_path, dpi):
    metrics = [("brier", "Brier loss (lower is better)"), ("f1", "F1 score (higher is better)"),
               ("average_precision", "mAP / average precision (higher is better)")]
    require_columns(raw, ["dataset", "model"] + [m[0] for m in metrics], "openml_raw.csv")
    models = [m for m in OPENML_MODELS if m in set(raw.model)]
    datasets = list(dict.fromkeys(raw.dataset))
    palette = dict(zip(models, sns.color_palette("colorblind", len(models))))
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    for ax, (metric, ylabel) in zip(axes, metrics):
        kwargs = dict(data=raw, x="dataset", y=metric, hue="model", order=datasets, hue_order=models,
                      palette=palette, errorbar="sd", capsize=0.08, ax=ax)
        try:
            sns.barplot(err_kws={"linewidth": 1.0}, **kwargs)
        except TypeError:  # seaborn < 0.13
            sns.barplot(**kwargs)
        if ax.get_legend() is not None:
            ax.get_legend().remove()
        ax.set_xlabel("")
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=15)
    handles = [Patch(facecolor=palette[m], label=OPENML_MODELS[m]) for m in models]
    fig.legend(handles=handles, loc="lower center", ncol=len(models), bbox_to_anchor=(0.5, -0.03))
    n_folds = raw.groupby(["dataset", "model"]).size().min()
    fig.suptitle(f"OpenML benchmark: outer-fold test performance (mean, error bars = sd over {n_folds} folds)",
                 y=1.02)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--correlation-dir", type=Path, default=Path("outputs/exp_correlation_small_penalty"),
                   help="directory containing results_raw.csv from exp_correlation.py")
    p.add_argument("--openml-raw", type=Path, default=Path("outputs/openml_raw.csv"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs/plots"))
    p.add_argument("--dpi", type=int, default=300)
    args = p.parse_args()

    set_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    corr_csv = args.correlation_dir / "results_raw.csv"
    if corr_csv.exists():
        raw = pd.read_csv(corr_csv)
        note = config_note(args.correlation_dir)
        for name, func in [("correlation_vs_brier.png", plot_correlation_brier),
                           ("correlation_metrics.png", plot_correlation_metrics)]:
            func(raw, args.output_dir / name, note, args.dpi)
            saved.append(args.output_dir / name)
    else:
        print(f"[skip] {corr_csv} not found (run experiments/exp_correlation.py first).")

    if args.openml_raw.exists():
        plot_openml(pd.read_csv(args.openml_raw), args.output_dir / "openml_benchmark.png", args.dpi)
        saved.append(args.output_dir / "openml_benchmark.png")
    else:
        print(f"[skip] {args.openml_raw} not found (run experiments/exp_openml.py first).")

    for path in saved:
        print(f"saved {path}")


if __name__ == "__main__":
    main()