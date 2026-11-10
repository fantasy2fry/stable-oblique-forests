"""Standalone plotter for the OpenML benchmark.

Reads the per-fold results written by ``exp_openml.py`` and saves grouped bar plots of
Brier loss, accuracy, F1 and mAP (one group per dataset, one bar per model; error bars =
standard deviation over the outer folds) to ``<output-dir>/plots/openml_benchmark.png``.

Inputs (default ``--output-dir outputs``):

    outputs/tables/openml_benchmark_results.csv   per-fold metrics (required)
    outputs/openml_config.json                    dataset sizes for the axis labels (optional)

Usage from the repository root:

    uv run python experiments/generate_openml_plots.py
    uv run python experiments/generate_openml_plots.py --output-dir outputs --dpi 200

``exp_openml.py`` calls ``plot_openml_benchmark`` automatically at the end of a run. This script
needs pandas, matplotlib and seaborn (``uv sync --extra experiments``).
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

RESULTS_NAME = "openml_benchmark_results.csv"
MODEL_ORDER = ["RF default", "RF size-matched", "Oblique baseline", "Oblique tuned"]
PANELS = [
    ("brier", "Brier loss (lower is better)"),
    ("accuracy", "Accuracy (higher is better)"),
    ("f1", "F1 score, minority class positive (higher is better)"),
    ("average_precision", "mAP / average precision (higher is better)"),
]


def _dataset_labels(datasets, config_json):
    """Axis labels 'name (n=..., p=...)'; plain names if the config file is unavailable."""
    try:
        loaded = json.loads(Path(config_json).read_text(encoding="utf-8")).get("datasets_loaded", {})
    except (OSError, ValueError):
        loaded = {}
    labels = {}
    for name in datasets:
        meta = loaded.get(name)
        labels[name] = f"{name}\n(n={meta['n_samples']}, p={meta['n_features']})" if meta else name
    return labels


def plot_openml_benchmark(results_csv, config_json, output_path, dpi=300):
    """Create the 2x2 grouped bar figure and return the path of the saved PNG."""
    results_csv, output_path = Path(results_csv), Path(output_path)
    if not results_csv.exists():
        raise FileNotFoundError(f"{results_csv} not found; run experiments/exp_openml.py first.")
    df = pd.read_csv(results_csv)
    missing = [c for c in ["dataset", "model"] + [p[0] for p in PANELS] if c not in df.columns]
    if missing:
        raise ValueError(f"{results_csv} is missing columns {missing}")

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    datasets = list(dict.fromkeys(df["dataset"]))  # order of appearance = order of the run
    labels = _dataset_labels(datasets, config_json)
    df["dataset_label"] = df["dataset"].map(labels)
    order = [labels[d] for d in datasets]
    models = [m for m in MODEL_ORDER if m in set(df["model"])]
    models += sorted(set(df["model"]) - set(models))
    palette = dict(zip(models, sns.color_palette("colorblind", len(models))))

    fig, axes = plt.subplots(2, 2, figsize=(max(14.0, 0.9 * len(order) + 6.0), 11.5), squeeze=False)
    for ax, (metric, title) in zip(axes.ravel(), PANELS):
        kwargs = dict(data=df, x="dataset_label", y=metric, hue="model", order=order,
                      hue_order=models, palette=palette, errorbar="sd", capsize=0.05, ax=ax)
        try:
            sns.barplot(err_kws={"linewidth": 0.8}, **kwargs)
        except TypeError:  # seaborn < 0.13 has no err_kws
            sns.barplot(**kwargs)
        if ax.get_legend() is not None:
            ax.get_legend().remove()
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=45)
        for tick in ax.get_xticklabels():
            tick.set_ha("right")
    fig.legend(handles=[Patch(facecolor=palette[m], label=m) for m in models], loc="lower center",
               ncol=len(models), frameon=False, bbox_to_anchor=(0.5, 0.0))
    n_folds = df.groupby(["dataset", "model"]).size().min()
    fig.suptitle(f"OpenML benchmark on {len(order)} binary datasets "
                 f"(mean, error bars = sd over {n_folds} outer folds)", fontsize=15, y=0.995)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir", type=Path, default=Path("outputs"),
                   help="directory used by exp_openml.py (tables/, openml_config.json, plots/)")
    p.add_argument("--dpi", type=int, default=300)
    args = p.parse_args()
    try:
        path = plot_openml_benchmark(args.output_dir / "tables" / RESULTS_NAME,
                                     args.output_dir / "openml_config.json",
                                     args.output_dir / "plots" / "openml_benchmark.png", args.dpi)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc))
    print(f"saved {path.resolve()}")


if __name__ == "__main__":
    main()
