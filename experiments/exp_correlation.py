"""Experiment 2: effect of feature correlation on sparse and stability-aware oblique forests.

Synthetic binary problems with p=50 features. The first four features (x1..x4) form an
equicorrelated block with correlation rho; the remaining 46 features are independent
N(0, 1) noise. Labels are Bernoulli(sigmoid(a * f(x))) for three boundary functions:

    axis     f(x) = x1
    oblique  f(x) = (x1 + x2 + x3) / sqrt(3)
    mixed    f(x) = x2                          if x1 <= 0
                    (x2 + x3 + x4) / sqrt(3)    otherwise

Because the block covers x1..x4 for every function, correlated *decoys* exist for the
axis and oblique functions, which is where a stability penalty could matter.
Note that Var f(x) changes with rho for the oblique and mixed functions (features in
the block are summed), so the effective signal strength changes with rho. The
``oracle_brier`` column (Brier loss of the true probability on the same test labels)
makes this visible and is the right reference line when reading the plots.

Models (identical tree-size limits, so differences come from split type and penalties):
    Random forest (sklearn), Oblique (lambda=0, gamma=0), Oblique + L0 (gamma=0),
    Oblique + L0 + stability (proposed). ``--include-controls`` adds the uniform and
    shuffled stability controls.

Run from the repository root:

    uv run python experiments/exp_correlation.py                # full run
    uv run python experiments/exp_correlation.py --quick        # smoke test, ~1 minute
    uv run python experiments/exp_correlation.py --n-jobs 4     # parallel over datasets

Outputs go to ``outputs/exp_correlation/``: config.json, results_raw.csv, summary.csv,
summary.json, brier_vs_rho.png and selection_vs_rho.png. Plots need matplotlib
(``uv pip install matplotlib``); the tables are written without it.

This script reports whatever the runs produce. Penalties are NOT tuned here.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import scipy
import sklearn
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier

from stable_oblique_forests import binary_classification_metrics, make_ablation_models

N_BLOCK = 4  # x1..x4 form the correlated block


def f_axis(X):
    return X[:, 0]


def f_oblique(X):
    return (X[:, 0] + X[:, 1] + X[:, 2]) / np.sqrt(3.0)


def f_mixed(X):
    return np.where(X[:, 0] <= 0, X[:, 1], (X[:, 1] + X[:, 2] + X[:, 3]) / np.sqrt(3.0))


def sigmoid(z):
    """Numerically stable logistic function (plain NumPy so worker processes can pickle it)."""
    return 0.5 * (1.0 + np.tanh(0.5 * z))


FUNCTIONS = {"axis": f_axis, "oblique": f_oblique, "mixed": f_mixed}
# Columns that actually enter f (0-based).
RELEVANT = {"axis": (0,), "oblique": (0, 1, 2), "mixed": (0, 1, 2, 3)}

MODEL_ORDER = [
    "Random forest",
    "Oblique",
    "Oblique + L0",
    "Oblique + L0 + stability",
    "Oblique + L0 + uniform stability",
    "Oblique + L0 + shuffled stability",
]
METRICS = ["brier", "accuracy", "f1", "average_precision", "balanced_accuracy"]
DIAGNOSTICS = [
    "oracle_brier", "nodes_per_tree", "mean_support_size", "relevant_fraction",
    "root_support_size", "root_relevant_fraction", "fit_seconds",
]


# ----------------------------------------------------------------------------- data
def draw_features(rng, n, n_features, rho):
    """Standard-normal features; columns 0..N_BLOCK-1 are equicorrelated with corr rho."""
    X = rng.standard_normal((n, n_features))
    shared = rng.standard_normal(n)
    X[:, :N_BLOCK] = np.sqrt(1.0 - rho) * X[:, :N_BLOCK] + np.sqrt(rho) * shared[:, None]
    return X


def make_dataset(rho, function, replicate, rho_index, args):
    """Train/test data. X and the label uniforms depend only on (seed, rho, replicate),
    so all functions and all models see paired data (common random numbers)."""
    rng = np.random.default_rng([args.seed, rho_index, replicate])
    X_train = draw_features(rng, args.n_train, args.n_features, rho)
    u_train = rng.random(args.n_train)
    X_test = draw_features(rng, args.n_test, args.n_features, rho)
    u_test = rng.random(args.n_test)
    f = FUNCTIONS[function]
    p_train = sigmoid(args.signal * f(X_train))
    p_test = sigmoid(args.signal * f(X_test))
    y_train = (u_train < p_train).astype(int)
    y_test = (u_test < p_test).astype(int)
    return X_train, y_train, X_test, y_test, p_test


# --------------------------------------------------------------------------- models
def build_models(args, seed):
    forest_kwargs = dict(
        n_estimators=args.n_estimators,
        n_bootstraps=args.n_bootstraps,
        random_state=seed,
        n_jobs=1,
    )
    ablations = make_ablation_models(
        sparsity_penalty=args.sparsity_penalty,
        stability_penalty=args.stability_penalty,
        **forest_kwargs,
    )
    models = {
        # Same size limits as the oblique forests' defaults (depth 8, leaf 15, split 60).
        "Random forest": RandomForestClassifier(
            n_estimators=args.n_estimators, max_depth=8, min_samples_split=60,
            min_samples_leaf=15, max_features="sqrt", random_state=seed, n_jobs=1,
        ),
        "Oblique": ablations["unpenalized"],
        "Oblique + L0": ablations["sparsity_only"],
        "Oblique + L0 + stability": ablations["sparsity_and_stability"],
    }
    if args.include_controls:
        models["Oblique + L0 + uniform stability"] = ablations["uniform_stability_control"]
        models["Oblique + L0 + shuffled stability"] = ablations["shuffled_stability_control"]
    return models


def selection_diagnostics(model, relevant):
    """Support size and share of relevant features among selected split features.

    Pooled over all accepted splits ("all") and over tree roots only ("root"). Roots see
    all construction data, so they are the least affected by small-node noise.
    """
    nan = float("nan")
    out = dict(mean_support_size=nan, relevant_fraction=nan,
               root_support_size=nan, root_relevant_fraction=nan)
    if not hasattr(model, "estimators_") or not hasattr(model, "node_count_"):
        return out  # sklearn forest: no oblique supports
    supports, roots = [], []
    for tree in model.estimators_:
        for d in tree.node_diagnostics_:
            supports.append(d["support"])
            if d["node_id"] == 0:
                roots.append(d["support"])

    def summarize(items):
        if not items:
            return nan, nan
        sizes = sum(len(s) for s in items)
        hits = sum(int(np.isin(s, relevant).sum()) for s in items)
        return sizes / len(items), hits / sizes

    out["mean_support_size"], out["relevant_fraction"] = summarize(supports)
    out["root_support_size"], out["root_relevant_fraction"] = summarize(roots)
    return out


def nodes_per_tree(model):
    if hasattr(model, "node_count_"):
        return model.node_count_ / len(model.estimators_)
    return float(np.mean([t.tree_.node_count for t in model.estimators_]))


def run_job(job, args):
    """One (rho, function, replicate) dataset: fit and evaluate every model."""
    rho, function, replicate, rho_index = job
    X_tr, y_tr, X_te, y_te, p_te = make_dataset(rho, function, replicate, rho_index, args)
    oracle = binary_classification_metrics(
        y_te, (p_te > 0.5).astype(int), p_te
    )["brier"]
    model_seed = int(np.random.default_rng([args.seed, rho_index, replicate, 1]).integers(2**31 - 1))
    rows = []
    for name, model in build_models(args, model_seed).items():
        start = time.perf_counter()
        model.fit(X_tr, y_tr)
        fit_seconds = time.perf_counter() - start
        proba = model.predict_proba(X_te)  # computed once: prediction is the slow step
        p1 = proba[:, 1]
        pred = model.classes_[np.argmax(proba, axis=1)]
        row = dict(function=function, rho=rho, replicate=replicate, model=name)
        row.update(binary_classification_metrics(y_te, pred, p1))
        row.update(selection_diagnostics(model, RELEVANT[function]))
        row.update(oracle_brier=oracle, nodes_per_tree=nodes_per_tree(model),
                   fit_seconds=fit_seconds)
        rows.append(row)
    return rows


# ------------------------------------------------------------------------ reporting
def summarize(rows):
    """Mean, standard deviation and replicate count per (function, rho, model)."""
    groups = {}
    for r in rows:
        groups.setdefault((r["function"], r["rho"], r["model"]), []).append(r)
    summary = []
    for (function, rho, model), items in groups.items():
        rec = dict(function=function, rho=rho, model=model, n_replicates=len(items))
        for key in METRICS + DIAGNOSTICS:
            vals = np.array([it[key] for it in items], dtype=float)
            rec[f"{key}_mean"] = float(np.nanmean(vals)) if np.isfinite(vals).any() else float("nan")
            rec[f"{key}_std"] = float(np.nanstd(vals, ddof=1)) if np.isfinite(vals).sum() > 1 else float("nan")
        summary.append(rec)
    order = {m: i for i, m in enumerate(MODEL_ORDER)}
    fn_order = {f: i for i, f in enumerate(FUNCTIONS)}
    summary.sort(key=lambda r: (fn_order[r["function"]], r["rho"], order[r["model"]]))
    return summary


def write_csv(path, records):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def print_summary(summary):
    print("\nTest-set means over replicates (Brier: lower is better; others: higher is better)")
    for function in FUNCTIONS:
        for rho in sorted({r["rho"] for r in summary}):
            block = [r for r in summary if r["function"] == function and r["rho"] == rho]
            if not block:
                continue
            print(f"\n[{function}, rho={rho}]  oracle Brier={block[0]['oracle_brier_mean']:.4f}")
            print(f"  {'Model':36} {'Brier':>15} {'Acc':>7} {'F1':>7} {'AP':>7} "
                  f"{'nodes/tree':>10} {'relev.':>7}")
            for r in block:
                rel = r["relevant_fraction_mean"]
                sd = r["brier_std"]
                brier = f"{r['brier_mean']:.4f}" + (f"±{sd:.4f}" if np.isfinite(sd) else "      ")
                print(f"  {r['model']:36} {brier:>15}"
                      f"  {r['accuracy_mean']:7.3f} {r['f1_mean']:7.3f}"
                      f" {r['average_precision_mean']:7.3f} {r['nodes_per_tree_mean']:10.1f} "
                      f"{'' if np.isnan(rel) else format(rel, '7.2f'):>7}")


MODEL_COLORS = {
    "Random forest": "tab:red",
    "Oblique": "tab:blue",
    "Oblique + L0": "tab:orange",
    "Oblique + L0 + stability": "tab:green",
    "Oblique + L0 + uniform stability": "tab:purple",
    "Oblique + L0 + shuffled stability": "tab:brown",
}

def make_plots(summary, raw_rows, out_dir, dpi=300):
    models = [m for m in MODEL_ORDER if any(r["model"] == m for r in summary)]
    rhos = sorted({r["rho"] for r in summary})
    functions = [f for f in FUNCTIONS if any(r["function"] == f for r in summary)]

    def panel(ax, function, key, ylabel, models_subset):
        for model in models_subset:
            rec = [r for r in summary if r["function"] == function and r["model"] == model]
            rec.sort(key=lambda r: r["rho"])
            mean = np.array([r[f"{key}_mean"] for r in rec])
            se = np.array([r[f"{key}_std"] / np.sqrt(r["n_replicates"]) for r in rec])
            ax.errorbar(
                [r["rho"] for r in rec],
                mean,
                yerr=np.nan_to_num(se),
                color=MODEL_COLORS[model],
                marker="o",
                capsize=3,
                label=model,
            )
        ax.set_xlabel(r"correlation $\rho$ within $x_1..x_4$")
        ax.set_ylabel(ylabel)
        ax.set_xticks(rhos)
        ax.grid(alpha=0.25)

    paths = []
    fig, axes = plt.subplots(1, len(functions), figsize=(5.2 * len(functions), 4.6), squeeze=False)
    for ax, function in zip(axes[0], functions):
        panel(ax, function, "brier", "Test Brier loss (lower is better)", models)
        oracle = sorted((r["rho"], r["oracle_brier_mean"]) for r in summary
                        if r["function"] == function and r["model"] == models[0])
        ax.plot(*zip(*oracle), "k--", lw=1, label="Oracle (true probability)")
        ax.set_title(f"f_{function}")
    axes[0][0].legend(fontsize=8)
    fig.suptitle("Brier loss vs feature correlation (mean ± s.e. over replicates)")
    fig.tight_layout()
    path = out_dir / "brier_vs_rho.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)

    oblique_models = [m for m in models if m != "Random forest"]
    rows = [("relevant_fraction", "Share of selected features that are relevant"),
            ("mean_support_size", "Mean support size of accepted splits"),
            ("nodes_per_tree", "Nodes per tree")]
    fig, axes = plt.subplots(len(rows), len(functions),
                             figsize=(5.2 * len(functions), 3.6 * len(rows)), squeeze=False)
    for i, (key, label) in enumerate(rows):
        for j, function in enumerate(functions):
            subset = models if key == "nodes_per_tree" else oblique_models
            panel(axes[i][j], function, key, label, subset)
            if i == 0:
                axes[i][j].set_title(f"f_{function}")
    handles, labels = axes[-1, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=3,
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    path = out_dir / "selection_vs_rho.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


    # Additional metrics, averaged over boundary functions
    # within each replicate, then summarized across replicates.
    metric_panels = [
        ("accuracy", "Accuracy (higher is better)"),
        ("f1", "F1 score (higher is better)"),
        ("average_precision", "Average precision (higher is better)"),
        ("nodes_per_tree", "Average nodes per tree"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    colors = {
        model: plt.get_cmap("tab10")(MODEL_ORDER.index(model) % 10)
        for model in models
    }

    for ax, (metric, title) in zip(axes.ravel(), metric_panels):
        for model in models:
            means = []
            errors = []
            available_rhos = []

            for rho in rhos:
                by_replicate = {}

                for row in raw_rows:
                    if row["model"] != model:
                        continue
                    if float(row["rho"]) != rho:
                        continue

                    replicate = int(row["replicate"])
                    by_replicate.setdefault(replicate, []).append(
                        float(row[metric])
                    )

                if not by_replicate:
                    continue

                # One value per replicate, averaged over
                # the boundary functions in the experiment.
                values = np.array([
                    np.mean(replicate_values)
                    for replicate_values in by_replicate.values()
                ])

                mean = float(np.mean(values))
                se = (
                    float(np.std(values, ddof=1) / np.sqrt(len(values)))
                    if len(values) > 1 else 0.0
                )

                available_rhos.append(rho)
                means.append(mean)
                errors.append(se)

            if available_rhos:
                ax.errorbar(
                    available_rhos,
                    means,
                    yerr=errors,
                    label=model,
                    color=colors[model],
                    marker="o",
                    linewidth=1.7,
                    capsize=3,
                )

        ax.set_title(title)
        ax.set_xlabel("Feature correlation $\\rho$")
        ax.set_xticks(rhos)
        ax.grid(alpha=0.25)

    handles, labels = axes[0, 0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=2,
        fontsize=9,
    )

    fig.suptitle(
        "Predictive metrics and tree complexity vs correlation",
        fontsize=14,
    )

    fig.tight_layout(rect=(0, 0.13, 1, 0.96))

    path = out_dir / "correlation_metrics.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)
    return paths


# ------------------------------------------------------------------------------ main
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rho", type=float, nargs="+", default=[0.0, 0.5, 0.9])
    p.add_argument("--functions", nargs="+", choices=list(FUNCTIONS), default=list(FUNCTIONS))
    p.add_argument("--n-features", type=int, default=50)
    p.add_argument("--n-train", type=int, default=1000)
    p.add_argument("--n-test", type=int, default=20_000)
    p.add_argument("--signal", type=float, default=2.0, help="signal strength a in sigmoid(a f(x))")
    p.add_argument("--n-replicates", type=int, default=5)
    p.add_argument("--n-estimators", type=int, default=100)
    p.add_argument("--sparsity-penalty", type=float, default=0.003, help="lambda")
    p.add_argument("--stability-penalty", type=float, default=0.007, help="gamma")
    p.add_argument("--n-bootstraps", type=int, default=10)
    p.add_argument("--include-controls", action="store_true",
                   help="also run the uniform and shuffled stability controls")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--n-jobs", type=int, default=1,
                   help="processes across datasets (each forest itself runs single-threaded)")
    p.add_argument("--output-dir", type=Path, default=Path("outputs/exp_correlation"))
    p.add_argument("--quick", action="store_true",
                   help="smoke test: 1 replicate, 10 trees, 2000 test points, 3 bootstraps")
    p.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Resolution of the saved plots.",
    )
    p.add_argument(
        "--plots-only",
        action="store_true",
        help="Regenerate plots from saved results without fitting models.",
    )
    args = p.parse_args()
    if args.quick:
        args.n_replicates, args.n_estimators, args.n_test, args.n_bootstraps = 1, 10, 2000, 3
    if not all(0.0 <= r < 1.0 for r in args.rho):
        p.error("--rho values must lie in [0, 1).")
    if args.n_features < N_BLOCK:
        p.error(f"--n-features must be at least {N_BLOCK}.")
    return args


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.plots_only:
        summary_path = args.output_dir / "summary.json"
        raw_path = args.output_dir / "results_raw.csv"

        summary = json.loads(
            summary_path.read_text(encoding="utf-8")
        )["summary"]

        with raw_path.open(newline="", encoding="utf-8") as file:
            raw_rows = list(csv.DictReader(file))

        plots = make_plots(
            summary,
            raw_rows,
            args.output_dir,
            dpi=args.dpi,
        )

        for path in plots:
            print(f"Saved {path.resolve()}")

        return
    jobs = [(rho, fn, rep, ri) for ri, rho in enumerate(args.rho)
            for fn in args.functions for rep in range(args.n_replicates)]
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    config["versions"] = dict(python=sys.version.split()[0], platform=platform.platform(),
                              numpy=np.__version__, scipy=scipy.__version__,
                              scikit_learn=sklearn.__version__)
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"{len(jobs)} datasets x {len(MODEL_ORDER) if args.include_controls else 4} models; "
          f"n={args.n_train}, test={args.n_test}, p={args.n_features}, "
          f"trees={args.n_estimators}, lambda={args.sparsity_penalty}, "
          f"gamma={args.stability_penalty}", flush=True)

    start = time.perf_counter()
    rows = []
    results = Parallel(n_jobs=args.n_jobs, return_as="generator")(
        delayed(run_job)(job, args) for job in jobs
    )
    for job, job_rows in zip(jobs, results):
        rows.extend(job_rows)
        print(f"  done rho={job[0]} function={job[1]} replicate={job[2]} "
              f"({time.perf_counter() - start:.0f}s elapsed)", flush=True)

    summary = summarize(rows)
    write_csv(args.output_dir / "results_raw.csv", rows)
    write_csv(args.output_dir / "summary.csv", summary)
    (args.output_dir / "summary.json").write_text(
        json.dumps({"config": config, "summary": summary}, indent=2, allow_nan=True),
        encoding="utf-8")
    print_summary(summary)
    plots = make_plots(
        summary,
        rows,
        args.output_dir,
        dpi=args.dpi,
    )
    print(f"\nSaved to {args.output_dir.resolve()}:")
    for name in ["config.json", "results_raw.csv", "summary.csv", "summary.json"] + [p.name for p in plots]:
        print(f"  {name}")


if __name__ == "__main__":
    main()