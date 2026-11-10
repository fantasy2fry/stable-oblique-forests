"""Experiment 3 (v2): OpenML benchmark with nested random search over (lambda, gamma, ...).

For each of 15 OpenML binary-classification datasets, a stratified 5-fold outer CV compares

    * sklearn random forest, default settings (the package's reference baseline),
    * sklearn random forest with the oblique forests' size limits (depth 8, leaf 15, split 60),
    * oblique forest with lambda = gamma = 0 (package defaults otherwise),
    * oblique forest TUNED by nested random search. Inside every outer training fold a
      RandomizedSearchCV (``--n-iter`` samples, stratified inner CV, selected by Brier loss)
      draws from

          sparsity_penalty   (lambda): 0.0 with probability 0.2, else log-uniform on [1e-4, 1e-1]
          stability_penalty  (gamma) : 0.0 with probability 0.2, else log-uniform on [1e-4, 1e-1]
          max_depth                  : {6, 8, 12, None}
          min_samples_leaf           : {5, 10, 20}

      The search uses ``--search-n-estimators`` trees per candidate to save time; the winning
      configuration is then refitted on the whole outer training fold with ``--n-estimators`` trees.

No leakage: imputation, scaling and one-hot encoding live inside scikit-learn Pipelines, so they
are fitted on training folds only, in the outer loop and in every inner search fold. Test folds
are never used for tuning. Numeric columns: median imputation + standardization; categorical
columns: most-frequent imputation + one-hot encoding. The positive class is the LESS frequent
class, so F1 and average precision (mAP) are informative.

Determinism: all seeds derive from ``--seed`` and the dataset NAME (not its position), so running
a subset of datasets reproduces exactly the same folds; results do not depend on ``--n-jobs``.
Library versions are written to the config file. A dataset that cannot be loaded is skipped,
recorded in ``datasets_skipped`` of the config JSON and reported at the end (``--strict`` aborts
instead). Errors during fitting or searching are NOT swallowed (``error_score="raise"``).

Runtime: all forests are pure Python and the nested search dominates (about ``n_iter`` x
``inner_folds`` forest fits per outer fold). Scale with ``--n-jobs`` (processes across folds),
``--max-samples`` (stratified, seeded cap on rows per dataset; 0 disables), ``--n-iter`` and
``--search-n-estimators``. Partial results are rewritten after every finished fold.

Run from the repository root (needs internet on first use; scikit-learn caches the data):

    uv sync --extra dev --extra experiments        # or: uv pip install pandas matplotlib seaborn
    uv run python experiments/exp_openml.py --quick
    uv run python experiments/exp_openml.py --n-jobs 4

Outputs (``--output-dir``, default ``outputs/``):

    plots/openml_benchmark.png                  grouped bar plots: Brier, accuracy, F1, mAP
    tables/openml_benchmark_results.csv         full metrics per outer fold and model
    tables/openml_benchmark_summary.csv         mean / sd per dataset and model
    tables/openml_benchmark_macro.csv           macro means and average ranks over datasets
    tables/openml_search_trials.csv             inner-CV scores of every sampled configuration
    openml_config.json                          arguments, search space, datasets, versions
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
import zlib
from collections import Counter
from pathlib import Path
import traceback

import joblib
import numpy as np
import scipy
import sklearn
from joblib import Parallel, delayed
from scipy.stats import rankdata
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import (
    RandomizedSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils import check_random_state

from stable_oblique_forests import (
    ObliqueRandomForestClassifier,
    binary_classification_metrics,
    make_random_forest_reference,
)

# key -> (OpenML data id, expected OpenML dataset name). Loading by id is unambiguous; the
# returned name is compared with the expected one so that a wrong id can never silently
# benchmark a different dataset (it is skipped and reported instead).
DATASETS = {
    "breast-w": (15, "breast-w"),
    "diabetes": (37, "diabetes"),
    "credit-g": (31, "credit-g"),
    "ionosphere": (59, "ionosphere"),
    "banknote-authentication": (1462, "banknote-authentication"),
    "kr-vs-kp": (3, "kr-vs-kp"),
    "spambase": (44, "spambase"),
    "tic-tac-toe": (50, "tic-tac-toe"),
    "qsar-biodeg": (1494, "qsar-biodeg"),
    "blood-transfusion": (1464, "blood-transfusion-service-center"),
    "haberman": (43, "haberman"),
    "phoneme": (1489, "phoneme"),
    "churn": (40701, "churn"),
    "ilpd": (1480, "ilpd"),
    "adult": (1590, "adult"),
}
QUICK_DATASETS = ("breast-w", "haberman", "ionosphere")

METRICS = ("brier", "accuracy", "f1", "average_precision")
SCORING = {
    "brier": "neg_brier_score",
    "accuracy": "accuracy",
    "f1": make_scorer(f1_score, zero_division=0),
    "average_precision": "average_precision",
}
RF_DEFAULT = "RF default"
RF_MATCHED = "RF size-matched"
OBLIQUE_PLAIN = "Oblique baseline"
OBLIQUE_TUNED = "Oblique tuned"
MODEL_ORDER = [RF_DEFAULT, RF_MATCHED, OBLIQUE_PLAIN, OBLIQUE_TUNED]
TUNED_KEYS = ("sparsity_penalty", "stability_penalty", "max_depth", "min_samples_leaf")


class LogUniformOrZero:
    """Log-uniform on [low, high] with probability ``p_zero`` of returning exactly 0.0.

    Implements the ``rvs`` protocol used by ``ParameterSampler`` / ``RandomizedSearchCV``.
    A plain module-level class, so it can be pickled for process-based parallelism.
    """

    def __init__(self, low, high, p_zero=0.2):
        self.low, self.high, self.p_zero = float(low), float(high), float(p_zero)

    def rvs(self, size=None, random_state=None):
        rng = check_random_state(random_state)
        if rng.random_sample() < self.p_zero:
            return 0.0
        return float(np.exp(rng.uniform(np.log(self.low), np.log(self.high))))


SEARCH_SPACE = {
    "clf__sparsity_penalty": LogUniformOrZero(1e-4, 1e-1),
    "clf__stability_penalty": LogUniformOrZero(1e-4, 1e-1),
    "clf__max_depth": [6, 8, 12, None],
    "clf__min_samples_leaf": [5, 10, 20],
}


# ----------------------------------------------------------------------------- data
def dataset_seed(base_seed, key):
    """Seed derived from the base seed and the dataset name (independent of dataset order)."""
    return (int(base_seed) * 1_000_003 + zlib.crc32(key.encode("utf-8"))) % (2**31 - 1)


class FetchError(RuntimeError):
    """Raised when no parser could load a dataset; the message lists every attempt."""


def describe_error(exc):
    """'ExcType: message [at file.py:line in function]' using the innermost traceback frame."""
    frames = traceback.extract_tb(exc.__traceback__) if exc.__traceback__ else []
    where = f" [at {Path(frames[-1].filename).name}:{frames[-1].lineno} in {frames[-1].name}]" if frames else ""
    return f"{type(exc).__name__}: {exc}{where}"


def fetch_openml_frame(data_id, data_home):
    """fetch_openml with a fallback parser.

    ``parser="auto"`` resolves to scikit-learn's pandas parser for dense data. If it fails for any
    reason other than a missing optional dependency or a network problem, the pure-Python
    ``liac-arff`` parser is tried. All errors are collected and reported together.
    """
    kwargs = dict(data_id=data_id, as_frame=True, data_home=data_home, n_retries=3, delay=2.0)
    errors = []
    for parser in ("auto", "liac-arff"):
        try:
            return fetch_openml(parser=parser, **kwargs)
        except (ImportError, OSError):  # missing pandas / network: a second parser cannot help
            raise
        except Exception as exc:
            errors.append(f"parser={parser}: {describe_error(exc)}")
    raise FetchError("; ".join(errors))

def load_dataset(key, data_home, seed, max_samples):
    """Download one OpenML dataset; return X (DataFrame), y in {0, 1} and metadata."""
    data_id, expected = DATASETS[key]
    bunch = fetch_openml_frame(data_id, data_home)
    actual = str(bunch.details.get("name", ""))
    if actual.lower() != expected.lower():
        raise ValueError(f"data_id={data_id} is dataset '{actual}', expected '{expected}'")

    X = bunch.data.copy()
    X.columns = [str(c) for c in X.columns]
    for col in X.select_dtypes(include="category").columns:  # plain object dtype for imputers
        X[col] = X[col].astype(object)

    target = bunch.target
    if target is None:
        raise ValueError("dataset has no default target column")
    if getattr(target, "ndim", 1) == 2:  # several target columns: use the first
        target = target.iloc[:, 0]
    has_label = ~np.asarray(target.isna())  # rows without a label cannot be used
    if not has_label.all():
        X, target = X[has_label].reset_index(drop=True), target[has_label]
    labels_raw = np.array([str(v) for v in target], dtype=str)  # safe for numeric-looking labels

    labels, counts = np.unique(labels_raw, return_counts=True)
    if len(labels) != 2:
        raise ValueError(f"expected 2 classes, found {len(labels)}: {labels[:5].tolist()}")
    positive = labels[int(np.argmin(counts))]  # minority class (ties: first label)
    y = (labels_raw == positive).astype(int)
    n_total = len(y)
    if max_samples and n_total > max_samples:  # stratified, seeded cap (labels only used to stratify)
        keep, _ = train_test_split(np.arange(n_total), train_size=int(max_samples),
                                   stratify=y, random_state=seed)
        keep = np.sort(keep)
        X = X.iloc[keep].reset_index(drop=True)
        y = y[keep]
    info = dict(openml_id=int(data_id), openml_name=actual, n_samples_total=int(n_total),
                n_samples=int(len(y)), n_features=int(X.shape[1]), positive_class=str(positive),
                positive_rate=float(y.mean()))
    return X, y, info


def make_preprocessor():
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric, make_column_selector(dtype_include=np.number)),
        ("cat", categorical, make_column_selector(dtype_exclude=np.number)),
    ])


def with_preprocessing(classifier):
    return Pipeline([("prep", make_preprocessor()), ("clf", classifier)])


# ------------------------------------------------------------------------ evaluation
def run_fold(key, ds_seed, repeat, fold, X, y, train, test, args):
    """Tune, fit and score every model on one outer fold. Returns (result rows, trial rows)."""
    seed = int(np.random.default_rng([ds_seed, repeat, fold]).integers(2**31 - 1))
    X_tr, X_te, y_tr, y_te = X.iloc[train], X.iloc[test], y[train], y[test]

    def oblique(n_estimators, **params):
        return ObliqueRandomForestClassifier(
            n_estimators=n_estimators, n_bootstraps=args.n_bootstraps,
            random_state=seed, n_jobs=1, **params)

    # Nested random search on the outer training fold only; selection by inner-CV Brier loss.
    search = RandomizedSearchCV(
        with_preprocessing(oblique(args.search_n_estimators)),
        param_distributions=SEARCH_SPACE, n_iter=args.n_iter, scoring=SCORING, refit=False,
        cv=StratifiedKFold(args.inner_folds, shuffle=True, random_state=seed),
        random_state=seed, error_score="raise", n_jobs=1,
    )
    start = time.perf_counter()
    search.fit(X_tr, y_tr)
    search_seconds = time.perf_counter() - start
    res = search.cv_results_
    best = int(np.argmax(res["mean_test_brier"]))  # neg_brier_score: larger is better
    best_params = {name.split("__", 1)[1]: value for name, value in res["params"][best].items()}

    models = {
        RF_DEFAULT: with_preprocessing(
            make_random_forest_reference(n_estimators=args.n_estimators, random_state=seed)),
        RF_MATCHED: with_preprocessing(RandomForestClassifier(
            n_estimators=args.n_estimators, max_depth=8, min_samples_split=60,
            min_samples_leaf=15, random_state=seed)),
        OBLIQUE_PLAIN: with_preprocessing(oblique(args.n_estimators)),
        OBLIQUE_TUNED: with_preprocessing(oblique(args.n_estimators, **best_params)),
    }
    rows = []
    for name, model in models.items():
        start = time.perf_counter()
        model.fit(X_tr, y_tr)
        fit_seconds = time.perf_counter() - start
        proba = model.predict_proba(X_te)
        row = dict(dataset=key, repeat=repeat, fold=fold, model=name,
                   n_train=len(y_tr), n_test=len(y_te), fit_seconds=fit_seconds,
                   search_seconds=search_seconds if name == OBLIQUE_TUNED else 0.0,
                   sparsity_penalty="", stability_penalty="", max_depth="", min_samples_leaf="")
        row.update(binary_classification_metrics(y_te, np.argmax(proba, axis=1), proba[:, 1]))
        if name == OBLIQUE_TUNED:
            row.update(sparsity_penalty=best_params["sparsity_penalty"],
                       stability_penalty=best_params["stability_penalty"],
                       max_depth=str(best_params["max_depth"]),
                       min_samples_leaf=best_params["min_samples_leaf"])
        rows.append(row)

    trials = [
        dict(dataset=key, repeat=repeat, fold=fold, trial=i, selected=int(i == best),
             sparsity_penalty=p["clf__sparsity_penalty"], stability_penalty=p["clf__stability_penalty"],
             max_depth=str(p["clf__max_depth"]), min_samples_leaf=p["clf__min_samples_leaf"],
             brier=-res["mean_test_brier"][i], accuracy=res["mean_test_accuracy"][i],
             f1=res["mean_test_f1"][i], average_precision=res["mean_test_average_precision"][i])
        for i, p in enumerate(res["params"])
    ]
    return rows, trials


# ------------------------------------------------------------------------- reporting
def mean_std(items, key):
    values = np.array([float(it[key]) for it in items], dtype=float)
    return float(values.mean()), (float(values.std(ddof=1)) if len(values) > 1 else float("nan"))


def summarize(rows, info):
    order = {m: i for i, m in enumerate(MODEL_ORDER)}
    ds_order = {d: i for i, d in enumerate(info)}
    groups = {}
    for r in rows:
        groups.setdefault((r["dataset"], r["model"]), []).append(r)
    summary = []
    for (dataset, model), items in sorted(
            groups.items(), key=lambda kv: (ds_order[kv[0][0]], order[kv[0][1]])):
        rec = dict(dataset=dataset, **info[dataset], model=model, n_folds=len(items))
        for m in METRICS:
            rec[f"{m}_mean"], rec[f"{m}_std"] = mean_std(items, m)
        rec["fit_seconds_mean"], _ = mean_std(items, "fit_seconds")
        rec["search_seconds_mean"], _ = mean_std(items, "search_seconds")
        rec.update(tuned_lambda_zero_folds="", tuned_lambda_median="", tuned_gamma_zero_folds="",
                   tuned_gamma_median="", tuned_max_depth_counts="", tuned_min_samples_leaf_counts="")
        if model == OBLIQUE_TUNED:
            lam = np.array([float(it["sparsity_penalty"]) for it in items])
            gam = np.array([float(it["stability_penalty"]) for it in items])
            rec.update(
                tuned_lambda_zero_folds=int(np.sum(lam == 0.0)), tuned_lambda_median=float(np.median(lam)),
                tuned_gamma_zero_folds=int(np.sum(gam == 0.0)), tuned_gamma_median=float(np.median(gam)),
                tuned_max_depth_counts="; ".join(
                    f"{k}:{v}" for k, v in sorted(Counter(str(it["max_depth"]) for it in items).items())),
                tuned_min_samples_leaf_counts="; ".join(
                    f"{k}:{v}" for k, v in sorted(Counter(str(it["min_samples_leaf"]) for it in items).items())),
            )
        summary.append(rec)
    return summary


def macro_summary(summary):
    """Macro means over datasets and average ranks (rank 1 = best; ties share the mean rank)."""
    models = [m for m in MODEL_ORDER if any(r["model"] == m for r in summary)]
    datasets = list(dict.fromkeys(r["dataset"] for r in summary))
    means = {m: {k: [] for k in METRICS} for m in models}
    ranks = {m: {k: [] for k in METRICS} for m in models}
    for dataset in datasets:
        block = {r["model"]: r for r in summary if r["dataset"] == dataset}
        if not all(m in block for m in models):
            continue
        for k in METRICS:
            values = np.array([block[m][f"{k}_mean"] for m in models], dtype=float)
            ranking = rankdata(values if k == "brier" else -values, method="average")
            for m, value, rank in zip(models, values, ranking):
                means[m][k].append(float(value))
                ranks[m][k].append(float(rank))
    out = []
    for m in models:
        rec = dict(model=m, n_datasets=len(means[m]["brier"]))
        for k in METRICS:
            rec[f"{k}_macro_mean"] = float(np.mean(means[m][k])) if means[m][k] else float("nan")
            rec[f"{k}_avg_rank"] = float(np.mean(ranks[m][k])) if ranks[m][k] else float("nan")
        out.append(rec)
    return out


def write_csv(path, records):
    if not records:
        return
    fieldnames = list(dict.fromkeys(key for record in records for key in record))
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def print_summary(summary, macro):
    print("\nOuter-fold test results, mean +/- sd over folds (Brier: lower is better; others: higher)")
    for dataset in dict.fromkeys(r["dataset"] for r in summary):
        block = [r for r in summary if r["dataset"] == dataset]
        head = block[0]
        print(f"\n[{dataset}] n={head['n_samples']} (of {head['n_samples_total']}), "
              f"p={head['n_features']}, positive='{head['positive_class']}' ({head['positive_rate']:.2f})")
        print(f"  {'Model':18} {'Brier':>14} {'Accuracy':>14} {'F1':>14} {'mAP':>14}")
        for r in block:
            cells = [f"{r[f'{m}_mean']:.3f}+/-{np.nan_to_num(r[f'{m}_std']):.3f}" for m in METRICS]
            print(f"  {r['model']:18} " + " ".join(f"{c:>14}" for c in cells))
    print("\nMacro mean over datasets (average rank in parentheses; rank 1 = best)")
    print(f"  {'Model':18} " + " ".join(f"{m:>20}" for m in ("Brier", "Accuracy", "F1", "mAP")))
    for rec in macro:
        cells = [f"{rec[f'{m}_macro_mean']:.3f} ({rec[f'{m}_avg_rank']:.2f})" for m in METRICS]
        print(f"  {rec['model']:18} " + " ".join(f"{c:>20}" for c in cells))


def make_plot(rows, info, path):
    """Grouped bar plots (mean, error bars = sd over folds); returns False if plotting is unavailable."""
    try:
        import matplotlib

        matplotlib.use("Agg")  # headless backend, selected before pyplot is imported
        import matplotlib.pyplot as plt
        import pandas as pd
        import seaborn as sns
        from matplotlib.patches import Patch
    except ImportError as exc:
        print(f"[plots skipped] {exc}. Install with: uv sync --extra experiments "
              "(or: uv pip install pandas matplotlib seaborn)")
        return False

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
    df = pd.DataFrame(rows)
    labels = {d: f"{d}\n(n={v['n_samples']}, p={v['n_features']})" for d, v in info.items()}
    df["dataset_label"] = df["dataset"].map(labels)
    present = set(df["dataset"])
    order = [labels[d] for d in info if d in present]
    models = [m for m in MODEL_ORDER if m in set(df["model"])]
    palette = dict(zip(models, sns.color_palette("colorblind", len(models))))
    panels = [("brier", "Brier loss (lower is better)"),
              ("accuracy", "Accuracy (higher is better)"),
              ("f1", "F1 score, minority class positive (higher is better)"),
              ("average_precision", "mAP / average precision (higher is better)")]
    fig, axes = plt.subplots(2, 2, figsize=(max(16.0, 1.5 * len(order) + 4.0), 12.0), squeeze=False)
    for ax, (metric, title) in zip(axes.ravel(), panels):
        kwargs = dict(data=df, x="dataset_label", y=metric, hue="model", order=order,
                      hue_order=models, palette=palette, errorbar="sd", capsize=0.05, ax=ax)
        try:
            sns.barplot(err_kws={"linewidth": 0.8}, **kwargs)
        except TypeError:  # seaborn < 0.13
            sns.barplot(**kwargs)
        if ax.get_legend() is not None:
            ax.get_legend().remove()
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=45)
        for tick in ax.get_xticklabels():
            tick.set_ha("right")
    handles = [Patch(facecolor=palette[m], label=m) for m in models]
    fig.legend(handles=handles, loc="lower center", ncol=len(models), bbox_to_anchor=(0.5, -0.01))
    n_folds = df.groupby(["dataset", "model"]).size().min()
    fig.suptitle(f"OpenML benchmark across {len(order)} binary datasets "
                 f"(mean, error bars = sd over {n_folds} outer folds)", fontsize=15, y=0.995)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return True


def build_config(args, info, skipped):
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    config.update(
        search_space={
            "sparsity_penalty": "0.0 with probability 0.2, else log-uniform on [1e-4, 1e-1]",
            "stability_penalty": "0.0 with probability 0.2, else log-uniform on [1e-4, 1e-1]",
            "max_depth": [6, 8, 12, None],
            "min_samples_leaf": [5, 10, 20],
        },
        datasets_loaded=info,
        datasets_skipped=skipped,
        versions=dict(python=sys.version.split()[0], platform=platform.platform(),
                      numpy=np.__version__, scipy=scipy.__version__,
                      scikit_learn=sklearn.__version__, joblib=joblib.__version__),
    )
    return config


# ----------------------------------------------------------------------------- main
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", nargs="+", default=None, choices=list(DATASETS),
                   help="subset of datasets (default: all 15)")
    p.add_argument("--n-estimators", type=int, default=100, help="trees of every final model")
    p.add_argument("--search-n-estimators", type=int, default=30,
                   help="trees per candidate during the random search")
    p.add_argument("--n-bootstraps", type=int, default=10, help="node-level bootstraps (B)")
    p.add_argument("--n-iter", type=int, default=25, help="random-search samples per outer fold")
    p.add_argument("--outer-folds", type=int, default=5)
    p.add_argument("--inner-folds", type=int, default=3)
    p.add_argument("--n-repeats", type=int, default=1, help="repeats of the outer CV")
    p.add_argument("--max-samples", type=int, default=3000,
                   help="stratified cap on rows per dataset (0 = use all rows)")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--n-jobs", type=int, default=1, help="processes across outer folds")
    p.add_argument("--output-dir", type=Path, default=Path("outputs"))
    p.add_argument("--data-home", default=None, help="scikit-learn data cache directory")
    p.add_argument("--strict", action="store_true",
                   help="abort if any requested dataset cannot be loaded")
    p.add_argument("--quick", action="store_true",
                   help="smoke test: 3 small datasets, 10 trees, 4 search samples, 3 outer / 2 inner folds")
    args = p.parse_args()
    if args.quick:
        args.n_estimators, args.search_n_estimators, args.n_bootstraps = 10, 5, 3
        args.n_iter, args.outer_folds, args.inner_folds, args.max_samples = 4, 3, 2, 500
    args.datasets = args.datasets or list(QUICK_DATASETS if args.quick else DATASETS)
    if args.outer_folds < 2 or args.inner_folds < 2 or args.n_iter < 1:
        p.error("--outer-folds and --inner-folds must be >= 2 and --n-iter >= 1.")
    return args


def main():
    args = parse_args()
    tables_dir, plots_dir = args.output_dir / "tables", args.output_dir / "plots"
    for directory in (tables_dir, plots_dir):
        directory.mkdir(parents=True, exist_ok=True)

    data, info, skipped = {}, {}, {}
    for key in args.datasets:
        try:
            X, y, meta = load_dataset(key, args.data_home, dataset_seed(args.seed, key), args.max_samples)
        except ImportError as exc:
            sys.exit(f"{exc}\nInstall the experiment dependencies first: uv sync --extra experiments "
                     "(or: uv pip install pandas matplotlib seaborn)")
        except Exception as exc:  # network problems, renamed datasets, unexpected targets, ...
            skipped[key] = str(exc) if isinstance(exc, FetchError) else describe_error(exc)
            print(f"[skip] {key}: {skipped[key]}", flush=True)
            continue
        data[key], info[key] = (X, y), meta
        print(f"loaded {key}: {meta}", flush=True)

    config = build_config(args, info, skipped)
    (args.output_dir / "openml_config.json").write_text(
        json.dumps(config, indent=2, default=str), encoding="utf-8")
    if skipped:
        print(f"\nSkipped {len(skipped)} dataset(s): {', '.join(skipped)} "
              "(reasons are recorded in openml_config.json).", flush=True)
        if args.strict:
            sys.exit("--strict given: aborting because requested datasets could not be loaded.")
    if not data:
        sys.exit("No dataset could be loaded (check the internet connection and the skip messages).")

    jobs = []
    for key, (X, y) in data.items():
        ds_seed = dataset_seed(args.seed, key)
        splitter = RepeatedStratifiedKFold(n_splits=args.outer_folds, n_repeats=args.n_repeats,
                                           random_state=ds_seed)
        for i, (train, test) in enumerate(splitter.split(X, y)):
            jobs.append((key, ds_seed, i // args.outer_folds, i % args.outer_folds, X, y, train, test))
    print(f"\n{len(jobs)} outer folds on {len(data)} datasets; random search with {args.n_iter} "
          f"samples x {args.inner_folds} inner folds per outer fold.", flush=True)

    start = time.perf_counter()
    rows, trial_rows = [], []
    results = Parallel(n_jobs=args.n_jobs, return_as="generator")(
        delayed(run_fold)(*job, args) for job in jobs)
    for job, (fold_rows, fold_trials) in zip(jobs, results):
        rows.extend(fold_rows)
        trial_rows.extend(fold_trials)
        write_csv(tables_dir / "openml_benchmark_results.csv", rows)  # checkpoint after every fold
        print(f"  done {job[0]} repeat={job[2]} fold={job[3]} "
              f"({time.perf_counter() - start:.0f}s elapsed)", flush=True)

    summary = summarize(rows, info)
    macro = macro_summary(summary)
    write_csv(tables_dir / "openml_benchmark_results.csv", rows)
    write_csv(tables_dir / "openml_benchmark_summary.csv", summary)
    write_csv(tables_dir / "openml_benchmark_macro.csv", macro)
    write_csv(tables_dir / "openml_search_trials.csv", trial_rows)
    print_summary(summary, macro)

    plotted = make_plot(rows, info, plots_dir / "openml_benchmark.png")
    print(f"\nSaved tables to {tables_dir.resolve()}"
          + (f" and the figure to {(plots_dir / 'openml_benchmark.png').resolve()}" if plotted else "")
          + f"; config in {(args.output_dir / 'openml_config.json').resolve()}")
    if skipped:
        print(f"Reminder: {len(skipped)} requested dataset(s) were skipped: {', '.join(skipped)}")


if __name__ == "__main__":
    main()