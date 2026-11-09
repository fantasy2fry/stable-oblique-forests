"""Experiment 3: OpenML binary-classification benchmark with a grid over (lambda, gamma).

For every dataset, outer repeated stratified K-fold CV compares

    * sklearn random forest, default settings (the package's reference baseline),
    * sklearn random forest with the oblique forests' size limits (depth 8, leaf 15, split 60),
    * oblique forest with lambda = gamma = 0,
    * oblique forest tuned by NESTED CV: inside each outer training fold, a grid search over
      lambda, gamma in {0, 0.001, 0.005} (9 cells, selected by inner-CV Brier loss).

The inner-CV scores of all 9 grid cells are also logged, so the whole penalty surface is
visible and not only the selected cell. Preprocessing is a scikit-learn Pipeline fitted on
training folds only: median imputation + standardization for numeric columns, most-frequent
imputation + one-hot encoding for categorical columns (credit-g).

The positive class is the LESS frequent class, so F1 and average precision (mAP) are
informative. Metrics: Brier loss, accuracy, F1, average precision.

Robustness and reproducibility
------------------------------
* ``GridSearchCV(error_score="raise")``: a failing grid cell aborts the run immediately,
  with the dataset / repeat / fold in the error message, instead of becoming a silent NaN.
* Each dataset is loaded with several fallback strategies (by name with two parsers, then by
  OpenML data id). Every attempt is logged. A dataset that cannot be loaded is NOT silently
  dropped: it is recorded as ``skipped`` (with the error details) in ``openml_summary.csv``
  and in ``openml_config.json``. Use ``--fail-on-skip`` to exit non-zero in that case.
* Seeds depend only on (--seed, canonical dataset index, repeat, fold), never on which
  datasets happened to load or on --n-jobs. Every fold runs with BLAS/OpenMP limited to one
  thread, so the numerical results do not depend on the number of worker processes either
  (only ``fit_seconds`` varies from run to run).

Run from the repository root (needs internet on first use; scikit-learn caches the data):

    uv pip install pandas                                   # once, required by fetch_openml
    uv run python experiments/exp_openml.py --quick         # 2 small datasets, a few minutes
    uv run python experiments/exp_openml.py --n-jobs 4      # full run

Outputs (in --output-dir, default ``outputs/``): openml_summary.csv, openml_grid_summary.csv,
openml_raw.csv, openml_config.json. Penalties are tuned only inside the training folds.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import sklearn
from joblib import Parallel, delayed
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

from stable_oblique_forests import (
    ObliqueRandomForestClassifier,
    binary_classification_metrics,
    make_random_forest_reference,
)

DATASETS = ("breast-w", "diabetes", "credit-g", "ionosphere", "banknote-authentication")
# Last-resort fallback: OpenML data ids (version 1 of each dataset). Verify on openml.org.
OPENML_IDS = {
    "breast-w": 15,
    "diabetes": 37,
    "credit-g": 31,
    "ionosphere": 59,
    "banknote-authentication": 1462,
}
PENALTIES = (0.0, 0.001, 0.005)
METRICS = ("brier", "accuracy", "f1", "average_precision")
SCORING = {
    "brier": "neg_brier_score",
    "accuracy": "accuracy",
    "f1": make_scorer(f1_score, zero_division=0),
    "average_precision": "average_precision",
}
RF_DEFAULT = "Random forest (sklearn default)"
RF_MATCHED = "Random forest (size-matched)"
OBLIQUE_PLAIN = "Oblique (lambda=0, gamma=0)"
OBLIQUE_TUNED = "Oblique tuned (nested CV)"
MODEL_ORDER = [RF_DEFAULT, RF_MATCHED, OBLIQUE_PLAIN, OBLIQUE_TUNED]
MAX_ERROR_CHARS = 300


# ----------------------------------------------------------------------------- data
class DatasetLoadError(Exception):
    """Raised when every loading strategy for a dataset failed; carries the attempt log."""

    def __init__(self, name, attempts):
        self.name = name
        self.attempts = attempts
        super().__init__(f"{name}: all {len(attempts)} loading attempts failed")


def _short(message):
    message = " ".join(str(message).split())
    return message if len(message) <= MAX_ERROR_CHARS else message[: MAX_ERROR_CHARS - 3] + "..."


def format_attempts(attempts):
    """One-line, CSV-friendly description of a list of loading attempts."""
    parts = []
    for a in attempts:
        text = f"{a['strategy']}: {a['outcome']}"
        if a["outcome"] != "ok":
            text += f" ({a['error_type']}: {a['error_message']})"
        parts.append(text)
    return " | ".join(parts)


def _prepare(bunch, name, strategy):
    """Convert a fetched OpenML bunch into (X, y in {0,1}, info). Raises on bad data."""
    X = bunch.data.copy()
    for col in X.select_dtypes(include="category").columns:  # plain object dtype for imputers
        X[col] = X[col].astype(object)
    labels_raw = np.asarray(bunch.target.astype(str))
    labels, counts = np.unique(labels_raw, return_counts=True)
    if len(labels) != 2:
        raise ValueError(f"{name}: expected 2 classes, found {len(labels)}")
    positive = labels[int(np.argmin(counts))]  # minority class (ties: first label)
    y = (labels_raw == positive).astype(int)
    info = dict(n_samples=len(y), n_features=X.shape[1], positive_class=str(positive),
                positive_rate=float(y.mean()), openml_id=str(bunch.details.get("id", "")),
                loaded_via=strategy)
    return X, y, info


def load_dataset(name, data_home):
    """Download one OpenML dataset, trying several strategies in order.

    Returns (X, y, info, attempts). Raises ImportError immediately if pandas is missing
    (fatal for every dataset) and DatasetLoadError if every strategy failed.
    """
    strategies = [
        ("name, version=1, parser=auto", dict(name=name, version=1, parser="auto")),
        ("name, version=1, parser=liac-arff", dict(name=name, version=1, parser="liac-arff")),
    ]
    if name in OPENML_IDS:
        strategies.append((f"data_id={OPENML_IDS[name]}",
                           dict(data_id=OPENML_IDS[name], parser="auto")))

    attempts = []
    for strategy, kwargs in strategies:
        try:
            bunch = fetch_openml(as_frame=True, data_home=data_home, n_retries=3, delay=1.0,
                                 **kwargs)
            X, y, info = _prepare(bunch, name, strategy)
        except ImportError:
            raise  # missing pandas: not a per-dataset problem
        except Exception as exc:  # network problems, renamed datasets, parser issues, ...
            attempts.append(dict(strategy=strategy, outcome="failed",
                                 error_type=type(exc).__name__, error_message=_short(exc)))
            print(f"  [attempt failed] {name} via {strategy}: "
                  f"{type(exc).__name__}: {_short(exc)}", flush=True)
            continue
        attempts.append(dict(strategy=strategy, outcome="ok", error_type="", error_message=""))
        return X, y, info, attempts
    raise DatasetLoadError(name, attempts)


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
def fold_seed(base_seed, ds_index, repeat, fold):
    """Seed for one outer fold: a pure function of its identity, independent of n_jobs."""
    return int(np.random.default_rng([base_seed, ds_index, repeat, fold]).integers(2**31 - 1))


def _run_fold_impl(ds_index, dataset, repeat, fold, X, y, train, test, args):
    seed = fold_seed(args.seed, ds_index, repeat, fold)
    X_tr, X_te, y_tr, y_te = X.iloc[train], X.iloc[test], y[train], y[test]

    def oblique():
        return ObliqueRandomForestClassifier(
            n_estimators=args.n_estimators, n_bootstraps=args.n_bootstraps,
            random_state=seed, n_jobs=1,
        )

    grid = {"clf__sparsity_penalty": list(PENALTIES), "clf__stability_penalty": list(PENALTIES)}
    tuned = GridSearchCV(
        with_preprocessing(oblique()), grid, scoring=SCORING, refit="brier",
        cv=StratifiedKFold(args.inner_folds, shuffle=True, random_state=seed), n_jobs=1,
        error_score="raise",  # never turn grid-search failures into silent NaN scores
    )
    models = {
        RF_DEFAULT: with_preprocessing(
            make_random_forest_reference(n_estimators=args.n_estimators, random_state=seed)),
        RF_MATCHED: with_preprocessing(RandomForestClassifier(
            n_estimators=args.n_estimators, max_depth=8, min_samples_split=60,
            min_samples_leaf=15, random_state=seed)),
        OBLIQUE_PLAIN: with_preprocessing(oblique()),
        OBLIQUE_TUNED: tuned,
    }
    rows = []
    for name, model in models.items():
        start = time.perf_counter()
        model.fit(X_tr, y_tr)
        fit_seconds = time.perf_counter() - start
        proba = model.predict_proba(X_te)
        row = dict(dataset=dataset, repeat=repeat, fold=fold, model=name,
                   fit_seconds=fit_seconds, sparsity_penalty="", stability_penalty="")
        row.update(binary_classification_metrics(y_te, np.argmax(proba, axis=1), proba[:, 1]))
        if name == OBLIQUE_TUNED:
            row["sparsity_penalty"] = tuned.best_params_["clf__sparsity_penalty"]
            row["stability_penalty"] = tuned.best_params_["clf__stability_penalty"]
        rows.append(row)

    res = tuned.cv_results_
    grid_rows = [
        dict(dataset=dataset, repeat=repeat, fold=fold,
             sparsity_penalty=p["clf__sparsity_penalty"], stability_penalty=p["clf__stability_penalty"],
             brier=-res["mean_test_brier"][i], accuracy=res["mean_test_accuracy"][i],
             f1=res["mean_test_f1"][i], average_precision=res["mean_test_average_precision"][i])
        for i, p in enumerate(res["params"])
    ]
    return rows, grid_rows


def run_fold(ds_index, dataset, repeat, fold, X, y, train, test, args):
    """Fit and score every model on one outer fold. Returns (result rows, grid rows).

    Runs with BLAS/OpenMP limited to one thread so results do not depend on --n-jobs, and
    re-raises any failure with the fold identity attached (original traceback is chained).
    """
    try:
        with threadpool_limits(limits=1):
            return _run_fold_impl(ds_index, dataset, repeat, fold, X, y, train, test, args)
    except Exception as exc:
        raise RuntimeError(
            f"fold failed: dataset={dataset} repeat={repeat} fold={fold}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


# ------------------------------------------------------------------------- reporting
def mean_std(items, key):
    v = np.array([it[key] for it in items], dtype=float)
    return float(v.mean()), (float(v.std(ddof=1)) if len(v) > 1 else float("nan"))


def summarize(rows, info):
    groups = {}
    for r in rows:
        groups.setdefault((r["dataset"], r["model"]), []).append(r)
    order = {m: i for i, m in enumerate(MODEL_ORDER)}
    summary = []
    for (dataset, model), items in sorted(groups.items(), key=lambda kv: (list(info).index(kv[0][0]), order[kv[0][1]])):
        rec = dict(dataset=dataset, status="ok", **info[dataset], model=model, n_folds=len(items))
        for m in METRICS:
            rec[f"{m}_mean"], rec[f"{m}_std"] = mean_std(items, m)
        rec["chosen_penalties"] = "; ".join(
            f"lambda={lam:g},gamma={gam:g} x{c}" for (lam, gam), c in Counter(
                (it["sparsity_penalty"], it["stability_penalty"]) for it in items).most_common()
        ) if model == OBLIQUE_TUNED else ""
        summary.append(rec)
    return summary


def skipped_records(load_status):
    """Summary-CSV rows for datasets that could not be loaded (metric columns stay empty)."""
    return [
        dict(dataset=name, status="skipped", error_type=s["error_type"],
             error_message=s["error_message"], attempts_log=format_attempts(s["attempts"]))
        for name, s in load_status.items() if s["status"] == "skipped"
    ]


def summarize_grid(grid_rows):
    groups = {}
    for r in grid_rows:
        groups.setdefault((r["dataset"], r["sparsity_penalty"], r["stability_penalty"]), []).append(r)
    out = []
    for (dataset, lam, gam), items in groups.items():
        rec = dict(dataset=dataset, sparsity_penalty=lam, stability_penalty=gam, n_folds=len(items))
        for m in METRICS:
            rec[f"inner_cv_{m}_mean"], _ = mean_std(items, m)
        out.append(rec)
    return out


def write_csv(path, records):
    """Write records to CSV; the header is the union of all keys (first-seen order)."""
    fieldnames = list(dict.fromkeys(key for rec in records for key in rec))
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(records)


def print_summary(summary, load_status):
    ok_rows = [r for r in summary if r["status"] == "ok"]
    print("\nOuter-fold test results, mean +/- std (Brier: lower is better; others: higher)")
    for dataset in dict.fromkeys(r["dataset"] for r in ok_rows):
        block = [r for r in ok_rows if r["dataset"] == dataset]
        print(f"\n[{dataset}] n={block[0]['n_samples']}, p={block[0]['n_features']}, "
              f"positive='{block[0]['positive_class']}' ({block[0]['positive_rate']:.2f}), "
              f"loaded via {block[0]['loaded_via']}")
        print(f"  {'Model':32} {'Brier':>14} {'Accuracy':>14} {'F1':>14} {'mAP':>14}")
        for r in block:
            cells = [f"{r[f'{m}_mean']:.3f}+/-{np.nan_to_num(r[f'{m}_std']):.3f}" for m in METRICS]
            print(f"  {r['model']:32} " + " ".join(f"{c:>14}" for c in cells))