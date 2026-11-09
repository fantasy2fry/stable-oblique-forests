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

from stable_oblique_forests import (
    ObliqueRandomForestClassifier,
    binary_classification_metrics,
    make_random_forest_reference,
)

DATASETS = ("breast-w", "diabetes", "credit-g", "ionosphere", "banknote-authentication")
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


# ----------------------------------------------------------------------------- data
def load_dataset(name, data_home):
    """Download one OpenML dataset (version 1); return X, y in {0,1}, and metadata."""
    bunch = fetch_openml(name=name, version=1, as_frame=True, data_home=data_home, parser="auto")
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
                positive_rate=float(y.mean()), openml_id=str(bunch.details.get("id", "")))
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
def run_fold(ds_index, dataset, repeat, fold, X, y, train, test, args):
    """Fit and score every model on one outer fold. Returns (result rows, grid rows)."""
    seed = int(np.random.default_rng([args.seed, ds_index, repeat, fold]).integers(2**31 - 1))
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
        rec = dict(dataset=dataset, **info[dataset], model=model, n_folds=len(items))
        for m in METRICS:
            rec[f"{m}_mean"], rec[f"{m}_std"] = mean_std(items, m)
        rec["chosen_penalties"] = "; ".join(
            f"lambda={lam:g},gamma={gam:g} x{c}" for (lam, gam), c in Counter(
                (it["sparsity_penalty"], it["stability_penalty"]) for it in items).most_common()
        ) if model == OBLIQUE_TUNED else ""
        summary.append(rec)
    return summary


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
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def print_summary(summary):
    print("\nOuter-fold test results, mean +/- std (Brier: lower is better; others: higher)")
    for dataset in dict.fromkeys(r["dataset"] for r in summary):
        block = [r for r in summary if r["dataset"] == dataset]
        print(f"\n[{dataset}] n={block[0]['n_samples']}, p={block[0]['n_features']}, "
              f"positive='{block[0]['positive_class']}' ({block[0]['positive_rate']:.2f})")
        print(f"  {'Model':32} {'Brier':>14} {'Accuracy':>14} {'F1':>14} {'mAP':>14}")
        for r in block:
            cells = [f"{r[f'{m}_mean']:.3f}+/-{np.nan_to_num(r[f'{m}_std']):.3f}" for m in METRICS]
            print(f"  {r['model']:32} " + " ".join(f"{c:>14}" for c in cells))
        tuned = [r for r in block if r["chosen_penalties"]]
        if tuned:
            print(f"  selected penalties: {tuned[0]['chosen_penalties']}")


# ----------------------------------------------------------------------------- main
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", nargs="+", default=None, choices=DATASETS)
    p.add_argument("--n-estimators", type=int, default=100)
    p.add_argument("--n-bootstraps", type=int, default=10)
    p.add_argument("--outer-folds", type=int, default=5)
    p.add_argument("--inner-folds", type=int, default=3)
    p.add_argument("--n-repeats", type=int, default=1, help="repeats of the outer CV")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--n-jobs", type=int, default=1, help="processes across outer folds")
    p.add_argument("--output-dir", type=Path, default=Path("outputs"))
    p.add_argument("--data-home", default=None, help="scikit-learn data cache directory")
    p.add_argument("--quick", action="store_true",
                   help="smoke test: first 2 datasets, 10 trees, 3 bootstraps, 3 outer / 2 inner folds")
    args = p.parse_args()
    if args.quick:
        args.n_estimators, args.n_bootstraps, args.outer_folds, args.inner_folds = 10, 3, 3, 2
    args.datasets = args.datasets or list(DATASETS[:2] if args.quick else DATASETS)
    return args


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data, info = {}, {}
    for name in args.datasets:
        try:
            data[name] = load_dataset(name, args.data_home)
            info[name] = data[name][2]
            print(f"loaded {name}: {info[name]}", flush=True)
        except ImportError as exc:
            sys.exit(f"{exc}\nInstall pandas first: uv pip install pandas")
        except Exception as exc:  # network problems, renamed datasets, ...
            print(f"[skip] {name}: {type(exc).__name__}: {exc}", flush=True)
    if not data:
        sys.exit("No dataset could be loaded (check the internet connection).")

    jobs = []
    for ds_index, (name, (X, y, _)) in enumerate(data.items()):
        splitter = RepeatedStratifiedKFold(n_splits=args.outer_folds, n_repeats=args.n_repeats,
                                           random_state=args.seed + ds_index)
        for i, (train, test) in enumerate(splitter.split(X, y)):
            jobs.append((ds_index, name, i // args.outer_folds, i % args.outer_folds, X, y, train, test))

    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    config.update(penalty_grid=list(PENALTIES), datasets_info=info,
                  versions=dict(python=sys.version.split()[0], platform=platform.platform(),
                                numpy=np.__version__, scikit_learn=sklearn.__version__))
    (args.output_dir / "openml_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    start = time.perf_counter()
    rows, grid_rows = [], []
    results = Parallel(n_jobs=args.n_jobs, return_as="generator")(
        delayed(run_fold)(*job, args) for job in jobs)
    for job, (fold_rows, fold_grid) in zip(jobs, results):
        rows.extend(fold_rows)
        grid_rows.extend(fold_grid)
        print(f"  done {job[1]} repeat={job[2]} fold={job[3]} "
              f"({time.perf_counter() - start:.0f}s elapsed)", flush=True)

    summary = summarize(rows, info)
    write_csv(args.output_dir / "openml_summary.csv", summary)
    write_csv(args.output_dir / "openml_grid_summary.csv", summarize_grid(grid_rows))
    write_csv(args.output_dir / "openml_raw.csv", rows)
    print_summary(summary)
    print(f"\nSaved openml_summary.csv, openml_grid_summary.csv, openml_raw.csv and "
          f"openml_config.json to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()