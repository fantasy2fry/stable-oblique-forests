# Stable Oblique Forests

Research implementation for **Local Feature-Selection Stability in Oblique Random Forests**.

This repository contains an implementation of sparse oblique decision trees and their ensembles. At each internal node, a tree chooses a split from a bank of sparse random projections. The choice can be regularized by the number of active features and by how often each feature is selected in repeated bootstrap fits **at that node**.

The code is designed as a Python package with scikit-learn-compatible estimators. It implements the proposed classifiers and controlled variants, together with experiment scripts for the synthetic correlation study and the OpenML benchmark (see [`docs/reproduce.md`](docs/reproduce.md)).

## Installation

Requires Python 3.10 or newer.

The recommended workflow uses [uv](https://docs.astral.sh/uv/). To install the package together with the development and experiment dependencies:

```bash
uv sync --extra dev --extra experiments
```

Alternatively, install into an existing environment with:

```bash
uv pip install -e ".[dev,experiments]"
```

A plain `pip` installation of the core package also works:

```bash
python -m pip install -e .
```

To run the tests (currently `31 passed`):

```bash
uv run python -m pytest -q
# or, without uv:
python -m pytest -q
```

The package has no compiled extensions. It uses NumPy and SciPy for tree construction and scikit-learn's estimator interface for model selection and pipelines. The `experiments` extra adds pandas, matplotlib and seaborn, which are needed only by the scripts in `experiments/` and `examples/`.

## Quick start

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss
from stable_oblique_forests import ObliqueRandomForestClassifier

X, y = make_classification(n_samples=500, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

model = ObliqueRandomForestClassifier(
    n_estimators=100,
    sparsity_penalty=0.01,
    stability_penalty=0.01,
    n_bootstraps=10,
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train, y_train)

p1 = model.predict_proba(X_test)[:, 1]
print(brier_score_loss(y_test, p1))
print(model.predict(X_test[:5]))
```

This snippet illustrates the API. It is not a reported experiment or a claim about predictive performance.

## Main objects

| Object | Purpose |
|---|---|
| `ObliqueTreeClassifier` | Single oblique tree with optional sparsity and local-stability penalties. |
| `ObliqueRandomForestClassifier` | Ensemble that averages class probabilities from independently randomized trees. |
| `make_ablation_models()` | Named model configurations with shared implementation and parameter defaults. |
| `make_random_forest_reference()` | Actual `sklearn.ensemble.RandomForestClassifier` baseline. |

### Named ablations

```python
from stable_oblique_forests import make_ablation_models

models = make_ablation_models(
    n_estimators=100,
    sparsity_penalty=0.01,
    stability_penalty=0.01,
    random_state=42,
)
print(list(models))
```

- `unpenalized`: neither penalty.
- `sparsity_only`: ordinary L0 penalty; no node-level bootstrap work.
- `stability_only`: node-local feature-selection stability penalty alone.
- `sparsity_and_stability`: the proposed combined criterion.
- `uniform_stability_control`: same bootstrap estimates and overall node-specific penalty, but a common stability weight for all available features.
- `shuffled_stability_control`: stability frequencies randomly reassigned among candidate features within a node.
- `training_selection_control`: sparse directions chosen using construction-sample Brier loss rather than held-out Brier loss. This is an ablation, **not** the published SPORF algorithm.

All custom variants share the same tree code and candidate-generation procedure. `make_ablation_models()` does not fit models or run experiments.

## Split construction

At node `v`, choose a finite bank of random sparse directions with support sizes `support_sizes` (default `(1, 2, 4, 8)`). Coefficients are independently sampled as `+1/sqrt(k)` or `-1/sqrt(k)` on `k` features. Thresholds minimize smoothed construction-sample Brier loss subject to a minimum child size.

The project investigates the split score

$$
S_v(c)=\widehat L_{v,H}(c)+\lambda\|w_c\|_0
+\gamma\sum_{j\in\mathrm{supp}(w_c)}(1-\widehat\pi_{v,j}),
$$

where the frequency estimate is

$$
\widehat\pi_{v,j}=\frac{1}{B}\sum_{b=1}^B
\mathbb{1}\{j\in\mathrm{supp}(w_v^{(b)})\}.
$$

To estimate these frequencies, the tree reuses the **same direction bank** on `B` bootstrap resamples of the node's construction observations, refits thresholds, and selects the candidate with the lowest construction Brier loss plus the ordinary L0 penalty. `no-split` is excluded from frequency estimation but included in the **final** choice. This prevents the meaning of frequency from being conflated with how often the node should terminate.

A separate assessment pool is reserved before each tree is built. The final candidate is selected by assessment Brier loss plus any penalties, and a node stops if no split beats predicting the parent's smoothed class probability. Each nonterminal node routes both construction and assessment observations to its children.

If no bootstrap repetition produces a valid split, the method falls back to the unweighted L0 penalty at that node. The number of successful repetitions is recorded in `node_diagnostics_`.

**Important independence qualification:** although construction and assessment observations are disjoint, descendant assessments reuse labels that may have influenced the selection of ancestor splits. The implementation does not claim a whole-tree generalization bound.

### scikit-learn interface

Both tree and forest provide:

- `fit(X, y)` returning `self`, `predict(X)`, `predict_proba(X)` and `predict_log_proba(X)`;
- `score(X, y)` (accuracy), `get_params()` and `set_params()`;
- `apply(X)` and `decision_path(X)`;
- `classes_`, `n_features_in_`, `feature_importances_`, and `feature_usage_` after fitting.

The forest exposes the fitted `estimators_`. Trees expose `nodes_`, `node_count_`, `node_diagnostics_`, `n_leaves_`, and `max_depth_`. `feature_usage_` counts how often features occur in accepted splits. `feature_importances_` is an **exploratory heuristic** from construction-sample Brier gain distributed across split features, not a permutation-importance estimate or an unbiased inference about relevance.

The estimators can be cloned, embedded in `Pipeline`, tuned by `GridSearchCV`, serialized with pickle/joblib, and trained with independent seeds across `n_jobs` values. Both implementations pass scikit-learn's built-in `check_estimator` suite in the development environment (scikit-learn 1.8.0).

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from stable_oblique_forests import ObliqueRandomForestClassifier

pipe = make_pipeline(StandardScaler(), ObliqueRandomForestClassifier(random_state=42))
search = GridSearchCV(
    pipe,
    {
        "obliquerandomforestclassifier__sparsity_penalty": [0, 0.01],
        "obliquerandomforestclassifier__stability_penalty": [0, 0.01],
    },
    cv=3,
    scoring="neg_brier_score",
)
search.fit(X_train, y_train)
```

## Current scope and implementation choices

- **Binary classification only** (training on a single observed class produces a constant classifier). No regression, multiclass or multilabel implementation.
- The raw estimator requires a dense, finite, numeric matrix. For missing and categorical data, impute/encode with a scikit-learn Pipeline. If one-hot encoding is used, set `sparse_output=False`.
- Fitting uses full per-node candidate searches in NumPy. This is a correctness-first research implementation, not a compiled or optimized alternative to industrial-grade forests. Large candidate banks and many bootstrap repetitions can be expensive.
- `n_jobs` parallelizes trees with joblib threads. It may not speed up small datasets because significant code executes in Python.
- The default `max_features="sqrt"` follows the plan's `min(p, max(8, ceil(sqrt(p))))` rule, which differs from scikit-learn's standard `sqrt` convention. Adjust it for high-dimensional experiments.
- No fitted estimators support incremental `partial_fit`, `warm_start`, `sample_weight`, or OOB scoring at present.
- Parameters `support_sizes`, `n_directions_per_k`, `min_samples_split`, `min_samples_leaf`, `min_assessment_samples`, `assessment_fraction`, `max_depth`, and both penalties should be treated as explicit parts of the experimental design.
- The method uses Laplace-smoothing `(positives+1)/(total+2)` for child and parent probability estimates.

### Published SPORF and Super Greedy Forests

The repository does **not** reimplement SPORF or Super Greedy Forests faithfully. Those published methods have different optimization and splitting details, and a claim of faithful reproduction would be misleading. For comparisons against the literature, use their authors' released implementations or document a separately validated adapter. The `training_selection_control` is only a simple construction-sample sparse projection reference using **Brier loss**.

- SPORF: https://jmlr.org/papers/v21/18-664.html
- Official code: https://github.com/neurodata/SPORF
- Super Greedy Trees: https://doi.org/10.1007/s10462-026-11541-6

## Repository layout

```text
src/stable_oblique_forests/
  _random.py       Local random-number generator compatibility
  _splits.py       Sparse candidate generation and optimized thresholds
  _validation.py   Parameter and input checks
  tree.py          Single tree, local bootstrap, diagnostics, prediction
  forest.py        Ensemble training and predictions
  models.py        Baseline and ablation configurations
tests/             Unit, integration, compatibility and reproducibility tests
experiments/       Experiment scripts (exp_correlation.py, exp_openml.py, generate_plots.py)
examples/          API usage and visual demo examples (e.g. visual_demo.py)
docs/              Method mapping, implementation notes and reproduction guide
```

## Experiments

The scripts in `experiments/` cover the synthetic correlation experiment (`exp_correlation.py`), the OpenML benchmark (`exp_openml.py`), and plot generation (`generate_plots.py`). They require the `experiments` extra (see [Installation](#installation)). Exact commands are listed in [`docs/reproduce.md`](docs/reproduce.md).