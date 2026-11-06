"""Scikit-learn-compatible ensembles of sparse oblique classification trees."""
from __future__ import annotations

from numbers import Integral
import numpy as np
from joblib import Parallel, delayed
from scipy.sparse import hstack
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted, validate_data

from .tree import ObliqueTreeClassifier
from ._validation import validate_parameters, validated_binary_data
from ._random import make_rng


def _train_one_tree(params, X, y, classes, seed):
    rng = np.random.default_rng(int(seed))
    permutation = rng.permutation(len(y))
    n_assess = max(1, min(len(y) - 1, int(np.ceil(params["assessment_fraction"] * len(y)))))
    assess = permutation[:n_assess]
    construct = permutation[n_assess:]
    tree = ObliqueTreeClassifier(**params, random_state=int(seed))
    tree._fit_encoded(X, y, classes, construct, assess, rng)
    return tree


class ObliqueRandomForestClassifier(ClassifierMixin, BaseEstimator):
    """Averaged sparse oblique binary classification trees.

    Every tree reserves a disjoint assessment pool *before* tree-level
    bootstrap. Each node searches a reusable bank of sparse random directions.
    Stability-aware selection adds a node-local bootstrap frequency penalty.

    Set ``sparsity_penalty=0, stability_penalty=0`` for the unpenalized ablation,
    ``stability_penalty=0`` for ordinary L0, ``sparsity_penalty=0`` for
    stability-only, or both positive for the proposed method.

    Parameters not documented below have the same semantics as
    :class:`ObliqueTreeClassifier`.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of trees in the ensemble.
    n_jobs : int or None, default=None
        Joblib thread parallelism across trees. Independent seeds ensure results
        are reproducible for fixed random_state irrespective of n_jobs.
    """

    def __init__(
        self,
        n_estimators=100,
        support_sizes=(1, 2, 4, 8),
        n_directions_per_k=8,
        sparsity_penalty=0.0,
        stability_penalty=0.0,
        n_bootstraps=10,
        max_features="sqrt",
        min_samples_split=60,
        min_samples_leaf=15,
        min_assessment_samples=20,
        assessment_fraction=0.2,
        max_depth=8,
        bootstrap=True,
        selection_mode="assessment",
        stability_mode="feature",
        n_jobs=None,
        random_state=None,
    ):
        self.n_estimators = n_estimators
        self.support_sizes = support_sizes
        self.n_directions_per_k = n_directions_per_k
        self.sparsity_penalty = sparsity_penalty
        self.stability_penalty = stability_penalty
        self.n_bootstraps = n_bootstraps
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_assessment_samples = min_assessment_samples
        self.assessment_fraction = assessment_fraction
        self.max_depth = max_depth
        self.bootstrap = bootstrap
        self.selection_mode = selection_mode
        self.stability_mode = stability_mode
        self.n_jobs = n_jobs
        self.random_state = random_state

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.classifier_tags.multi_class = False
        tags.classifier_tags.multi_label = False
        return tags

    def fit(self, X, y):
        """Fit all trees using independent bootstrap and assessment partitions."""
        validate_parameters(self)
        if self.selection_mode not in ("assessment", "training"):
            raise ValueError("selection_mode must be 'assessment' or 'training'.")
        if self.stability_mode not in ("feature", "uniform", "shuffled"):
            raise ValueError("stability_mode must be 'feature', 'uniform', or 'shuffled'.")
        if not isinstance(self.n_estimators, Integral) or self.n_estimators < 1:
            raise ValueError("n_estimators must be a positive integer.")
        if self.n_jobs is not None and (
            not isinstance(self.n_jobs, Integral) or self.n_jobs == 0
        ):
            raise ValueError("n_jobs must be None or a nonzero integer.")
        X, y_encoded, classes = validated_binary_data(self, X, y)
        self.classes_ = classes
        self.n_classes_ = len(classes)
        self.n_outputs_ = 1
        params = {
            name: getattr(self, name) for name in (
                "support_sizes", "n_directions_per_k", "sparsity_penalty",
                "stability_penalty", "n_bootstraps", "max_features",
                "min_samples_split", "min_samples_leaf", "min_assessment_samples",
                "assessment_fraction", "max_depth", "bootstrap", "selection_mode",
                "stability_mode",
            )
        }
        rng = make_rng(self.random_state)
        seeds = rng.integers(0, np.iinfo(np.uint32).max, size=self.n_estimators)
        if self.n_jobs in (None, 1):
            self.estimators_ = [
                _train_one_tree(params, X, y_encoded, classes, seed) for seed in seeds
            ]
        else:
            self.estimators_ = Parallel(n_jobs=self.n_jobs, prefer="threads")(
                delayed(_train_one_tree)(params, X, y_encoded, classes, seed)
                for seed in seeds
            )
        self.feature_importances_ = np.mean(
            [tree.feature_importances_ for tree in self.estimators_], axis=0
        )
        self.feature_usage_ = np.sum(
            [tree.feature_usage_ for tree in self.estimators_], axis=0
        )
        self.n_bootstrap_fits_ = sum(tree.n_bootstrap_fits_ for tree in self.estimators_)
        self.node_count_ = sum(tree.node_count_ for tree in self.estimators_)
        return self

    def _validated_X(self, X):
        check_is_fitted(self, "estimators_")
        return validate_data(self, X, reset=False, dtype=np.float64)

    def predict_proba(self, X):
        """Mean predicted class probabilities across trees."""
        X = self._validated_X(X)
        probabilities = np.zeros((len(X), self.n_classes_), dtype=np.float64)
        for tree in self.estimators_:
            probabilities += tree.predict_proba(X)
        probabilities /= len(self.estimators_)
        return probabilities

    def predict(self, X):
        check_is_fitted(self, "estimators_")
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def predict_log_proba(self, X):
        return np.log(self.predict_proba(X))

    def apply(self, X):
        """Return shape (n_samples, n_estimators) with one leaf ID per tree."""
        X = self._validated_X(X)
        return np.column_stack([tree.apply(X) for tree in self.estimators_])

    def decision_path(self, X):
        """Return (sparse node-indicator matrix, cumulative tree node offsets)."""
        X = self._validated_X(X)
        matrices = [tree.decision_path(X) for tree in self.estimators_]
        offsets = np.cumsum([0] + [tree.node_count_ for tree in self.estimators_])
        return hstack(matrices, format="csr"), offsets
