"""Greedy sparse oblique binary classification trees with node-wise stability weights."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted, validate_data

from ._splits import (
    Candidate,
    assessment_loss,
    brier_constant,
    generate_direction_bank,
    optimize_threshold,
    smoothed_probability,
)
from ._validation import n_features_to_sample, validate_parameters, validated_binary_data
from ._random import make_rng


@dataclass(slots=True)
class _Node:
    node_id: int
    depth: int
    probability: float
    n_samples: int
    n_assessment: int
    direction: np.ndarray | None = None
    threshold: float | None = None
    left_id: int | None = None
    right_id: int | None = None
    training_gain: float = 0.0

    @property
    def is_leaf(self) -> bool:
        return self.direction is None


class ObliqueTreeClassifier(ClassifierMixin, BaseEstimator):
    """Binary oblique tree using sparse random projections and penalized Brier loss.

    Parameters
    ----------
    support_sizes : tuple of int, default=(1, 2, 4, 8)
        Candidate nonzero coefficient counts. Values above the sampled feature-pool
        size are skipped. 1-feature projections are always eligible if 1 is included.
    n_directions_per_k : int, default=8
        Number of directions sampled for each eligible support size at a node.
    sparsity_penalty, stability_penalty : float, default=0
        The lambda and gamma terms in the project proposal. When gamma is zero,
        no node-level stability bootstrap is performed.
    n_bootstraps : int, default=10
        Bootstrap repetitions within each candidate node if gamma is positive.
    max_features : {"sqrt", "log2", None} or int or float, default="sqrt"
        Number of features offered to a node. "sqrt" follows the research-plan
        rule max(8, ceil(sqrt(p))), capped at p. A float is a fraction of p.
    min_samples_split : int, default=60
        Minimum construction observations for attempting a split.
    min_samples_leaf : int, default=15
        Minimum construction observations in either child.
    min_assessment_samples : int, default=20
        Minimum held-out observations at a node when selection_mode="assessment".
    assessment_fraction : float, default=0.2
        Fraction of fit data reserved as assessment data before tree bootstrapping.
    max_depth : int or None, default=8
    bootstrap : bool, default=False
        Bootstrap construction rows for the tree. The forest sets this True by
        default. This is independent of node-level bootstrapping.
    selection_mode : {"assessment", "training"}, default="assessment"
        Whether final candidates are scored on a disjoint assessment pool or
        their construction data. "training" is an ablation, not a faithful SPORF.
    stability_mode : {"feature", "uniform", "shuffled"}, default="feature"
        "feature" uses bootstrap frequencies attached to individual features.
        "uniform" uses their mean for a node-wise but feature-agnostic penalty.
        "shuffled" permutes the weights within the node's feature pool as a
        negative control. All three pay the same bootstrap estimation cost.
    random_state : int, numpy.random.Generator, RandomState, or None

    Notes
    -----
    No-split is included in FINAL selection and excluded when estimating local
    feature-selection frequencies. Candidate directions are held fixed across
    bootstrap repetitions, but their thresholds are refitted each time.

    The same assessment pool is routed through descendant nodes. Thus, it is
    disjoint from construction data but not statistically independent of all
    ancestor decisions. No generalization guarantee is claimed.

    This prototype requires numeric, finite, dense input. Use a scikit-learn
    preprocessing Pipeline for mixed-type or missing-value inputs.
    """

    def __init__(
        self,
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
        bootstrap=False,
        selection_mode="assessment",
        stability_mode="feature",
        random_state=None,
    ):
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
        self.random_state = random_state

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.classifier_tags.multi_class = False
        tags.classifier_tags.multi_label = False
        return tags

    def fit(self, X, y):
        """Fit the tree, reserving a disjoint assessment pool from the input rows."""
        validate_parameters(self)
        if self.selection_mode not in ("assessment", "training"):
            raise ValueError("selection_mode must be 'assessment' or 'training'.")
        if self.stability_mode not in ("feature", "uniform", "shuffled"):
            raise ValueError("stability_mode must be 'feature', 'uniform', or 'shuffled'.")
        X, encoded, classes = validated_binary_data(self, X, y)
        rng = make_rng(self.random_state)
        indices = rng.permutation(len(encoded))
        n_assess = max(1, min(len(encoded) - 1, int(np.ceil(self.assessment_fraction * len(encoded)))))
        assessment_indices = indices[:n_assess]
        construction_indices = indices[n_assess:]
        return self._fit_encoded(X, encoded, classes, construction_indices, assessment_indices, rng)

    def _fit_encoded(self, X, y, classes, construction_indices, assessment_indices, rng):
        """Internal forest entry point. y is already encoded as {0,1}."""
        validate_parameters(self)
        if self.selection_mode not in ("assessment", "training"):
            raise ValueError("selection_mode must be 'assessment' or 'training'.")
        if self.stability_mode not in ("feature", "uniform", "shuffled"):
            raise ValueError("stability_mode must be 'feature', 'uniform', or 'shuffled'.")
        self.classes_ = np.asarray(classes)
        self.n_classes_ = len(classes)
        self.n_outputs_ = 1
        self.n_features_in_ = X.shape[1]
        self.nodes_: list[_Node] = []
        self.node_diagnostics_: list[dict] = []
        self.n_bootstrap_fits_ = 0
        self._raw_importances = np.zeros(X.shape[1], dtype=float)
        construct = np.asarray(construction_indices, dtype=np.int64)
        assess = np.asarray(assessment_indices, dtype=np.int64)
        if self.bootstrap:
            construct = rng.choice(construct, size=len(construct), replace=True)
        if len(construct) == 0:
            raise ValueError("The construction pool cannot be empty.")
        root_seed = int(rng.integers(0, np.iinfo(np.uint32).max))
        self._build(X, y, construct, assess, depth=0, node_seed=root_seed)
        self.node_count_ = len(self.nodes_)
        self.n_leaves_ = sum(node.is_leaf for node in self.nodes_)
        self.max_depth_ = max(node.depth for node in self.nodes_)
        total = self._raw_importances.sum()
        self.feature_importances_ = (
            self._raw_importances / total if total > 0 else self._raw_importances.copy()
        )
        self.feature_usage_ = np.zeros(X.shape[1], dtype=int)
        for node in self.nodes_:
            if node.direction is not None:
                self.feature_usage_[np.flatnonzero(node.direction)] += 1
        del self._raw_importances
        return self

    def _bootstrap_selection_frequencies(self, X, y, A, bank, rng):
        """Node-level bootstrap frequency of selection, conditional on the bank."""
        frequency = np.zeros(X.shape[1], dtype=float)
        n_valid = 0
        for _ in range(self.n_bootstraps):
            sampled = rng.choice(A, size=len(A), replace=True)
            winner = None
            winner_score = float("inf")
            for w in bank:
                candidate = optimize_threshold(X, y, sampled, w, self.min_samples_leaf)
                if candidate is None:
                    continue
                score = candidate.training_loss + self.sparsity_penalty * candidate.sparsity
                if score < winner_score:
                    winner_score = score
                    winner = candidate
            self.n_bootstrap_fits_ += 1
            if winner is not None:
                frequency[winner.support] += 1
                n_valid += 1
        if n_valid:
            frequency /= n_valid
        return frequency, n_valid

    def _build(self, X, y, A, H, depth, node_seed):
        # Node-local RNG isolates descendant directions from extra bootstrap
        # draws in a parent. This matters for paired ablation experiments.
        rng = np.random.default_rng(node_seed)
        left_seed, right_seed = rng.integers(
            0, np.iinfo(np.uint32).max, size=2
        )
        node_id = len(self.nodes_)
        parent_probability = smoothed_probability(y[A])
        node = _Node(
            node_id=node_id,
            depth=depth,
            probability=parent_probability,
            n_samples=len(A),
            n_assessment=len(H),
        )
        self.nodes_.append(node)
        if (
            (self.max_depth is not None and depth >= self.max_depth)
            or len(A) < max(self.min_samples_split, 2 * self.min_samples_leaf)
            or (self.selection_mode == "assessment" and len(H) < self.min_assessment_samples)
            or np.all(y[A] == y[A[0]])
        ):
            return node_id

        pool_size = n_features_to_sample(self.max_features, X.shape[1])
        feature_pool = rng.choice(X.shape[1], size=pool_size, replace=False)
        k_choices = tuple(sorted({int(k) for k in self.support_sizes if k <= pool_size}))
        bank, exposure = generate_direction_bank(
            feature_pool, k_choices, self.n_directions_per_k, rng, X.shape[1]
        )
        fitted = [optimize_threshold(X, y, A, w, self.min_samples_leaf) for w in bank]
        fitted = [c for c in fitted if c is not None]
        if not fitted:
            return node_id

        frequencies = np.zeros(X.shape[1], dtype=float)
        n_valid_bootstraps = 0
        if self.stability_penalty > 0:
            frequencies, n_valid_bootstraps = self._bootstrap_selection_frequencies(
                X, y, A, bank, rng
            )
            # Without any valid bootstrap selections we cannot estimate stability:
            # use ordinary L0 instead of systematically penalizing every feature.
            if n_valid_bootstraps == 0:
                frequencies[:] = 1.0

        effective_frequencies = frequencies.copy()
        if self.stability_penalty > 0:
            if self.stability_mode == "uniform":
                effective_frequencies[feature_pool] = np.mean(frequencies[feature_pool])
            elif self.stability_mode == "shuffled":
                effective_frequencies[feature_pool] = rng.permutation(frequencies[feature_pool])

        if self.selection_mode == "assessment":
            base_loss = brier_constant(y[H], parent_probability)
        else:
            base_loss = brier_constant(y[A], parent_probability)
        best = None
        best_score = base_loss
        for c in fitted:
            loss = (
                assessment_loss(c, X, y, H)
                if self.selection_mode == "assessment" else c.training_loss
            )
            instability = (
                float(np.sum(1.0 - effective_frequencies[c.support]))
                if self.stability_penalty > 0 else 0.0
            )
            score = (
                loss + self.sparsity_penalty * c.sparsity
                + self.stability_penalty * instability
            )
            if score < best_score:
                best_score = score
                best = c
        if best is None:
            return node_id

        # Direction <= t goes to the left, > t to the right.
        construction_right = X[A] @ best.direction > best.threshold
        assessment_right = X[H] @ best.direction > best.threshold
        left_A, right_A = A[~construction_right], A[construction_right]
        left_H, right_H = H[~assessment_right], H[assessment_right]
        node.direction = best.direction.copy()
        node.threshold = best.threshold
        node.training_gain = max(
            0.0, brier_constant(y[A], parent_probability) - best.training_loss
        )
        self._raw_importances[best.support] += (
            len(A) * node.training_gain / best.sparsity
        )
        self.node_diagnostics_.append(
            {
                "node_id": node_id,
                "depth": depth,
                "n_construction": len(A),
                "n_assessment": len(H),
                "support": best.support.copy(),
                "support_size": best.sparsity,
                "threshold": best.threshold,
                "selected_score": best_score,
                "no_split_score": base_loss,
                "n_valid_bootstraps": n_valid_bootstraps,
                "feature_frequencies": {
                    int(j): float(frequencies[j]) for j in feature_pool
                } if self.stability_penalty > 0 else {},
                "feature_exposure": {
                    int(j): int(exposure[j]) for j in feature_pool
                },
                "stability_mode": self.stability_mode,
            }
        )
        node.left_id = self._build(
            X, y, left_A, left_H, depth + 1, int(left_seed)
        )
        node.right_id = self._build(
            X, y, right_A, right_H, depth + 1, int(right_seed)
        )
        return node_id

    def _validated_X(self, X):
        check_is_fitted(self, "nodes_")
        return validate_data(self, X, reset=False, dtype=np.float64)

    def _traverse(self, row):
        path = [0]
        current = self.nodes_[0]
        while not current.is_leaf:
            next_id = (
                current.right_id if row @ current.direction > current.threshold
                else current.left_id
            )
            current = self.nodes_[next_id]
            path.append(current.node_id)
        return current, path

    def predict_proba(self, X):
        """Return class probabilities with columns ordered by classes_."""
        X = self._validated_X(X)
        if self.n_classes_ == 1:
            return np.ones((len(X), 1), dtype=float)
        positive = np.fromiter(
            (self._traverse(row)[0].probability for row in X),
            dtype=float, count=len(X)
        )
        return np.column_stack([1.0 - positive, positive])

    def predict(self, X):
        check_is_fitted(self, "nodes_")
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def predict_log_proba(self, X):
        return np.log(self.predict_proba(X))

    def apply(self, X):
        """Return the leaf ID reached by each observation."""
        X = self._validated_X(X)
        return np.fromiter(
            (self._traverse(row)[0].node_id for row in X), dtype=np.int64, count=len(X)
        )

    def decision_path(self, X):
        """Return an (n_samples, node_count_) CSR matrix of visited nodes."""
        X = self._validated_X(X)
        indices, indptr = [], [0]
        for row in X:
            _, path = self._traverse(row)
            indices.extend(path)
            indptr.append(len(indices))
        return csr_matrix(
            (np.ones(len(indices), dtype=np.int8), np.asarray(indices), np.asarray(indptr)),
            shape=(len(X), self.node_count_),
        )

    def get_depth(self):
        check_is_fitted(self, "nodes_")
        return self.max_depth_

    def get_n_leaves(self):
        check_is_fitted(self, "nodes_")
        return self.n_leaves_
