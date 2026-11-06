"""Sparse-projection candidate generation and one-dimensional split optimization.

Implementation detail: no dependence on the forest and no global RNG state.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True, slots=True)
class Candidate:
    direction: np.ndarray
    threshold: float
    left_probability: float
    right_probability: float
    training_loss: float
    n_left: int
    n_right: int

    @property
    def support(self) -> np.ndarray:
        return np.flatnonzero(self.direction)

    @property
    def sparsity(self) -> int:
        return int(np.count_nonzero(self.direction))


def smoothed_probability(y: np.ndarray) -> float:
    """Binary class-one probability with Beta(1,1) smoothing."""
    return float((np.sum(y) + 1.0) / (len(y) + 2.0))


def brier_constant(y: np.ndarray, probability: float) -> float:
    return float(np.mean((y - probability) ** 2))


def generate_direction_bank(
    pool: np.ndarray,
    support_sizes: tuple[int, ...],
    candidates_per_support: int,
    rng: np.random.Generator,
    n_features: int,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Build a reusable candidate bank with balanced feature exposure.

    Each selected support has k elements with independent Rademacher signs / sqrt(k).
    For fixed k, features with the fewest prior appearances are sampled first.
    """
    pool = np.asarray(pool, dtype=int)
    exposure = np.zeros(n_features, dtype=np.int64)
    directions: list[np.ndarray] = []
    for k in support_sizes:
        if k > len(pool):
            continue
        for _ in range(candidates_per_support):
            # Random tie-breaking makes the bank seed-dependent but reproducible.
            shuffled = rng.permutation(pool)
            features = shuffled[np.argsort(exposure[shuffled], kind="stable")[:k]]
            signs = rng.choice(np.array([-1.0, 1.0]), size=k)
            w = np.zeros(n_features, dtype=np.float64)
            w[features] = signs / np.sqrt(k)
            exposure[features] += 1
            directions.append(w)
    return directions, exposure


def optimize_threshold(
    X: np.ndarray,
    y: np.ndarray,
    indices: np.ndarray,
    direction: np.ndarray,
    min_samples_leaf: int,
) -> Candidate | None:
    """Choose the threshold minimizing construction-sample Brier loss.

    Candidate predictions on either side use Laplace-smoothed class proportions.
    Every eligible threshold is between two distinct projected values. Repeated
    bootstrap indices are intentionally counted with multiplicity.
    """
    n = len(indices)
    if n < 2 * min_samples_leaf:
        return None
    z = X[indices] @ direction
    yy = y[indices].astype(np.float64, copy=False)
    order = np.argsort(z, kind="mergesort")
    zs = z[order]
    ys = yy[order]
    cuts = np.arange(min_samples_leaf, n - min_samples_leaf + 1, dtype=np.int64)
    if len(cuts) == 0:
        return None
    valid = zs[cuts - 1] < zs[cuts]
    cuts = cuts[valid]
    if len(cuts) == 0:
        return None

    positives = np.cumsum(ys)
    left_n = cuts.astype(np.float64)
    right_n = n - left_n
    left_pos = positives[cuts - 1]
    right_pos = positives[-1] - left_pos
    p_left = (left_pos + 1.0) / (left_n + 2.0)
    p_right = (right_pos + 1.0) / (right_n + 2.0)
    left_neg = left_n - left_pos
    right_neg = right_n - right_pos
    squared_error = (
        left_pos * (1.0 - p_left) ** 2 + left_neg * p_left**2
        + right_pos * (1.0 - p_right) ** 2 + right_neg * p_right**2
    ) / n
    best = int(np.argmin(squared_error))
    cut = int(cuts[best])
    # This midpoint is robust to overflow in the sum of two large endpoints.
    threshold = float(zs[cut - 1] + 0.5 * (zs[cut] - zs[cut - 1]))
    return Candidate(
        direction=direction,
        threshold=threshold,
        left_probability=float(p_left[best]),
        right_probability=float(p_right[best]),
        training_loss=float(squared_error[best]),
        n_left=cut,
        n_right=n - cut,
    )


def assessment_loss(
    candidate: Candidate, X: np.ndarray, y: np.ndarray, indices: np.ndarray
) -> float:
    if len(indices) == 0:
        return float("inf")
    right = X[indices] @ candidate.direction > candidate.threshold
    predictions = np.where(right, candidate.right_probability, candidate.left_probability)
    return float(np.mean((y[indices] - predictions) ** 2))
