"""Evaluation helpers shared by examples and future experiment scripts."""
from __future__ import annotations

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
)


def binary_classification_metrics(y_true, y_pred, p_positive, *, pos_label=1) -> dict[str, float]:
    """Return the evaluation metrics logged for every binary model.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True labels.
    y_pred : array-like of shape (n_samples,)
        Hard predictions from ``predict`` (argmax of the probabilities).
    p_positive : array-like of shape (n_samples,)
        Predicted probability of the positive class, i.e. column
        ``list(model.classes_).index(pos_label)`` of ``predict_proba``.
    pos_label : label, default=1
        The positive class. For this package it is ``classes_[1]``.

    Returns
    -------
    dict with keys ``accuracy``, ``balanced_accuracy``, ``brier`` (lower is better),
    ``f1`` and ``average_precision``.

    Notes
    -----
    ``f1`` uses the hard predictions. ``average_precision`` (the binary mAP / PR-AUC
    summary) uses the probabilities and is scikit-learn's step-wise estimate, not a
    trapezoidal area, so it can differ slightly from ``auc(recall, precision)``.
    ``f1`` is 0 when no positive is predicted (``zero_division=0``).
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "brier": float(brier_score_loss(y_true, p_positive, pos_label=pos_label)),
        "f1": float(f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0)),
        "average_precision": float(
            average_precision_score(y_true, p_positive, pos_label=pos_label)
        ),
    }
