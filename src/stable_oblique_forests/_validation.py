"""Parameter validation and common input checks."""
from __future__ import annotations

from numbers import Integral, Real
import numpy as np
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import validate_data


def validated_binary_data(estimator, X, y):
    X, y = validate_data(estimator, X, y, dtype=np.float64, ensure_min_samples=2)
    check_classification_targets(y)
    classes, y_enc = np.unique(y, return_inverse=True)
    if len(classes) > 2:
        raise ValueError("Only binary classification is supported; y must have at most two classes.")
    return X, y_enc.astype(np.int8), classes


def validate_parameters(model):
    for key in ("n_directions_per_k", "min_samples_split", "min_samples_leaf",
                "min_assessment_samples", "n_bootstraps"):
        val = getattr(model, key)
        if not isinstance(val, Integral) or isinstance(val, bool) or val < 1:
            raise ValueError(f"{key} must be a positive integer; got {val!r}.")
    if model.min_samples_split < 2:
        raise ValueError("min_samples_split must be at least 2.")
    if model.max_depth is not None and (
        not isinstance(model.max_depth, Integral) or model.max_depth < 1
    ):
        raise ValueError("max_depth must be None or a positive integer.")
    if not isinstance(model.support_sizes, (tuple, list)) or not model.support_sizes:
        raise ValueError("support_sizes must be a nonempty tuple/list of positive integers.")
    for k in model.support_sizes:
        if not isinstance(k, Integral) or isinstance(k, bool) or k < 1:
            raise ValueError("support_sizes must contain only positive integers.")
    for key in ("sparsity_penalty", "stability_penalty"):
        val = getattr(model, key)
        if not isinstance(val, Real) or not np.isfinite(val) or val < 0:
            raise ValueError(f"{key} must be finite and nonnegative.")
    if not isinstance(model.assessment_fraction, Real) or not 0 < model.assessment_fraction < 1:
        raise ValueError("assessment_fraction must be strictly between 0 and 1.")
    max_features = model.max_features
    if max_features not in ("sqrt", "log2", None):
        if isinstance(max_features, bool) or not isinstance(max_features, Real):
            raise ValueError("max_features must be 'sqrt', 'log2', None, int or float.")
        if isinstance(max_features, Integral):
            if max_features < 1:
                raise ValueError("Integer max_features must be >= 1.")
        elif not 0 < max_features <= 1:
            raise ValueError("Float max_features must lie in (0, 1].")


def n_features_to_sample(max_features, p):
    if max_features is None:
        return p
    if max_features == "sqrt":
        return min(p, max(8, int(np.ceil(np.sqrt(p)))))
    if max_features == "log2":
        return min(p, max(1, int(np.ceil(np.log2(p)))))
    if isinstance(max_features, Integral):
        return min(p, int(max_features))
    return min(p, max(1, int(np.ceil(float(max_features) * p))))
