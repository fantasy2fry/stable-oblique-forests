"""Named baseline/ablation configurations, all using the same candidate engine.

These factories return ordinary estimators, so GridSearchCV, clone, Pipeline,
and serialization work exactly as for direct class construction.
"""
from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from .forest import ObliqueRandomForestClassifier


def make_ablation_models(
    *, sparsity_penalty: float = 0.01,
    stability_penalty: float = 0.01,
    **forest_kwargs,
):
    """Create preconfigured ablation estimators without fitting any models.

    All custom variants differ only in their penalties or final selection rule.
    ``training_selection_control`` uses the same sparse projection generator but
    scores candidates on construction samples; it is NOT official SPORF.
    """
    if "sparsity_penalty" in forest_kwargs or "stability_penalty" in forest_kwargs:
        raise TypeError("Pass penalty values as named arguments to make_ablation_models.")
    base = dict(forest_kwargs)
    return {
        "unpenalized": ObliqueRandomForestClassifier(
            **base, sparsity_penalty=0.0, stability_penalty=0.0
        ),
        "sparsity_only": ObliqueRandomForestClassifier(
            **base, sparsity_penalty=sparsity_penalty, stability_penalty=0.0
        ),
        "stability_only": ObliqueRandomForestClassifier(
            **base, sparsity_penalty=0.0, stability_penalty=stability_penalty
        ),
        "sparsity_and_stability": ObliqueRandomForestClassifier(
            **base, sparsity_penalty=sparsity_penalty, stability_penalty=stability_penalty
        ),
        "uniform_stability_control": ObliqueRandomForestClassifier(
            **{**base, "stability_mode": "uniform"},
            sparsity_penalty=sparsity_penalty, stability_penalty=stability_penalty,
        ),
        "shuffled_stability_control": ObliqueRandomForestClassifier(
            **{**base, "stability_mode": "shuffled"},
            sparsity_penalty=sparsity_penalty, stability_penalty=stability_penalty,
        ),
        "training_selection_control": ObliqueRandomForestClassifier(
            **{**base, "selection_mode": "training"},
            sparsity_penalty=0.0, stability_penalty=0.0,
        ),
    }


def make_random_forest_reference(*, n_estimators=100, random_state=None, n_jobs=None):
    """Return the actual sklearn RandomForestClassifier, rather than a replica."""
    return RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=n_jobs,
    )
