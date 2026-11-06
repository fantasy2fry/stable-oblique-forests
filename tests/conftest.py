import numpy as np
import pytest


@pytest.fixture
def classification_data():
    rng = np.random.default_rng(2026)
    X = rng.normal(size=(240, 9))
    y = (1.5 * X[:, 0] + X[:, 1] - 0.4 * X[:, 2] > 0).astype(int)
    return X, y


@pytest.fixture
def fast_settings():
    return dict(
        n_estimators=4,
        support_sizes=(1, 2, 4),
        n_directions_per_k=5,
        max_features=None,
        min_samples_split=18,
        min_samples_leaf=5,
        min_assessment_samples=5,
        max_depth=4,
        n_bootstraps=5,
        random_state=123,
    )
