import numpy as np
import pytest
from sklearn.metrics import average_precision_score, f1_score

from stable_oblique_forests import ObliqueRandomForestClassifier, binary_classification_metrics


def test_metrics_match_sklearn_on_known_values():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.6, 0.4, 0.9])
    pred = (p > 0.5).astype(int)
    out = binary_classification_metrics(y, pred, p)
    assert set(out) == {"accuracy", "balanced_accuracy", "brier", "f1", "average_precision"}
    assert out["f1"] == pytest.approx(f1_score(y, pred))
    assert out["average_precision"] == pytest.approx(average_precision_score(y, p))
    assert out["accuracy"] == pytest.approx(0.5)
    assert out["brier"] == pytest.approx(np.mean((y - p) ** 2))


def test_metrics_with_string_labels(classification_data, fast_settings):
    X, y = classification_data
    y = np.where(y == 0, "negative", "positive")
    model = ObliqueRandomForestClassifier(**fast_settings).fit(X, y)
    pos = model.classes_[1]
    p1 = model.predict_proba(X)[:, 1]
    out = binary_classification_metrics(y, model.predict(X), p1, pos_label=pos)
    assert all(np.isfinite(v) for v in out.values())
    assert 0 <= out["f1"] <= 1 and 0 <= out["average_precision"] <= 1
