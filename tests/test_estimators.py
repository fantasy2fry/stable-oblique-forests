import pickle
import numpy as np
import pytest
from scipy.sparse import csr_matrix
from sklearn.base import clone
from sklearn.datasets import make_classification
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from stable_oblique_forests import (
    ObliqueTreeClassifier, ObliqueRandomForestClassifier,
    make_ablation_models, make_random_forest_reference,
)


def forest(fast_settings, **changes):
    args = dict(fast_settings)
    args.update(changes)
    return ObliqueRandomForestClassifier(**args)


def test_binary_probabilities_and_classes(classification_data, fast_settings):
    X, y = classification_data
    y = np.where(y == 0, "negative", "positive")
    model = forest(fast_settings, stability_penalty=0.01, sparsity_penalty=0.01).fit(X, y)
    proba = model.predict_proba(X[:12])
    assert proba.shape == (12, 2)
    assert np.allclose(proba.sum(axis=1), 1)
    assert np.all((proba > 0) & (proba < 1))
    assert set(model.predict(X[:12])).issubset(set(y))
    assert model.classes_.tolist() == ["negative", "positive"]
    assert np.allclose(np.exp(model.predict_log_proba(X[:12])), proba)
    assert 0 <= model.score(X, y) <= 1


def test_tree_fits_and_exposes_structure(classification_data):
    X, y = classification_data
    tree = ObliqueTreeClassifier(
        support_sizes=(1, 2), n_directions_per_k=10,
        min_samples_split=20, min_samples_leaf=5,
        min_assessment_samples=5, max_depth=3, random_state=13,
    ).fit(X, y)
    assert tree.n_leaves_ >= 1
    assert tree.node_count_ == 2 * tree.n_leaves_ - 1
    assert tree.get_depth() <= 3
    assert tree.get_n_leaves() == tree.n_leaves_
    leaves = tree.apply(X[:10])
    path = tree.decision_path(X[:10])
    assert path.shape == (10, tree.node_count_)
    assert isinstance(path, csr_matrix)
    assert np.all(path[np.arange(10), leaves].A1 == 1)
    assert np.allclose(tree.predict_proba(X[:10]).sum(axis=1), 1)


def test_forest_apply_decision_path(classification_data, fast_settings):
    X, y = classification_data
    model = forest(fast_settings).fit(X, y)
    leaves = model.apply(X[:7])
    path, offsets = model.decision_path(X[:7])
    assert leaves.shape == (7, model.n_estimators)
    assert path.shape == (7, model.node_count_)
    assert offsets.shape == (model.n_estimators + 1,)
    for j in range(model.n_estimators):
        assert np.all(path[np.arange(7), offsets[j] + leaves[:, j]].A1 == 1)


def test_gamma_zero_skips_bootstrap(classification_data, fast_settings):
    X, y = classification_data
    a = forest(fast_settings, stability_penalty=0.0).fit(X, y)
    b = forest(fast_settings, stability_penalty=0.01).fit(X, y)
    assert a.n_bootstrap_fits_ == 0
    assert b.n_bootstrap_fits_ > 0
    assert any(n["n_valid_bootstraps"] > 0 for t in b.estimators_ for n in t.node_diagnostics_)


def test_gamma_zero_is_independent_of_n_bootstraps(classification_data, fast_settings):
    X, y = classification_data
    a = forest(fast_settings, n_bootstraps=2).fit(X, y)
    b = forest(fast_settings, n_bootstraps=25).fit(X, y)
    assert np.array_equal(a.predict_proba(X), b.predict_proba(X))


def test_seed_and_parallel_determinism(classification_data, fast_settings):
    X, y = classification_data
    args = dict(stability_penalty=0.01, sparsity_penalty=0.01)
    a = forest(fast_settings, n_jobs=1, **args).fit(X, y)
    b = forest(fast_settings, n_jobs=2, **args).fit(X, y)
    c = forest(fast_settings, n_jobs=1, **args).fit(X, y)
    assert np.array_equal(a.predict_proba(X), b.predict_proba(X))
    assert np.array_equal(a.predict_proba(X), c.predict_proba(X))


def test_clone_and_set_params(classification_data, fast_settings):
    X, y = classification_data
    orig = forest(fast_settings)
    fresh = clone(orig)
    assert fresh.get_params() == orig.get_params()
    fresh.set_params(sparsity_penalty=0.003, n_estimators=3)
    assert fresh.n_estimators == 3
    assert fresh.sparsity_penalty == 0.003
    fresh.fit(X, y)
    check_is_fitted(fresh)


def test_pipeline_and_gridsearch(classification_data, fast_settings):
    X, y = classification_data
    args = dict(fast_settings)
    args.update(n_estimators=2, n_directions_per_k=3, max_depth=2)
    pipe = Pipeline([("scale", StandardScaler()), ("forest", ObliqueRandomForestClassifier(**args))])
    grid = GridSearchCV(pipe, {"forest__sparsity_penalty": [0., 0.01]}, cv=2, n_jobs=1)
    grid.fit(X, y)
    assert 0 <= grid.best_estimator_.score(X, y) <= 1
    scores = cross_val_score(pipe, X, y, cv=2)
    assert len(scores) == 2 and np.all(np.isfinite(scores))


def test_pickle_roundtrip(classification_data, fast_settings):
    X, y = classification_data
    m = forest(fast_settings).fit(X, y)
    p = pickle.loads(pickle.dumps(m))
    assert np.array_equal(p.predict_proba(X[:15]), m.predict_proba(X[:15]))


def test_unbalanced_and_constant_features(fast_settings):
    rng = np.random.default_rng(4)
    X = np.column_stack([np.ones(140), rng.normal(size=(140, 4))])
    y = np.zeros(140, dtype=int)
    y[-8:] = 1
    m = forest(fast_settings).fit(X, y)
    assert np.all(np.isfinite(m.predict_proba(X)))


def test_pure_local_root_is_leaf():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(10, 3))
    y = np.array([0] * 9 + [1])
    tree = ObliqueTreeClassifier(min_samples_split=60, random_state=5).fit(X, y)
    assert tree.node_count_ == 1
    assert tree.get_n_leaves() == 1


def test_label_and_feature_validation(classification_data, fast_settings):
    X, y = classification_data
    with pytest.raises(ValueError, match="binary"):
        forest(fast_settings).fit(X, np.arange(len(y)) % 3)
    with pytest.raises(ValueError):
        forest(fast_settings).fit(np.full_like(X, np.nan), y)
    one_class = forest(fast_settings).fit(X, np.zeros_like(y))
    assert one_class.predict_proba(X[:3]).shape == (3, 1)
    assert np.all(one_class.predict(X[:3]) == 0)
    fitted = forest(fast_settings).fit(X, y)
    with pytest.raises(ValueError):
        fitted.predict_proba(X[:, :-1])


@pytest.mark.parametrize("name,value", [
    ("sparsity_penalty", -1), ("stability_penalty", -1),
    ("n_bootstraps", 0), ("min_samples_leaf", 0),
    ("assessment_fraction", 1.0), ("max_features", -1),
])
def test_invalid_parameters(classification_data, fast_settings, name, value):
    X, y = classification_data
    with pytest.raises(ValueError):
        forest(fast_settings, **{name: value}).fit(X, y)


def test_ablation_factory(fast_settings):
    args = dict(fast_settings)
    args.pop("n_estimators")
    models = make_ablation_models(n_estimators=2, **args)
    assert len(models) == 7
    assert models["unpenalized"].sparsity_penalty == 0
    assert models["sparsity_only"].stability_penalty == 0
    assert models["stability_only"].sparsity_penalty == 0
    assert models["sparsity_and_stability"].stability_penalty > 0
    assert models["training_selection_control"].selection_mode == "training"
    assert models["uniform_stability_control"].stability_mode == "uniform"
    assert models["shuffled_stability_control"].stability_mode == "shuffled"
    assert make_random_forest_reference(n_estimators=5).n_estimators == 5


def test_feature_importances_shape_and_usage(classification_data, fast_settings):
    X, y = classification_data
    m = forest(fast_settings).fit(X, y)
    assert m.feature_importances_.shape == (X.shape[1],)
    assert m.feature_usage_.shape == (X.shape[1],)
    assert np.all(m.feature_importances_ >= 0)
    assert np.sum(m.feature_importances_) <= 1.0000001
    assert np.sum(m.feature_usage_) > 0


def test_brier_is_finite(classification_data, fast_settings):
    X, y = classification_data
    m = forest(fast_settings, stability_penalty=0.01).fit(X, y)
    assert np.isfinite(brier_score_loss(y, m.predict_proba(X)[:, 1]))


def test_repeated_fit_replaces_old_model(classification_data, fast_settings):
    X, y = classification_data
    m = forest(fast_settings).fit(X, y)
    p1 = m.predict_proba(X)
    m.fit(X, y)
    assert np.array_equal(p1, m.predict_proba(X))


def test_uniform_and_shuffled_stability_controls(classification_data, fast_settings):
    X, y = classification_data
    for mode in ("feature", "uniform", "shuffled"):
        m = forest(fast_settings, sparsity_penalty=0.01,
                   stability_penalty=0.01, stability_mode=mode).fit(X, y)
        assert m.n_bootstrap_fits_ > 0
        assert np.isfinite(m.predict_proba(X)).all()
        for t in m.estimators_:
            assert all(d["stability_mode"] == mode for d in t.node_diagnostics_)


def test_legacy_random_state_accepted(classification_data, fast_settings):
    X, y = classification_data
    model = forest(fast_settings, random_state=np.random.RandomState(17)).fit(X, y)
    assert np.isfinite(model.predict_proba(X[:3])).all()
