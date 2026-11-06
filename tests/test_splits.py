import numpy as np
from stable_oblique_forests._splits import (
    assessment_loss, generate_direction_bank, optimize_threshold, smoothed_probability,
)


def test_probability_smoothing():
    assert smoothed_probability(np.array([0, 1, 1])) == 3 / 5
    assert smoothed_probability(np.array([0, 0])) == 1 / 4


def test_threshold_finds_separable_problem():
    X = np.array([[-3.], [-2.], [-1.], [1.], [2.], [3.]])
    y = np.array([0, 0, 0, 1, 1, 1])
    idx = np.arange(6)
    c = optimize_threshold(X, y, idx, np.array([1.]), min_samples_leaf=2)
    assert c is not None
    assert -1 < c.threshold < 1
    assert c.n_left == 3 and c.n_right == 3
    assert c.left_probability == 1 / 5
    assert c.right_probability == 4 / 5
    assert np.isfinite(assessment_loss(c, X, y, idx))


def test_no_valid_threshold_for_constant_projection():
    X = np.ones((20, 3))
    y = np.array([0, 1] * 10)
    assert optimize_threshold(X, y, np.arange(20), np.array([1., 0., 0.]), 3) is None


def test_minimum_leaf_size_is_enforced():
    X = np.arange(7, dtype=float)[:, None]
    y = np.array([0, 1, 1, 1, 1, 1, 1])
    c = optimize_threshold(X, y, np.arange(7), np.array([1.]), 3)
    assert c.n_left >= 3 and c.n_right >= 3


def test_bank_supports_and_exposure_are_reproducible():
    kw = dict(pool=np.arange(10), support_sizes=(1, 2, 4), candidates_per_support=8, n_features=12)
    bank1, exposure1 = generate_direction_bank(rng=np.random.default_rng(4), **kw)
    bank2, exposure2 = generate_direction_bank(rng=np.random.default_rng(4), **kw)
    assert len(bank1) == 24
    assert np.array_equal(exposure1, exposure2)
    assert np.max(exposure1[:10]) - np.min(exposure1[:10]) <= 2
    assert np.all(exposure1[10:] == 0)
    for i, (a, b) in enumerate(zip(bank1, bank2)):
        assert np.array_equal(a, b)
        assert np.count_nonzero(a) == (1, 2, 4)[i // 8]
        assert np.isclose(np.linalg.norm(a), 1.0)
