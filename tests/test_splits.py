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


def _estimate_pi_hat(X, y, construction_idx, bank, n_bootstraps, min_samples_leaf,
                     sparsity_penalty, seed):
    """Selection frequencies of each feature over bootstrap resamples of the construction sample.

    Mirrors the local-stability estimate: the same direction bank is refit on each bootstrap
    resample of the construction indices, and the candidate with the lowest construction Brier
    loss plus the L0 penalty is selected ("no-split" is excluded). The full label vector `y` is
    passed through on purpose; only entries at `construction_idx` may influence the result.
    """
    rng = np.random.default_rng(seed)
    counts = np.zeros(X.shape[1])
    n_valid = 0
    for _ in range(n_bootstraps):
        boot = rng.choice(construction_idx, size=len(construction_idx), replace=True)
        best_w, best_score = None, np.inf
        for w in bank:
            cand = optimize_threshold(X, y, boot, w, min_samples_leaf)
            if cand is None:
                continue
            score = assessment_loss(cand, X, y, boot) + sparsity_penalty * np.count_nonzero(w)
            if score < best_score:
                best_w, best_score = w, score
        if best_w is None:
            continue
        n_valid += 1
        counts[np.flatnonzero(best_w)] += 1
    assert n_valid > 0, "test setup produced no valid bootstrap split"
    return counts / n_valid


def test_pi_hat_independent_of_assessment_labels():
    """pi_hat depends only on the construction sample A_v, never on labels in H_v."""
    rng = np.random.default_rng(0)
    n_construction, n_assessment, p = 80, 40, 8
    n = n_construction + n_assessment
    X = rng.normal(size=(n, p))
    logits = 2.0 * X[:, 0] - 1.5 * X[:, 1]
    y = (logits + 0.5 * rng.normal(size=n) > 0).astype(int)

    construction_idx = np.arange(n_construction)       # A_v
    assessment_idx = np.arange(n_construction, n)      # H_v
    assert np.intersect1d(construction_idx, assessment_idx).size == 0

    bank, _ = generate_direction_bank(
        rng=np.random.default_rng(1), pool=np.arange(p), support_sizes=(1, 2, 4),
        candidates_per_support=8, n_features=p,
    )
    kw = dict(bank=bank, n_bootstraps=10, min_samples_leaf=3, sparsity_penalty=0.001, seed=7)

    pi_ref = _estimate_pi_hat(X, y, construction_idx, **kw)
    assert np.all((pi_ref >= 0) & (pi_ref <= 1))

    # 1) flip every assessment label
    y_flipped = y.copy()
    y_flipped[assessment_idx] = 1 - y_flipped[assessment_idx]
    assert not np.array_equal(y_flipped, y)
    np.testing.assert_array_equal(
        pi_ref, _estimate_pi_hat(X, y_flipped, construction_idx, **kw))

    # 2) shuffle assessment labels, several independent permutations
    for s in range(3):
        y_shuffled = y.copy()
        y_shuffled[assessment_idx] = np.random.default_rng(100 + s).permutation(
            y[assessment_idx])
        np.testing.assert_array_equal(
            pi_ref, _estimate_pi_hat(X, y_shuffled, construction_idx, **kw))

    # 3) constant labels on H_v (extreme mutation)
    for value in (0, 1):
        y_const = y.copy()
        y_const[assessment_idx] = value
        np.testing.assert_array_equal(
            pi_ref, _estimate_pi_hat(X, y_const, construction_idx, **kw))