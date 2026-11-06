# Method implementation notes

This document maps the research plan to the implementation and records choices that need to remain explicit when experiments are developed.

## Mathematical objective

Given a construction sample `A_v` and a disjoint assessment sample `H_v` at node `v`, a candidate `c=(w_c,t_c)` partitions the observations. The sample-Brier objective used for final selection is

$$
S_v(c)=\widehat L_{v,H}(c)+\lambda\|w_c\|_0
+\gamma\sum_{j\in\operatorname{supp}(w_c)}(1-\widehat\pi_{v,j}).
$$

`lambda=0, gamma=0` is the unpenalized candidate selection, `gamma=0` is ordinary L0, and `lambda=0` is stability-only. If the best candidate does not improve on a smoothed no-split prediction of the parent node, the node remains a leaf.

## Code path

1. `forest.py` draws tree-local disjoint construction/assessment data partitions. The tree-level bootstrap operates only on the construction partition.
2. `_splits.py` generates node-local, balanced-exposure sparse random directions, and `optimize_threshold` evaluates every eligible one-dimensional threshold using prefix-sum counts.
3. `tree.py` uses exactly that candidate bank on `B` construction-sample bootstraps. It refits the thresholds each time and selects one direction by bootstrap training Brier plus `lambda * k`.
4. `tree.py` converts selection counts into `pi_hat` estimates, scores the fitted candidates on `H_v`, and adds any chosen penalties.
5. The no-split criterion has no complexity penalty. Trees recurse and the forest averages their leaf probability predictions.

Node-local RNG streams are derived in advance for the left and right children. This prevents the extra draws needed for stability estimation at an ancestor from changing descendant random-projection banks when the paths otherwise agree. The candidate bank is fixed **within** the bootstrap procedure for one node, not fixed for all nodes/trees. Exposures are approximately balanced only within each node's sampled feature pool. The positive-class label is encoded relative to `classes_` and predicted probabilities use add-one smoothing.

## Risks and open issues for paper-grade results

- The fixed direction bank changes the interpretation of stability: it measures resampling variation conditional on a bank, not variation over all possible candidate searches. This is intentional and can be explored later in a separate ablation.
- Per-feature selection rates are not conditional on candidate exposure. Balanced supports and exposure diagnostics reduce a source of bias, but do not completely eliminate it when `Q` is small.
- When features are strongly correlated, reliable alternative supports can have low marginal selection frequencies.
- The assessment pool is disjoint from construction data at the root but reused adaptively down the tree. This prevents a simple independent-assessment guarantee at deep nodes.
- Because penalization competes with no-split, increasing `lambda` or `gamma` can strongly reduce tree depth. This is a meaningful effect of the proposed method and should be recorded, not silently corrected.
- The stability penalty can be decomposed into an additional uniform L0 term minus a frequency reward. Hence `uniform_stability_control` and `shuffled_stability_control` are especially important for isolating the feature-specific information in the frequencies.
- The outer comparison between a model with stability bootstraps and the cheaper L0-only model should include compute-matched baselines in later experiments. This package does not execute that comparison.

## Estimator API policy

Constructors store parameters unchanged. Validation, RNG creation and data-dependent attributes occur in `fit`. Unfitted calls raise `NotFittedError`. Fitted objects support common sklearn meta-estimators via `BaseEstimator`. No global random seed is modified.

Do not interpret `feature_importances_` as a formal variable selection procedure or a bootstrap stability statistic. It is only a training-gain bookkeeping measure.
