# Experimental Results: Local Feature-Selection Stability in Oblique Random Forests

## 1. Executive Summary

We evaluated a node-wise stability-aware split criterion for sparse oblique random forests, in which the assessment-sample Brier loss of a candidate split is penalised by an L0 term (λ‖w‖₀) and by a feature-specific term γ Σⱼ (1 − π̂ⱼ), where π̂ⱼ is the bootstrap selection frequency of feature j at that node. On synthetic data with p = 50 features and a correlated block of four features (ρ ∈ {0, 0.5, 0.9}), the penalised forests did not improve beyond noise on the unpenalised oblique forest or on a size-matched scikit-learn random forest in any of the four metrics we recorded (Brier loss, accuracy, F1, mAP), and their Brier loss was consistently worse. The penalties lowered probabilistic accuracy (Brier excess over the oracle rose by about 0.004 for L0 and about 0.019 for L0 + stability, independently of ρ) while leaving accuracy, F1 and mAP essentially unchanged. The decisive control experiment shows that the stability-aware model is practically indistinguishable from a *uniform-weight* control (same bootstrap estimates, one common weight for all features) and from a *shuffled-frequency* control (frequencies randomly reassigned among features): Brier differences are at most 0.0012 against replicate standard deviations of 0.003–0.009, and tree sizes are identical. A very small and consistent advantage in the share of truly relevant features among selected split features (about +0.005 in absolute terms) is the only trace of feature-specific information in π̂; it is too small to support a claim without further replication. Reducing the penalties from (λ, γ) = (0.003, 0.007) to (0.001, 0.002) removed the near-single-leaf trees (1.6–4.4 nodes per tree) but did not eliminate shrinkage: stability-aware trees still contain 28–48 % of the nodes of unpenalised trees.

On four real OpenML binary tasks (breast-w, diabetes, credit-g, ionosphere), nested cross-validation over λ, γ ∈ {0, 0.001, 0.005} produced a tuned oblique forest whose Brier loss differs from the unpenalised oblique forest by at most 0.0012, well within fold-to-fold variation. The inner-CV Brier surface is almost flat in γ (range ≤ 0.0019 at λ = 0) and degrades only for λ = 0.005; the search selected λ = 0 in 16 of 20 outer folds. Compared with scikit-learn's default random forest, the oblique forests were on par on breast-w and diabetes but worse in Brier loss on credit-g and ionosphere in every outer fold; a size-matched random forest shows that most of this gap stems from the tree-size limits shared by the oblique forests, not from the oblique splits. Against that size-matched forest, the oblique forest had lower Brier loss on three of four datasets. Overall, we find no empirical evidence that the stability penalty improves predictive performance, and only weak, unconfirmed evidence that it improves the precision of feature selection; the tuned model is roughly 60–90 times more expensive to fit than the unpenalised one.

---

## 2. Synthetic Correlation Analysis

### 2.1 Design

Each dataset has n = 1,000 training observations and p = 50 standard-normal features. The first four features (x₁–x₄) form an equicorrelated block with correlation ρ; the other 46 are independent noise. Labels are Bernoulli(sigmoid(2 f(x))) for three boundary functions: f_axis = x₁; f_oblique = (x₁ + x₂ + x₃)/√3; f_mixed = x₂ if x₁ ≤ 0, else (x₂ + x₃ + x₄)/√3. Models are evaluated on 20,000 held-out points with five replicates per (function, ρ) cell, using common random numbers across models. All forests have 100 trees. Oblique forests use support sizes (1, 2, 4, 8), eight directions per size, `max_features="sqrt"` (a pool of eight features per node), B = 10 bootstrap repetitions, depth ≤ 8, minimum leaf size 15 and minimum split size 60. The scikit-learn random forest uses the same size limits. The penalised models use **λ = 0.001 and γ = 0.002**; an earlier run with (λ, γ) = (0.003, 0.007) and no controls is cited only for comparison of tree sizes. The six compared models are: random forest (RF); unpenalised oblique (Obl); oblique + L0 (L0); oblique + L0 + stability (Stab); and the uniform (Unif) and shuffled (Shuf) stability controls.

Two properties of the design matter for interpretation. First, because the block features are summed in f_oblique and f_mixed, Var f(x) grows with ρ, so the problem becomes *easier* as ρ increases (the oracle Brier falls from about 0.151 to 0.107 for f_oblique). Results are therefore reported both in raw Brier loss and as excess over the oracle. Second, for f_axis and f_oblique the block contains correlated decoys (x₂–x₄ and x₄, respectively); the "relevant fraction" diagnostic counts only the true support of f, so selecting a near-equivalent proxy counts as an irrelevant selection.

### 2.2 Predictive performance under increasing correlation

**Brier loss (mean over 5 replicates; replicate sd 0.001–0.009):**

| Function | ρ | RF | Obl | L0 | Stab | Unif | Shuf | Oracle |
|---|---|---|---|---|---|---|---|---|
| axis | 0.0 | 0.1817 | 0.1807 | 0.1862 | 0.2117 | 0.2118 | 0.2116 | 0.1507 |
| axis | 0.5 | 0.1752 | 0.1769 | 0.1798 | 0.1924 | 0.1932 | 0.1929 | 0.1506 |
| axis | 0.9 | 0.1608 | 0.1630 | 0.1663 | 0.1767 | 0.1779 | 0.1773 | 0.1505 |
| oblique | 0.0 | 0.1951 | 0.2040 | 0.2062 | 0.2174 | 0.2184 | 0.2181 | 0.1513 |
| oblique | 0.5 | 0.1442 | 0.1489 | 0.1539 | 0.1708 | 0.1719 | 0.1696 | 0.1197 |
| oblique | 0.9 | 0.1138 | 0.1158 | 0.1199 | 0.1366 | 0.1372 | 0.1371 | 0.1066 |
| mixed | 0.0 | 0.1987 | 0.2058 | 0.2093 | 0.2213 | 0.2223 | 0.2225 | 0.1512 |
| mixed | 0.5 | 0.1654 | 0.1682 | 0.1721 | 0.1875 | 0.1873 | 0.1873 | 0.1367 |
| mixed | 0.9 | 0.1370 | 0.1389 | 0.1426 | 0.1574 | 0.1576 | 0.1569 | 0.1279 |

**Mean excess Brier over the oracle (average of the three functions):**

| ρ | RF | Obl | L0 | Stab | Unif | Shuf |
|---|---|---|---|---|---|---|
| 0.0 | 0.041 | 0.046 | 0.050 | 0.066 | 0.066 | 0.066 |
| 0.5 | 0.026 | 0.029 | 0.033 | 0.048 | 0.048 | 0.048 |
| 0.9 | 0.009 | 0.011 | 0.015 | 0.029 | 0.029 | 0.029 |

*Effect of ρ.* Every model improves as ρ increases, and the excess Brier shrinks by a factor of four to five for the unpenalised models. Two mechanisms contribute and cannot be separated with this design: the larger signal variance noted above, and the fact that correlated block features act as proxies, so that a randomly sampled pool of eight features contains a usable feature far more often (if the pool is a uniform random sample of 8 of the 50 features, it contains at least one of the four block features with probability ≈ 0.51, versus ≈ 0.16 for x₁ alone). The *cost of the penalties is approximately constant in ρ*: L0 adds 0.004 relative to the unpenalised model at every ρ, and Stab adds 0.020, 0.019 and 0.018 at ρ = 0, 0.5, 0.9. We therefore find no evidence that the stability penalty becomes more beneficial (or more harmful) as feature redundancy increases.

*Model ranking.* In Brier loss the ordering RF ≲ Obl < L0 < Stab holds in all nine cells (the only near-tie is f_axis at ρ = 0, where Obl and RF differ by 0.001). L0 costs 0.002–0.006 over Obl, and Stab costs 0.013–0.031 over Obl (mean 0.019) and 0.010–0.026 over L0 (mean 0.015). The unpenalised oblique forest does not outperform the random forest even on the oblique and mixed boundaries; RF is better by 0.007–0.009 in Brier at ρ = 0 for those two functions and the two are within 0.005 elsewhere.

*Accuracy, F1 and mAP.* These metrics barely separate the models. At ρ = 0.9 all six models are within 0.002 in accuracy for every function. Within any single cell the spread is at most 0.010 for mAP and 0.012 for F1. The largest accuracy differences occur at ρ = 0 for f_oblique and f_mixed, where the three stability variants trail Obl by about 0.003–0.007 (replicate sd ≈ 0.005–0.014); F1 for these variants at f_oblique also has a larger replicate standard deviation (≈ 0.04 versus 0.02 for Obl), which suggests that a few replicates predict the positive class less often. The pattern across ρ in accuracy, F1 and mAP follows the oracle: it is flat for f_axis (accuracy ≈ 0.77) and rises with ρ for f_oblique (0.74 → 0.84) and f_mixed (0.73 → 0.81). The penalties therefore degrade the *calibration/sharpness* of the probability estimates (Brier) but not the *ranking* (mAP) or the thresholded decisions.

### 2.3 Does π̂ carry feature-specific signal? Stability vs. uniform and shuffled controls

The stability term γ Σⱼ (1 − π̂ⱼ) can be decomposed into a uniform L0-like penalty minus a reward for frequently selected features. The uniform control replaces all feature weights by a common value and the shuffled control randomly reassigns the observed frequencies among candidate features within a node; both retain the same bootstrap computation and the same overall penalty scale. If π̂ carried genuine information about which features are reliable, the proposed model should outperform both controls.

**Predictive performance.** Brier differences between the proposed model and the controls are negligible. Relative to Unif, Stab has the lower Brier loss in 8 of 9 cells by a mean of 0.0006 (maximum absolute difference 0.0012). Relative to Shuf, the mean difference is 0.0002, with Stab better in 5 of 9 cells. Replicate standard deviations are 0.003–0.009 (standard errors ≈ 0.001–0.004), so these differences are well inside the noise. Accuracy, F1 and mAP are indistinguishable among Stab, Unif and Shuf in every cell (all pairwise differences ≤ 0.006).

**Tree topology and support size.** Stab, Unif and Shuf grow trees of the same size (9.2–16.4, 9.2–16.0 and 9.3–16.5 nodes per tree) with the same mean support size (1.48–1.77, 1.47–1.77 and 1.48–1.75 selected features per split, compared with 1.88–2.44 for L0 and 3.1–3.5 for the unpenalised model). The shrinkage and sparsification relative to L0 are thus reproduced entirely by the uniform control.

**Feature-selection precision.** The fraction of selected split features that belong to the true support of f is:

| Function | ρ | L0 | Stab | Unif | Shuf |
|---|---|---|---|---|---|
| axis | 0.0 | 0.063 | 0.110 | 0.108 | 0.107 |
| axis | 0.5 | 0.062 | 0.093 | 0.091 | 0.092 |
| axis | 0.9 | 0.052 | 0.080 | 0.077 | 0.076 |
| oblique | 0.0 | 0.136 | 0.188 | 0.183 | 0.185 |
| oblique | 0.5 | 0.167 | 0.242 | 0.233 | 0.234 |
| oblique | 0.9 | 0.174 | 0.249 | 0.243 | 0.245 |
| mixed | 0.0 | 0.136 | 0.191 | 0.185 | 0.184 |
| mixed | 0.5 | 0.176 | 0.256 | 0.249 | 0.247 |
| mixed | 0.9 | 0.195 | 0.277 | 0.277 | 0.274 |

Relative to L0, all three stability variants roughly increase this fraction by 0.03–0.08, but the uniform control achieves essentially the same improvement, so this gain is attributable to the heavier, generic penalty (smaller supports, fewer splits) and not to feature-specific frequencies. Stab exceeds Unif by +0.004 on average (non-negative in 8 of 9 cells, with one difference of −0.0000) and exceeds Shuf by +0.005 on average (non-negative in all 9 cells); the individual differences range from 0.000 to 0.009. At tree roots the comparison is inconsistent in sign (for example f_axis at ρ = 0.9: 0.187 for Stab versus 0.191 for Unif), and replicate standard deviations at the root are 0.03–0.09.

**Interpretation.** In terms of predictive performance, tree size and support size, the stability-aware model behaves like an additional uniform penalty. The only signature compatible with a feature-specific effect is a very small, directionally consistent advantage in selection precision (about 2–3 % in relative terms). Because the nine cells share data-generating seeds and are not independent, and because we did not run a paired test on the replicate-level results, this should be reported as *suggestive but unconfirmed*. A plausible explanation, which we did not test, is that at γ = 0.002 the frequency-dependent part of the penalty (at most 0.002 per selected feature) is small relative to typical assessment-loss gains of deep splits, and that B = 10 bootstrap repetitions give coarse frequency estimates (multiples of 0.1). Testing larger γ with λ = 0 (the `stability_only` configuration), more bootstrap repetitions and more replicates, always together with both controls, is the natural next step.

### 2.4 Tree topology: was the depth collapse resolved?

| Setting | Unpenalised Obl | L0 | Stab | Unif | Shuf |
|---|---|---|---|---|---|
| Initial run, (λ, γ) = (0.003, 0.007) | 30.6–34.2 | 6.6–13.4 | **1.6–4.4** | n/a | n/a |
| Small-penalty run, (λ, γ) = (0.001, 0.002) | 30.6–34.2 | 23.2–29.5 | **9.2–16.4** | 9.2–16.0 | 9.3–16.5 |

Nodes per tree are reported as ranges over the nine (function, ρ) cells; the random forest has 32.6–35.7. In the initial run, stability-aware trees averaged 1.6 nodes at f_axis with ρ = 0, which implies that at most about 31 % of the trees contained a split at all. With the smaller penalties this degenerate behaviour is gone: the smallest mean tree has 9.2 nodes. The collapse is therefore **substantially mitigated but not eliminated**. Stability-aware trees still contain only 28–48 % of the nodes of unpenalised trees (the smallest ratio, 28 %, occurs for f_axis at ρ = 0), and the Brier cost over the unpenalised forest remains 0.013–0.031. The dependence on ρ is not uniform: for f_axis, stability-aware trees are smallest at ρ = 0 and larger for ρ > 0 (9.2 → 16.4 → 15.9 nodes), which may reflect correlated proxies making a usable candidate available more often, whereas for f_oblique (15.9 → 15.2 → 12.6) and f_mixed (13.6 → 15.0 → 13.6) the node count is flat or decreasing in ρ. The uniform and shuffled controls show the same topology, so the residual shrinkage is a property of the penalty scale, not of the frequency weighting.

### 2.5 Computational cost

Mean fit times per forest were 0.3–0.4 s (RF), 6.5–9.2 s (Obl), 5.4–7.5 s (L0) and 16.6–38.7 s for each of the three stability variants (roughly three to five times the L0 cost). These wall-clock times were recorded while datasets were processed in parallel and vary between identical configurations (for example 16.6 s versus 38.7 s for Stab at ρ = 0 and ρ = 0.9), so only the order of magnitude is reliable. A compute-matched comparison, in which the L0 model receives the extra budget (more trees or candidates), has not been performed.

---

## 3. OpenML Real-World Benchmark Analysis

### 3.1 Protocol

Four binary datasets were evaluated: breast-w (n = 699, p = 9), diabetes (n = 768, p = 8), credit-g (n = 1,000, p = 20) and ionosphere (n = 351, p = 34). The configuration also requested banknote-authentication, but it does not appear in the outputs (the dataset could not be loaded in our run), so it is excluded. The positive class is the less frequent class in each dataset (breast-w: malignant, 34 %; diabetes: tested_positive, 35 %; credit-g: bad, 30 %; ionosphere: b, 36 %). Preprocessing (median imputation and standardisation for numeric columns, most-frequent imputation and one-hot encoding for categorical columns) is part of a scikit-learn pipeline fitted on training folds only. We used stratified 5-fold outer cross-validation (one repeat) with 100 trees per forest. The "tuned" oblique forest selects (λ, γ) ∈ {0, 0.001, 0.005}² (nine cells) by 3-fold inner-CV Brier loss within each outer training fold (nested CV); the penalties are never selected on test folds. Baselines are scikit-learn's default random forest, a random forest with the oblique forests' size limits (depth 8, leaf 15, split 60), and the unpenalised oblique forest (λ = γ = 0). The uniform and shuffled controls were not run on these data.

### 3.2 Results (mean ± sd over 5 outer folds)

**Brier loss (lower is better)**

| Dataset | RF default | RF size-matched | Oblique (λ = γ = 0) | Oblique tuned |
|---|---|---|---|---|
| breast-w | 0.0277 ± 0.0091 | 0.0353 ± 0.0072 | 0.0268 ± 0.0062 | 0.0268 ± 0.0062 |
| diabetes | 0.1617 ± 0.0137 | 0.1624 ± 0.0098 | 0.1592 ± 0.0092 | 0.1601 ± 0.0086 |
| credit-g | 0.1618 ± 0.0095 | 0.1741 ± 0.0045 | 0.1766 ± 0.0049 | 0.1763 ± 0.0053 |
| ionosphere | 0.0514 ± 0.0163 | 0.0862 ± 0.0175 | 0.0769 ± 0.0137 | 0.0781 ± 0.0134 |

**Accuracy**

| Dataset | RF default | RF size-matched | Oblique | Oblique tuned |
|---|---|---|---|---|
| breast-w | 0.966 ± 0.014 | 0.969 ± 0.010 | 0.971 ± 0.005 | 0.971 ± 0.005 |
| diabetes | 0.766 ± 0.032 | 0.756 ± 0.019 | 0.763 ± 0.026 | 0.759 ± 0.015 |
| credit-g | 0.763 ± 0.026 | 0.718 ± 0.014 | 0.715 ± 0.016 | 0.716 ± 0.016 |
| ionosphere | 0.934 ± 0.024 | 0.929 ± 0.025 | 0.932 ± 0.028 | 0.926 ± 0.033 |

**F1 score (positive = minority class, threshold 0.5)**

| Dataset | RF default | RF size-matched | Oblique | Oblique tuned |
|---|---|---|---|---|
| breast-w | 0.950 ± 0.021 | 0.955 ± 0.015 | 0.960 ± 0.007 | 0.960 ± 0.007 |
| diabetes | 0.636 ± 0.059 | 0.601 ± 0.031 | 0.623 ± 0.045 | 0.608 ± 0.033 |
| credit-g | 0.481 ± 0.067 | 0.149 ± 0.064 | 0.115 ± 0.080 | 0.116 ± 0.078 |
| ionosphere | 0.906 ± 0.033 | 0.896 ± 0.036 | 0.900 ± 0.040 | 0.893 ± 0.047 |

**mAP (average precision, higher is better)**

| Dataset | RF default | RF size-matched | Oblique | Oblique tuned |
|---|---|---|---|---|
| breast-w | 0.981 ± 0.016 | 0.977 ± 0.014 | 0.981 ± 0.014 | 0.981 ± 0.014 |
| diabetes | 0.695 ± 0.053 | 0.715 ± 0.052 | 0.718 ± 0.048 | 0.713 ± 0.048 |
| credit-g | 0.638 ± 0.076 | 0.612 ± 0.054 | 0.618 ± 0.057 | 0.610 ± 0.062 |
| ionosphere | 0.975 ± 0.022 | 0.946 ± 0.028 | 0.965 ± 0.025 | 0.962 ± 0.024 |

### 3.3 Effect of the penalties and the grid search

*Tuned versus unpenalised.* The mean paired difference in outer-fold Brier loss (tuned minus unpenalised) is 0.0000 on breast-w, +0.0009 on diabetes, −0.0003 on credit-g and +0.0012 on ionosphere, with individual fold differences between −0.0013 and +0.0064. Tuning therefore neither helped nor hurt measurably. The selected penalties were unstable across folds: λ = 0 was chosen in 16 of 20 folds (λ = 0.001 in 4; λ = 0.005 never), and γ was 0, 0.001 and 0.005 in 8, 7 and 5 folds, respectively. On diabetes, the five folds chose five different cells.

*Shape of the penalty surface.* Averaged over outer folds, the inner-CV Brier surface is nearly flat in γ: at λ = 0 it varies by only 0.0006 (breast-w), 0.0009 (diabetes), 0.0019 (credit-g) and 0.0006 (ionosphere) over γ ∈ {0, 0.001, 0.005}. The best cell improves on (0, 0) by at most 0.0010 (diabetes). λ is the sensitive direction: at γ = 0, increasing λ from 0 to 0.005 raises inner-CV Brier loss by 0.0027 on breast-w, 0.0022 on ionosphere and 0.0122 on credit-g (diabetes is unaffected, −0.0001). These results agree with the synthetic study: for the penalties examined, the L0 component drives the loss of probabilistic accuracy, and the stability component is, at these magnitudes, nearly inert.

### 3.4 Oblique versus random forest baselines

*Default random forest.* On breast-w and diabetes the unpenalised oblique forest matches the default random forest (paired fold differences in Brier of −0.0008 and −0.0024, with large fold-level variation). On credit-g and ionosphere the oblique forest is worse in Brier loss in all five folds (mean paired differences +0.0148 and +0.0254), a consistent but small-sample result (the folds share training data, so we do not report p-values).

*Size-matched random forest.* The size-matched forest behaves much like the oblique forest and is clearly worse than the default random forest on credit-g (Brier 0.1741 versus 0.1618) and ionosphere (0.0862 versus 0.0514). Hence most of the gap to the default random forest comes from the tree-size limits (depth ≤ 8, leaf ≥ 15, split ≥ 60) that the oblique forests share, and not from the oblique split search. Against the size-matched forest, the oblique forest has a lower Brier loss on breast-w (−0.0084, lower in 5 of 5 folds), diabetes (−0.0032, 5 of 5) and ionosphere (−0.0093, 4 of 5) and a higher one on credit-g (+0.0025, worse in 4 of 5 folds). mAP is higher for the oblique forest on all four datasets (by 0.003–0.019), although these differences are smaller than the fold standard deviations.

### 3.5 Calibration versus ranking

The datasets illustrate that probability quality, ranking and thresholded decisions can disagree. On credit-g, the oblique forests have a Brier loss 0.015 above the default random forest and an F1 of only 0.12 (versus 0.48), yet their mAP (0.610–0.618) is within about 0.03 of the random forest (0.638), well inside the fold standard deviation of 0.05–0.08. The low F1 is a threshold effect: with a 30 % positive class and averaged, heavily regularised leaf probabilities, few observations exceed 0.5. The size-matched random forest shows the same pattern (F1 0.149), which again points to the size limits and not to the split type. F1 at the default threshold is therefore a poor basis for comparing probabilistic models under class imbalance; Brier loss and mAP are more informative, and a tuned decision threshold should be used when F1 matters. On breast-w, diabetes and ionosphere, the four models differ by at most 0.04 in F1 and 0.03 in mAP, within fold variation. Finally, tuning is expensive: the nested search costs 178–420 s per outer fold versus 2.4–6.8 s for the unpenalised forest (roughly 60–90 times; timings measured with four parallel processes).

---

## 4. Summary and Guidelines for Practitioners

**What the evidence supports.**
1. Under the tested settings, the L0 and stability penalties reduce probabilistic accuracy (L0 by about 0.004 and L0 + stability by about 0.019 excess Brier on synthetic data) without measurable gains in accuracy, F1 or mAP. No improvement over the unpenalised oblique forest beyond noise was observed on synthetic or real data.
2. The stability weighting is not distinguishable from uniform or shuffled weighting in predictive performance, tree size or support size. A very small advantage in selection precision (≈ +0.005 absolute) is suggestive only.
3. The penalty cost is roughly independent of the feature correlation ρ in the tested range.
4. On four real datasets, nested tuning of (λ, γ) matches but does not beat the unpenalised oblique forest, and the choice of γ matters very little within {0, 0.001, 0.005}.

**Practical advice.**
- *Start without penalties.* Use the unpenalised oblique forest as the reference model and only add penalties when sparsity or interpretability is the goal, not to improve accuracy.
- *Treat λ as the primary knob, and keep it small.* λ = 0.005 already harmed Brier loss on credit-g (+0.012 inner-CV) and λ = 0.003 combined with γ = 0.007 produced trees of only 1.6–4.4 nodes on synthetic data. A starting range of λ ≤ 0.001 and γ ≤ 0.002 avoided collapse but still shrank trees to roughly half the unpenalised size.
- *Monitor tree size.* Report nodes per tree (`node_count_ / n_estimators`) next to every penalised result; it is the quickest diagnostic of whether the penalty is acting as a stopping rule.
- *Always include the uniform and shuffled controls* when claiming a benefit from the stability term; without them, the effect of extra penalisation cannot be separated from the effect of feature-specific frequencies.
- *Choose metrics deliberately.* Use Brier loss and mAP for probabilistic comparisons; interpret F1 at a fixed 0.5 threshold with caution under class imbalance.
- *Mind the budget.* The stability variant needs roughly three to five times the fit time of the L0 model (B = 10), and the nested grid search is far more expensive; consider a coarse grid in λ and a fixed small γ.
- *Dataset suitability.* The tested real datasets are small (n = 351–1,000) and low-dimensional; with fold standard deviations of 0.005–0.016 in Brier loss, differences below about 0.005 should not be interpreted. Benefits of stability regularisation, if any, are more likely to show up in larger, higher-dimensional and more strongly redundant problems, which these experiments did not cover.

---

## 5. Limitations and Threats to Validity

- **Small replication:** five replicates per synthetic cell and five outer folds (one repeat) per real dataset. No hypothesis tests were performed; statements about "consistency" refer to signs across cells or folds that share seeds or training data and are not independent.
- **Limited penalty exploration:** the synthetic study has one small-penalty setting (plus one earlier large-penalty run without controls); the real-data grid is 3 × 3. The properties of the stability term at larger γ with λ = 0, with more bootstrap repetitions (B > 10), or with larger n, remain untested.
- **One correlation structure and signal strength:** an equicorrelated block of four features and a = 2.0. Because Var f(x) changes with ρ, correlation and effective signal strength are confounded in f_oblique and f_mixed.
- **Diagnostic definitions:** the "relevant fraction" counts only the true support of f and treats correlated proxies as irrelevant; root-level diagnostics are conditional on a root split being accepted, so they do not measure recall.
- **Candidate-bank design:** stability is measured conditionally on a fixed node-local direction bank, and assessment samples are reused adaptively below the root (see `docs/implementation_notes.md`).
- **Baselines:** the oblique forests do not outperform a size-matched random forest on the oblique and mixed synthetic boundaries, and the default random forest (fully grown trees) is stronger on two real datasets. No class weighting or threshold tuning was applied.
- **Real-data coverage:** four of five requested datasets were evaluated (banknote-authentication did not load), and the controls were not run on real data.
- **Timing:** wall-clock times come from parallel runs on one machine and are indicative only.
