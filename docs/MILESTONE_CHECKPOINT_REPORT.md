# Milestone 1 Checkpoint Report

**Project:** Local Feature-Selection Stability in Oblique Random Forests

**Authors:** Frydrysiak & Kukla

**Date:** 9 November 2026

**Status:** Interim progress checkpoint (Milestone 1), end of first research phase


---

## 1. Executive Summary and Milestone Scope

### 1.1 Purpose

This checkpoint closes the first research phase. It has three goals:

1. Establish an **audited baseline implementation** of a sparse oblique random forest whose split criterion includes a node-wise stability penalty.
2. Provide a first **empirical validation** of the stability regularisation on synthetic correlated-feature designs and on four OpenML binary tasks.
3. Record **sanity checks and diagnostics** that indicate what the penalty does mechanically, so that the next phase targets regimes where it could plausibly help.

### 1.2 Core takeaways

| # | Takeaway | Strength of evidence |
|---|---|---|
| 1 | In the tested regime (p = 50, n = 1,000, one correlated block of four features; four small real datasets) the stability-aware forest did **not** improve Brier loss, accuracy, F1 or mAP over the unpenalised oblique forest. Its Brier loss was consistently worse (about +0.019 excess over the oracle; L0 alone about +0.004). | 5 replicates per synthetic cell, 5 outer folds per real dataset; no hypothesis tests |
| 2 | The **uniform** and **shuffled** controls reproduce the stability model's predictive performance, tree size and support size. The feature-specific frequencies π̂ change which near-tied candidate wins (about one third of trees at γ = 0.002), but with no systematic predictive gain. | Synthetic only; the π̂-reordering check is exploratory (one dataset, one seed) |
| 3 | The Brier degradation is largely a **calibration** effect (smaller trees give under-confident probabilities), not a loss of ranking power. mAP is nearly unchanged, and a 1-D logistic recalibration removes most of the Brier gap. | Exploratory (one dataset, one seed per ρ) |
| 4 | The stability term costs 3–5 times the fit time of the L0 model. The nested (λ, γ) grid search costs about 60–90 times the fit time of one unpenalised forest. | Timings from parallel runs; order of magnitude only |

These are **negative results for the tested settings**, not a refutation of the idea. The designs have few relevant features, n well above the number of relevant features, B = 10 bootstrap repetitions and a depth cap of 8. Regimes with many redundant, weakly informative features are untested.

### 1.3 What is and is not verified

- The code was audited against the formulas in the project brief, `README.md` and `docs/implementation_notes.md`. The **research plan PDF was not available** during the audit. Items still to be reconciled with it are listed in §2.5.
- The experiments use 5 replicates or folds, share seeds across cells, and include no hypothesis tests. Statements of "consistency" refer to signs across cells.

---

## 2. Theoretical and Implementation Audit

### 2.1 Split criterion

For a candidate split c at node v the implemented score is

$$
S_v(c) = \widehat{\mathcal{L}}_{v,H}(c) + \lambda \,\|w_c\|_0 + \gamma \sum_{j \in \mathrm{supp}(w_c)} \bigl(1 - \widehat{\pi}_{v,j}\bigr),
$$

where $\widehat{\mathcal{L}}_{v,H}$ is the mean Brier loss on the assessment pool $H_v$, with child probabilities fitted on the construction pool $A_v$. Each term was checked in `tree.py::_build` (lines 264–287):

| Term | Implementation | Status |
|---|---|---|
| Loss | `assessment_loss(c, X, y, H)`, mean Brier on $H_v$ | Verified |
| L0 term | `sparsity_penalty * c.sparsity` | Verified |
| Stability term | `stability_penalty * Σ(1 − effective_frequencies[c.support])`, a **sum**, not a mean | Verified in code; sum-versus-mean still to confirm against the PDF |
| No-split option | Scored as the parent's smoothed probability on $H_v$ with no penalty; a split is accepted only if `score < best_score` (strictly better) | Verified |

Penalties are **absolute** and are added to a node-local mean loss. A given λ therefore means the same at the root and at a 25-sample leaf. This is a design property, not a bug, but it should be stated in the paper.

### 2.2 Sample partitioning, π̂ and Laplace smoothing

- **Partitioning.** The assessment pool H is reserved before the tree-level bootstrap, and the bootstrap acts on the construction pool only. Descendant nodes reuse the same H. This is documented as a non-independence of assessment data at depth.
- **Frequencies π̂.** `_bootstrap_selection_frequencies` (lines 186–208) resamples $A_v$ only, refits thresholds for the same fixed candidate bank, selects the winner by bootstrap training Brier plus λ·k, and counts the winner's support. π̂ is therefore **estimated on $A_v$**; the final loss is **evaluated on $H_v$**. No-split is excluded from the frequency estimate and included in the final choice.
- **Laplace smoothing.** Both node probabilities and child probabilities use $(n_1+1)/(n_0+n_1+2)$. Because the candidate's child probability and the child node's probability are computed on the same partition of $A_v$, they coincide. With tree-level bootstrap, $n_0$ and $n_1$ are bootstrap-weighted counts.
- **Pool rule.** The feature pool has size max(8, ⌈√p⌉), capped at p, drawn uniformly without replacement. This confirms the pool-coverage probabilities used in the results (about 0.51 for at least one of four block features, about 0.16 for x₁ alone).

### 2.3 The γ = 0 fast path

When `stability_penalty == 0`, the node-level bootstrap is skipped (line 248), the instability sum is skipped (lines 275–278) and `n_bootstrap_fits_` stays 0. Descendant seeds are drawn before any bootstrap draws, so skipping the bootstrap does not change the candidate banks of descendants. This is covered by `test_gamma_zero_skips_bootstrap` and `test_gamma_zero_is_independent_of_n_bootstraps`.

### 2.4 Independence of π̂ from assessment labels, and test status

- **Code inspection (done):** π̂, threshold fitting and probability fitting use $A_v$ only. No label from $H_v$ enters π̂. No leakage was found. Preprocessing is fitted inside the pipeline on training folds, and outer test folds are never used for tuning.
- **Unit tests:** **31 tests pass** (Windows and Linux); the scikit-learn `check_estimator` suite is reported as passing. Forests are identical for `n_jobs=1` and `n_jobs=2`.
- **Gap to close:** there is **no dedicated unit test** asserting that π̂ is independent of the assessment labels. The independence is established by reading the code, not by a test. Adding this test is on the Sprint 2 list (§5.5).

### 2.5 Items still to confirm against the research plan PDF

1. π̂ defined over bootstrap resamples of $A_v$ with the L0-penalised training Brier as selection rule.
2. Stability term is a **sum** of $(1-\widehat\pi_j)$.
3. Defaults: support sizes {1, 2, 4, 8}, 8 directions per size, B = 10, assessment fraction 0.2, minimum leaf 15, minimum split 60, depth 8.
4. The λ, γ grid {0, 0.001, 0.005}.
5. Absolute penalties (implemented) versus node-size-scaled penalties.

### 2.6 Reproducibility and design strengths

- Node-local RNG streams (child seeds drawn first) mean that penalised variants and controls share partitions, banks and bootstrap draws until their decisions first differ. This paired design is what makes the control comparison possible.
- Experimental hygiene: common random numbers across models, an oracle Brier reference, nested CV with preprocessing inside the pipeline, a size-matched random-forest baseline, minority class as positive class, and raw plus summary outputs with a config file recording library versions (Python 3.12.10, NumPy 2.5.4, scikit-learn 1.9.1).

### 2.7 Known open issues (none affects the reported numbers)

- **Seeding by position:** `exp_correlation.py` seeds data by the position of ρ in `--rho`, and `exp_openml.py` seeds by the position of the dataset among those loaded. Reproduce reported numbers only with the default ρ order `0.0 0.5 0.9` and the full dataset list. A fix (seeds derived from the ρ value and from a hash of the dataset name) requires re-running everything.
- **Silent dataset skip:** `banknote-authentication` was requested but did not load. It is listed in `openml_config.json` under `datasets` but absent from `datasets_info` and from all results. The cause is not yet recorded.
- **`GridSearchCV`** uses the default `error_score=np.nan`; it should be set to `"raise"`.
- **Diagnostics log accepted splits only** (`node_diagnostics_`), so the no-split rate that drives tree-size collapse is inferred, not measured.
- **Bootstrap multiplicities:** with tree-level bootstrap about 63 % of observations are unique, so a "15-sample" leaf may hold about 9–10 distinct observations, and the node-level resampling is a bootstrap of a bootstrap.
- **Packaging:** `pandas`, `matplotlib` and `seaborn` are not declared as dependencies, and `README.md` and `docs/tested_environment.md` are stale (see the pre-sharing checklist in `METHODOLOGICAL_ASSESSMENT.md` §5).

---

## 3. Empirical Results and Plot Analysis

### 3.1 Protocols

**Synthetic.** n = 1,000 training points, p = 50 standard-normal features. Features x₁–x₄ form an equicorrelated block with correlation ρ ∈ {0.0, 0.5, 0.9}; the other 46 are noise. Labels are Bernoulli(sigmoid(2f(x))) with f_axis = x₁, f_oblique = (x₁+x₂+x₃)/√3, and f_mixed piecewise (x₂ if x₁ ≤ 0, else (x₂+x₃+x₄)/√3). Five replicates per (function, ρ) cell, 20,000 test points, 100 trees, depth ≤ 8, minimum leaf 15, minimum split 60, B = 10. Penalised models use **λ = 0.001, γ = 0.002**. An earlier run with (0.003, 0.007) and no controls is used only for tree-size comparison.

Models: random forest (RF, same size limits), unpenalised oblique (Obl), oblique + L0 (L0), oblique + L0 + stability (Stab), uniform-weight control (Unif), shuffled-frequency control (Shuf).

**OpenML.** breast-w, diabetes, credit-g and ionosphere (positive class = minority class). Stratified 5-fold outer CV, one repeat, 100 trees. The tuned oblique forest chooses (λ, γ) ∈ {0, 0.001, 0.005}² by 3-fold inner-CV Brier loss. Baselines: default RF, size-matched RF, unpenalised oblique forest. **The Unif and Shuf controls were not run on real data.**

### 3.2 Synthetic results: Brier loss (mean over 5 replicates, replicate sd 0.001–0.009)

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

Mean excess Brier over the oracle, averaged over the three functions:

| ρ | RF | Obl | L0 | Stab | Unif | Shuf |
|---|---|---|---|---|---|---|
| 0.0 | 0.041 | 0.046 | 0.050 | 0.066 | 0.066 | 0.066 |
| 0.5 | 0.026 | 0.029 | 0.033 | 0.048 | 0.048 | 0.048 |
| 0.9 | 0.009 | 0.011 | 0.015 | 0.029 | 0.029 | 0.029 |

**Reading of the table.**

- The ordering RF ≲ Obl < L0 < Stab holds in all nine cells (the only near-tie is f_axis at ρ = 0). L0 costs 0.002–0.006 over Obl; Stab costs 0.013–0.031 over Obl.
- The penalty cost is roughly constant in ρ (L0 about +0.004; Stab +0.020 / +0.019 / +0.018). We find no evidence that stability becomes more useful as redundancy increases in this design.
- All models improve with ρ, but ρ is confounded with signal strength: in f_oblique and f_mixed, Var f grows with ρ, and correlated proxies make a usable feature appear more often in a pool of eight.
- The unpenalised oblique forest does **not** beat the random forest even on oblique and mixed boundaries (RF better by 0.007–0.009 at ρ = 0).

### 3.3 Stability versus controls

| Comparison | Brier result | Other metrics |
|---|---|---|
| Stab vs Unif | Stab lower in 8 of 9 cells, mean 0.0006, max \|Δ\| = 0.0012 | Accuracy, F1, mAP differ by at most 0.006 |
| Stab vs Shuf | Mean difference 0.0002, Stab better in 5 of 9 cells | Same |

Replicate standard deviations (0.003–0.009) are larger than these differences. Tree sizes are the same (9.2–16.4 / 9.2–16.0 / 9.3–16.5 nodes), and so are support sizes (1.48–1.77 features per split for Stab, versus 1.88–2.44 for L0 and 3.1–3.5 unpenalised). The sparsification relative to L0 is therefore reproduced by the uniform control.

**Feature-selection precision** (share of selected split features in the true support of f): Stab exceeds L0 by 0.03–0.08, but Unif does too. Stab exceeds Unif by about +0.004 and Shuf by about +0.005 on average (non-negative in 8–9 of 9 cells). This is the only trace of feature-specific information, and it is **suggestive but unconfirmed**.

### 3.4 Accuracy, F1 and mAP

These metrics barely separate the models. At ρ = 0.9 all six models are within 0.002 in accuracy. Within any cell the spread is at most 0.010 in mAP and 0.012 in F1. The largest gaps are at ρ = 0 for f_oblique and f_mixed, where the three stability variants trail Obl by about 0.003–0.007 in accuracy (replicate sd 0.005–0.014).

### 3.5 Real data (OpenML), mean ± sd over 5 outer folds

| Dataset | Metric | RF default | RF size-matched | Oblique (λ=γ=0) | Oblique tuned |
|---|---|---|---|---|---|
| breast-w | Brier | 0.0277 ± 0.0091 | 0.0353 ± 0.0072 | 0.0268 ± 0.0062 | 0.0268 ± 0.0062 |
| diabetes | Brier | 0.1617 ± 0.0137 | 0.1624 ± 0.0098 | 0.1592 ± 0.0092 | 0.1601 ± 0.0086 |
| credit-g | Brier | 0.1618 ± 0.0095 | 0.1741 ± 0.0045 | 0.1766 ± 0.0049 | 0.1763 ± 0.0053 |
| ionosphere | Brier | 0.0514 ± 0.0163 | 0.0862 ± 0.0175 | 0.0769 ± 0.0137 | 0.0781 ± 0.0134 |
| credit-g | F1 | 0.481 ± 0.067 | 0.149 ± 0.064 | 0.115 ± 0.080 | 0.116 ± 0.078 |
| credit-g | mAP | 0.638 ± 0.076 | 0.612 ± 0.054 | 0.618 ± 0.057 | 0.610 ± 0.062 |
| ionosphere | mAP | 0.975 ± 0.022 | 0.946 ± 0.028 | 0.965 ± 0.025 | 0.962 ± 0.024 |

(Full accuracy, F1 and mAP tables are in `EXPERIMENTAL_RESULTS.md` §3.2.)

**Reading.**

- Tuned versus unpenalised oblique Brier (paired mean): 0.0000 (breast-w), +0.0009 (diabetes), −0.0003 (credit-g), +0.0012 (ionosphere). Tuning neither helped nor hurt measurably.
- λ = 0 was chosen in 16 of 20 outer folds; on diabetes the five folds chose five different cells. The inner-CV surface is nearly flat in γ (range ≤ 0.0019 at λ = 0), while λ = 0.005 costs +0.012 inner-CV Brier on credit-g. Selecting among near-tied cells with a noisy 3-fold inner CV is itself noisy.
- Versus the default RF, the oblique forest is on par on breast-w and diabetes and worse in Brier on credit-g and ionosphere in all five folds (mean paired +0.0148 and +0.0254). The size-matched RF reproduces most of that gap, so the shared **tree-size limits**, not the oblique splits, are the main driver.
- Against the size-matched RF, the oblique forest has lower Brier on three of four datasets.
- The credit-g F1 of about 0.12 (versus 0.48 for the default RF) is a threshold artefact of compressed probabilities at 30 % prevalence; mAP differs by only about 0.03.

### 3.6 Visual Trend Analysis of the Generated Figures

Three figures were generated: predictive metrics and tree size versus ρ, Brier loss versus ρ per boundary function, and OpenML bar charts.

![Predictive Metrics vs Correlation Rho](../outputs/plots/correlation_metrics.png)

*Figure 1: Predictive performance (Accuracy, F1, mAP) across feature correlation values ($\rho \in [0.0, 0.5, 0.9]$). The fourth panel shows average nodes per tree. Means over boundary functions, error bars are standard errors over 5 replicates, λ = 0.001, γ = 0.002.*

**Metrics versus ρ (accuracy, F1, mAP).** All six curves rise together with ρ and lie almost on top of each other; the visible separation is a small downward offset of the three stability variants at ρ = 0 and 0.5, within overlapping error bars. Unif and Shuf lie on the Stab curve throughout.
**Tree size versus ρ** (Figure 1, bottom-right panel). Four tiers are visible: RF (about 34–35 nodes), unpenalised oblique (about 32–34), L0 (about 25–28) and the three stability variants (about 13–15, overlapping). The stability tier is flat to slightly rising in ρ. This is the tree-size shrinkage; the earlier (0.003, 0.007) run collapsed to 1.6–4.4 nodes per tree.
![Brier Loss vs Correlation Rho](../outputs/plots/correlation_vs_brier.png)

*Figure 2: Brier Loss across boundary functions and correlation levels. Mean ± standard error over 5 replicates; the dash-dotted line is the oracle (true-probability) Brier loss.*

**Brier versus ρ per function.** The three stability curves coincide and sit above L0, Obl and RF in every panel, with a gap that stays roughly constant as ρ rises, and all curves approach the oracle at ρ = 0.9.
![OpenML Benchmark Performance](../outputs/plots/openml_benchmark.png)

*Figure 3: OpenML benchmark evaluation across breast-w, diabetes, credit-g, and ionosphere datasets. Outer-fold means, error bars are standard deviations over 5 folds; panels show Brier loss, F1 and mAP.*

**OpenML bars.** Brier bars for the four oblique/RF variants are close within each dataset. The F1 panel shows the credit-g collapse for all three non-default models, while the mAP panel shows near-equal bars with error bars that overlap.

**Note on probability compression and calibration curves.** The generated figures do **not** include a calibration (reliability) plot or a prediction-spread plot. The evidence for probability compression comes from the exploratory Murphy decomposition table in §4.2, not from a figure. Producing a reliability diagram is a Sprint 2 deliverable.

### 3.7 Tree-size collapse

| Setting | Unpenalised | L0 | Stab | Brier excess, Stab minus unpenalised |
|---|---|---|---|---|
| (λ, γ) = (0.003, 0.007) | 30.6–34.2 nodes | 6.6–13.4 | **1.6–4.4** | +0.032 / +0.037 / +0.034 |
| (λ, γ) = (0.001, 0.002) | 30.6–34.2 | 23.2–29.5 | **9.2–16.4** | +0.020 / +0.019 / +0.018 |

(Brier excess listed for ρ = 0 / 0.5 / 0.9.) At the initial setting at most about 31 % of f_axis trees at ρ = 0 contained any split. The smaller setting removes degenerate trees, but stability-aware trees still hold only 28–48 % of the unpenalised node count. The response is non-linear: cutting λ threefold reduced the L0 Brier cost by about 83 %, while cutting γ 3.5-fold reduced the stability cost by only about 45 %. In the exploratory diagnostic fit, accepted-split margins concentrate near zero (10th percentile 0.0018, median 0.0116), so small penalty increments delete many marginal splits.

---

## 4. Scientific Discussion and Critical Interpretation

### 4.1 Why the stability penalty behaves like generic complexity shrinkage

Three observations support the reading that, in this regime, the stability term acts mainly as an additional L0-type shrinkage:

1. **Controls match.** Stab, Unif and Shuf agree in predictive metrics, tree size and support size (§3.3).
2. **π̂ is a weak signal here** (exploratory, f_axis, ρ = 0.5, one dataset). π̂ takes values in multiples of 1/B = 0.1. The chosen features' mean π̂ has median 0.35, and only 38 % of accepted splits have a mean chosen π̂ ≥ 0.5, so (1 − π̂) is close to its pool mean for most chosen features. The difference between the feature-specific and uniform penalty has a median of 0.0004, against a median split margin of 0.0116, and exceeded that margin in only 1.8 % of accepted splits.
3. **Selection noise dominates at depth.** The median assessment sample at accepted splits is 43 points (about 25 at depth ≥ 5). Assuming a per-observation sd of about 0.15, the standard error of a mean-Brier difference is of order 0.02–0.03, so choice among up to 32 candidates is driven by noise rather than by penalties of order 0.001–0.003.

**The weights are not inert.** In an exploratory paired check (identical seeds, banks and bootstraps; 40 trees; f_axis and f_oblique at ρ = 0.5), replacing uniform by feature weights changed 5 % of root splits and 32 % of whole trees at (0.001, 0.002), and 10–12 % of roots and 52–55 % of trees at (0, 0.005). The mean absolute change in predicted probability was 0.010–0.023. The weights reorder near-equivalent candidates, but this reordering produced no systematic predictive gain, consistent with π̂ acting as noise among near-ties.

**Scope of the claim.** This is a statement about regimes with few relevant features, ample n and B = 10. A plausible but **untested** explanation is that there is little selection instability for the stability term to repair here, and that benefits would require high-dimensional, strongly redundant, small-n problems and larger B.

### 4.2 Separating Brier degradation from ranking capability

Exploratory Murphy decomposition (f_oblique, 40 trees, 20,000 test points, 20 equal-frequency bins; REL = calibration error, RES = resolution; one dataset and one seed per ρ):

| ρ | Model | Brier | AP | sd of predictions | REL | RES | Nodes/tree |
|---|---|---|---|---|---|---|---|
| 0.9 | unpenalised | 0.1123 | 0.9222 | 0.314 | 0.0043 | 0.1417 | 30.4 |
| 0.9 | (0.001, 0.002) | 0.1306 | 0.9249 | 0.227 | 0.0226 | 0.1416 | 11.3 |
| 0.9 | (0.003, 0.007) | 0.1481 | 0.9251 | 0.180 | 0.0393 | 0.1410 | 5.5 |
| 0.0 | unpenalised | 0.2085 | 0.8098 | 0.095 | 0.0305 | 0.0717 | 34.8 |
| 0.0 | (0.001, 0.002) | 0.2204 | 0.8025 | 0.065 | 0.0395 | 0.0689 | 14.2 |
| 0.0 | (0.003, 0.007) | 0.2255 | 0.7857 | 0.055 | 0.0405 | 0.0647 | 3.7 |

At strong signal (ρ = 0.9), resolution and AP are unchanged while calibration error rises almost tenfold and the spread of predictions contracts. Smaller trees give coarser, more strongly averaged leaves, which means **under-confident probabilities**. Accuracy is unaffected while predictions stay on the correct side of 0.5, and AP depends only on ordering. At weak signal (ρ = 0), the strong penalty also lowers resolution and AP, so discrimination is not entirely preserved.

**Recalibration check.** A one-dimensional logistic recalibration (fitted on 10,000 held-out points, scored on 10,000 others) shrinks the Brier gap of the penalised forests relative to the unpenalised forest at ρ = 0.9 from **+0.018 / +0.035 to +0.001 / +0.004** (recalibrated Brier 0.1117 / 0.1148 versus 0.1107). At ρ = 0 the gap shrinks from +0.012 / +0.017 to +0.003 / +0.009. The unpenalised forest is itself poorly calibrated at ρ = 0 (0.209 raw versus 0.179 recalibrated).

**Consequence.** Raw Brier differences between these forests are largely calibration differences. Comparisons should be reported both raw and after recalibration, and ranking power (AP) interpreted separately. This check is exploratory (one dataset, one seed per ρ) and should be repeated across replicates before it is cited as a result.

### 4.3 Computational cost-benefit

| Quantity | Measured |
|---|---|
| RF fit time (100 trees) | 0.3–0.4 s |
| Unpenalised oblique | 6.5–9.2 s |
| L0 | 5.4–7.5 s |
| Each stability variant (Stab, Unif, Shuf) | 16.6–38.7 s, about 3–5 times the L0 cost |
| Nested tuning on real data | 178–420 s per outer fold versus 2.4–6.8 s for one unpenalised forest, about 60–90 times |

**Attribution.** The 60–90× figure is the cost of the **whole nested grid search** (nine (λ, γ) cells with a 3-fold inner CV, plus the refit), not of node-level bootstrapping alone. The bootstrapping overhead itself is the 3–5× factor over L0 at B = 10, and it is skipped entirely when γ = 0.

Timings come from parallel runs on one machine and vary between identical configurations (for example 16.6 s versus 38.7 s for Stab at ρ = 0 and ρ = 0.9), so only orders of magnitude are reliable. No compute-matched comparison (giving L0 the extra budget in trees or candidates) has been run.

**Cost-benefit at this milestone.** The extra cost bought no measurable predictive gain and no confirmed selection-precision gain. At present the stability term is not justified on accuracy grounds; its justification must come from a regime where it demonstrably improves selection stability.

### 4.4 Threats to validity

- Five replicates or folds, no hypothesis tests, and cells that share seeds.
- One correlation structure (block of four) and one signal strength (a = 2.0); ρ is confounded with signal strength in f_oblique and f_mixed.
- Two penalty settings on synthetic data and a 3 × 3 grid on real data. Four of five requested datasets were evaluated, and controls were not run on real data.
- The "relevant fraction" counts correlated proxies as irrelevant, and root diagnostics are conditional on an accepted split, so they measure precision, not recall.
- Stability is defined conditionally on a node-local fixed candidate bank, and assessment data are reused down the tree.
- For p ≤ 9 (diabetes, breast-w) the pool rule selects 8 of 8 or 8 of 9 features per node, so feature subsampling gives much less diversity than the √p rule of the RF baseline. Oblique-versus-RF differences on those datasets mix split type and feature randomisation.
- The exploratory diagnostics (π̂-reordering, π̂ statistics, Murphy decomposition, recalibration) use one dataset and one seed.

---

## 5. Sprint 2 Research Roadmap (Next Two Weeks)

### 5.1 Priority order

| Priority | Workstream | Output |
|---|---|---|
| 1 | Reporting: raw and recalibrated Brier plus Murphy decomposition for all models; reliability diagram | Tables and figure for the paper |
| 2 | Stability-only setting (λ = 0), larger γ, B = 30–50, 20+ replicates, both controls, paired analysis on `results_raw.csv` | Decisive test of feature-specific signal |
| 3 | High-dimensional redundant-block scenario | Test in the regime where the idea could help |
| 4 | Relaxed depth limits | Separate penalty effects from tree-size limits |
| 5 | Continuous joint regularisation (elastic-net style) | Candidate fix for probability compression |
| 6 | Integrated leaf-probability recalibration | Protocol compatible with penalised trees |

### 5.2 High-dimensional regimes (p ≫ n)

- Start with a redundant-block design such as p = 200, n = 200, blocks of 10 features at ρ = 0.9; then extend to p ≥ 1,000.
- Cost warning: with B = 10 and the current implementation, fit time grows with the pool and candidate bank, so budget runs and consider fewer trees for screening.
- Fix the pool rule deliberately: max(8, ⌈√p⌉) gives pools of about 32 at p = 1,000, which changes pool-coverage probabilities for block features and must be recomputed.
- Keep the Unif and Shuf controls in every run.

### 5.3 Relaxed depth limits (max_depth > 8, unbounded trees)

- Rerun the synthetic grid and the real datasets with deeper or unbounded trees, keeping a size-matched RF at the same limits.
- Motivation: the default RF (fully grown) is stronger on credit-g and ionosphere, and the size-matched RF explains most of that gap.
- Caution: with absolute penalties and shrinking assessment samples at depth, penalty behaviour at depth will change. Log rejected nodes (reason and margin) so that the no-split rate is measured rather than inferred.

### 5.4 Elastic-net style joint regularisation

- Explore a continuous penalty combining L1/L2 terms on the direction weights with the stability term, instead of the discrete ‖w‖₀ count.
- **Hypothesis to test, not a result:** a continuous penalty may avoid the abrupt deletion of marginal splits that drives tree-size collapse, and thereby reduce probability compression. The Sprint 1 evidence shows the compression comes with tree shrinkage, but it does not show that the discrete L0 form is the cause.
- Decide consciously between absolute and node-size-scaled penalties, since this changes the method.

### 5.5 Recalibration and test additions

- Compare post-hoc logistic or isotonic recalibration on a held-out calibration split with leaf-level shrinkage corrections, always scored on separate data.
- Add the missing unit tests: π̂ independent of assessment labels; generator correlation matches ρ; `uniform` mode with equal frequencies reproduces `feature` mode.
- Housekeeping before sharing the repository: declare `pandas`, `matplotlib`, `seaborn` as an `experiments` extra and commit `uv.lock`; update `README.md` and `docs/tested_environment.md`; add `.gitattributes`; set `error_score="raise"`; record skipped datasets in `openml_config.json` and investigate the `banknote-authentication` load failure; decide on seed derivation (and re-run everything if changed).

### 5.6 Milestone 2 decision rule (proposal)

Continue developing the stability term in its current form only if, in the high-dimensional and stability-only experiments, Stab beats **both** controls in a paired analysis across 20+ replicates. Otherwise, reposition the contribution as a characterisation of when node-wise stability penalties reduce to generic shrinkage.

---

## Appendix A. Source Materials

- `METHODOLOGICAL_ASSESSMENT.md`: code audit, exploratory diagnostics, review findings.
- `EXPERIMENTAL_RESULTS.md`: full tables for synthetic and OpenML experiments.
- `openml_config.json`, `openml_raw.csv`, `openml_summary.csv`, `openml_grid_summary.csv`: real-data configuration and outputs.
- Generated figures: metrics and tree size versus ρ; Brier versus ρ by boundary function; OpenML benchmark bars.

## Appendix B. Exact Reproduction Commands (from the audit)

- Initial run: default `exp_correlation.py`.
- Small-penalty run: `--sparsity-penalty 0.001 --stability-penalty 0.002 --include-controls --output-dir outputs/exp_correlation_small_penalty --n-jobs 4`.
- Real data: `exp_openml.py --n-jobs 4`; plots: `generate_plots.py`.
- Use the default ρ order `0.0 0.5 0.9` and the locked environment (`uv.lock`).
