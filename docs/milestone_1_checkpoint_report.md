# Milestone 1 Checkpoint Report

**Project:** Local Feature-Selection Stability in Oblique Random Forests

**Authors:** Frydrysiak & Kukla

**Date:** 10 November 2026

**Status:** Interim progress checkpoint (Milestone 1), end of first research phase


---

## 1. Executive Summary and Milestone Scope

### 1.1 Purpose

This checkpoint closes the first research phase. It has three goals:

1. Establish an **audited baseline implementation** of a sparse oblique random forest whose split criterion includes a node-wise stability penalty.
2. Provide a first **empirical validation** of the stability regularisation on synthetic correlated-feature designs and on an expanded benchmark of 15 OpenML binary tasks.
3. Record **sanity checks and diagnostics** that indicate what the penalty does mechanically, so that the next phase targets regimes where it could plausibly help.

### 1.2 Core takeaways

| # | Takeaway | Strength of evidence |
|---|---|---|
| 1 | In the tested regime (p = 50, n = 1,000, one correlated block of four features; 15 real datasets) the stability-aware forest did **not** consistently improve Brier loss, accuracy, F1 or mAP over the unpenalised oblique forest. On synthetic data, its Brier loss was worse (about +0.019 excess over the oracle; L0 alone about +0.004). | 5 replicates per synthetic cell, 5 outer folds per 15 real datasets; no hypothesis tests |
| 2 | The **uniform** and **shuffled** controls reproduce the stability model's predictive performance, tree size and support size. The feature-specific frequencies $\widehat{\pi}$ change which near-tied candidate wins (about one third of trees at $\gamma = 0.002$), but with no systematic predictive gain. | Synthetic only; the $\widehat{\pi}$-reordering check is exploratory (one dataset, one seed) |
| 3 | The Brier degradation is largely a **calibration** effect (smaller trees give under-confident probabilities), not a loss of ranking power. mAP is nearly unchanged, and a 1-D logistic recalibration removes most of the Brier gap. | Exploratory (one dataset, one seed per $\rho$) |
| 4 | The stability term costs 3–5 times the fit time of the L0 model. The nested randomized search over $(\lambda, \gamma, \text{max\_depth}, \text{min\_samples\_leaf})$ costs significantly more than a single fit, dominating the runtime on real datasets. | Timings from parallel runs; order of magnitude only |

These are **negative results for the tested settings**, not a refutation of the idea. The designs have few relevant features, n well above the number of relevant features, B = 10 bootstrap repetitions and a depth cap of 8. Regimes with many redundant, weakly informative features are untested.

### 1.3 What is and is not verified

- The implementation was checked against the mathematical specification, the
  research plan, `README.md` and `docs/implementation_notes.md`. The main
  algorithmic requirements of the plan are implemented; the deviations from
  the full planned evaluation are listed in §2.5 and §4.4.
- This is a Milestone 1 checkpoint, not the final study. The synthetic pilot
  uses 5 replicates per cell rather than the 20 planned for the full study.
  The OpenML benchmark uses one seed per outer fold, while the plan proposed
  two. No confirmatory hypothesis tests are reported at this stage.

---

## 2. Theoretical and Implementation Audit

### 2.1 Split criterion

For a candidate split $c$ at node $v$ the implemented score is

$$
S_v(c) = \widehat{\mathcal{L}}_{v,H}(c) + \lambda \,\|w_c\|_0 + \gamma \sum_{j \in \mathrm{supp}(w_c)} \bigl(1 - \widehat{\pi}_{v,j}\bigr),
$$

where $\widehat{\mathcal{L}}_{v,H}$ is the mean Brier loss on the assessment pool $H_v$, with child probabilities fitted on the construction pool $A_v$. Each term was checked in `tree.py::_build` (lines 264–287):

| Term | Implementation | Status |
|---|---|---|
| Loss | `assessment_loss(c, X, y, H)`, mean Brier on $H_v$ | Verified |
| L0 term | `sparsity_penalty * c.sparsity` | Verified |
| Stability term | `stability_penalty * Σ(1 − effective_frequencies[c.support])`, a **sum**, not a mean | Verified against the plan |
| No-split option | Scored as the parent's smoothed probability on $H_v$ with no penalty; a split is accepted only if `score < best_score` (strictly better) | Verified |

Penalties are **absolute** and are added to a node-local mean loss. A given $\lambda$ therefore means the same at the root and at a 25-sample leaf. This is a design property, not a bug, but it should be stated in the paper.

### 2.2 Sample partitioning, $\widehat{\pi}$ and Laplace smoothing

- **Partitioning.** The assessment pool $H$ is reserved before the tree-level bootstrap, and the bootstrap acts on the construction pool only. Descendant nodes reuse the same $H$. This is documented as a non-independence of assessment data at depth.
- **Frequencies $\widehat{\pi}$.** `_bootstrap_selection_frequencies` (lines 186–208) resamples $A_v$ only, refits thresholds for the same fixed candidate bank, selects the winner by bootstrap training Brier plus $\lambda \cdot k$, and counts the winner's support. $\widehat{\pi}$ is therefore **estimated on $A_v$**; the final loss is **evaluated on $H_v$**. No-split is excluded from the frequency estimate and included in the final choice.
- **Laplace smoothing.** Both node probabilities and child probabilities use $(n_1+1)/(n_0+n_1+2)$. Because the candidate's child probability and the child node's probability are computed on the same partition of $A_v$, they coincide. With tree-level bootstrap, $n_0$ and $n_1$ are bootstrap-weighted counts.
- **Pool rule.** The feature pool has size $\max(8, \lceil\sqrt{p}\rceil)$, capped at $p$, drawn uniformly without replacement. This confirms the pool-coverage probabilities used in the results (about 0.51 for at least one of four block features, about 0.16 for $x_1$ alone).

### 2.3 The $\gamma = 0$ fast path

When `stability_penalty == 0`, the node-level bootstrap is skipped (line 248), the instability sum is skipped (lines 275–278) and `n_bootstrap_fits_` stays 0. Descendant seeds are drawn before any bootstrap draws, so skipping the bootstrap does not change the candidate banks of descendants. This is covered by `test_gamma_zero_skips_bootstrap` and `test_gamma_zero_is_independent_of_n_bootstraps`.

### 2.4 Independence of $\widehat{\pi}$ from assessment labels, and test status

- **Code inspection (done):** $\widehat{\pi}$, threshold fitting and probability fitting use $A_v$ only. No label from $H_v$ enters $\widehat{\pi}$. No leakage was found. Preprocessing is fitted inside the pipeline on training folds, and outer test folds are never used for tuning.
- **Unit tests:** `uv run python -m pytest -q` passed all 32 tests in 13.44
  seconds. An earlier development run also reported that the scikit-learn
  `check_estimator` suite passed. Forests were reported identical for
  `n_jobs=1` and `n_jobs=2`.
- **Test coverage qualification:** `test_pi_hat_independent_of_assessment_labels`
  verifies the equivalent bootstrap-frequency procedure using a fixed bank.
  A higher-level test that exercises the complete tree-building path would
  still strengthen this guarantee.

### 2.5 Conformance with the research plan

The core method conforms to the plan:

1. $\widehat{\pi}$ is estimated from bootstrap resamples of $A_v$, using
   training Brier loss plus the L0 penalty to select each bootstrap winner.
2. The stability term is a **sum** of $(1-\widehat\pi_j)$ over the candidate
   support.
3. The implementation uses the planned defaults: support sizes
   {1, 2, 4, 8}, 8 directions per size, B = 10, assessment fraction 0.2,
   minimum leaf 15, minimum split 60 and depth 8.
4. The OpenML tuning protocol uses `RandomizedSearchCV`, with 25 sampled
   configurations per outer fold. The two penalty distributions are
   log-uniform on $[10^{-4}, 10^{-1}]$ with a 20% probability of exactly
   zero.
5. The implemented penalties are absolute node-local penalties, as specified
   by the current objective.

The full evaluation plan is not yet complete: the synthetic study has not
covered all planned values of $p$, $n$ and signal strength; the planned 20
replicates and two seeds per outer fold have not been run; and the planned
diversity, compute-matched and formal statistical analyses remain future work.

### 2.6 Reproducibility and design strengths

- Node-local RNG streams (child seeds drawn first) mean that penalised variants and controls share partitions, banks and bootstrap draws until their decisions first differ. This paired design is what makes the control comparison possible.
- Experimental hygiene: common random numbers across models, an oracle Brier reference, nested CV with preprocessing inside the pipeline, a size-matched random-forest baseline, minority class as positive class, and raw plus summary outputs with a config file recording library versions (Python 3.12.10, NumPy 2.5.4, scikit-learn 1.9.1).

### 2.7 Known limitations of this checkpoint

- **Synthetic seed identity:** `exp_correlation.py` derives data seeds partly
  from the position of $\rho$ in the command-line list. Reordering or
  subsetting `--rho` therefore changes the data seeds. This does not affect
  the reported run, but should be fixed before the extended study.
- **Outer-fold seeds:** the current OpenML run uses one deterministic seed per
  outer fold. Two independent seeds per fold, as proposed in the plan, are
  still missing.
- **Search protocol:** the current OpenML experiment intentionally uses
  randomized search rather than the earlier pilot grid. The report's results
  and configuration files should be read as describing the randomized-search
  protocol.
- **Diagnostics log accepted splits only** (`node_diagnostics_`), so the no-split rate that drives tree-size collapse is inferred, not measured.
- **Bootstrap multiplicities:** with tree-level bootstrap about 63 % of observations are unique, so a "15-sample" leaf may hold about 9–10 distinct observations, and the node-level resampling is a bootstrap of a bootstrap.
- **Dataset truncation:** the OpenML command defaults to a stratified
  `max_samples=3000` cap for larger datasets. This is recorded in
  `openml_config.json` and is a computational compromise, not the full
  dataset protocol described in the research plan.

---

## 3. Empirical Results and Plot Analysis

### 3.1 Protocols

**Synthetic.** $n = 1,000$ training points, $p = 50$ standard-normal features. Features $x_1$–$x_4$ form an equicorrelated block with correlation $\rho \in \{0.0, 0.5, 0.9\}$; the other 46 are noise. Labels are Bernoulli(sigmoid(2f(x))) with $f_{axis} = x_1$, $f_{oblique} = (x_1+x_2+x_3)/\sqrt{3}$, and $f_{mixed}$ piecewise ($x_2$ if $x_1 \leq 0$, else $(x_2+x_3+x_4)/\sqrt{3}$). Five replicates per (function, $\rho$) cell, 20,000 test points, 100 trees, depth $\leq 8$, minimum leaf 15, minimum split 60, B = 10. Penalised models use **$\lambda = 0.001, \gamma = 0.002$**. 

Models: random forest (RF, same size limits), unpenalised oblique (Obl), oblique + L0 (L0), oblique + L0 + stability (Stab), uniform-weight control (Unif), shuffled-frequency control (Shuf).

**OpenML.** 15 binary datasets (e.g., breast-w, diabetes, credit-g, adult, etc.) with the positive class defined as the minority class. Evaluated via a stratified 5-fold outer CV (1 repeat). The tuned oblique forest chooses $(\lambda, \gamma, \text{max\_depth}, \text{min\_samples\_leaf})$ using a 25-iteration `RandomizedSearchCV` (3-fold inner-CV Brier loss). Baselines: default RF, size-matched RF, unpenalised oblique forest. **The Unif and Shuf controls were not run on real data.**

### 3.2 Synthetic results: Brier loss (mean over 5 replicates, replicate sd 0.001–0.009)

| Function | $\rho$ | RF | Obl | L0 | Stab | Unif | Shuf | Oracle |
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

**Reading of the table.**

- The ordering RF $\lesssim$ Obl < L0 < Stab holds in all nine cells. L0 costs 0.002–0.006 over Obl; Stab costs 0.013–0.031 over Obl.
- The penalty cost is roughly constant in $\rho$ (L0 about +0.004; Stab +0.020 / +0.019 / +0.018). We find no evidence that stability becomes more useful as redundancy increases in this design.
- The unpenalised oblique forest does **not** beat the random forest even on oblique and mixed boundaries (RF better by 0.007–0.009 at $\rho = 0$).

### 3.3 Stability versus controls

| Comparison | Brier result | Other metrics |
|---|---|---|
| Stab vs Unif | Stab lower in 8 of 9 cells, mean 0.0006, max $\| \Delta \| = 0.0012$ | Accuracy, F1, mAP differ by at most 0.006 |
| Stab vs Shuf | Mean difference 0.0002, Stab better in 5 of 9 cells | Same |

Replicate standard deviations (0.003–0.009) are larger than these differences. Tree sizes are the same, and so are support sizes. The sparsification relative to L0 is therefore reproduced by the uniform control.

### 3.4 Accuracy, F1 and mAP

These metrics barely separate the models. At $\rho = 0.9$ all six models are within 0.002 in accuracy. Within any cell the spread is at most 0.010 in mAP and 0.012 in F1. 

### 3.5 Real data benchmark (15 OpenML datasets)

Across the expanded 15-dataset OpenML benchmark, performance highlights include (see Figure 3 and `openml_benchmark_results.csv` for exact fold variances):

- **Tuned vs. Baseline Oblique:** Tuning the hyperparameters via nested randomized search yields predictions that are on par with or marginally different from the baseline oblique forest (λ=γ=0). The noisy 3-fold inner CV on Brier loss often selects zero penalties, flattening the predictive gains.
- **RF Default Dominance:** The unconstrained, default Random Forest frequently achieves the lowest Brier loss and highest F1 across datasets (e.g., noticeably outperforming others on credit-g and ionosphere).
- **The Tree Size Factor:** The `RF size-matched` control closely trails the Oblique variants. This confirms that the observed performance gap between the Oblique Forest and the Default RF is largely a consequence of the rigid tree-size limits (depth 8, min leaf 15) enforced on the oblique models, rather than an inherent weakness of the oblique splits.

### 3.6 Visual Trend Analysis of the Generated Figures

Three figures were generated: Brier loss versus $\rho$ per boundary function,
selection diagnostics versus $\rho$, and OpenML bar charts.

![Brier Loss vs Correlation Rho](../outputs/exp_correlation_small_penalty/brier_vs_rho.png)
*Figure 1: Brier loss across boundary functions and feature correlation values ($\rho \in [0.0, 0.5, 0.9]$).*
The stability curves sit above the unpenalised and L0 models in every panel. The oracle line shows how the effective signal changes with correlation for the oblique and mixed functions.

![Selection Diagnostics vs Correlation Rho](../outputs/exp_correlation_small_penalty/selection_vs_rho.png)
*Figure 2: Selection diagnostics and tree size across correlation levels. Mean ± standard error over 5 replicates.*
The stability variants show substantially smaller trees than the unpenalised model. The uniform and shuffled controls closely track the proposed model.

![OpenML Benchmark Performance](../outputs/plots/openml_benchmark.png)
*Figure 3: OpenML benchmark evaluation across 15 binary datasets. Outer-fold means, error bars are standard deviations over 5 folds.*
The expanded benchmark shows densely packed Brier bars for the oblique variants. F1 and mAP scores reveal some threshold artefacts and demonstrate that the size constraints are the primary bottleneck compared to the default Random Forest.

### 3.7 Tree-size collapse

| Setting | Unpenalised | L0 | Stab | Brier excess, Stab minus unpenalised |
|---|---|---|---|---|
| $(\lambda, \gamma) = (0.003, 0.007)$ | 30.6–34.2 nodes | 6.6–13.4 | **1.6–4.4** | +0.032 / +0.037 / +0.034 |
| $(\lambda, \gamma) = (0.001, 0.002)$ | 30.6–34.2 | 23.2–29.5 | **9.2–16.4** | +0.020 / +0.019 / +0.018 |

(Brier excess listed for $\rho = 0 / 0.5 / 0.9$.) The smaller setting removes degenerate trees, but stability-aware trees still hold only 28–48 % of the unpenalised node count. The response is highly non-linear.

---

## 4. Scientific Discussion and Critical Interpretation

### 4.1 Why the stability penalty behaves like generic complexity shrinkage

Three observations support the reading that, in this regime, the stability term acts mainly as an additional L0-type shrinkage:
1. **Controls match.** Stab, Unif and Shuf agree in predictive metrics and tree size.
2. **$\widehat{\pi}$ is a weak signal here.** $\widehat{\pi}$ takes values in multiples of 1/B = 0.1. The difference between the feature-specific and uniform penalty is smaller than the median split margin.
3. **Selection noise dominates at depth.** The standard error of a mean-Brier difference on small assessment samples outweighs penalties of order 0.001.

**Scope of the claim.** This is a statement about regimes with few relevant features, ample $n$ and $B = 10$. High-dimensional, strongly redundant problems remain untested.

### 4.2 Separating Brier degradation from ranking capability

At strong signal ($\rho = 0.9$), resolution and AP are unchanged while calibration error rises almost tenfold and the spread of predictions contracts. Smaller trees give coarser, more strongly averaged leaves, which means **under-confident probabilities**. Accuracy is unaffected, and AP depends only on ordering. 

**Recalibration check.** A one-dimensional logistic recalibration shrinks the Brier gap of the penalised forests relative to the unpenalised forest at $\rho = 0.9$ from **+0.018 / +0.035 to +0.001 / +0.004**. 

**Consequence.** Raw Brier differences between these forests are largely calibration differences. Comparisons should be reported both raw and after recalibration.

### 4.3 Computational cost-benefit

The bootstrapping overhead costs 3–5× over L0 at $B = 10$. Implementing a 25-sample `RandomizedSearchCV` on a nested loop significantly amplifies the cost, running roughly 60–90 times longer than fitting one unpenalised forest.

**Cost-benefit at this milestone.** The extra compute budget bought no measurable predictive gain and no confirmed selection-precision gain. Its justification must come from a regime where it demonstrably improves selection stability.

### 4.4 Threats to validity

- Five replicates or folds, no hypothesis tests.
- One correlation structure (block of four) and one signal strength on the synthetic side.
- Random Search across 15 datasets is subject to the variance of inner-CV selections (frequent ties at 0.0).
- Stability is defined conditionally on a node-local fixed candidate bank, and assessment data are reused down the tree.
- The OpenML results use a stratified cap of 3,000 rows for larger datasets,
  one seed per outer fold, and a manually specified set of 15 datasets rather
  than the full dataset-selection procedure from the plan.
- Diversity metrics, compute-matched baselines and formal cross-dataset
  statistical tests are not part of this checkpoint.

### 4.5 Milestone 1 conclusion

Milestone 1 delivers a working implementation with a test suite for the
proposed node-wise stability criterion, its main ablations and a substantially
reproducible preliminary evaluation on synthetic correlated features and 15
OpenML binary classification tasks. The evidence does not support a predictive advantage
for the stability penalty under the tested settings. Instead, the penalty
primarily reduces tree size and worsens raw probability calibration, while
the uniform and shuffled controls suggest that the feature-specific
frequency signal is weak in this regime.

This conclusion is intentionally limited. It does not establish that the
method cannot help in high-dimensional, low-sample or strongly redundant
settings, which are the main targets for Milestone 2. The current result is
therefore a valid negative checkpoint result and a basis for a more targeted
follow-up study, not a final rejection of the method.

---

## 5. Milestone 2 Research Roadmap

### 5.1 Priority order

| Priority | Workstream | Output |
|---|---|---|
| 1 | Reporting: raw and recalibrated Brier plus Murphy decomposition for all models; reliability diagram | Tables and figure for the paper |
| 2 | Stability-only setting ($\lambda = 0$), larger $\gamma$, $B = 30–50$, 20+ replicates, both controls | Decisive test of feature-specific signal |
| 3 | High-dimensional redundant-block scenario | Test in the regime where the idea could help |
| 4 | Relaxed depth limits | Separate penalty effects from tree-size limits |
| 5 | Continuous joint regularisation (elastic-net style) | Candidate fix for probability compression |
| 6 | Integrated leaf-probability recalibration | Protocol compatible with penalised trees |

### 5.2 High-dimensional regimes ($p \gg n$)
Start with a redundant-block design such as $p = 200$, $n = 200$, blocks of 10 features at $\rho = 0.9$; then extend to $p \geq 1,000$.

### 5.3 Relaxed depth limits (max_depth > 8, unbounded trees)
The new 15-dataset benchmark highlighted that RF default is stronger on several sets. We must test deeper or unbounded trees to isolate the effect of the stability penalty from the hard size constraints.

### 5.4 Elastic-net style joint regularisation
Explore a continuous penalty combining L1/L2 terms on the direction weights with the stability term to avoid the abrupt deletion of marginal splits that drives tree-size collapse.

### 5.5 Recalibration and test additions
Compare post-hoc logistic or isotonic recalibration. Add a higher-level
tree-path test for $\widehat{\pi}$ independence in addition to the existing
fixed-bank procedure test.

### 5.6 Milestone 2 decision rule (proposal)
Continue developing the stability term in its current form only if, in the high-dimensional and stability-only experiments, Stab beats **both** controls in a paired analysis across 20+ replicates.

---

## Appendix A. Source Materials

- Research plan: `Ensemble_project_Frydrysiak_Kukla.pdf`
- `README.md`, `docs/implementation_notes.md` and `docs/reproduce.md`
- `outputs/openml_config.json`
- `outputs/tables/openml_benchmark_results.csv`
- `outputs/tables/openml_benchmark_summary.csv`
- `outputs/tables/openml_benchmark_macro.csv`
- `outputs/tables/openml_search_trials.csv`
- Synthetic experiment outputs under `outputs/exp_correlation*/`

## Appendix B. Exact Reproduction Commands

- Synthetic pilot, reproducing the reported small-penalty tables:
  `uv run python experiments/exp_correlation.py --sparsity-penalty 0.001 --stability-penalty 0.002 --include-controls --n-jobs 4 --output-dir outputs/exp_correlation_small_penalty`
- Expanded real-data benchmark: `uv run python experiments/exp_openml.py --n-jobs 4`

Both experiment scripts write their result tables and figures during the same
run. A separate plotting step is not required for the Milestone 1 protocol.