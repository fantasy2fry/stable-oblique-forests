# Milestone 1 Checkpoint Report

**Project:** Local Feature-Selection Stability in Oblique Random Forests  
**Authors:** Frydrysiak & Kukla  
**Date:** 10 November 2026  
**Status:** Interim research checkpoint — revised to distinguish completed experiments from subsequent code changes

---

## 1. Executive Summary

### 1.1 Objective and scope

This milestone establishes a research implementation of sparse oblique classification forests with node-local feature-selection stability regularization. It also provides preliminary synthetic and real-data experiments, diagnostic controls, and a plan for a more conclusive evaluation.

The main research question is whether penalizing features that are selected inconsistently across bootstrap resamples improves predictive performance or feature-selection behavior **beyond ordinary sparsity regularization**.

The present document is a checkpoint, not a final empirical or theoretical validation. In particular, the numerical results reported below belong to the original pilot runs; changes made later to the OpenML protocol do not retroactively update those results.

### 1.2 Principal findings

1. **No predictive advantage in the synthetic pilot.** Across nine synthetic settings, the stability-aware forest had an average Brier-loss increase of approximately **0.0188** relative to the unpenalized oblique forest; the $L_0$-only variant increased it by approximately **0.0038**. These are equal-weight averages of cell means, **not differences relative to the oracle**. Penalties were fixed rather than tuned.
2. **Strong reduction in tree size.** Stability regularization substantially reduced the number of accepted splits. This can improve sparsity diagnostics while simultaneously producing overly simple trees.
3. **Weak evidence for feature-specific stability information.** Uniform-penalty and shuffled-frequency controls closely matched the proposed variant in the synthetic pilot. This suggests that, in the tested regime, much of the effect may arise from additional complexity shrinkage rather than informative feature-specific frequencies.
4. **A possible calibration mechanism.** Predictive ranking changed relatively little in several settings while Brier loss worsened. Probability miscalibration is a plausible explanation, but the exploratory recalibration result requires a reproducible, leakage-free evaluation before it can support a firm conclusion.
5. **A stronger real-data comparison is now implemented.** The OpenML script includes separate hyperparameter searches for an $L_0$-only forest and an $L_0$-plus-stability forest. The earlier four-model OpenML results must not be presented as results of this updated five-model protocol.

**Interpretation:** These are preliminary negative results under specific modeling and tuning choices. They neither establish a benefit of stability regularization nor rule out benefits in high-dimensional, low-sample, or strongly redundant settings.

---

## 2. Method and Implementation Review

### 2.1 Split-selection objective

For a candidate split $c$ at node $v$, the implemented selection criterion is

$$
S_v(c)=\widehat{\mathcal L}_{v,H}(c)
+\lambda\|w_c\|_0
+\gamma\sum_{j\in\mathrm{supp}(w_c)}\left(1-\widehat\pi_{v,j}\right).
$$

Here, $w_c$ is a sparse projection direction, $\widehat{\mathcal L}_{v,H}(c)$ is the mean assessment-sample Brier loss, $\lambda\ge 0$ controls sparsity, and $\gamma\ge 0$ controls the additional stability penalty. The final term is a **sum over selected features**, not an average.

The implementation generates a finite bank of sparse directions with nonzero coefficients $\pm1/\sqrt{k}$ for support size $k$. For each direction, `optimize_threshold()` chooses a threshold by minimizing construction-sample Brier loss with Laplace-smoothed child probabilities and a minimum-child-size constraint.

The final selection compares each penalized split score with the Brier loss of predicting the parent's smoothed probability. A split is accepted only if it strictly improves this no-split alternative.

### 2.2 Construction, assessment, and bootstrap samples

Each tree first reserves a disjoint assessment pool $H$ and a construction pool $A$. When tree-level bootstrapping is enabled, it resamples the construction observations only. At node $v$, the local feature-selection frequency is estimated as

$$
\widehat\pi_{v,j}
=\frac{1}{B_{v,\mathrm{valid}}}
\sum_{b\in\mathcal B_{v,\mathrm{valid}}}
\mathbb{1}\!\left\{j\in\mathrm{supp}(w_v^{(b)})\right\},
$$

where $\mathcal B_{v,\mathrm{valid}}$ contains bootstrap repetitions with a valid candidate split. Each bootstrap repetition reuses the node's fixed direction bank, refits candidate thresholds on its resampled construction data, and selects a winner using training Brier loss plus the $L_0$ penalty. The no-split option is excluded when estimating frequencies but included in the final node decision.

This gives a **conditional stability estimate for a fixed candidate bank**. It is not an unconditional selection probability over independently regenerated banks. If no bootstrap repetition yields a valid split, the implementation removes the stability contribution at that node.

The assessment labels are not used to estimate these bootstrap frequencies. However, assessment observations are routed to descendant nodes after influencing ancestor split selection. Therefore the assessment subsets are not fully independent of the tree-building history, and this design alone does not establish a generalization guarantee.

### 2.3 Implemented variants

| Variant | Sparsity penalty | Stability penalty | Purpose |
|---|---|---|---|
| Unpenalized oblique | $\lambda=0$ | $\gamma=0$ | Common custom-model baseline |
| $L_0$ only | $\lambda>0$ | $\gamma=0$ | Isolate sparsity regularization |
| Stability only | $\lambda=0$ | $\gamma>0$ | Isolate the stability term |
| $L_0$ + stability | $\lambda>0$ | $\gamma>0$ | Proposed criterion |
| Uniform stability control | $\lambda\ge0$ | Uniformized frequencies | Test generic node-level shrinkage |
| Shuffled stability control | $\lambda\ge0$ | Permuted frequencies | Test feature specificity |

All custom variants share the same split-generation and tree-building code. With $\gamma=0$, local bootstrap estimation is skipped. Tree-specific and node-specific random seeds support paired comparisons until the fitted structures diverge. The reference Random Forest is the scikit-learn implementation, not a reimplementation of SPORF.

### 2.4 Implementation checks and remaining risks

Code inspection indicates that the three objective terms, the no-split comparison, and the separation of construction-data frequency estimation from assessment-data scoring are consistent with the stated design. Existing tests exercise estimator interfaces, parameter checks, reproducibility, parallel execution, split optimization, and several stability-related invariants.

The original checkpoint recorded **32 passing tests** in a local run. This report revision does not claim a new test execution or a formal proof of correctness. The following checks remain important:

- **Numerical split consistency:** the midpoint threshold currently used between adjacent projected floating-point values can round to the upper endpoint. This can make the executed `<=`/`>` partition differ from the cut evaluated during threshold optimization. Replace it with a threshold representation guaranteed to preserve the partition, and add a regression test.
- **Brute-force threshold oracle:** compare optimized thresholds and losses with exhaustive enumeration on small random datasets, including tied values and bootstrap multiplicities.
- **End-to-end assessment isolation:** supplement the fixed-bank frequency test with an integration test exercising the actual tree-building implementation.
- **Accepted versus rejected nodes:** diagnostics currently emphasize accepted splits, limiting direct estimation of the no-split rate.

The package is a correctness-oriented research prototype, not a performance-optimized production forest. It supports binary classification and a scikit-learn-style estimator interface; it does not claim faithful reproduction of published SPORF.

---

## 3. Experimental Protocols

### 3.1 Synthetic correlation study — completed pilot

The synthetic pilot used $n=1{,}000$ training observations, $p=50$ features, $20{,}000$ test observations, and an equicorrelated block $X_1,\ldots,X_4$ with $\rho\in\{0,0.5,0.9\}$. The remaining 46 features were independent standard-normal noise. Three data-generating functions were considered:

$$
\begin{aligned}
f_{\mathrm{axis}}(x)&=x_1,\\
f_{\mathrm{oblique}}(x)&=(x_1+x_2+x_3)/\sqrt{3},\\
f_{\mathrm{mixed}}(x)&=
\begin{cases}
x_2,&x_1\le0,\\
(x_2+x_3+x_4)/\sqrt{3},&x_1>0.
\end{cases}
\end{aligned}
$$

Labels were generated from Bernoulli probabilities $\sigma(2f(x))$. The pilot comprised five replicates per function-correlation setting, forests with 100 trees, maximum depth 8, minimum leaf size 15, minimum split size 60, and $B=10$ local bootstraps. The principal small-penalty run used **$\lambda=0.001$ and $\gamma=0.002$**, with uniform and shuffled controls enabled. No penalty tuning was performed in this study.

The same random draws were paired across competing models within a setting. The oracle reference is the Brier loss obtained from the known data-generating probabilities, not an attainable fitted-model baseline.

**Important confounding factor:** for the oblique and mixed functions, varying $\rho$ also changes the distribution and effective strength of the signal. Therefore differences across correlation levels cannot be attributed exclusively to feature redundancy. A follow-up should normalize the signal variance across $\rho$.

### 3.2 OpenML benchmark — original results and updated protocol

The original preliminary benchmark covered 15 binary OpenML datasets, with minority-class positive labels, stratified five-fold outer cross-validation, and three-fold inner cross-validation for model selection. Larger datasets were capped at 3,000 observations by stratified subsampling. One deterministic seed was used per outer fold.

The **original results** corresponded to four model families: default Random Forest, depth/leaf-matched Random Forest, unpenalized oblique forest, and jointly tuned oblique forest. The current `exp_openml.py` has since been extended to evaluate five families:

1. `RF default`
2. `RF size-matched` (matched on selected depth and leaf constraints, not on every effective-sample-size property)
3. `Oblique baseline`
4. `Oblique L0 tuned` ($\gamma=0$)
5. `Oblique L0+stability tuned` (both penalties eligible for tuning)

The two tuned oblique variants have separate `RandomizedSearchCV` procedures, each using 25 sampled configurations per outer fold by default. The full variant tunes $(\lambda,\gamma,\texttt{max\_depth},\texttt{min\_samples\_leaf})$, whereas the $L_0$-only variant fixes $\gamma=0$. The penalty search distributions include zero as well as positive values.

**Result-status distinction:** the numerical findings from the earlier OpenML run are historical pilot observations. They must not be interpreted as a completed evaluation of the new five-model comparison. In particular, the incremental benefit of stability relative to a separately tuned $L_0$ forest on real data remains to be established from the updated run's saved results.

---

## 4. Preliminary Results

### 4.1 Synthetic Brier loss

The following table reproduces the **original small-penalty pilot**. Entries are mean test Brier losses across five replicates in each cell; lower is better.

| Function | $\rho$ | RF | Oblique | $L_0$ | $L_0$ + Stability | Uniform | Shuffled | Oracle |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Axis | 0.0 | 0.1817 | 0.1807 | 0.1862 | 0.2117 | 0.2118 | 0.2116 | 0.1507 |
| Axis | 0.5 | 0.1752 | 0.1769 | 0.1798 | 0.1924 | 0.1932 | 0.1929 | 0.1506 |
| Axis | 0.9 | 0.1608 | 0.1630 | 0.1663 | 0.1767 | 0.1779 | 0.1773 | 0.1505 |
| Oblique | 0.0 | 0.1951 | 0.2040 | 0.2062 | 0.2174 | 0.2184 | 0.2181 | 0.1513 |
| Oblique | 0.5 | 0.1442 | 0.1489 | 0.1539 | 0.1708 | 0.1719 | 0.1696 | 0.1197 |
| Oblique | 0.9 | 0.1138 | 0.1158 | 0.1199 | 0.1366 | 0.1372 | 0.1371 | 0.1066 |
| Mixed | 0.0 | 0.1987 | 0.2058 | 0.2093 | 0.2213 | 0.2223 | 0.2225 | 0.1512 |
| Mixed | 0.5 | 0.1654 | 0.1682 | 0.1721 | 0.1875 | 0.1873 | 0.1873 | 0.1367 |
| Mixed | 0.9 | 0.1370 | 0.1389 | 0.1426 | 0.1574 | 0.1576 | 0.1569 | 0.1279 |

Averaging the nine cell means with equal weights gives:

| Comparison | Mean Brier difference |
|---|---:|
| $L_0$ minus unpenalized Oblique | **+0.0038** |
| $L_0$ + Stability minus unpenalized Oblique | **+0.0188** |

These are descriptive differences across experimental cells, not confidence intervals, significance tests, or estimates of the effects of optimally tuned penalties. In this pilot the stability-aware model worsened Brier loss relative to both unpenalized and $L_0$-only oblique forests in every cell.

### 4.2 Uniform and shuffled controls

The pilot report found very small performance differences between feature-specific stability, uniform stability, and shuffled-frequency controls. Stability-aware Brier loss was lower than the uniform control in eight of nine cells, by roughly 0.0006 on average, and lower than the shuffled control in five of nine cells, by roughly 0.0002 on average. These differences were smaller than the reported replicate-level variability.

This supports a **limited interpretation**: under this particular design and parameter setting, the extra penalty behaves largely like generic complexity shrinkage. It does not prove that feature-specific bootstrap frequencies are uninformative in other regimes. Frequency resolution also depends on the number of valid bootstrap repetitions, which may be smaller than the requested $B$.

### 4.3 Tree complexity and feature-selection diagnostics

The original experiments indicated substantial reductions in average tree size as penalties increased:

| Penalties $(\lambda,\gamma)$ | Unpenalized nodes/tree | $L_0$ nodes/tree | Stability nodes/tree |
|---|---:|---:|---:|
| $(0.003, 0.007)$ | 30.6–34.2 | 6.6–13.4 | **1.6–4.4** |
| $(0.001, 0.002)$ | 30.6–34.2 | 23.2–29.5 | **9.2–16.4** |

Ranges summarize the settings reported in the original checkpoint; they should not be treated as uncertainty intervals. The response to penalty strength is strongly nonlinear: relatively small parameter changes may remove many internal nodes.

A high proportion of relevant features among accepted splits is not, by itself, evidence of superior feature recovery when the model accepts very few splits. Future studies should report both selection precision and recall, together with the number of accepted splits and the frequency of stopping without splitting.

### 4.4 Other predictive metrics and calibration

The original report indicated that accuracy, F1, and binary average precision (AP) often changed much less than Brier loss. This is compatible with altered probability calibration, but it does not establish calibration as the primary causal mechanism.

An exploratory one-dimensional logistic recalibration was reported to substantially reduce selected Brier gaps. **However, the checkpoint does not provide a reproducible calibration procedure or a documented independent calibration split.** Consequently, these exploratory numbers are not treated here as a confirmed quantitative result.

A defensible follow-up should use calibration data separate from the final evaluation data, provide reliability diagrams, and report Brier loss before and after calibration. Where feasible, the Brier decomposition into uncertainty, resolution, and reliability can help distinguish sources of degradation.

### 4.5 Historical OpenML observations

The earlier four-family OpenML pilot suggested that the default Random Forest was frequently competitive or better in Brier loss and that hard depth/leaf constraints influenced the comparison. These are qualitative observations from the previous protocol, not updated results for the five-model benchmark.

The label `RF size-matched` should be read narrowly: matching selected hyperparameter limits does **not** equalize construction-sample usage, candidate-search budgets, or all aspects of model complexity. Conclusions attributing the full performance gap to tree size alone would therefore be too strong.


### 4.6 Figures from the original checkpoint

The original figure references are retained below in their original order and with their original relative paths. These figures document the earlier pilot experiments. In particular, the OpenML figure is **not** evidence from the updated five-model protocol. The images render when the corresponding files exist at the paths shown relative to this report in `docs/`.

![Brier Loss vs Correlation Rho](../outputs/exp_correlation/brier_vs_rho.png)

*Figure 1. Test Brier loss versus feature correlation for the three data-generating functions. The oracle curve uses the known data-generating probabilities; lower Brier loss is better.*

![Selection Diagnostics vs Correlation Rho](../outputs/exp_correlation/selection_vs_rho.png)

*Figure 2. Feature-selection diagnostics and average tree size versus correlation for the original synthetic pilot. The Random Forest line is shown only for tree size because oblique support diagnostics do not apply directly to axis-aligned splits.*

![OpenML Benchmark Performance](../outputs/plots/openml_benchmark.png)

*Figure 3. Historical OpenML benchmark figure from the earlier protocol. It should not be confused with results from the subsequently extended five-model benchmark.*

---

## 5. Limitations and Threats to Validity

1. **Limited replication:** five synthetic replicates per cell and one outer-CV seed per dataset/fold in the original pilot; no confirmatory paired tests.
2. **Restricted synthetic scenarios:** one four-feature correlated block, one main signal-strength setting, and a relatively favorable sample-to-relevant-feature ratio.
3. **Confounded correlation effect:** effective signal strength changes with $\rho$ for some target functions.
4. **Fixed direction bank:** estimated stability is conditional on a node's randomly generated bank, rather than stability under repeated full candidate generation.
5. **Assessment reuse:** assessment labels guide ancestor decisions and therefore cannot be treated as wholly fresh information for descendant decisions.
6. **Small effective samples in deep nodes:** tree-level bootstrap multiplicities and subsequent node bootstraps can reduce the number of distinct observations supporting a split.
7. **Unequal model resources:** the classical Random Forest and custom oblique forests differ in feature-pool rules, assessment allocation, and split-search computations.
8. **Real-data subsampling:** the 3,000-observation cap trades dataset coverage for computational feasibility; conclusions may not transfer to the full datasets.
9. **Implementation verification:** test coverage is useful but not a formal correctness guarantee; the numerical threshold edge case requires a fix and targeted regression tests.
10. **Calibration evidence:** exploratory recalibration findings require a fully documented, held-out evaluation before inclusion as confirmed results.

---

## 6. Milestone 2 Research Plan

| Priority | Workstream | Main deliverable |
|---|---|---|
| 1 | Correctness and reproducibility | Threshold edge-case fix, brute-force oracle tests, end-to-end isolation test, updated dependencies and documentation |
| 2 | Complete updated OpenML benchmark | Five-model nested-CV results including separately tuned $L_0$ baseline and selected hyperparameters |
| 3 | High-dimensional synthetic regimes | $p/n$ sensitivity, redundant blocks, weak signals, and increased replication |
| 4 | Test feature-specific stability | Stability-only model, $B=30$–$50$, uniform/shuffled controls, paired comparisons |
| 5 | Separate calibration and structural effects | Held-out recalibration, Brier decomposition, tree-size and selection-recall diagnostics |
| 6 | Sensitivity and cost | Relaxed depth/leaf constraints, compute-matched comparisons, runtime and memory measurements |

### 6.1 High-dimensional and redundant-feature experiments

Start with controlled settings such as $n=200$, $p=200$, correlated blocks of ten variables, and $\rho=0.9$; extend to $p\ge1{,}000$ if computationally feasible. Include both genuinely relevant redundant variables and correlated decoys. Vary $n$, signal strength, and the number of relevant features separately.

For oblique targets, normalize signal variance across correlation settings to avoid conflating correlation with signal-to-noise ratio.

### 6.2 Stability-specific ablation

Run paired comparisons of the feature-specific, uniform, and shuffled penalties at fixed $\lambda$ and $\gamma$; also evaluate the stability-only case $\lambda=0$. Increase $B$ and use at least 20 replicates for primary synthetic comparisons, with uncertainty intervals based on paired differences.

Interpret any advantage over the uniform and shuffled controls as evidence specific to the tested distribution and configuration, not as a universal superiority claim.

### 6.3 Probability estimation and alternative regularization

Examine whether the decrease in tree size causes underconfident probabilities, using held-out recalibration and calibrated versus uncalibrated scores. Consider adaptive or node-size-aware scaling of penalties as a direct extension of the existing objective.

A conventional elastic-net penalty is **not an immediately informative modification** of the current fixed-magnitude projection bank. For $k$ nonzero coefficients of magnitude $1/\sqrt{k}$, $\|w\|_1=\sqrt{k}$ and $\|w\|_2^2=1$. Meaningful continuous weight regularization would first require a different direction-parameterization or optimization procedure.

### 6.4 Decision criterion for continuing the method

The next milestone should prioritize whether feature-specific stability delivers a measurable improvement **beyond a well-tuned $L_0$ baseline and beyond both controls**, rather than comparing only against an unpenalized forest. Report predictive quality, stability/selection diagnostics, and compute cost jointly. If no consistent benefit emerges across targeted regimes, report the negative result and investigate alternative formulations rather than increasing complexity without supporting evidence.

---

## 7. Reproducibility and Source Files

### 7.1 Main code

- `src/stable_oblique_forests/_splits.py` — random sparse directions, candidate thresholds, Brier losses.
- `src/stable_oblique_forests/tree.py` — node construction, local bootstrap frequencies, split selection and diagnostics.
- `src/stable_oblique_forests/forest.py` — ensemble fitting and prediction.
- `src/stable_oblique_forests/models.py` — model and control configurations.
- `tests/` — implementation and estimator tests.
- `experiments/exp_correlation.py` — synthetic pilot, CSV outputs and figures.
- `experiments/exp_openml.py` — nested-CV benchmark, CSV outputs and figures.

### 7.2 Reproduction commands

Install the project and its experiment dependencies:

```bash
uv sync --extra dev --extra experiments
uv run python -m pytest -q
```

Reproduce the original small-penalty synthetic protocol:

```bash
uv run python experiments/exp_correlation.py \
    --sparsity-penalty 0.001 \
    --stability-penalty 0.002 \
    --include-controls \
    --n-jobs 4 \
    --output-dir outputs/exp_correlation_small_penalty
```

Regenerate its figures without retraining:

```bash
uv run python experiments/exp_correlation.py \
    --plots-only \
    --output-dir outputs/exp_correlation_small_penalty
```

Run the **updated** five-model real-data benchmark:

```bash
uv run python experiments/exp_openml.py --n-jobs 4
```

The synthetic script writes `results_raw.csv`, `summary.csv`, `summary.json`, and figures such as `brier_vs_rho.png`, `selection_vs_rho.png`, and `correlation_metrics.png`. The OpenML script writes per-fold results, summary tables, search trials, a configuration file, and an OpenML benchmark figure.

Output directories may be excluded from version control. Consequently, locally generated figures and CSV files should not be presented as repository-hosted artifacts unless explicitly committed or otherwise shared. The original figures and historical OpenML summary should be regenerated from the exact pilot configurations before citation in a paper.

### 7.3 Provenance of reported results

- **Section 4.1:** numerical cell means reproduced from the original Milestone 1 checkpoint; average differences recomputed from those nine displayed cells.
- **Sections 4.2–4.4:** descriptive pilot findings and caveats; the exploratory recalibration result is not treated as confirmed.
- **Section 4.5:** historical qualitative OpenML observations from the earlier four-model evaluation.
- **Updated five-model OpenML results:** not reported in this revision because the revised benchmark's completed results were not supplied or independently verified.
- **Figures:** the three Markdown image references in Section 4.6 are preserved verbatim from the original report. Their rendering depends on the corresponding image files being available at the specified relative paths.

**Checkpoint conclusion:** Milestone 1 provides a coherent implementation and useful preliminary negative evidence. The next scientific step is to validate numerical correctness, complete the separately tuned $L_0$ comparison, and test whether feature-specific stability improves performance or selection behavior in regimes where redundant features and limited sample size create a genuine stability problem.

````
