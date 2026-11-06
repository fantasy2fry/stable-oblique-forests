# Quality assurance

## Checks supplied

- Candidate bank support sizes and per-node exposure accounting.
- Threshold search, Laplace probabilities and minimum child sizes.
- Empty/constant-projection and pure-class handling.
- Model prediction shapes, probabilities, class labels and path traversal.
- No local bootstrap work when `gamma=0`.
- Deterministic results for fixed seeds, including across `n_jobs=1` and `n_jobs=2`.
- `get_params`, `set_params`, `clone`, `Pipeline`, and `GridSearchCV`.
- Pickle round trip and repeated `fit`.
- Three stability weighting modes: feature-level, uniform and shuffled.
- scikit-learn built-in estimator compatibility suite.

## Running checks

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
python - <<'PY'
from sklearn.utils.estimator_checks import check_estimator
from stable_oblique_forests import ObliqueTreeClassifier, ObliqueRandomForestClassifier
check_estimator(ObliqueTreeClassifier())
check_estimator(ObliqueRandomForestClassifier())
print('Both sklearn estimator check suites passed')
PY
```

These checks verify expected API and behavior, not predictive superiority, statistical independence of adaptively selected assessments, or fidelity to external papers.
