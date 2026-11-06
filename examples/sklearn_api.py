"""Usage demonstration, NOT an experiment or a reported benchmark."""
from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from stable_oblique_forests import (
    ObliqueRandomForestClassifier,
    make_ablation_models,
)

X, y = make_classification(n_samples=300, n_features=10, random_state=7)

clf = make_pipeline(
    StandardScaler(),
    ObliqueRandomForestClassifier(
        n_estimators=12,
        sparsity_penalty=0.01,
        stability_penalty=0.01,
        n_bootstraps=5,
        min_samples_split=30,
        min_samples_leaf=8,
        min_assessment_samples=8,
        random_state=7,
    ),
)
clf.fit(X, y)
print("classes:", clf.classes_)
print("probabilities:", clf.predict_proba(X[:3]))

models = make_ablation_models(n_estimators=5, random_state=7)
print("available configurations:", list(models))

search = GridSearchCV(
    clf,
    {"obliquerandomforestclassifier__sparsity_penalty": [0.0, 0.01]},
    cv=2,
)
search.fit(X, y)
print("chosen penalty:", search.best_params_)
