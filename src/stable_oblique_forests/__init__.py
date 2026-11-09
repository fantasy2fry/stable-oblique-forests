"""Stable Oblique Forests: research-grade prototypes of local stability penalties."""
from .forest import ObliqueRandomForestClassifier
from .tree import ObliqueTreeClassifier
from .models import make_ablation_models, make_random_forest_reference
from .metrics import binary_classification_metrics

__all__ = [
    "ObliqueTreeClassifier",
    "ObliqueRandomForestClassifier",
    "make_ablation_models",
    "make_random_forest_reference",
    "binary_classification_metrics",
]
__version__ = "0.1.0"