"""ru-stylometry — interpretable stylometric features for Russian text.

Public API:
    - :func:`extract_features` — named stylometric features of one text.
    - :func:`feature_names` and :func:`describe_features` — the feature catalogue.
    - :class:`StylometricVectorizer` — scikit-learn transformer: texts -> feature matrix.
    - :class:`StylometricClassifier` — classifier over those features with explanations.

Features are grouped by the level of analysis (characters, typography, vocabulary, syntax,
readability, repetition, morphology); see :data:`ALL_GROUPS` and :data:`DEFAULT_GROUPS`.
The submodules ``evaluation``, ``explain`` and ``perturb`` hold the quality criteria, the
explanations and the text transformations used in robustness checks.
"""

from ru_stylometry.features import (
    ALL_GROUPS,
    DEFAULT_GROUPS,
    describe_features,
    extract_features,
    feature_names,
)
from ru_stylometry.model import StylometricClassifier
from ru_stylometry.vectorizer import StylometricVectorizer

__all__ = [
    "ALL_GROUPS",
    "DEFAULT_GROUPS",
    "StylometricClassifier",
    "StylometricVectorizer",
    "describe_features",
    "extract_features",
    "feature_names",
]
__version__ = "0.1.0"
