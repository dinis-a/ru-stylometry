"""End-to-end classifier: Russian texts in, text sources out.

:class:`StylometricClassifier` joins the feature extractor with a scikit-learn estimator and keeps
what is needed to explain a decision. The estimator is fixed by name (``"hgb"`` gradient boosting,
``"logreg"`` standardised logistic regression, ``"rf"`` random forest) and works for two classes
(for example ``"human"`` and ``"ai"``) as well as for several sources.
"""

from __future__ import annotations

from typing import IO, Any, Dict, Iterable, List, Optional, Union

import joblib
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from ru_stylometry.explain import explain_row
from ru_stylometry.features import Groups, describe_features
from ru_stylometry.vectorizer import StylometricVectorizer

ESTIMATORS = ("hgb", "logreg", "rf")


def make_estimator(
    name: str, n_classes: int, class_weight: Optional[str] = None, random_state: int = 0
) -> Any:
    """Build one of the supported scikit-learn estimators."""
    if name == "hgb":
        return HistGradientBoostingClassifier(
            max_iter=300 if n_classes == 2 else 150,
            learning_rate=0.1,
            max_leaf_nodes=31,
            class_weight=class_weight,
            random_state=random_state,
        )
    if name == "logreg":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight=class_weight),
        )
    if name == "rf":
        weight = "balanced_subsample" if class_weight == "balanced" else class_weight
        return RandomForestClassifier(
            n_estimators=100,
            min_samples_leaf=3,
            class_weight=weight,
            n_jobs=-1,
            random_state=random_state,
        )
    raise ValueError(f"Unknown estimator {name!r}; choose one of {list(ESTIMATORS)}.")


class StylometricClassifier(ClassifierMixin, BaseEstimator):
    """Classify texts by source using interpretable stylometric features.

    Args:
        estimator: ``"hgb"``, ``"logreg"`` or ``"rf"``.
        groups: Feature groups (``None`` selects ``ru_stylometry.DEFAULT_GROUPS``).
        max_words: Analyse only the first ``max_words`` words of a text, in training and in use.
        class_weight: ``None`` or ``"balanced"``; the latter makes every class count equally.
        n_jobs: Worker processes for feature extraction.
        random_state: Seed of the estimator.

    Attributes:
        classes_: Class labels in the column order of :meth:`predict_proba`.
        feature_names_: Names of the features the estimator was trained on.
        baseline_: Median of every feature on the training texts, the reference for explanations.
    """

    def __init__(
        self,
        estimator: str = "hgb",
        groups: Groups = None,
        max_words: Optional[int] = 500,
        class_weight: Optional[str] = None,
        n_jobs: Optional[int] = None,
        random_state: int = 0,
    ) -> None:
        self.estimator = estimator
        self.groups = groups
        self.max_words = max_words
        self.class_weight = class_weight
        self.n_jobs = n_jobs
        self.random_state = random_state

    def fit(self, X: Iterable[str], y: Iterable[Any]) -> "StylometricClassifier":
        """Extract features from the texts ``X`` and train the estimator on the labels ``y``."""
        labels = np.asarray(list(y))
        self.vectorizer_ = StylometricVectorizer(
            groups=self.groups, max_words=self.max_words, n_jobs=self.n_jobs
        ).fit()
        features = self.vectorizer_.transform(X)
        if len(features) != len(labels):
            raise ValueError(f"X has {len(features)} texts but y has {len(labels)} labels.")
        self.estimator_ = make_estimator(
            self.estimator, len(set(labels.tolist())), self.class_weight, self.random_state
        ).fit(features, labels)
        self.classes_ = self.estimator_.classes_
        self.feature_names_ = list(self.vectorizer_.feature_names_)
        self.feature_groups_ = [g for g, _, _ in describe_features(self.vectorizer_.groups_)]
        self.baseline_ = np.median(features, axis=0)
        return self

    def features(self, X: Iterable[str]) -> np.ndarray:
        """Feature matrix of the texts, as seen by the estimator."""
        check_is_fitted(self, "estimator_")
        return self.vectorizer_.transform(X)

    def predict(self, X: Iterable[str]) -> np.ndarray:
        """Predicted class of every text."""
        features = self.features(X)
        return self.estimator_.predict(features)

    def predict_proba(self, X: Iterable[str]) -> np.ndarray:
        """Class probabilities of every text, columns ordered as ``classes_``."""
        features = self.features(X)
        return self.estimator_.predict_proba(features)

    def explain(self, text: str, target: Optional[Any] = None, top_k: int = 10) -> Dict[str, Any]:
        """Explain the decision for one text with the effect of every feature and group.

        The effect of a feature is the fall of the log-odds of ``target`` when the feature is
        replaced by its typical (training median) value; positive values push towards the class.
        The accompanying ``probability_change`` is the same fall measured in probability, which
        is small for confident decisions.

        Args:
            text: The text to explain.
            target: Class to explain; defaults to the predicted class.
            top_k: Number of features to list, ordered by absolute effect.

        Returns:
            dict with the predicted ``label``, its ``probability``, all ``probabilities``, the
            explained ``target``, the ``top_k`` ``features`` (name, group, value, typical value,
            ``effect``, ``probability_change``) and the effect of every feature ``groups``
            (``effect``, ``probability_change``), ordered by absolute effect.
        """
        row = self.features([text])[0]
        probabilities = self.estimator_.predict_proba(row[None, :])[0]
        predicted = int(np.argmax(probabilities))
        classes = list(self.classes_)
        target_index = predicted if target is None else classes.index(target)
        explained = explain_row(
            self.estimator_.predict_proba, row, self.baseline_, target_index, self.feature_groups_
        )
        effects = explained["feature_effects"]
        changes = explained["feature_probability_changes"]
        order = np.argsort(-np.abs(effects))[:top_k]
        groups = {
            group: {
                "effect": effect,
                "probability_change": explained["group_probability_changes"][group],
            }
            for group, effect in explained["group_effects"].items()
        }
        return {
            "label": _plain(classes[predicted]),
            "probability": float(probabilities[predicted]),
            "probabilities": {str(c): float(p) for c, p in zip(classes, probabilities)},
            "target": _plain(classes[target_index]),
            "features": [
                {
                    "feature": self.feature_names_[i],
                    "group": self.feature_groups_[i],
                    "value": float(row[i]),
                    "typical": float(self.baseline_[i]),
                    "effect": float(effects[i]),
                    "probability_change": float(changes[i]),
                }
                for i in order
            ],
            "groups": dict(sorted(groups.items(), key=lambda kv: -abs(kv[1]["effect"]))),
        }

    def save(self, file: Union[str, IO[bytes]]) -> None:
        """Write the fitted model to a path or file object (pickle format)."""
        check_is_fitted(self, "estimator_")
        joblib.dump(self, file)

    @staticmethod
    def load(file: Union[str, IO[bytes]]) -> "StylometricClassifier":
        """Read a model written by :meth:`save`. Only load files from sources you trust."""
        model = joblib.load(file)
        if not isinstance(model, StylometricClassifier):
            raise TypeError(f"File does not hold a StylometricClassifier but {type(model)}.")
        return model


def _plain(value: Any) -> Any:
    """Turn numpy scalars into Python ones so results can be serialised to JSON."""
    return value.item() if hasattr(value, "item") else value
