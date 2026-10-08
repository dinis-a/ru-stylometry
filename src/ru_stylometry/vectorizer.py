"""scikit-learn compatible transformer that turns texts into stylometric feature vectors."""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from joblib import Parallel, delayed, effective_n_jobs
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from ru_stylometry.features import (
    Groups,
    check_max_words,
    compute_features,
    feature_names,
    resolve_groups,
)

# Several chunks per worker keep all workers busy when documents differ a lot in length.
_CHUNKS_PER_WORKER = 8


def _compute_rows(
    texts: Sequence[object], groups: Tuple[str, ...], max_words: Optional[int]
) -> List[List[float]]:
    return [list(compute_features(text, groups, max_words).values()) for text in texts]


class StylometricVectorizer(TransformerMixin, BaseEstimator):
    """Convert texts into a dense matrix of named stylometric features.

    The transformer is stateless: ``fit`` only fixes the feature layout, so it can be used in a
    :class:`~sklearn.pipeline.Pipeline` or stacked next to other vectorizers, for example with
    transformer embeddings.

    Args:
        groups: Feature groups to use (see ``ru_stylometry.ALL_GROUPS``). ``None`` selects
            ``ru_stylometry.DEFAULT_GROUPS``.
        max_words: Analyse only the first ``max_words`` words of every text (``None`` means all).
        n_jobs: Number of worker processes (``None`` means 1, ``-1`` uses all cores). Every worker
            keeps its own cache of morphological analyses and loads the pymorphy3 dictionaries.

    Attributes:
        groups_: Resolved group names in registry order.
        feature_names_: Names of the output columns.

    Example:
        >>> vectorizer = StylometricVectorizer(groups=["char", "lexical"])
        >>> vectorizer.fit_transform(["Мама мыла раму.", "Раму мыла мама."]).shape
        (2, 14)
    """

    def __init__(
        self, groups: Groups = None, max_words: Optional[int] = None, n_jobs: Optional[int] = None
    ) -> None:
        self.groups = groups
        self.max_words = max_words
        self.n_jobs = n_jobs

    def fit(self, X: Optional[Iterable[str]] = None, y: object = None) -> "StylometricVectorizer":
        """Validate the parameters and fix the feature layout; ``X`` and ``y`` are ignored."""
        check_max_words(self.max_words)
        self.groups_ = resolve_groups(self.groups)
        self.feature_names_ = feature_names(self.groups_)
        return self

    def transform(self, X: Iterable[str]) -> np.ndarray:
        """Compute features for every text; blank or non-string items give zero rows."""
        check_is_fitted(self, "feature_names_")
        if isinstance(X, str):
            raise TypeError("Expected an iterable of texts, got a single string.")
        texts = list(X)
        jobs = effective_n_jobs(self.n_jobs)
        if jobs == 1 or len(texts) <= jobs:
            rows = _compute_rows(texts, self.groups_, self.max_words)
        else:
            size = -(-len(texts) // (jobs * _CHUNKS_PER_WORKER))
            chunks = [texts[start : start + size] for start in range(0, len(texts), size)]
            parts = Parallel(n_jobs=jobs)(
                delayed(_compute_rows)(chunk, self.groups_, self.max_words) for chunk in chunks
            )
            rows = [row for part in parts for row in part]
        if not rows:
            return np.empty((0, len(self.feature_names_)))
        return np.asarray(rows, dtype=float)

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        """Names of the output columns."""
        check_is_fitted(self, "feature_names_")
        return np.asarray(self.feature_names_, dtype=object)
