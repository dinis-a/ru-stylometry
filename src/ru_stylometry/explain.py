"""Explanations of classifier decisions in terms of stylometric features.

Two complementary views are provided:

* **local** (:func:`explain_row`): for one text, how much does each feature, and each whole feature
  group, move the evidence for a class? A feature is replaced by its typical value (the training
  median) and the fall of the class log-odds is the *effect* of the feature. Positive effects push
  towards the class, negative ones away from it. Log-odds are used because the probability of a
  confident decision is saturated (0.998 stays 0.998 whichever single feature is changed), while
  the log-odds still show which features carry the evidence. The effects are not additive:
  correlated features share the credit, so the effect of a group is usually larger than any single
  feature of it.
* **global** (:func:`group_permutation_importance`): how much does a quality metric fall on a
  labelled set when the columns of one feature group are shuffled together?

Both work with any model that offers ``predict`` and ``predict_proba`` on a feature matrix.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from ru_stylometry.evaluation import evaluate_classification


def _log_odds(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(probability, 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def explain_row(
    predict_proba: Callable[[np.ndarray], np.ndarray],
    row: np.ndarray,
    baseline: np.ndarray,
    target: int,
    feature_groups: Sequence[str],
    individual: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """Effect of every feature and feature group on the evidence for one class.

    Args:
        predict_proba: Maps a feature matrix to class probabilities.
        row: Feature vector of the text.
        baseline: Typical value of every feature (for example the training median).
        target: Column index of the explained class in the output of ``predict_proba``.
        feature_groups: Group name of every feature, in the order of ``row``.
        individual: Columns whose effect is measured one by one (default: all). The columns of
            a block of meaningless coordinates, such as a text embedding, are best left out and
            explained as a group only.

    Returns:
        dict with ``probability`` of the class for the text, and for every feature and every group
        the fall of the class log-odds (``feature_effects`` array, ``group_effects`` dict) and of
        the class probability (``feature_probability_changes``, ``group_probability_changes``) when
        it is replaced by its baseline value. The feature arrays follow the order of
        ``individual``. Probabilities are clipped to [1e-6, 1 - 1e-6] before the log-odds are
        taken.
    """
    columns_one_by_one = list(range(row.size)) if individual is None else list(individual)
    n_features = len(columns_one_by_one)
    groups = list(dict.fromkeys(feature_groups))
    variants = np.tile(row, (1 + n_features + len(groups), 1))
    for i, column in enumerate(columns_one_by_one):
        variants[1 + i, column] = baseline[column]
    members = np.asarray(feature_groups)
    for j, group in enumerate(groups):
        columns = np.flatnonzero(members == group)
        variants[1 + n_features + j, columns] = baseline[columns]

    probabilities = predict_proba(variants)[:, target]
    odds = _log_odds(probabilities)
    return {
        "probability": float(probabilities[0]),
        "feature_effects": odds[0] - odds[1 : 1 + n_features],
        "feature_probability_changes": probabilities[0] - probabilities[1 : 1 + n_features],
        "group_effects": {
            group: float(odds[0] - odds[1 + n_features + j]) for j, group in enumerate(groups)
        },
        "group_probability_changes": {
            group: float(probabilities[0] - probabilities[1 + n_features + j])
            for j, group in enumerate(groups)
        },
    }


def group_permutation_importance(
    model: Any,
    X: np.ndarray,
    y: Sequence[Any],
    groups: Dict[str, List[int]],
    metric: str = "macro_f1",
    n_repeats: int = 3,
    seed: int = 0,
) -> Dict[str, Dict[str, float]]:
    """Fall of a quality metric when the columns of one feature group are shuffled together.

    The columns of a group are permuted with the same row order, so dependencies inside the group
    stay intact and only its link to the label and to other groups is broken.

    Args:
        model: Fitted model with a ``predict`` method.
        X: Feature matrix of a labelled evaluation set.
        y: Reference labels.
        groups: Column indices of every group.
        metric: ``"macro_f1"`` or ``"accuracy"``.
        n_repeats: Number of random permutations per group.
        seed: Seed of the random generator.

    Returns:
        ``{"baseline": score without shuffling, "groups": {group: {"drop": mean drop, "std":
        its standard deviation}}}``; groups are ordered from the most to the least important.
    """
    if metric not in ("macro_f1", "accuracy"):
        raise ValueError(f"Unknown metric {metric!r}.")
    X = np.asarray(X)
    y = np.asarray(y)

    def score(matrix: np.ndarray) -> float:
        return evaluate_classification(y, model.predict(matrix))[metric]

    baseline = score(X)
    rng = np.random.default_rng(seed)
    result: Dict[str, Dict[str, float]] = {}
    for group, columns in groups.items():
        drops = []
        for _ in range(n_repeats):
            shuffled = X.copy()
            shuffled[:, columns] = X[rng.permutation(len(X))][:, columns]
            drops.append(baseline - score(shuffled))
        result[group] = {"drop": float(np.mean(drops)), "std": float(np.std(drops))}
    ordered = dict(sorted(result.items(), key=lambda item: -item[1]["drop"]))
    return {"baseline": float(baseline), "groups": ordered}
