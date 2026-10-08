"""Quality criteria for text-source classification.

Implements the measures used to judge a classifier in this project: accuracy and macro-averaged
F1 over all classes, precision and recall of every class separately, time per document, and
bootstrap confidence intervals that resample whole groups of related texts (for example texts
generated from the same source) instead of single texts.

Macro averaging gives every source the same weight, so weak recognition of rare sources is not
hidden by the large classes; micro-averaged F1 equals accuracy for single-label problems.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

Report = Dict[str, Any]


def evaluate_classification(
    y_true: Sequence[Any], y_pred: Sequence[Any], labels: Optional[Sequence[Any]] = None
) -> Report:
    """Compute the quality criteria of a single-label classifier.

    Args:
        y_true: Reference labels.
        y_pred: Predicted labels.
        labels: Classes in the order used for the confusion matrix. Defaults to the sorted union
            of the labels seen in ``y_true`` and ``y_pred``.

    Returns:
        dict with ``n``, ``accuracy``, ``macro_precision``, ``macro_recall``, ``macro_f1``,
        ``labels``, ``per_class`` (precision, recall, f1 and support of every class) and
        ``confusion_matrix`` (rows are true classes). Macro averages cover the classes that occur
        in ``y_true``.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    class_labels = list(labels) if labels is not None else sorted(set(y_true) | set(y_pred))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=class_labels, zero_division=0
    )
    present = support > 0
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision[present].mean()) if present.any() else 0.0,
        "macro_recall": float(recall[present].mean()) if present.any() else 0.0,
        "macro_f1": float(f1[present].mean()) if present.any() else 0.0,
        "labels": [str(label) for label in class_labels],
        "per_class": {
            str(label): {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, label in enumerate(class_labels)
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=class_labels).tolist(),
    }


def format_report(report: Report) -> str:
    """Render the output of :func:`evaluate_classification` as a Markdown table."""
    lines = [
        "| class | precision | recall | F1 | support |",
        "|:--|--:|--:|--:|--:|",
    ]
    for label, scores in report["per_class"].items():
        lines.append(
            f"| {label} | {scores['precision']:.3f} | {scores['recall']:.3f} | "
            f"{scores['f1']:.3f} | {scores['support']} |"
        )
    lines.append(
        f"| **macro** | {report['macro_precision']:.3f} | {report['macro_recall']:.3f} | "
        f"{report['macro_f1']:.3f} | {report['n']} |"
    )
    lines.append("")
    lines.append(f"Accuracy: {report['accuracy']:.3f}")
    return "\n".join(lines)


def latency_per_document(
    func: Callable[[List[Any]], Any], items: Sequence[Any], repeats: int = 3, warmup: int = 1
) -> Dict[str, float]:
    """Measure the processing time per document of ``func`` applied to a batch of ``items``.

    Args:
        func: Callable that processes a whole list of items (for example ``pipeline.predict``).
        items: Documents to process.
        repeats: Number of timed runs.
        warmup: Number of untimed runs before them (fills caches, loads dictionaries).

    Returns:
        dict with ``ms_per_doc_mean`` and ``ms_per_doc_min`` over the timed runs.
    """
    batch = list(items)
    if not batch:
        raise ValueError("items must not be empty.")
    for _ in range(warmup):
        func(batch)
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        func(batch)
        timings.append((time.perf_counter() - start) * 1000.0 / len(batch))
    return {"ms_per_doc_mean": float(np.mean(timings)), "ms_per_doc_min": float(np.min(timings))}


def _scores_from_confusion(matrix: np.ndarray) -> Tuple[float, float]:
    """Accuracy and macro-F1 (over classes with support) from a confusion matrix."""
    true_positive = np.diag(matrix)
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    precision = np.divide(
        true_positive, predicted, out=np.zeros_like(true_positive), where=predicted > 0
    )
    recall = np.divide(true_positive, support, out=np.zeros_like(true_positive), where=support > 0)
    denominator = precision + recall
    f1 = np.divide(
        2 * precision * recall, denominator, out=np.zeros_like(denominator), where=denominator > 0
    )
    present = support > 0
    return float(true_positive.sum() / matrix.sum()), float(f1[present].mean())


def bootstrap_ci(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    groups: Optional[Sequence[Any]] = None,
    labels: Optional[Sequence[Any]] = None,
    n_boot: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Dict[str, Dict[str, float]]:
    """Percentile bootstrap confidence intervals for accuracy and macro-F1.

    Whole groups are resampled with replacement. Texts that share a group (for example a human
    source and the machine texts generated from it) are not independent, and resampling single
    texts would make the intervals too narrow.

    Args:
        y_true: Reference labels.
        y_pred: Predicted labels.
        groups: Group id of every text; ``None`` treats every text as its own group.
        labels: Classes; defaults to the sorted union of the labels seen.
        n_boot: Number of bootstrap replicates.
        confidence: Confidence level of the interval.
        seed: Seed of the random generator.

    Returns:
        ``{"accuracy": {"estimate", "low", "high"}, "macro_f1": {...}}``.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    class_labels = list(labels) if labels is not None else sorted(set(y_true) | set(y_pred))
    k = len(class_labels)
    position, true_index, group_index = _index(y_true, class_labels, groups)
    counts = _group_counts(true_index, y_pred, position, group_index, k)
    estimate = _scores_from_confusion(counts.sum(axis=0).reshape(k, k))

    rng = np.random.default_rng(seed)
    uniform = np.full(len(counts), 1.0 / len(counts))
    samples = np.empty((n_boot, 2))
    for i in range(n_boot):
        weights = rng.multinomial(len(counts), uniform)
        samples[i] = _scores_from_confusion((weights @ counts).reshape(k, k))

    low, high = _percentile_interval(samples, confidence)
    return {
        name: {"estimate": estimate[j], "low": float(low[j]), "high": float(high[j])}
        for j, name in enumerate(("accuracy", "macro_f1"))
    }


def bootstrap_difference_ci(
    y_true: Sequence[Any],
    pred_a: Sequence[Any],
    pred_b: Sequence[Any],
    groups: Optional[Sequence[Any]] = None,
    labels: Optional[Sequence[Any]] = None,
    n_boot: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Dict[str, Dict[str, float]]:
    """Paired bootstrap interval for the difference of accuracy and macro-F1 of two classifiers.

    Both classifiers are scored on the same resampled groups of texts, so the interval reflects
    the difference between them and not the sampling noise of each. Two models may therefore
    differ significantly although their separate intervals overlap.

    Args:
        y_true: Reference labels.
        pred_a: Predictions of the first classifier.
        pred_b: Predictions of the second classifier on the same texts.
        groups: Group id of every text; ``None`` treats every text as its own group.
        labels: Classes; defaults to the sorted union of the labels seen.
        n_boot: Number of bootstrap replicates.
        confidence: Confidence level of the interval.
        seed: Seed of the random generator.

    Returns:
        ``{"accuracy": {"difference", "low", "high"}, "macro_f1": {...}}`` where the difference is
        the score of ``pred_a`` minus the score of ``pred_b``.
    """
    y_true = np.asarray(y_true)
    pred_a = np.asarray(pred_a)
    pred_b = np.asarray(pred_b)
    class_labels = (
        list(labels) if labels is not None else sorted(set(y_true) | set(pred_a) | set(pred_b))
    )
    k = len(class_labels)
    position, true_index, group_index = _index(y_true, class_labels, groups)
    counts_a = _group_counts(true_index, pred_a, position, group_index, k)
    counts_b = _group_counts(true_index, pred_b, position, group_index, k)
    estimate = np.array(_scores_from_confusion(counts_a.sum(axis=0).reshape(k, k))) - np.array(
        _scores_from_confusion(counts_b.sum(axis=0).reshape(k, k))
    )

    rng = np.random.default_rng(seed)
    uniform = np.full(len(counts_a), 1.0 / len(counts_a))
    samples = np.empty((n_boot, 2))
    for i in range(n_boot):
        weights = rng.multinomial(len(counts_a), uniform)
        samples[i] = np.array(
            _scores_from_confusion((weights @ counts_a).reshape(k, k))
        ) - np.array(_scores_from_confusion((weights @ counts_b).reshape(k, k)))

    low, high = _percentile_interval(samples, confidence)
    return {
        name: {"difference": float(estimate[j]), "low": float(low[j]), "high": float(high[j])}
        for j, name in enumerate(("accuracy", "macro_f1"))
    }


def _index(y_true: np.ndarray, class_labels: List[Any], groups: Optional[Sequence[Any]]):
    position = {label: i for i, label in enumerate(class_labels)}
    true_index = np.array([position[v] for v in y_true])
    if groups is None:
        group_index = np.arange(len(y_true))
    else:
        _, group_index = np.unique(np.asarray(groups), return_inverse=True)
    return position, true_index, group_index


def _group_counts(
    true_index: np.ndarray,
    y_pred: np.ndarray,
    position: Dict[Any, int],
    group_index: np.ndarray,
    k: int,
) -> np.ndarray:
    """Confusion counts of every group, flattened: array of shape (groups, k * k)."""
    pred_index = np.array([position[v] for v in y_pred])
    counts = np.zeros((int(group_index.max()) + 1, k * k))
    np.add.at(counts, (group_index, true_index * k + pred_index), 1.0)
    return counts


def _percentile_interval(samples: np.ndarray, confidence: float) -> Tuple[np.ndarray, np.ndarray]:
    tail = (1.0 - confidence) / 2.0 * 100.0
    low, high = np.percentile(samples, [tail, 100.0 - tail], axis=0)
    return low, high
