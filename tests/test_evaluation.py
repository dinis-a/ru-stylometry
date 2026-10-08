"""Tests for the evaluation helpers."""

import numpy as np
import pytest

from ru_stylometry.evaluation import (
    bootstrap_ci,
    bootstrap_difference_ci,
    evaluate_classification,
    format_report,
    latency_per_document,
)

Y_TRUE = [0, 0, 1, 1, 1]
Y_PRED = [0, 1, 1, 1, 0]


def test_binary_metrics_match_hand_computation():
    report = evaluate_classification(Y_TRUE, Y_PRED)
    assert report["n"] == 5
    assert report["accuracy"] == pytest.approx(3 / 5)
    # class 0: tp=1, predicted=2, support=2; class 1: tp=2, predicted=3, support=3
    assert report["per_class"]["0"]["precision"] == pytest.approx(1 / 2)
    assert report["per_class"]["0"]["recall"] == pytest.approx(1 / 2)
    assert report["per_class"]["1"]["precision"] == pytest.approx(2 / 3)
    assert report["per_class"]["1"]["recall"] == pytest.approx(2 / 3)
    assert report["macro_f1"] == pytest.approx((1 / 2 + 2 / 3) / 2)
    assert report["confusion_matrix"] == [[1, 1], [1, 2]]


def test_macro_f1_ignores_classes_absent_from_the_reference():
    report = evaluate_classification([0, 0, 1, 1], [0, 0, 1, 2], labels=[0, 1, 2])
    assert report["per_class"]["2"]["support"] == 0
    assert report["macro_recall"] == pytest.approx((1.0 + 0.5) / 2)


def test_macro_f1_exposes_a_rare_class_that_accuracy_hides():
    y_true = ["human"] * 98 + ["rare"] * 2
    y_pred = ["human"] * 100
    report = evaluate_classification(y_true, y_pred)
    assert report["accuracy"] == pytest.approx(0.98)
    assert report["per_class"]["rare"]["recall"] == 0.0
    assert report["macro_f1"] < 0.5


def test_string_labels_and_explicit_label_order():
    report = evaluate_classification(
        ["ai", "human", "ai"], ["ai", "ai", "ai"], labels=["human", "ai"]
    )
    assert report["labels"] == ["human", "ai"]
    assert report["confusion_matrix"] == [[0, 1], [0, 2]]


def test_format_report_is_a_markdown_table():
    text = format_report(evaluate_classification(Y_TRUE, Y_PRED))
    assert text.startswith("| class | precision | recall | F1 | support |")
    assert "| **macro** |" in text
    assert "Accuracy: 0.600" in text


def test_latency_per_document():
    result = latency_per_document(lambda batch: [len(x) for x in batch], ["текст"] * 50, repeats=2)
    assert set(result) == {"ms_per_doc_mean", "ms_per_doc_min"}
    assert 0 < result["ms_per_doc_min"] <= result["ms_per_doc_mean"]


def test_latency_requires_documents():
    with pytest.raises(ValueError, match="empty"):
        latency_per_document(len, [])


def test_bootstrap_of_perfect_predictions_is_degenerate():
    labels = [0, 1] * 50
    result = bootstrap_ci(labels, labels, n_boot=50)
    assert result["accuracy"] == {"estimate": 1.0, "low": 1.0, "high": 1.0}
    assert result["macro_f1"]["low"] == 1.0


def test_bootstrap_estimate_matches_the_point_metrics_and_lies_inside_the_interval():
    rng = np.random.default_rng(1)
    y_true = rng.integers(0, 3, size=400)
    y_pred = np.where(rng.random(400) < 0.7, y_true, rng.integers(0, 3, size=400))
    result = bootstrap_ci(y_true, y_pred, n_boot=300, seed=2)
    report = evaluate_classification(y_true, y_pred)
    assert result["accuracy"]["estimate"] == pytest.approx(report["accuracy"])
    assert result["macro_f1"]["estimate"] == pytest.approx(report["macro_f1"])
    for name in ("accuracy", "macro_f1"):
        assert result[name]["low"] <= result[name]["estimate"] <= result[name]["high"]


def test_resampling_groups_gives_wider_intervals_than_resampling_texts():
    rng = np.random.default_rng(3)
    n_groups, size = 40, 10
    group_correct = rng.random(n_groups) < 0.6  # texts of a group are all right or all wrong
    groups = np.repeat(np.arange(n_groups), size)
    y_true = np.zeros(n_groups * size, dtype=int)
    y_pred = np.where(np.repeat(group_correct, size), 0, 1)
    by_text = bootstrap_ci(y_true, y_pred, n_boot=400, seed=4)["accuracy"]
    by_group = bootstrap_ci(y_true, y_pred, groups=groups, n_boot=400, seed=4)["accuracy"]
    assert by_group["high"] - by_group["low"] > 2 * (by_text["high"] - by_text["low"])


def test_difference_of_identical_predictions_is_zero():
    result = bootstrap_difference_ci(Y_TRUE, Y_PRED, Y_PRED, n_boot=50)
    for name in ("accuracy", "macro_f1"):
        assert result[name] == {"difference": 0.0, "low": 0.0, "high": 0.0}


def test_paired_interval_detects_a_small_but_consistent_difference():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=5000)
    pred_a = np.where(rng.random(5000) < 0.8, y_true, 1 - y_true)
    pred_b = pred_a.copy()
    wrong_for_b = rng.choice(np.flatnonzero(pred_a == y_true), 100, replace=False)
    pred_b[wrong_for_b] = 1 - pred_b[wrong_for_b]  # b errs wherever it differs from a
    paired = bootstrap_difference_ci(y_true, pred_a, pred_b, n_boot=300, seed=1)
    assert paired["accuracy"]["difference"] == pytest.approx(0.02)
    assert paired["accuracy"]["low"] > 0
    # Separate intervals of the two models overlap, the paired one does not contain zero.
    alone_a = bootstrap_ci(y_true, pred_a, n_boot=300, seed=1)["accuracy"]
    alone_b = bootstrap_ci(y_true, pred_b, n_boot=300, seed=1)["accuracy"]
    assert alone_a["low"] < alone_b["high"]


def test_difference_sign_follows_the_argument_order():
    rng = np.random.default_rng(2)
    y_true = rng.integers(0, 3, size=600)
    good = np.where(rng.random(600) < 0.9, y_true, rng.integers(0, 3, size=600))
    bad = rng.integers(0, 3, size=600)
    forward = bootstrap_difference_ci(y_true, good, bad, n_boot=100, seed=3)
    backward = bootstrap_difference_ci(y_true, bad, good, n_boot=100, seed=3)
    assert forward["accuracy"]["difference"] > 0 > backward["accuracy"]["difference"]
    assert forward["accuracy"]["difference"] == pytest.approx(-backward["accuracy"]["difference"])


def test_bootstrap_is_reproducible_for_a_seed():
    first = bootstrap_ci(Y_TRUE, Y_PRED, n_boot=100, seed=5)
    second = bootstrap_ci(Y_TRUE, Y_PRED, n_boot=100, seed=5)
    assert first == second
