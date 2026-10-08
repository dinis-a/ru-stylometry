"""Tests for the explanation helpers."""

import numpy as np
import pytest

from ru_stylometry.explain import explain_row, group_permutation_importance


def logistic_model(weights):
    """Class 1 has log-odds ``weights @ x``; the output has the columns (class 0, class 1)."""
    weights = np.asarray(weights, dtype=float)

    def predict_proba(X):
        p = 1.0 / (1.0 + np.exp(-(X @ weights)))
        return np.column_stack([1.0 - p, p])

    return predict_proba


def test_explain_row_effects_of_features_and_groups():
    result = explain_row(
        logistic_model([2.0, 0.0, 0.0]),
        row=np.array([1.0, 7.0, 3.0]),
        baseline=np.zeros(3),
        target=1,
        feature_groups=["signal", "noise", "noise"],
    )
    # The log-odds are 2 * x0, so replacing x0 = 1 by 0 costs exactly 2 units.
    assert result["probability"] == pytest.approx(1 / (1 + np.exp(-2)))
    assert result["feature_effects"] == pytest.approx([2.0, 0.0, 0.0])
    assert result["group_effects"] == pytest.approx({"signal": 2.0, "noise": 0.0})
    assert result["feature_probability_changes"][0] == pytest.approx(result["probability"] - 0.5)
    assert result["group_probability_changes"]["noise"] == 0.0


def test_effect_has_the_opposite_sign_for_the_other_class():
    result = explain_row(
        logistic_model([2.0, 0.0]), np.array([1.0, 7.0]), np.zeros(2), 0, ["a", "b"]
    )
    assert result["feature_effects"][0] == pytest.approx(-2.0)


def test_log_odds_effects_do_not_saturate_for_confident_decisions():
    result = explain_row(logistic_model([8.0, 4.0]), np.ones(2), np.zeros(2), 1, ["a", "b"])
    assert result["probability"] > 0.99999
    assert result["feature_effects"] == pytest.approx([8.0, 4.0])
    # In probability the second feature seems irrelevant, in log-odds it carries 4 units.
    assert abs(result["feature_probability_changes"][1]) < 1e-3


def test_individual_columns_limit_the_single_feature_effects():
    result = explain_row(
        logistic_model([2.0, 3.0, 0.0]),
        np.ones(3),
        np.zeros(3),
        1,
        ["a", "a", "b"],
        individual=[1],
    )
    assert result["feature_effects"] == pytest.approx([3.0])  # only column 1 was examined
    assert result["group_effects"] == pytest.approx({"a": 5.0, "b": 0.0})  # groups are unaffected


def test_extreme_probabilities_stay_finite():
    def certain(X):
        p = (X[:, 0] > 0.5).astype(float)
        return np.column_stack([1.0 - p, p])

    result = explain_row(certain, np.array([1.0]), np.zeros(1), 1, ["a"])
    assert np.isfinite(result["feature_effects"]).all()
    assert result["feature_effects"][0] > 10


class FirstColumnRule:
    def predict(self, X):
        return (X[:, 0] > 0).astype(int)


def test_group_permutation_importance_ranks_the_informative_group_first():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 3))
    y = (X[:, 0] > 0).astype(int)
    result = group_permutation_importance(
        FirstColumnRule(), X, y, {"noise": [1, 2], "signal": [0]}, metric="accuracy", n_repeats=3
    )
    assert result["baseline"] == 1.0
    assert list(result["groups"]) == ["signal", "noise"]
    assert result["groups"]["signal"]["drop"] > 0.3
    assert result["groups"]["noise"]["drop"] == 0.0


def test_group_permutation_importance_does_not_modify_the_input():
    X = np.arange(20, dtype=float).reshape(10, 2)
    original = X.copy()
    group_permutation_importance(FirstColumnRule(), X, [0, 1] * 5, {"g": [0]}, n_repeats=2)
    assert np.array_equal(X, original)


def test_unknown_metric_raises():
    with pytest.raises(ValueError, match="metric"):
        group_permutation_importance(FirstColumnRule(), np.zeros((4, 1)), [0] * 4, {}, metric="auc")
