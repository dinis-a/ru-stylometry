"""Tests for the numeric helpers."""

import pytest

from ru_stylometry._stats import mattr, mean, mtld, pstdev, safe_div, yule_k


def test_safe_div():
    assert safe_div(1, 2) == 0.5
    assert safe_div(1, 0) == 0.0
    assert safe_div(1, 0, default=-1.0) == -1.0


def test_mean_and_pstdev():
    assert mean([]) == 0.0
    assert mean([1, 2, 3]) == 2.0
    assert pstdev([]) == 0.0
    assert pstdev([2, 2, 2]) == 0.0
    assert pstdev([1, 3]) == pytest.approx(1.0)


def test_mattr_of_short_text_is_plain_type_token_ratio():
    assert mattr(["a", "b", "a", "b"], window=50) == 0.5
    assert mattr([]) == 0.0


def test_mattr_sliding_window():
    assert mattr(["a", "b"] * 30, window=50) == pytest.approx(2 / 50)


def test_mattr_matches_naive_computation():
    tokens = list("abcabdabcaabbcdeabcdeeedcba" * 4)
    window = 7
    windows = [tokens[i : i + window] for i in range(len(tokens) - window + 1)]
    naive = sum(len(set(w)) / window for w in windows) / len(windows)
    assert mattr(tokens, window) == pytest.approx(naive)


def test_mtld_repeated_word():
    # TTR drops to 0.5 after every second token: five factors in ten tokens.
    assert mtld(["да"] * 10) == pytest.approx(2.0)


def test_mtld_of_all_unique_tokens_equals_length():
    assert mtld(list("abcdefghij")) == pytest.approx(10.0)


def test_mtld_of_empty_text():
    assert mtld([]) == 0.0


def test_yule_k():
    assert yule_k(["да"] * 5) == pytest.approx(8000.0)
    assert yule_k(["раз", "два", "три"]) == 0.0
    assert yule_k([]) == 0.0
