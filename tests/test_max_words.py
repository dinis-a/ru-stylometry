"""Tests for truncation to the first N words and for parallel feature extraction."""

import json

import numpy as np
import pytest

from ru_stylometry import StylometricVectorizer, extract_features
from ru_stylometry._text import truncate_words
from ru_stylometry.cli import main

TEXT = "Один два три четыре пять. Шесть семь восемь."


@pytest.mark.parametrize(
    "text, limit, expected",
    [
        (TEXT, 3, "Один два три"),
        (TEXT, 5, "Один два три четыре пять"),
        ("Привет, мир! Как дела?", 2, "Привет, мир"),
        (TEXT, 8, TEXT),
        (TEXT, 100, TEXT),
        ("!!! ???", 1, "!!! ???"),
        ("", 3, ""),
    ],
)
def test_truncate_words(text, limit, expected):
    assert truncate_words(text, limit) == expected


def test_extract_features_uses_only_the_first_words():
    assert extract_features(TEXT, ["length"], max_words=5)["word_count"] == 5.0
    assert extract_features(TEXT, ["length"])["word_count"] == 8.0
    assert extract_features(TEXT, max_words=5) == extract_features("Один два три четыре пять")


@pytest.mark.parametrize("bad", [0, -1, True])
def test_invalid_max_words_raises(bad):
    with pytest.raises(ValueError, match="max_words"):
        extract_features(TEXT, max_words=bad)
    with pytest.raises(ValueError, match="max_words"):
        StylometricVectorizer(max_words=bad).fit()


def test_vectorizer_max_words_matches_manual_truncation():
    texts = [TEXT, "Мама мыла раму. Рама была чистой! " * 5]
    groups = ["lexical", "syntax"]
    truncated = StylometricVectorizer(groups=groups, max_words=4).fit_transform(texts)
    manual = StylometricVectorizer(groups=groups).fit_transform(
        [truncate_words(text, 4) for text in texts]
    )
    assert np.array_equal(truncated, manual)


def test_parallel_transform_matches_sequential_and_keeps_order():
    texts = [f"Текст номер {i}. Мама мыла раму {i} раз, а рама была чистой!" for i in range(60)]
    groups = ["char", "lexical", "syntax", "morph"]
    sequential = StylometricVectorizer(groups=groups).fit_transform(texts)
    parallel = StylometricVectorizer(groups=groups, n_jobs=2).fit_transform(texts)
    assert np.array_equal(sequential, parallel)


def test_cli_max_words(capsys):
    assert main(["features", TEXT, "-g", "length", "--max-words", "3"]) == 0
    assert json.loads(capsys.readouterr().out)["word_count"] == 3
