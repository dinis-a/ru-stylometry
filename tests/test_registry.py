"""Tests for the feature registry and the public extraction API."""

import math
import subprocess
import sys

import pytest

from ru_stylometry import (
    ALL_GROUPS,
    DEFAULT_GROUPS,
    describe_features,
    extract_features,
    feature_names,
)

SAMPLE = "Мама мыла раму. Рама была чистой!"

EDGE_TEXTS = [
    " ",
    "\n\n",
    "\t\t\t",
    "!!!",
    "12345",
    "а",
    "ё",
    "…",
    "a b c",
    "Привет",
    "😀😀😀",
    "- - -",
    "— — —",
    "***",
    ". . .",
    "А. Б. В.",
    "«»",
    "()",
    "да " * 500,
    "Слово" * 2000,
    "Один. " * 300,
]


@pytest.mark.parametrize("group", ALL_GROUPS)
def test_group_returns_exactly_its_declared_features(group):
    assert list(extract_features(SAMPLE, [group])) == feature_names([group])


def test_catalogue_names_are_unique_and_described():
    rows = describe_features(ALL_GROUPS)
    names = [name for _, name, _ in rows]
    assert len(names) == len(set(names))
    assert all(description.strip() for _, _, description in rows)


def test_default_vector_size_is_in_the_expected_range():
    assert 50 <= len(feature_names()) <= 80


@pytest.mark.parametrize("text", EDGE_TEXTS)
def test_features_are_finite_floats_on_edge_cases(text):
    values = extract_features(text, ALL_GROUPS)
    assert list(values) == feature_names(ALL_GROUPS)
    assert all(isinstance(v, float) and math.isfinite(v) for v in values.values())


@pytest.mark.parametrize("text", ["", "   ", "\n\t", None, 42, ["список"]])
def test_blank_or_non_string_input_gives_zero_vector(text):
    assert extract_features(text) == {name: 0.0 for name in feature_names()}


def test_group_order_given_by_caller_does_not_change_the_layout():
    first = extract_features(SAMPLE, ["morph", "char"])
    second = extract_features(SAMPLE, ["char", "morph"])
    assert list(first) == list(second) == feature_names(["char", "morph"])


def test_single_group_name_may_be_a_string():
    assert list(extract_features(SAMPLE, "char")) == feature_names(["char"])


def test_unknown_group_raises():
    with pytest.raises(ValueError, match="Unknown feature group"):
        extract_features(SAMPLE, ["nope"])


def test_empty_group_selection_raises():
    with pytest.raises(ValueError, match="At least one"):
        extract_features(SAMPLE, [])


def test_corpus_dependent_groups_are_opt_in():
    assert set(ALL_GROUPS) - set(DEFAULT_GROUPS) == {"formatting", "length"}


def test_extraction_is_deterministic():
    assert extract_features(SAMPLE) == extract_features(SAMPLE)


def test_morphology_is_not_loaded_unless_requested():
    code = (
        "import sys\n"
        "from ru_stylometry import extract_features\n"
        "extract_features('Мама мыла раму.', ['char', 'lexical', 'syntax'])\n"
        "assert 'pymorphy3' not in sys.modules\n"
        "extract_features('Мама мыла раму.', ['morph'])\n"
        "assert 'pymorphy3' in sys.modules\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
