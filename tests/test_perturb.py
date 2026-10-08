"""Tests for the text perturbations."""

import pytest

from ru_stylometry import extract_features
from ru_stylometry._text import split_sentences
from ru_stylometry.perturb import (
    PERTURBATIONS,
    add_typos,
    clean_up,
    normalize_typography,
    remove_discourse_markers,
    shuffle_sentences,
    strip_formatting,
)

TYPESET = "«Привет» — сказал он… Ёлка зелёная."


def test_normalize_typography_uses_keyboard_characters():
    assert normalize_typography(TYPESET) == '"Привет" - сказал он... Елка зеленая.'


def test_normalize_typography_removes_the_typographic_features():
    before = extract_features(TYPESET, ["typography", "char"])
    after = extract_features(normalize_typography(TYPESET), ["typography", "char"])
    assert before["em_dash_share"] == 1.0 and after["em_dash_share"] == 0.0
    assert before["guillemet_share"] == 1.0 and after["guillemet_share"] == 0.0
    assert before["yo_ratio"] > 0 and after["yo_ratio"] == 0.0


def test_strip_formatting_joins_lines_and_drops_markers():
    text = "# Заголовок\n\n- **важно** помнить\n- второе\n1. третье"
    assert strip_formatting(text) == "Заголовок важно помнить второе третье"


def test_clean_up_combines_both():
    assert clean_up("**«Да»** — нет\n- пункт") == '"Да" - нет пункт'


def test_remove_discourse_markers_and_restore_capitals():
    text = "Кроме того, это важно. Однако мы согласны."
    assert remove_discourse_markers(text) == "Это важно. Мы согласны."


def test_remove_discourse_markers_leaves_other_text_untouched():
    assert remove_discourse_markers("Мама мыла раму.") == "Мама мыла раму."


def test_remove_discourse_markers_lowers_the_marker_feature():
    text = "Таким образом, мы справились. В целом, всё хорошо. Важно отметить, что так."
    before = extract_features(text, ["syntax"])["discourse_marker_per100w"]
    after = extract_features(remove_discourse_markers(text), ["syntax"])["discourse_marker_per100w"]
    assert before > 0 and after == 0.0


def test_shuffle_sentences_keeps_the_sentences_and_is_reproducible():
    text = "Первое предложение. Второе предложение. Третье предложение. Четвёртое. Пятое."
    shuffled = shuffle_sentences(text, seed=3)
    assert sorted(split_sentences(shuffled)) == sorted(split_sentences(text))
    assert shuffled == shuffle_sentences(text, seed=3)
    assert any(shuffle_sentences(text, seed=s) != text for s in range(5))


def test_add_typos_with_zero_rate_changes_nothing():
    assert add_typos("Мама мыла раму.", rate=0.0) == "Мама мыла раму."


def test_add_typos_is_reproducible_and_keeps_non_letters():
    text = "Мама мыла раму 12 раз, а потом отдыхала!" * 5
    noisy = add_typos(text, seed=1, rate=0.3)
    assert noisy != text
    assert noisy == add_typos(text, seed=1, rate=0.3)
    for symbol in "0123456789,!":
        assert noisy.count(symbol) == text.count(symbol)


def test_typos_raise_the_unknown_word_share():
    text = "Мама мыла раму, а потом долго отдыхала на диване. " * 10
    noisy = add_typos(text, seed=0, rate=0.15)
    clean = extract_features(text, ["morph"])["unknown_word_ratio"]
    assert extract_features(noisy, ["morph"])["unknown_word_ratio"] > clean


@pytest.mark.parametrize("name", PERTURBATIONS)
def test_every_registered_perturbation_maps_text_to_text(name):
    result = PERTURBATIONS[name]("Кроме того, «важно» — помнить… Ёлка!", 0)
    assert isinstance(result, str) and result
