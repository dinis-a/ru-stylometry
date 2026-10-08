"""Tests for tokenisation, sentence splitting and syllable counting."""

import pytest

from ru_stylometry._text import WORD_RE, count_syllables, split_sentences, word_length


def test_words_keep_hyphenated_forms_and_drop_digits():
    text = "Кто-то сказал: don't worry, ГОСТ-12!"
    assert WORD_RE.findall(text) == ["Кто-то", "сказал", "don't", "worry", "ГОСТ"]


@pytest.mark.parametrize("word, expected", [("кто-то", 5), ("don't", 4), ("ёлка", 4), ("", 0)])
def test_word_length_counts_letters_only(word, expected):
    assert word_length(word) == expected


@pytest.mark.parametrize(
    "word, expected", [("корова", 3), ("ёлка", 2), ("в", 0), ("Информационный", 6)]
)
def test_count_syllables(word, expected):
    assert count_syllables(word) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Привет, мир. Как дела? Хорошо!", 3),
        ("А. С. Пушкин родился в 1799 г. в Москве. Он писал стихи.", 2),
        ("Он думал… и молчал. Потом ушёл.", 2),
        ("Он сказал: «Иди.» Потом ушёл.", 2),
        ("Заголовок\n\nТекст здесь.", 2),
        ("- пункт один\n- пункт два", 2),
        ("Строка, перенесённая\nпосле запятой, продолжается.", 1),
        ("Что?! Не может быть!!! Да.", 3),
        ("Без точки в конце", 1),
        ("", 0),
        ("   \n  ", 0),
    ],
)
def test_split_sentences_count(text, expected):
    assert len(split_sentences(text)) == expected


def test_split_sentences_keeps_closing_quote_with_its_sentence():
    assert split_sentences("Он сказал: «Иди.» Потом ушёл.") == ["Он сказал: «Иди.»", "Потом ушёл."]


def test_split_sentences_does_not_break_after_abbreviation():
    assert split_sentences("Это было в 1990 г. в Москве.") == ["Это было в 1990 г. в Москве."]
