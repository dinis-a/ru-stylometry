"""Lazily computed analysis of a single text, shared by all feature groups."""

from __future__ import annotations

from collections import Counter
from functools import cached_property
from typing import List

from ru_stylometry._morph import WordInfo, analyze_word
from ru_stylometry._text import CYRILLIC_RE, WORD_RE, split_sentences


def normalize_word(word: str) -> str:
    """Lowercase a word and fold 'ё' into 'е' (the two are interchangeable in everyday writing)."""
    return word.lower().replace("ё", "е")


class Document:
    """A text with cached tokenisation, sentence splitting and morphological analysis.

    Every attribute is computed on first access, so feature groups that do not need morphology
    never trigger loading of the pymorphy3 dictionaries.
    """

    def __init__(self, text: str) -> None:
        self.text = text

    @cached_property
    def char_counts(self) -> "Counter[str]":
        """Frequency of every distinct character."""
        return Counter(self.text)

    @cached_property
    def words(self) -> List[str]:
        """Word tokens in their original case."""
        return WORD_RE.findall(self.text)

    @cached_property
    def norm_words(self) -> List[str]:
        """Word tokens lowercased, with 'ё' folded into 'е'."""
        return [normalize_word(word) for word in self.words]

    @cached_property
    def sentences(self) -> List[str]:
        """Sentences that contain at least one word."""
        return [s for s in split_sentences(self.text) if WORD_RE.search(s)]

    @cached_property
    def sentence_words(self) -> List[List[str]]:
        """Word tokens of every sentence."""
        return [WORD_RE.findall(sentence) for sentence in self.sentences]

    @cached_property
    def sentence_lengths(self) -> List[int]:
        """Number of words in every sentence."""
        return [len(words) for words in self.sentence_words]

    @cached_property
    def cyrillic_words(self) -> List[str]:
        """Lowercased tokens containing Cyrillic letters ('ё' is kept for morphology)."""
        return [word.lower() for word in self.words if CYRILLIC_RE.search(word)]

    @cached_property
    def word_infos(self) -> List[WordInfo]:
        """Morphological analysis (most probable parse) of every Cyrillic word."""
        return [analyze_word(word) for word in self.cyrillic_words]
