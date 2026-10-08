"""Readability features built from syllable counts and word/sentence length.

Flesch-type formulas are deliberately omitted: they are linear combinations of the mean sentence
length and the mean number of syllables per word, which are already part of the feature vector,
and published coefficients for Russian differ between sources.
"""

from __future__ import annotations

from typing import Dict

from ru_stylometry._stats import mean, safe_div
from ru_stylometry._text import LONG_WORD_LETTERS, count_syllables, word_length
from ru_stylometry.document import Document

DESCRIPTION = "Syllable-based and LIX readability measures."

POLYSYLLABIC_MIN = 4

FEATURES: Dict[str, str] = {
    "avg_syllables_per_word": "Mean number of syllables (vowel letters) per word.",
    "polysyllabic_ratio": "Share of words with four or more syllables.",
    "lix": "LIX index (Björnsson, 1968): mean sentence length plus percentage of long words.",
}


def compute(doc: Document) -> Dict[str, float]:
    words = doc.words
    syllables = [count_syllables(word) for word in words]
    long_words = sum(1 for word in words if word_length(word) > LONG_WORD_LETTERS)
    return {
        "avg_syllables_per_word": mean(syllables),
        "polysyllabic_ratio": safe_div(
            sum(1 for n in syllables if n >= POLYSYLLABIC_MIN), len(syllables)
        ),
        "lix": mean(doc.sentence_lengths) + 100.0 * safe_div(long_words, len(words)),
    }
