"""Syntactic and function-word features: sentence rhythm, connectives, negation."""

from __future__ import annotations

import re
from typing import Dict

from ru_stylometry._lexicons import (
    CONJUNCTION_STARTERS,
    DISCOURSE_MARKERS,
    FUNCTION_WORDS,
    NEGATIONS,
)
from ru_stylometry._stats import mean, pstdev, safe_div
from ru_stylometry.document import Document, normalize_word

DESCRIPTION = "Sentence length distribution, function words, negation and discourse markers."

SHORT_SENTENCE_WORDS = 5
LONG_SENTENCE_WORDS = 25

FEATURES: Dict[str, str] = {
    "avg_sentence_len": "Mean sentence length in words.",
    "std_sentence_len": "Standard deviation of sentence length in words (rhythm variability).",
    "short_sentence_ratio": "Share of sentences with at most 5 words.",
    "long_sentence_ratio": "Share of sentences with at least 25 words.",
    "function_word_ratio": "Share of prepositions, conjunctions, particles, pronouns and 'быть'.",
    "negation_ratio": "Share of negation words (не, ни, нет, никогда, ...).",
    "conj_start_ratio": "Share of sentences opening with a conjunction (и, а, но, да, ...).",
    "opener_diversity": "Distinct first words divided by number of sentences (low means repetitive).",
    "discourse_marker_per100w": "Stock connectives (кроме того, таким образом, ...) per 100 words.",
}

# Words are joined with single spaces before matching, so a space (or an edge) is the boundary.
_MARKER_RE = re.compile(
    r"(?<!\S)(?:"
    + "|".join(re.escape(marker) for marker in sorted(DISCOURSE_MARKERS, key=len, reverse=True))
    + r")(?!\S)"
)


def compute(doc: Document) -> Dict[str, float]:
    lengths = doc.sentence_lengths
    tokens = doc.norm_words
    n_words = len(tokens)
    openers = [normalize_word(words[0]) for words in doc.sentence_words]
    markers = len(_MARKER_RE.findall(" ".join(tokens)))
    return {
        "avg_sentence_len": mean(lengths),
        "std_sentence_len": pstdev(lengths),
        "short_sentence_ratio": safe_div(
            sum(1 for n in lengths if n <= SHORT_SENTENCE_WORDS), len(lengths)
        ),
        "long_sentence_ratio": safe_div(
            sum(1 for n in lengths if n >= LONG_SENTENCE_WORDS), len(lengths)
        ),
        "function_word_ratio": safe_div(sum(1 for w in tokens if w in FUNCTION_WORDS), n_words),
        "negation_ratio": safe_div(sum(1 for w in tokens if w in NEGATIONS), n_words),
        "conj_start_ratio": safe_div(
            sum(1 for w in openers if w in CONJUNCTION_STARTERS), len(openers)
        ),
        "opener_diversity": safe_div(len(set(openers)), len(openers)),
        "discourse_marker_per100w": 100.0 * safe_div(markers, n_words),
    }
