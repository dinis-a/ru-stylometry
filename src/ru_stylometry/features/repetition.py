"""Repetition features: compressibility and repeated word n-grams."""

from __future__ import annotations

import zlib
from collections import Counter
from typing import Dict, Sequence

from ru_stylometry._stats import safe_div
from ru_stylometry.document import Document

DESCRIPTION = "Redundancy of the text: compression ratio and repeated word n-grams."

FEATURES: Dict[str, str] = {
    "zlib_ratio": "Deflate-compressed size divided by raw UTF-8 size (lower means more redundant).",
    "repeated_bigram_ratio": "Share of word bigram occurrences whose bigram appears more than once.",
    "repeated_trigram_ratio": "Share of word trigram occurrences whose trigram appears more than once.",
}


def _compression_ratio(text: str) -> float:
    raw = text.encode("utf-8")
    # Raw deflate (negative wbits) avoids the zlib header and checksum on short texts.
    compressor = zlib.compressobj(level=9, wbits=-15)
    compressed = compressor.compress(raw) + compressor.flush()
    return safe_div(len(compressed), len(raw))


def _repeated_ngram_ratio(tokens: Sequence[str], n: int) -> float:
    total = len(tokens) - n + 1
    if total <= 0:
        return 0.0
    ngrams = Counter(zip(*(tokens[i:] for i in range(n))))
    return sum(count for count in ngrams.values() if count > 1) / total


def compute(doc: Document) -> Dict[str, float]:
    tokens = doc.norm_words
    return {
        "zlib_ratio": _compression_ratio(doc.text),
        "repeated_bigram_ratio": _repeated_ngram_ratio(tokens, 2),
        "repeated_trigram_ratio": _repeated_ngram_ratio(tokens, 3),
    }
