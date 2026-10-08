"""Lexical features: word length and lexical diversity."""

from __future__ import annotations

from collections import Counter
from typing import Dict

from ru_stylometry._stats import mattr, mean, mtld, pstdev, safe_div, yule_k
from ru_stylometry._text import LONG_WORD_LETTERS, word_length
from ru_stylometry.document import Document

DESCRIPTION = "Word length and lexical richness measured on word forms."

FEATURES: Dict[str, str] = {
    "avg_word_len": "Mean word length in letters.",
    "std_word_len": "Standard deviation of word length in letters.",
    "long_word_ratio": "Share of words longer than six letters.",
    "mattr_50": "Moving-average type-token ratio, window of 50 words (robust to text length).",
    "mtld": "Measure of textual lexical diversity (McCarthy & Jarvis), threshold 0.72.",
    "yule_k": "Yule's K: repetitiveness of vocabulary (higher means less diverse).",
    "hapax_ratio": "Share of tokens whose word form occurs exactly once.",
}


def compute(doc: Document) -> Dict[str, float]:
    lengths = [word_length(word) for word in doc.words]
    tokens = doc.norm_words
    hapaxes = sum(1 for count in Counter(tokens).values() if count == 1)
    return {
        "avg_word_len": mean(lengths),
        "std_word_len": pstdev(lengths),
        "long_word_ratio": safe_div(sum(1 for n in lengths if n > LONG_WORD_LETTERS), len(lengths)),
        "mattr_50": mattr(tokens, 50),
        "mtld": mtld(tokens),
        "yule_k": yule_k(tokens),
        "hapax_ratio": safe_div(hapaxes, len(tokens)),
    }
