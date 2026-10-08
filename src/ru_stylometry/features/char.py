"""Character-level features: composition of the text by character class."""

from __future__ import annotations

import unicodedata
from typing import Dict

from ru_stylometry._stats import safe_div
from ru_stylometry.document import Document

DESCRIPTION = "Composition of the text by character class."

FEATURES: Dict[str, str] = {
    "letter_ratio": "Share of letters among all characters.",
    "digit_ratio": "Share of decimal digits among all characters.",
    "punct_ratio": "Share of punctuation (Unicode category P) among all characters.",
    "symbol_ratio": "Share of symbols (Unicode category S, incl. emoji) among all characters.",
    "upper_ratio": "Share of uppercase letters among all letters.",
    "latin_ratio": "Share of Latin letters among all letters.",
    "yo_ratio": "Share of 'ё' among all 'е' and 'ё' letters.",
}


def compute(doc: Document) -> Dict[str, float]:
    total = len(doc.text)
    letters = digits = punct = symbols = upper = latin = e_letters = yo = 0
    for char, count in doc.char_counts.items():
        category = unicodedata.category(char)
        if category[0] == "L":
            letters += count
            if category == "Lu":
                upper += count
            if unicodedata.name(char, "").startswith("LATIN"):
                latin += count
            if char in "еЕ":
                e_letters += count
            elif char in "ёЁ":
                e_letters += count
                yo += count
        elif category == "Nd":
            digits += count
        elif category[0] == "P":
            punct += count
        elif category[0] == "S":
            symbols += count
    return {
        "letter_ratio": safe_div(letters, total),
        "digit_ratio": safe_div(digits, total),
        "punct_ratio": safe_div(punct, total),
        "symbol_ratio": safe_div(symbols, total),
        "upper_ratio": safe_div(upper, letters),
        "latin_ratio": safe_div(latin, letters),
        "yo_ratio": safe_div(yo, e_letters),
    }
