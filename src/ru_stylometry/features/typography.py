"""Typographic habits: punctuation density, dash and quotation-mark conventions."""

from __future__ import annotations

import re
from typing import Dict

from ru_stylometry._stats import safe_div
from ru_stylometry.document import Document

DESCRIPTION = "Punctuation density and typographic conventions (dashes, quotation marks)."

FEATURES: Dict[str, str] = {
    "comma_per100w": "Commas per 100 words.",
    "colon_per100w": "Colons per 100 words.",
    "semicolon_per100w": "Semicolons per 100 words.",
    "dash_per100w": "Dashes (em dash, en dash, spaced hyphen) per 100 words.",
    "em_dash_share": "Share of em dashes (—) among all dash-like marks.",
    "quote_per100w": "Quotation marks per 100 words.",
    "guillemet_share": "Share of guillemets («») among all double quotation marks.",
    "paren_per100w": "Opening parentheses per 100 words.",
    "question_per100w": "Question marks per 100 words.",
    "exclam_per100w": "Exclamation marks per 100 words.",
    "ellipsis_per100w": "Ellipses (… or three and more dots) per 100 words.",
}

# A hyphen with a space on both sides is how many people type a dash.
_SPACED_HYPHEN_RE = re.compile(r"(?<=\S) -{1,2} (?=\S)")
_ELLIPSIS_RE = re.compile(r"\.{3,}|…")


def compute(doc: Document) -> Dict[str, float]:
    chars = doc.char_counts
    n_words = len(doc.words)

    def per100(count: float) -> float:
        return 100.0 * safe_div(count, n_words)

    em_dashes = chars["—"] + chars["―"]
    en_dashes = chars["–"] + chars["‒"]
    dashes = em_dashes + en_dashes + len(_SPACED_HYPHEN_RE.findall(doc.text))
    guillemets = chars["«"] + chars["»"]
    quotes = guillemets + chars['"'] + chars["“"] + chars["”"] + chars["„"] + chars["‟"]

    return {
        "comma_per100w": per100(chars[","]),
        "colon_per100w": per100(chars[":"]),
        "semicolon_per100w": per100(chars[";"]),
        "dash_per100w": per100(dashes),
        "em_dash_share": safe_div(em_dashes, dashes),
        "quote_per100w": per100(quotes),
        "guillemet_share": safe_div(guillemets, quotes),
        "paren_per100w": per100(chars["("]),
        "question_per100w": per100(chars["?"]),
        "exclam_per100w": per100(chars["!"]),
        "ellipsis_per100w": per100(len(_ELLIPSIS_RE.findall(doc.text))),
    }
