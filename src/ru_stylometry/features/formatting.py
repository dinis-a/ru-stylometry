"""Layout features: line breaks, list items and Markdown leftovers.

These are strong markers of chat-style machine output but also easy to confound with the way a
corpus was collected, so the group is opt-in and excluded from the default feature vector.
"""

from __future__ import annotations

import re
from typing import Dict

from ru_stylometry._stats import safe_div
from ru_stylometry.document import Document

DESCRIPTION = "Layout of the text: line breaks, list items, Markdown symbols (opt-in)."

FEATURES: Dict[str, str] = {
    "newline_per100w": "Line breaks per 100 words.",
    "list_line_ratio": "Share of non-empty lines that start a list item (-, *, •, 1.).",
    "markdown_per100w": "Markdown symbols (*, #, `, |) per 100 words.",
}

_LIST_ITEM_RE = re.compile(r"^\s*(?:[-*•–—]|\d+[.)])\s+")


def compute(doc: Document) -> Dict[str, float]:
    n_words = len(doc.words)
    lines = [line for line in doc.text.splitlines() if line.strip()]
    markdown = sum(doc.char_counts[symbol] for symbol in "*#`|")
    return {
        "newline_per100w": 100.0 * safe_div(doc.char_counts["\n"], n_words),
        "list_line_ratio": safe_div(sum(1 for ln in lines if _LIST_ITEM_RE.match(ln)), len(lines)),
        "markdown_per100w": 100.0 * safe_div(markdown, n_words),
    }
