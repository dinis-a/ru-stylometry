"""Raw size counts.

Absolute counts mostly encode how long a text is, and corpora often differ in length between
classes, so this group is opt-in and excluded from the default feature vector. The previous
baseline (``stylometric-ai-detector``) relied on such counts.
"""

from __future__ import annotations

from typing import Dict

from ru_stylometry.document import Document

DESCRIPTION = "Absolute size of the text (opt-in)."

FEATURES: Dict[str, str] = {
    "char_count": "Number of characters.",
    "word_count": "Number of words.",
    "sentence_count": "Number of sentences.",
    "line_count": "Number of non-empty lines.",
}


def compute(doc: Document) -> Dict[str, float]:
    return {
        "char_count": float(len(doc.text)),
        "word_count": float(len(doc.words)),
        "sentence_count": float(len(doc.sentences)),
        "line_count": float(sum(1 for line in doc.text.splitlines() if line.strip())),
    }
