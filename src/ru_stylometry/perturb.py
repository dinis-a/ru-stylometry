"""Deterministic text transformations for robustness checks.

They imitate what a person who passes machine text off as their own may do, or what a channel does
to a text by itself (copying through a messenger, stripping Markdown). Every function maps a string
to a string and takes a ``seed``, which only matters for the random ones. :data:`PERTURBATIONS`
lists them by name; none of them calls a language model, so they are cheap and reproducible.
"""

from __future__ import annotations

import random
import re
from typing import Callable, Dict

from ru_stylometry._lexicons import DISCOURSE_MARKERS
from ru_stylometry._text import split_sentences

_KEYBOARD = str.maketrans(
    {
        "—": "-",
        "–": "-",
        "‒": "-",
        "―": "-",
        "«": '"',
        "»": '"',
        "“": '"',
        "”": '"',
        "„": '"',
        "‟": '"',
        "…": "...",
        "ё": "е",
        "Ё": "Е",
        " ": " ",
        " ": " ",
    }
)
_BULLET_RE = re.compile(r"^[ \t]*(?:[-*•–—]|\d+[.)])[ \t]+", re.MULTILINE)
_MARKDOWN_RE = re.compile(r"\*\*|__|`+|^[ \t]*#{1,6}[ \t]*", re.MULTILINE)
# Markers are stored with 'ё' folded into 'е', so every 'е' may be written either way.
_MARKER_RE = re.compile(
    r"\b(?:"
    + "|".join(
        re.escape(marker).replace("е", "[её]")
        for marker in sorted(DISCOURSE_MARKERS, key=len, reverse=True)
    )
    + r")\b,?[ \t]*",
    re.IGNORECASE,
)
_SENTENCE_START_RE = re.compile(r"(^|[.!?…]\s+)([а-яё])")


def normalize_typography(text: str, seed: int = 0) -> str:
    """Replace typographic marks by what a plain keyboard types: dashes, quotes, ellipsis, 'ё'."""
    return text.translate(_KEYBOARD)


def strip_formatting(text: str, seed: int = 0) -> str:
    """Remove Markdown markers and list bullets and join all lines into one paragraph."""
    text = _BULLET_RE.sub("", text)
    text = _MARKDOWN_RE.sub("", text)
    return " ".join(text.split())


def remove_discourse_markers(text: str, seed: int = 0) -> str:
    """Delete stock connectives (see ``ru_stylometry._lexicons.DISCOURSE_MARKERS``)."""
    cleaned, removed = _MARKER_RE.subn("", text)
    if not removed:
        return text
    return _SENTENCE_START_RE.sub(lambda m: m.group(1) + m.group(2).upper(), cleaned)


def shuffle_sentences(text: str, seed: int = 0) -> str:
    """Put the sentences in a random order, which breaks the discourse structure."""
    sentences = split_sentences(text)
    random.Random(seed).shuffle(sentences)
    return " ".join(sentences)


def add_typos(text: str, seed: int = 0, rate: float = 0.02) -> str:
    """Delete, double or swap letters at random: about ``rate`` of the letters are affected."""
    rng = random.Random(seed)
    out = []
    i = 0
    while i < len(text):
        char = text[i]
        if char.isalpha() and rng.random() < rate:
            operation = rng.choice(("delete", "double", "swap"))
            if operation == "delete":
                i += 1
                continue
            if operation == "double":
                out.append(char * 2)
                i += 1
                continue
            if i + 1 < len(text) and text[i + 1].isalpha():
                out.append(text[i + 1] + char)
                i += 2
                continue
        out.append(char)
        i += 1
    return "".join(out)


def clean_up(text: str, seed: int = 0) -> str:
    """Typography and formatting normalised together, as after pasting into a plain-text field."""
    return strip_formatting(normalize_typography(text))


PERTURBATIONS: Dict[str, Callable[[str, int], str]] = {
    "typography": normalize_typography,
    "formatting": strip_formatting,
    "clean_up": clean_up,
    "no_connectives": remove_discourse_markers,
    "shuffled": shuffle_sentences,
    "typos": add_typos,
}
