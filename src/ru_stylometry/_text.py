"""Low-level text helpers: tokenisation, sentence splitting and syllable counting."""

from __future__ import annotations

import re
from typing import List

# A word is a run of letters, optionally joined by hyphens or apostrophes ("кто-то", "don't").
WORD_RE = re.compile(r"[^\W\d_]+(?:[-’'][^\W\d_]+)*")
CYRILLIC_RE = re.compile(r"[А-Яа-яЁё]")

# Words *longer* than this many letters are "long" (Björnsson's definition used by LIX).
LONG_WORD_LETTERS = 6

# Paragraph break, or a line break in front of a list item / heading marker.
_BLOCK_RE = re.compile(r"\n\s*\n|\n(?=[ \t]*(?:[-*•–—#]+|\d+[.)])[ \t]+)")
# Sentence terminator, optional closing quotes/brackets (captured) and the whitespace after them.
_BOUNDARY_RE = re.compile(r"(?<=[.!?…])([\"»”’)\]]*)\s+")
_INITIALS_RE = re.compile(r"^(?:[А-ЯЁA-Z]\.){1,3}$")

# Abbreviations after which a full stop does not end the sentence (lowercase, one token each).
_ABBREVIATION_LIST = """
    т.д. т.п. т.е. т.к. т.н. т.о. т.ч. и.о. н.э. г. гг. в. вв. ул. им. пр. др. см. рис. табл. стр.
    с. т. тт. тыс. млн. млрд. руб. коп. проф. доц. акад. канд. докт. напр. англ. лат. нем. франц.
    ок. ср. изд. вып. гл. п. пп. ч. ст. обл. пос. д. кв. корп. оф. тел. тов.
"""
_ABBREVIATIONS = frozenset(_ABBREVIATION_LIST.split())

_VOWELS = frozenset("аеёиоуыэюяaeiouy")


def word_length(word: str) -> int:
    """Number of letters in a word (hyphens and apostrophes are not counted)."""
    return sum(1 for char in word if char.isalpha())


def count_syllables(word: str) -> int:
    """Count syllables as vowel letters (Cyrillic and basic Latin vowels)."""
    return sum(1 for char in word.lower() if char in _VOWELS)


def truncate_words(text: str, max_words: int) -> str:
    """Keep the beginning of the text up to the end of its ``max_words``-th word.

    Texts with at most ``max_words`` words are returned unchanged. Cutting before analysis bounds
    the cost of long documents and puts texts of very different lengths on the same footing.
    """
    end = 0
    for count, match in enumerate(WORD_RE.finditer(text), start=1):
        if count > max_words:
            return text[:end]
        end = match.end()
    return text


def _ends_with_abbreviation(chunk: str) -> bool:
    last = chunk.rsplit(None, 1)[-1]
    return last.lower() in _ABBREVIATIONS or bool(_INITIALS_RE.match(last))


def _continues_after_ellipsis(previous: str, following: str) -> bool:
    return previous.endswith(("…", "...")) and following[:1].islower()


def _split_block(block: str) -> List[str]:
    parts = _BOUNDARY_RE.split(block)
    # parts = [sentence, closers, sentence, closers, ..., sentence]; glue closers back on.
    chunks = [parts[i] + parts[i + 1] for i in range(0, len(parts) - 1, 2)] + [parts[-1]]
    merged: List[str] = []
    for chunk in chunks:
        if merged and (
            _ends_with_abbreviation(merged[-1]) or _continues_after_ellipsis(merged[-1], chunk)
        ):
            merged[-1] = f"{merged[-1]} {chunk}"
        else:
            merged.append(chunk)
    return merged


def split_sentences(text: str) -> List[str]:
    """Split Russian text into sentences with a rule-based splitter.

    Boundaries are paragraph breaks, line breaks before list items, and sentence-final
    punctuation followed by whitespace. A full stop after a known abbreviation ("т.д.", "г.") or
    after initials ("А. С.") does not end a sentence, nor does an ellipsis followed by a lowercase
    letter. Single line breaks inside a paragraph are treated as spaces.

    Args:
        text: Input text.

    Returns:
        Non-empty sentences in order of appearance.
    """
    sentences: List[str] = []
    for block in _BLOCK_RE.split(text):
        block = " ".join(block.split())
        if block:
            sentences.extend(_split_block(block))
    return sentences
