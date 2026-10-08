"""Registry of feature groups and the functions that compute features from it.

Every group is a module with three members: ``DESCRIPTION``, ``FEATURES`` (ordered mapping of
feature name to description) and ``compute(doc)`` returning a value for every name in ``FEATURES``.
"""

from __future__ import annotations

from types import ModuleType
from typing import Dict, Iterable, List, Optional, Tuple, Union

from ru_stylometry._text import truncate_words
from ru_stylometry.document import Document
from ru_stylometry.features import (
    char,
    formatting,
    length,
    lexical,
    morph,
    readability,
    repetition,
    syntax,
    typography,
)

Groups = Optional[Union[str, Iterable[str]]]

_GROUPS: Dict[str, ModuleType] = {
    "char": char,
    "typography": typography,
    "lexical": lexical,
    "syntax": syntax,
    "readability": readability,
    "repetition": repetition,
    "morph": morph,
    "formatting": formatting,
    "length": length,
}

ALL_GROUPS: Tuple[str, ...] = tuple(_GROUPS)
# Opt-in groups ("formatting", "length") depend on how a corpus was collected rather than on style.
DEFAULT_GROUPS: Tuple[str, ...] = (
    "char",
    "typography",
    "lexical",
    "syntax",
    "readability",
    "repetition",
    "morph",
)


def resolve_groups(groups: Groups = None) -> Tuple[str, ...]:
    """Validate group names and return them in registry order.

    Registry order, not caller order, fixes the layout of feature vectors, so the same set of
    groups always produces the same columns.

    Raises:
        ValueError: If a name is unknown or no group is selected.
    """
    if groups is None:
        return DEFAULT_GROUPS
    requested = {groups} if isinstance(groups, str) else set(groups)
    unknown = sorted(requested - set(ALL_GROUPS))
    if unknown:
        raise ValueError(f"Unknown feature group(s) {unknown}; available: {list(ALL_GROUPS)}.")
    if not requested:
        raise ValueError("At least one feature group is required.")
    return tuple(group for group in ALL_GROUPS if group in requested)


def feature_names(groups: Groups = None) -> List[str]:
    """Names of the features in the selected groups, in output order."""
    return [name for group in resolve_groups(groups) for name in _GROUPS[group].FEATURES]


def describe_features(groups: Groups = None) -> List[Tuple[str, str, str]]:
    """``(group, feature name, description)`` for every feature in the selected groups."""
    return [
        (group, name, description)
        for group in resolve_groups(groups)
        for name, description in _GROUPS[group].FEATURES.items()
    ]


def group_description(group: str) -> str:
    """One-line description of a feature group."""
    return _GROUPS[resolve_groups(group)[0]].DESCRIPTION


def check_max_words(max_words: Optional[int]) -> None:
    """Raise ``ValueError`` unless ``max_words`` is ``None`` or a positive integer."""
    if max_words is not None and (isinstance(max_words, bool) or max_words < 1):
        raise ValueError(f"max_words must be a positive integer or None, got {max_words!r}.")


def compute_features(
    text: object, groups: Tuple[str, ...], max_words: Optional[int] = None
) -> Dict[str, float]:
    """Compute features for groups already validated by :func:`resolve_groups`."""
    if not isinstance(text, str) or not text.strip():
        return {name: 0.0 for group in groups for name in _GROUPS[group].FEATURES}
    doc = Document(text if max_words is None else truncate_words(text, max_words))
    values: Dict[str, float] = {}
    for group in groups:
        values.update(_GROUPS[group].compute(doc))
    return {name: float(value) for name, value in values.items()}


def extract_features(
    text: Optional[str], groups: Groups = None, max_words: Optional[int] = None
) -> Dict[str, float]:
    """Extract named stylometric features from a text.

    Args:
        text: Input text. Non-string and blank values give an all-zero vector.
        groups: Feature groups to compute (names from :data:`ALL_GROUPS`). Defaults to
            :data:`DEFAULT_GROUPS`. The order of the result follows the registry, not ``groups``.
        max_words: Analyse only the first ``max_words`` words of the text (``None`` means all).

    Returns:
        dict mapping feature name to a finite float.

    Raises:
        ValueError: If ``groups`` contains an unknown name or is empty, or ``max_words`` is not a
            positive integer.
    """
    check_max_words(max_words)
    return compute_features(text, resolve_groups(groups), max_words)
