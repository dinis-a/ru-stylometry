"""Small numeric helpers shared by feature groups."""

from __future__ import annotations

from collections import Counter
from typing import Hashable, Sequence


def safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Divide, returning ``default`` when the denominator is zero."""
    return numerator / denominator if denominator else default


def mean(values: Sequence[float]) -> float:
    """Arithmetic mean; 0.0 for an empty sequence."""
    return sum(values) / len(values) if values else 0.0


def pstdev(values: Sequence[float]) -> float:
    """Population standard deviation; 0.0 for an empty sequence."""
    if not values:
        return 0.0
    mu = mean(values)
    return (sum((v - mu) ** 2 for v in values) / len(values)) ** 0.5


def mattr(tokens: Sequence[Hashable], window: int = 50) -> float:
    """Moving-average type-token ratio (Covington & McFall, 2010).

    Mean share of distinct tokens over all windows of ``window`` consecutive tokens. Unlike the
    plain type-token ratio it does not fall as the text gets longer. Texts shorter than the window
    fall back to the plain type-token ratio.
    """
    n = len(tokens)
    if n == 0:
        return 0.0
    if n <= window:
        return len(set(tokens)) / n
    counts = Counter(tokens[:window])
    unique = len(counts)
    total = unique
    for i in range(window, n):
        leaving = tokens[i - window]
        counts[leaving] -= 1
        if counts[leaving] == 0:
            unique -= 1
        if counts[tokens[i]] == 0:
            unique += 1
        counts[tokens[i]] += 1
        total += unique
    return total / ((n - window + 1) * window)


def _mtld_pass(tokens: Sequence[Hashable], threshold: float) -> float:
    factors = 0.0
    types: set = set()
    count = 0
    for token in tokens:
        types.add(token)
        count += 1
        if len(types) / count <= threshold:
            factors += 1.0
            types = set()
            count = 0
    if count:
        factors += (1.0 - len(types) / count) / (1.0 - threshold)
    return len(tokens) / factors if factors else float(len(tokens))


def mtld(tokens: Sequence[Hashable], threshold: float = 0.72) -> float:
    """Measure of textual lexical diversity (McCarthy & Jarvis, 2010).

    Average length of a token run over which the type-token ratio stays above ``threshold``,
    computed forwards and backwards. A text in which no run ever drops to the threshold gets a value
    equal to its length.
    """
    if not tokens:
        return 0.0
    forward = _mtld_pass(tokens, threshold)
    backward = _mtld_pass(tokens[::-1], threshold)
    return (forward + backward) / 2.0


def yule_k(tokens: Sequence[Hashable]) -> float:
    """Yule's K characteristic (Yule, 1944): higher values mean more repetitive vocabulary."""
    n = len(tokens)
    if n == 0:
        return 0.0
    spectrum = Counter(Counter(tokens).values())
    s2 = sum(m * m * v for m, v in spectrum.items())
    return 1e4 * (s2 - n) / (n * n)
