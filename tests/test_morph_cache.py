"""Tests for the cached word analysis."""

from ru_stylometry._morph import analyze_word


def test_analyze_word_returns_the_grammatical_facts_used_by_features():
    info = analyze_word("книгу")
    assert info.pos == "NOUN"
    assert info.case == "accs"
    assert info.lemma == "книга"
    assert info.is_known


def test_analyze_word_flags_unknown_words():
    assert not analyze_word("ыыыкр").is_known


def test_analyze_word_is_cached_per_word():
    analyze_word.cache_clear()
    analyze_word("девушка")
    analyze_word("девушка")
    info = analyze_word.cache_info()
    assert (info.misses, info.hits) == (1, 1)
