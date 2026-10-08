"""Morphological features from pymorphy3 analysis: part-of-speech and grammar profile.

Only the most probable parse of each word is used, without disambiguation in context, so
shares of homonymous forms are approximate. The profile follows the word-level characteristics
used for Russian text classification by Lagutina et al. (autosemantic and analytic words,
verb tenses, grammatical categories).
"""

from __future__ import annotations

from collections import Counter
from typing import Dict

from ru_stylometry._stats import mattr, safe_div
from ru_stylometry.document import Document

DESCRIPTION = "Part-of-speech and grammatical profile of Cyrillic words (pymorphy3)."

FEATURES: Dict[str, str] = {
    "noun_ratio": "Share of nouns among Cyrillic words.",
    "verb_ratio": "Share of finite verbs.",
    "infinitive_ratio": "Share of infinitives.",
    "adj_ratio": "Share of full and short adjectives.",
    "adv_ratio": "Share of adverbs.",
    "pronoun_ratio": "Share of noun pronouns (я, он, это, ...).",
    "prep_ratio": "Share of prepositions.",
    "conj_ratio": "Share of conjunctions.",
    "particle_ratio": "Share of particles.",
    "participle_ratio": "Share of participles (причастия).",
    "gerund_ratio": "Share of adverbial participles (деепричастия).",
    "autosemantic_ratio": "Share of content words (nouns, verbs, adjectives, adverbs, numerals).",
    "analytic_ratio": "Share of prepositions, conjunctions and particles.",
    "case_nomn_share": "Share of the nominative among case-marked words.",
    "case_gent_share": "Share of the genitive among case-marked words.",
    "case_datv_share": "Share of the dative among case-marked words.",
    "case_accs_share": "Share of the accusative among case-marked words.",
    "case_ablt_share": "Share of the instrumental among case-marked words.",
    "case_loct_share": "Share of the prepositional among case-marked words.",
    "verb_past_share": "Share of past-tense forms among finite verbs.",
    "verb_pres_share": "Share of present-tense forms among finite verbs.",
    "first_person_ratio": "Share of first-person forms (я, мы, мой, наш, 1st-person verbs).",
    "second_person_ratio": "Share of second-person forms (ты, вы, твой, ваш, 2nd-person verbs).",
    "unknown_word_ratio": "Share of words missing from the dictionary (typos, slang, rare names).",
    "lemma_mattr_50": "Moving-average type-token ratio over lemmas, window of 50 words.",
    "wordforms_per_lemma": "Distinct word forms per distinct lemma (morphological variety).",
}

_AUTOSEMANTIC = frozenset(
    {"NOUN", "VERB", "INFN", "ADJF", "ADJS", "COMP", "ADVB", "PRTF", "PRTS", "GRND", "NUMR"}
)
_ANALYTIC = frozenset({"PREP", "CONJ", "PRCL"})
_CASES = ("nomn", "gent", "datv", "accs", "ablt", "loct")
# Second genitive/accusative/locative are folded into the main case.
_CASE_ALIASES = {"gen2": "gent", "acc2": "accs", "loc2": "loct"}
_FIRST_PERSON_LEMMAS = frozenset({"я", "мы", "мой", "наш"})
_SECOND_PERSON_LEMMAS = frozenset({"ты", "вы", "твой", "ваш"})


def compute(doc: Document) -> Dict[str, float]:
    infos = doc.word_infos
    n = len(infos)
    if not n:
        return {name: 0.0 for name in FEATURES}

    pos = Counter(info.pos for info in infos)

    def ratio(*tags: str) -> float:
        return safe_div(sum(pos[tag] for tag in tags), n)

    cases: Counter = Counter()
    for info in infos:
        if info.case:
            cases[_CASE_ALIASES.get(info.case, info.case)] += 1
    case_total = sum(cases[case] for case in _CASES)

    verbs = [info for info in infos if info.pos == "VERB"]
    first_person = sum(1 for i in infos if i.lemma in _FIRST_PERSON_LEMMAS or i.person == "1per")
    second_person = sum(1 for i in infos if i.lemma in _SECOND_PERSON_LEMMAS or i.person == "2per")
    lemmas = [info.lemma for info in infos]

    features = {
        "noun_ratio": ratio("NOUN"),
        "verb_ratio": ratio("VERB"),
        "infinitive_ratio": ratio("INFN"),
        "adj_ratio": ratio("ADJF", "ADJS"),
        "adv_ratio": ratio("ADVB"),
        "pronoun_ratio": ratio("NPRO"),
        "prep_ratio": ratio("PREP"),
        "conj_ratio": ratio("CONJ"),
        "particle_ratio": ratio("PRCL"),
        "participle_ratio": ratio("PRTF", "PRTS"),
        "gerund_ratio": ratio("GRND"),
        "autosemantic_ratio": ratio(*_AUTOSEMANTIC),
        "analytic_ratio": ratio(*_ANALYTIC),
    }
    for case in _CASES:
        features[f"case_{case}_share"] = safe_div(cases[case], case_total)
    features.update(
        {
            "verb_past_share": safe_div(sum(1 for v in verbs if v.tense == "past"), len(verbs)),
            "verb_pres_share": safe_div(sum(1 for v in verbs if v.tense == "pres"), len(verbs)),
            "first_person_ratio": safe_div(first_person, n),
            "second_person_ratio": safe_div(second_person, n),
            "unknown_word_ratio": safe_div(sum(1 for i in infos if not i.is_known), n),
            "lemma_mattr_50": mattr(lemmas, 50),
            "wordforms_per_lemma": safe_div(len(set(doc.cyrillic_words)), len(set(lemmas))),
        }
    )
    return features
