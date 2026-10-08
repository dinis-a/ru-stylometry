"""Tests for the individual feature groups, with values derived by hand from the definitions."""

import pytest

from ru_stylometry import extract_features

SAMPLE = "Мама мыла раму. Рама была чистой!"  # 33 characters, 6 words, 2 sentences


# --- char ------------------------------------------------------------------------------------


def test_char_composition():
    f = extract_features(SAMPLE, ["char"])
    assert f["letter_ratio"] == pytest.approx(26 / 33)
    assert f["punct_ratio"] == pytest.approx(2 / 33)
    assert f["upper_ratio"] == pytest.approx(2 / 26)
    assert f["digit_ratio"] == 0.0
    assert f["symbol_ratio"] == 0.0
    assert f["latin_ratio"] == 0.0
    assert f["yo_ratio"] == 0.0


def test_char_yo_ratio_counts_consistent_use_of_yo():
    # "ё": Ёлка, зелёная; plain "е": зелёная, ель, нет.
    assert extract_features("Ёлка зелёная, а ель — нет.", ["char"])["yo_ratio"] == pytest.approx(
        0.4
    )


def test_char_latin_and_upper_ratio():
    f = extract_features("Hello мир", ["char"])
    assert f["latin_ratio"] == pytest.approx(5 / 8)
    assert f["upper_ratio"] == pytest.approx(1 / 8)


def test_char_symbol_ratio_counts_emoji():
    assert extract_features("ок 😀", ["char"])["symbol_ratio"] == pytest.approx(1 / 4)


# --- typography ------------------------------------------------------------------------------


def test_typography_of_typeset_text():
    f = extract_features("«Привет» — сказал он — и ушёл.", ["typography"])  # 5 words
    assert f["dash_per100w"] == pytest.approx(40.0)
    assert f["em_dash_share"] == 1.0
    assert f["quote_per100w"] == pytest.approx(40.0)
    assert f["guillemet_share"] == 1.0


def test_typography_of_keyboard_typed_text():
    f = extract_features('Он сказал "привет" - и ушёл.', ["typography"])  # 5 words
    assert f["dash_per100w"] == pytest.approx(20.0)
    assert f["em_dash_share"] == 0.0
    assert f["quote_per100w"] == pytest.approx(40.0)
    assert f["guillemet_share"] == 0.0


@pytest.mark.parametrize(
    "name",
    [
        "comma_per100w",
        "colon_per100w",
        "semicolon_per100w",
        "paren_per100w",
        "question_per100w",
        "exclam_per100w",
        "ellipsis_per100w",
    ],
)
def test_typography_punctuation_density(name):
    # Seven words and exactly one mark of every kind.
    f = extract_features("Да, нет; может быть: да?! Ну... (ладно)", ["typography"])
    assert f[name] == pytest.approx(100 / 7)


def test_typography_list_bullet_is_not_a_dash():
    assert extract_features("- пункт один\n- пункт два", ["typography"])["dash_per100w"] == 0.0


# --- lexical ---------------------------------------------------------------------------------


def test_lexical_basic_values():
    f = extract_features(SAMPLE, ["lexical"])
    assert f["avg_word_len"] == pytest.approx(26 / 6)
    assert f["std_word_len"] == pytest.approx((5 / 9) ** 0.5)
    assert f["long_word_ratio"] == 0.0
    assert f["mattr_50"] == 1.0
    assert f["hapax_ratio"] == 1.0
    assert f["yule_k"] == 0.0


def test_lexical_repetitive_text_has_low_diversity():
    f = extract_features("да да да да да", ["lexical"])
    assert f["mattr_50"] == pytest.approx(0.2)
    assert f["hapax_ratio"] == 0.0
    assert f["yule_k"] == pytest.approx(8000.0)
    assert f["mtld"] == pytest.approx(2.5)


def test_lexical_case_and_yo_do_not_create_new_types():
    assert extract_features("Ёлка ёлка ЕЛКА", ["lexical"])["hapax_ratio"] == pytest.approx(0.0)


# --- syntax ----------------------------------------------------------------------------------


def test_syntax_sentence_statistics():
    f = extract_features(SAMPLE, ["syntax"])
    assert f["avg_sentence_len"] == 3.0
    assert f["std_sentence_len"] == 0.0
    assert f["short_sentence_ratio"] == 1.0
    assert f["long_sentence_ratio"] == 0.0
    assert f["opener_diversity"] == 1.0
    assert f["function_word_ratio"] == pytest.approx(1 / 6)  # "была"


def test_syntax_long_sentence_ratio():
    long_sentence = " ".join(["слово"] * 25) + "."
    f = extract_features(f"{long_sentence} Коротко.", ["syntax"])
    assert f["long_sentence_ratio"] == pytest.approx(0.5)
    assert f["short_sentence_ratio"] == pytest.approx(0.5)


def test_syntax_conjunction_openers_and_repeated_openers():
    f = extract_features("Но так. Но этак. А там.", ["syntax"])
    assert f["conj_start_ratio"] == 1.0
    assert f["opener_diversity"] == pytest.approx(2 / 3)


def test_syntax_negation():
    assert extract_features("Я не знаю ничего.", ["syntax"])["negation_ratio"] == pytest.approx(0.5)


def test_syntax_discourse_markers():
    f = extract_features("Кроме того, это важно. Однако мы согласны.", ["syntax"])  # 7 words
    assert f["discourse_marker_per100w"] == pytest.approx(200 / 7)


# --- readability -----------------------------------------------------------------------------


def test_readability_of_short_words():
    f = extract_features("Мама мыла раму.", ["readability"])
    assert f["avg_syllables_per_word"] == 2.0
    assert f["polysyllabic_ratio"] == 0.0
    assert f["lix"] == pytest.approx(3.0)


def test_readability_long_words_raise_lix():
    f = extract_features("Информационные технологии развиваются.", ["readability"])
    assert f["polysyllabic_ratio"] == 1.0
    assert f["lix"] == pytest.approx(3.0 + 100.0)


# --- repetition ------------------------------------------------------------------------------


def test_repetition_of_repeated_ngrams():
    f = extract_features("да да да да", ["repetition"])
    assert f["repeated_bigram_ratio"] == 1.0
    assert f["repeated_trigram_ratio"] == 1.0


def test_repetition_of_distinct_ngrams():
    f = extract_features("один два три четыре", ["repetition"])
    assert f["repeated_bigram_ratio"] == 0.0
    assert f["repeated_trigram_ratio"] == 0.0


def test_repetition_compression_is_lower_for_redundant_text():
    redundant = extract_features("да " * 100, ["repetition"])["zlib_ratio"]
    varied = extract_features(
        "Каждое предложение здесь говорит о чём-то новом: погода, цены, футбол, музыка, грибы.",
        ["repetition"],
    )["zlib_ratio"]
    assert redundant < 0.1 < varied


# --- morph -----------------------------------------------------------------------------------


def test_morph_part_of_speech_profile():
    f = extract_features("Красивая девушка читает интересную книгу.", ["morph"])
    assert f["noun_ratio"] == pytest.approx(0.4)
    assert f["adj_ratio"] == pytest.approx(0.4)
    assert f["verb_ratio"] == pytest.approx(0.2)
    assert f["autosemantic_ratio"] == 1.0
    assert f["analytic_ratio"] == 0.0
    assert f["unknown_word_ratio"] == 0.0


def test_morph_function_words():
    f = extract_features("Он пошёл в лес и не вернулся.", ["morph"])  # 7 words
    assert f["analytic_ratio"] == pytest.approx(3 / 7)  # в, и, не
    assert f["prep_ratio"] == pytest.approx(1 / 7)


def test_morph_case_shares():
    f = extract_features("Красивая девушка читает интересную книгу.", ["morph"])
    assert f["case_nomn_share"] == pytest.approx(0.5)
    assert f["case_accs_share"] == pytest.approx(0.5)
    assert f["case_gent_share"] == 0.0


def test_morph_verb_tense_shares():
    f = extract_features("Он читал, читает и прочтёт.", ["morph"])
    assert f["verb_past_share"] == pytest.approx(1 / 3)
    assert f["verb_pres_share"] == pytest.approx(1 / 3)


def test_morph_first_and_second_person():
    first = extract_features("Я думаю, что мы правы.", ["morph"])
    assert first["first_person_ratio"] == pytest.approx(3 / 5)
    assert first["second_person_ratio"] == 0.0
    second = extract_features("Ты думаешь, что вы правы.", ["morph"])
    assert second["second_person_ratio"] == pytest.approx(3 / 5)
    assert second["first_person_ratio"] == 0.0


def test_morph_lemma_level_features():
    f = extract_features("Стол стола столу столом", ["morph"])
    assert f["wordforms_per_lemma"] == pytest.approx(4.0)
    assert f["lemma_mattr_50"] == pytest.approx(0.25)


def test_morph_unknown_words():
    assert extract_features("Ыыыкр блярк фыва", ["morph"])["unknown_word_ratio"] > 0.5


def test_morph_ignores_latin_words():
    assert set(extract_features("Python and Java", ["morph"]).values()) == {0.0}


# --- formatting and length (opt-in groups) ---------------------------------------------------


def test_formatting_detects_lists_and_markdown():
    text = "# Итоги\n\n- первый пункт\n- второй пункт\n\n**важно** помнить"  # 7 words
    f = extract_features(text, ["formatting"])
    assert f["list_line_ratio"] == pytest.approx(2 / 4)
    assert f["markdown_per100w"] == pytest.approx(500 / 7)
    assert f["newline_per100w"] == pytest.approx(500 / 7)


def test_length_counts():
    assert extract_features(SAMPLE, ["length"]) == {
        "char_count": 33.0,
        "word_count": 6.0,
        "sentence_count": 2.0,
        "line_count": 1.0,
    }
