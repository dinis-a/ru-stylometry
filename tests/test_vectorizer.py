"""Tests for the scikit-learn transformer."""

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ru_stylometry import StylometricVectorizer, extract_features, feature_names

TEXTS = ["Мама мыла раму.", "Красивая девушка читает интересную книгу.", ""]


def test_fit_transform_returns_finite_matrix_with_named_columns():
    vectorizer = StylometricVectorizer()
    X = vectorizer.fit_transform(TEXTS)
    assert X.shape == (3, len(feature_names()))
    assert X.dtype == float
    assert np.isfinite(X).all()
    assert list(vectorizer.get_feature_names_out()) == feature_names()


def test_blank_text_gives_a_zero_row():
    X = StylometricVectorizer().fit_transform(TEXTS)
    assert not X[2].any()
    assert X[0].any()


def test_groups_parameter_selects_columns():
    vectorizer = StylometricVectorizer(groups=["char", "lexical"]).fit(TEXTS)
    assert list(vectorizer.get_feature_names_out()) == feature_names(["char", "lexical"])
    assert vectorizer.transform(TEXTS).shape == (3, 14)


def test_rows_match_extract_features():
    X = StylometricVectorizer().fit_transform([TEXTS[1]])
    assert X[0] == pytest.approx(list(extract_features(TEXTS[1]).values()))


def test_empty_input_gives_empty_matrix():
    vectorizer = StylometricVectorizer().fit()
    assert vectorizer.transform([]).shape == (0, len(feature_names()))


def test_a_single_string_is_rejected():
    with pytest.raises(TypeError, match="iterable of texts"):
        StylometricVectorizer().fit().transform("Мама мыла раму.")


def test_transform_before_fit_raises():
    with pytest.raises(NotFittedError):
        StylometricVectorizer().transform(TEXTS)


def test_unknown_group_fails_at_fit_time():
    with pytest.raises(ValueError, match="Unknown feature group"):
        StylometricVectorizer(groups=["nope"]).fit()


def test_set_output_pandas_keeps_feature_names():
    pd = pytest.importorskip("pandas")
    frame = (
        StylometricVectorizer(groups=["char"]).set_output(transform="pandas").fit_transform(TEXTS)
    )
    assert isinstance(frame, pd.DataFrame)
    assert list(frame.columns) == feature_names(["char"])


def test_can_be_cloned_and_used_in_a_pipeline():
    texts = ["Кроме того, важно отметить следующее.", "ну короче я пошёл, а ты как?"] * 10
    labels = [1, 0] * 10
    model = make_pipeline(
        StylometricVectorizer(), StandardScaler(), LogisticRegression(max_iter=1000)
    )
    model.fit(texts, labels)
    assert list(model.predict(texts)) == labels
    assert clone(model).named_steps["stylometricvectorizer"].groups is None
