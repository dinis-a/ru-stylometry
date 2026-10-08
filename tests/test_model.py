"""Tests for the end-to-end classifier and its explanations."""

import csv
import io
from pathlib import Path

import numpy as np
import pytest
from sklearn.base import clone

from ru_stylometry import StylometricClassifier
from ru_stylometry.model import make_estimator

TOY = Path(__file__).parent / "data" / "toy.csv"
EXCITED_TEXT = "Какой прекрасный день! Мы победили! Это невероятно!"


def toy_rows():
    with TOY.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [r["text"] for r in rows], [r["label"] for r in rows]


def imbalanced_corpus():
    """20 excited and 60 calm texts: the typical value of the only informative feature is 0."""
    texts, labels = toy_rows()
    calm = [(t, y) for t, y in zip(texts, labels) if y == "calm"]
    excited = [(t, y) for t, y in zip(texts, labels) if y == "excited"]
    pairs = excited + calm * 3
    return [t for t, _ in pairs], [y for _, y in pairs]


@pytest.mark.parametrize("estimator", ["hgb", "logreg", "rf"])
def test_fit_predict_and_probabilities(estimator):
    texts, labels = toy_rows()
    model = StylometricClassifier(estimator=estimator, max_words=None).fit(texts, labels)
    assert list(model.classes_) == ["calm", "excited"]
    assert list(model.predict(texts)) == labels
    probabilities = model.predict_proba(texts)
    assert probabilities.shape == (len(texts), 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_unknown_estimator_raises():
    with pytest.raises(ValueError, match="Unknown estimator"):
        make_estimator("svm", n_classes=2)


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError, match="labels"):
        StylometricClassifier().fit(["один текст", "второй текст"], ["a"])


def test_multiclass_and_balanced_weights():
    texts = ["Раз два три.", "Раз два три!", "Раз два три?"] * 10
    labels = ["a", "b", "c"] * 10
    model = StylometricClassifier(estimator="logreg", class_weight="balanced", max_words=None)
    assert list(model.fit(texts, labels).predict(texts)) == labels
    assert list(model.classes_) == ["a", "b", "c"]


def test_max_words_is_applied_at_prediction_time():
    texts, labels = toy_rows()
    model = StylometricClassifier(estimator="logreg", max_words=3).fit(texts, labels)
    # Only the first three words ("Какой прекрасный день") are analysed, so the tail is ignored.
    assert np.array_equal(model.features([EXCITED_TEXT]), model.features(["Какой прекрасный день"]))


def test_explanation_points_to_the_only_informative_feature():
    texts, labels = imbalanced_corpus()
    model = StylometricClassifier(estimator="hgb", max_words=None).fit(texts, labels)
    result = model.explain(EXCITED_TEXT, top_k=3)
    assert result["label"] == result["target"] == "excited"
    best = result["features"][0]
    assert best["feature"] == "exclam_per100w"
    assert best["group"] == "typography"
    assert best["effect"] > 5  # log-odds: removing the exclamation marks flips a sure decision
    assert 0 < best["probability_change"] <= 1
    assert next(iter(result["groups"])) == "typography"
    assert result["groups"]["typography"]["effect"] == pytest.approx(best["effect"], rel=0.2)
    assert len(result["features"]) == 3
    assert result["probabilities"]["excited"] == pytest.approx(result["probability"])


def test_explaining_the_other_class_flips_the_sign():
    texts, labels = imbalanced_corpus()
    model = StylometricClassifier(estimator="hgb", max_words=None).fit(texts, labels)
    result = model.explain(EXCITED_TEXT, target="calm", top_k=1)
    assert result["target"] == "calm"
    assert result["features"][0]["feature"] == "exclam_per100w"
    assert result["features"][0]["effect"] < -5


def test_save_and_load_roundtrip_in_memory():
    texts, labels = toy_rows()
    model = StylometricClassifier(estimator="logreg", max_words=None).fit(texts, labels)
    buffer = io.BytesIO()
    model.save(buffer)
    buffer.seek(0)
    restored = StylometricClassifier.load(buffer)
    assert np.allclose(restored.predict_proba(texts), model.predict_proba(texts))


def test_loading_a_foreign_object_is_rejected():
    import joblib

    buffer = io.BytesIO()
    joblib.dump({"not": "a model"}, buffer)
    buffer.seek(0)
    with pytest.raises(TypeError, match="StylometricClassifier"):
        StylometricClassifier.load(buffer)


def test_unfitted_model_raises():
    from sklearn.exceptions import NotFittedError

    with pytest.raises(NotFittedError):
        StylometricClassifier().predict(["текст"])


def test_params_are_clonable():
    model = StylometricClassifier(estimator="rf", groups=["char"], max_words=100)
    assert clone(model).get_params() == model.get_params()
