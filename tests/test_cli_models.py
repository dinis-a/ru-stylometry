"""Tests for the model-related commands of the command-line interface."""

import json
from pathlib import Path

import joblib
import pytest

from ru_stylometry.cli import main

TOY = Path(__file__).parent / "data" / "toy.csv"
EXCITED_TEXT = "Какой прекрасный день! Мы победили! Это невероятно!"


@pytest.fixture(scope="module")
def model_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("models") / "toy.joblib"
    exit_code = main(
        ["train", str(TOY), "-o", str(path), "--estimator", "logreg", "--max-words", "50"]
    )
    assert exit_code == 0
    return path


def test_train_reports_a_summary(tmp_path, capsys):
    out = tmp_path / "model.joblib"
    assert (
        main(["train", str(TOY), "-o", str(out), "--estimator", "logreg", "--valid", str(TOY)]) == 0
    )
    summary = json.loads(capsys.readouterr().out)
    assert out.exists()
    assert summary["classes"] == ["calm", "excited"]
    assert summary["texts"] == 40
    assert summary["valid"]["accuracy"] == 1.0


def test_evaluate_prints_a_markdown_report(model_path, capsys):
    assert main(["evaluate", str(TOY), "-m", str(model_path)]) == 0
    output = capsys.readouterr().out
    assert "| **macro** |" in output
    assert "Accuracy: 1.000" in output
    assert "Time per document" in output


def test_evaluate_json(model_path, capsys):
    assert main(["evaluate", str(TOY), "-m", str(model_path), "--format", "json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["accuracy"] == 1.0 and report["ms_per_document"] > 0


def test_predict(model_path, capsys):
    assert main(["predict", EXCITED_TEXT, "-m", str(model_path)]) == 0
    prediction = json.loads(capsys.readouterr().out)
    assert prediction["label"] == "excited"
    assert sum(prediction["probabilities"].values()) == pytest.approx(1.0)


def test_explain_as_text(model_path, capsys):
    assert main(["explain", EXCITED_TEXT, "-m", str(model_path), "--top", "3"]) == 0
    output = capsys.readouterr().out
    assert output.startswith("label: excited")
    assert "exclam_per100w" in output
    assert "by group:" in output


def test_explain_as_json_for_the_other_class(model_path, capsys):
    args = ["explain", EXCITED_TEXT, "-m", str(model_path), "--target", "calm", "--format", "json"]
    assert main(args) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["target"] == "calm"
    assert len(result["features"]) == 10


def test_train_with_a_missing_column_fails_cleanly(tmp_path, capsys):
    args = ["train", str(TOY), "-o", str(tmp_path / "m.joblib"), "--label-column", "nope"]
    assert main(args) == 2
    assert "Column 'nope' not found" in capsys.readouterr().err


def test_unsupported_table_format_fails_cleanly(tmp_path, capsys):
    table = tmp_path / "data.xlsx"
    table.write_bytes(b"")
    assert main(["train", str(table), "-o", str(tmp_path / "m.joblib")]) == 2
    assert "Unsupported file type" in capsys.readouterr().err


def test_a_file_that_is_not_a_model_fails_cleanly(tmp_path, capsys):
    path = tmp_path / "other.joblib"
    joblib.dump({"not": "a model"}, path)
    assert main(["predict", "текст", "-m", str(path)]) == 2
    assert "StylometricClassifier" in capsys.readouterr().err


def test_missing_model_file_fails_cleanly(tmp_path, capsys):
    assert main(["predict", "текст", "-m", str(tmp_path / "absent.joblib")]) == 2
    assert "ru-stylometry: error" in capsys.readouterr().err
