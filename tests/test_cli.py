"""Tests for the command-line interface."""

import io
import json
from pathlib import Path

from ru_stylometry import feature_names
from ru_stylometry.cli import main

SAMPLE_FILE = Path(__file__).parent / "data" / "sample.txt"


def test_features_from_argument_prints_json(capsys):
    assert main(["features", "Мама мыла раму.", "--groups", "char"]) == 0
    assert list(json.loads(capsys.readouterr().out)) == feature_names(["char"])


def test_features_from_file(capsys):
    assert main(["features", "--file", str(SAMPLE_FILE), "-g", "length"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["word_count"] == 6
    assert result["sentence_count"] == 2


def test_features_from_stdin(capsys, monkeypatch):
    monkeypatch.setattr("sys.stdin", io.StringIO("Мама мыла раму."))
    assert main(["features", "-g", "length"]) == 0
    assert json.loads(capsys.readouterr().out)["word_count"] == 3


def test_unknown_group_is_reported_with_exit_code_2(capsys):
    assert main(["features", "текст", "-g", "nope"]) == 2
    assert "Unknown feature group" in capsys.readouterr().err


def test_list_features_markdown(capsys):
    assert main(["list-features", "--format", "markdown"]) == 0
    output = capsys.readouterr().out
    assert output.startswith("# Feature catalogue")
    assert "| `letter_ratio` |" in output
    assert "(opt-in, 4 features)" in output


def test_list_features_json(capsys):
    assert main(["list-features", "--format", "json", "-g", "length"]) == 0
    records = json.loads(capsys.readouterr().out)
    assert [record["name"] for record in records] == feature_names(["length"])


def test_list_features_plain_text(capsys):
    assert main(["list-features", "-g", "char"]) == 0
    assert "letter_ratio" in capsys.readouterr().out
