"""Command-line interface of ``ru-stylometry``.

Commands: ``features`` and ``list-features`` work with the feature extractor; ``train``,
``evaluate``, ``predict`` and ``explain`` work with a classifier trained on a labelled table.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ru_stylometry import __version__
from ru_stylometry.evaluation import evaluate_classification, format_report
from ru_stylometry.features import (
    ALL_GROUPS,
    DEFAULT_GROUPS,
    describe_features,
    extract_features,
    group_description,
)
from ru_stylometry.model import ESTIMATORS, StylometricClassifier

Row = Tuple[str, str, str]


def _read_text(args: argparse.Namespace) -> str:
    if args.file is not None:
        return args.file.read_text(encoding="utf-8")
    if args.text is not None:
        return args.text
    return sys.stdin.read()


def _read_table(path: Path, text_column: str, label_column: str) -> Tuple[List[str], List[str]]:
    """Read texts and labels from a ``.csv``, ``.jsonl`` or ``.parquet`` file."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        csv.field_size_limit(sys.maxsize)
        with path.open(encoding="utf-8", newline="") as handle:
            records: List[Dict[str, Any]] = list(csv.DictReader(handle))
    elif suffix in (".jsonl", ".ndjson"):
        lines = path.read_text(encoding="utf-8").splitlines()
        records = [json.loads(line) for line in lines if line.strip()]
    elif suffix == ".parquet":
        try:
            import pandas as pd
        except ImportError as error:
            raise ValueError(
                "Reading Parquet needs pandas and pyarrow: pip install ru-stylometry[experiments]"
            ) from error
        records = pd.read_parquet(path, columns=[text_column, label_column]).to_dict("records")
    else:
        raise ValueError(f"Unsupported file type {suffix!r}; use .csv, .jsonl or .parquet.")
    for column in (text_column, label_column):
        if not records or column not in records[0]:
            available = sorted(records[0]) if records else []
            raise ValueError(f"Column {column!r} not found in {path}; available: {available}.")
    return [str(r[text_column]) for r in records], [str(r[label_column]) for r in records]


def _markdown_catalogue(rows: Sequence[Row]) -> str:
    by_group: Dict[str, List[Tuple[str, str]]] = {}
    for group, name, description in rows:
        by_group.setdefault(group, []).append((name, description))
    default_count = sum(1 for group, _, _ in rows if group in DEFAULT_GROUPS)
    lines = [
        "# Feature catalogue",
        "",
        f"{len(rows)} features in {len(by_group)} groups. The {default_count} features of the "
        "default groups form the default feature vector; groups marked *opt-in* are excluded "
        "from it.",
    ]
    for group, items in by_group.items():
        status = "default" if group in DEFAULT_GROUPS else "opt-in"
        lines += [
            "",
            f"## `{group}` ({status}, {len(items)} features)",
            "",
            group_description(group),
            "",
            "| Feature | Description |",
            "|---|---|",
        ]
        lines += [f"| `{name}` | {description} |" for name, description in items]
    return "\n".join(lines)


def _dump(data: Any) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2))


def _cmd_features(args: argparse.Namespace) -> int:
    _dump(extract_features(_read_text(args), groups=args.groups, max_words=args.max_words))
    return 0


def _cmd_list_features(args: argparse.Namespace) -> int:
    rows = describe_features(args.groups or ALL_GROUPS)
    if args.format == "json":
        _dump([{"group": g, "name": n, "description": d} for g, n, d in rows])
    elif args.format == "markdown":
        print(_markdown_catalogue(rows))
    else:
        width = max(len(name) for _, name, _ in rows)
        for group, name, description in rows:
            print(f"{group:<12} {name:<{width}}  {description}")
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    texts, labels = _read_table(args.data, args.text_column, args.label_column)
    model = StylometricClassifier(
        estimator=args.estimator,
        groups=args.groups,
        max_words=args.max_words,
        class_weight="balanced" if args.balanced else None,
        n_jobs=args.n_jobs,
    )
    started = time.perf_counter()
    model.fit(texts, labels)
    summary: Dict[str, Any] = {
        "model": str(args.out),
        "classes": [str(c) for c in model.classes_],
        "texts": len(texts),
        "features": len(model.feature_names_),
        "seconds": round(time.perf_counter() - started, 1),
    }
    model.save(str(args.out))
    if args.valid is not None:
        valid_texts, valid_labels = _read_table(args.valid, args.text_column, args.label_column)
        report = evaluate_classification(valid_labels, model.predict(valid_texts), model.classes_)
        summary["valid"] = {k: report[k] for k in ("n", "accuracy", "macro_f1")}
    _dump(summary)
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    model = StylometricClassifier.load(str(args.model))
    texts, labels = _read_table(args.data, args.text_column, args.label_column)
    started = time.perf_counter()
    predicted = model.predict(texts)
    seconds = time.perf_counter() - started
    report = evaluate_classification(labels, predicted, labels=[str(c) for c in model.classes_])
    report["ms_per_document"] = 1000.0 * seconds / max(len(texts), 1)
    if args.format == "json":
        _dump(report)
    else:
        print(format_report(report))
        print(f"\nTime per document: {report['ms_per_document']:.1f} ms")
    return 0


def _cmd_predict(args: argparse.Namespace) -> int:
    model = StylometricClassifier.load(str(args.model))
    text = _read_text(args)
    probabilities = model.predict_proba([text])[0]
    best = int(probabilities.argmax())
    _dump(
        {
            "label": str(model.classes_[best]),
            "probability": float(probabilities[best]),
            "probabilities": {str(c): float(p) for c, p in zip(model.classes_, probabilities)},
        }
    )
    return 0


def _cmd_explain(args: argparse.Namespace) -> int:
    model = StylometricClassifier.load(str(args.model))
    explanation = model.explain(_read_text(args), target=args.target, top_k=args.top)
    if args.format == "json":
        _dump(explanation)
        return 0
    print(f"label: {explanation['label']} (probability {explanation['probability']:.3f})")
    print(f"effect on the log-odds of class '{explanation['target']}' (positive: towards it):")
    for item in explanation["features"]:
        print(
            f"  {item['feature']:<26} {item['effect']:+.2f}   value {item['value']:.3f}, "
            f"typical {item['typical']:.3f}   [{item['group']}]"
        )
    print(
        "by group: "
        + ", ".join(f"{g} {v['effect']:+.2f}" for g, v in explanation["groups"].items())
    )
    return 0


def _add_text_input(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("text", nargs="?", help="text to analyse (default: read from stdin)")
    parser.add_argument("-f", "--file", type=Path, help="read the text from a UTF-8 file")


def _add_table_columns(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--text-column", default="text", help="column with the texts")
    parser.add_argument("--label-column", default="label", help="column with the class labels")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ru-stylometry",
        description="Interpretable stylometric features and classifiers for Russian text.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    features = commands.add_parser("features", help="extract features from a text as JSON")
    _add_text_input(features)
    features.add_argument("-g", "--groups", nargs="+", metavar="GROUP", help="feature groups")
    features.add_argument(
        "--max-words", type=int, metavar="N", help="analyse only the first N words"
    )
    features.set_defaults(handler=_cmd_features)

    catalogue = commands.add_parser("list-features", help="print the feature catalogue")
    catalogue.add_argument("-g", "--groups", nargs="+", metavar="GROUP", help="feature groups")
    catalogue.add_argument("--format", choices=["text", "markdown", "json"], default="text")
    catalogue.set_defaults(handler=_cmd_list_features)

    train = commands.add_parser("train", help="train a classifier on a labelled table")
    train.add_argument(
        "data", type=Path, help=".csv, .jsonl or .parquet file with texts and labels"
    )
    train.add_argument("-o", "--out", type=Path, required=True, help="where to save the model")
    _add_table_columns(train)
    train.add_argument("--estimator", choices=ESTIMATORS, default="hgb")
    train.add_argument("-g", "--groups", nargs="+", metavar="GROUP", help="feature groups")
    train.add_argument("--max-words", type=int, default=500, metavar="N")
    train.add_argument("--balanced", action="store_true", help="weight classes equally")
    train.add_argument("--n-jobs", type=int, help="worker processes for feature extraction")
    train.add_argument("--valid", type=Path, help="labelled file to score the model on")
    train.set_defaults(handler=_cmd_train)

    evaluate = commands.add_parser("evaluate", help="score a saved model on a labelled table")
    evaluate.add_argument("data", type=Path, help=".csv, .jsonl or .parquet file")
    evaluate.add_argument("-m", "--model", type=Path, required=True)
    _add_table_columns(evaluate)
    evaluate.add_argument("--format", choices=["markdown", "json"], default="markdown")
    evaluate.set_defaults(handler=_cmd_evaluate)

    predict = commands.add_parser("predict", help="classify one text")
    _add_text_input(predict)
    predict.add_argument("-m", "--model", type=Path, required=True)
    predict.set_defaults(handler=_cmd_predict)

    explain = commands.add_parser("explain", help="explain the decision for one text")
    _add_text_input(explain)
    explain.add_argument("-m", "--model", type=Path, required=True)
    explain.add_argument("--target", help="class to explain (default: the predicted class)")
    explain.add_argument("--top", type=int, default=10, help="number of features to list")
    explain.add_argument("--format", choices=["text", "json"], default="text")
    explain.set_defaults(handler=_cmd_explain)

    parser.epilog = f"feature groups: {', '.join(ALL_GROUPS)}"
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point of the ``ru-stylometry`` command."""
    args = _build_parser().parse_args(argv)
    try:
        return args.handler(args)
    except (ValueError, TypeError, OSError) as error:
        print(f"ru-stylometry: error: {error}", file=sys.stderr)
        return 2
