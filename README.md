# ru-stylometry

Interpretable stylometric features for Russian text, and a classifier that tells human-written text
from machine-generated text (and, if asked, the model family behind it) with an explanation of
every decision.

> **Status: alpha.** The feature extractor, the scikit-learn transformer, the classifier with
> explanations, the evaluation tools and the command-line interface are implemented. The package is
> not on the main PyPI index yet; test builds are published to TestPyPI.

It continues [`stylometric-ai-detector`](https://pypi.org/project/stylometric-ai-detector/), an
English baseline built on 8 surface features. Those features are mostly raw counts that reflect
text length, and nothing in them is specific to Russian. `ru-stylometry` replaces them with a
documented vector of 66 features that are normalised by text length and use Russian morphology.

## Installation

Test builds come from TestPyPI; the second index supplies the dependencies:

```bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ ru-stylometry
```

Once the package is released on PyPI, `pip install ru-stylometry` is enough. The latest code can be
installed straight from GitHub, and from a clone of the repository with extras:

```bash
pip install git+https://github.com/dinis-a/ru-stylometry.git   # the latest code from GitHub
pip install .                # from a clone: library and command-line tool
pip install .[fast]          # + C backend that speeds up morphological analysis
pip install .[experiments]   # + pandas, pyarrow, ...: reading Parquet tables
pip install -e .[dev]        # editable install + pytest, black, isort, build
```

Python 3.9+ is required. Dependencies: numpy, scikit-learn, joblib, pymorphy3.

## Quick start

### Features

```python
from ru_stylometry import StylometricVectorizer, extract_features, feature_names

features = extract_features("Мама мыла раму. Рама была чистой!")
features["avg_sentence_len"]   # 3.0
features["exclam_per100w"]     # 16.67
features["noun_ratio"]         # 0.67

len(feature_names())           # 66

# scikit-learn transformer: iterable of texts -> (n_texts, n_features) matrix
X = StylometricVectorizer(max_words=500, n_jobs=-1).fit_transform(["Первый текст.", "Второй текст."])
```

Pick feature groups explicitly, for example for an ablation study:

```python
extract_features(text, groups=["char", "lexical", "syntax"])
StylometricVectorizer(groups=["morph"])
```

The order of the columns is fixed by the registry, not by the order of `groups`, so the same
selection always gives the same layout. Blank or non-string input gives an all-zero vector.

### Classifier

```python
from ru_stylometry import StylometricClassifier

model = StylometricClassifier(estimator="hgb", class_weight="balanced")
model.fit(train_texts, train_labels)          # labels: any strings, two or more classes
model.predict(["Текст для проверки."])
model.predict_proba(["Текст для проверки."])

model.explain("Текст для проверки.", top_k=5)
# effect of every feature and feature group on the evidence for the predicted class

model.save("model.joblib")
StylometricClassifier.load("model.joblib")    # load only files you trust: the format is pickle
```

The *effect* of a feature is the fall of the class log-odds when the feature is replaced by its
typical value (the training median); positive effects push towards the class. Log-odds are used
because the probability of a confident decision is saturated and hides which features carry the
evidence. Effects are not additive, because correlated features share the credit; the effect of a
whole group is therefore reported separately.

Output for a machine-generated review from the LLMTrace test split, with a model trained on
30,000 texts of the same corpus:

```text
$ ru-stylometry explain --file review.txt -m model.joblib --top 3
label: ai (probability 0.998)
effect on the log-odds of class 'ai' (positive: towards it):
  lemma_mattr_50             +0.82   value 0.934, typical 0.857   [morph]
  conj_ratio                 +0.82   value 0.116, typical 0.084   [morph]
  verb_pres_share            +0.68   value 1.000, typical 0.429   [morph]
by group: morph +4.59, lexical +0.79, char +0.51, readability +0.43, syntax +0.21, ...
```

## Command line

```bash
ru-stylometry features "Мама мыла раму." --groups length syntax    # JSON to stdout
ru-stylometry list-features --format markdown                       # the feature catalogue

ru-stylometry train corpus.parquet -o model.joblib --estimator hgb --balanced --valid dev.parquet
ru-stylometry evaluate test.parquet -m model.joblib                 # metrics and time per document
ru-stylometry predict "Текст для проверки." -m model.joblib
ru-stylometry explain "Текст для проверки." -m model.joblib --top 10
```

Tables may be `.csv`, `.jsonl` or `.parquet` with a text column and a label column
(`--text-column`, `--label-column`).

## Feature groups

| Group         | Features | Default | What it measures                                                  |
|---------------|---------:|:-------:|-------------------------------------------------------------------|
| `char`        |        7 |   yes   | letters, digits, punctuation, symbols, capitals, Latin, use of ё  |
| `typography`  |       11 |   yes   | punctuation density; dash, quotation-mark and ellipsis habits     |
| `lexical`     |        7 |   yes   | word length; MATTR, MTLD, Yule's K, hapax share                   |
| `syntax`      |        9 |   yes   | sentence length distribution, function words, negation, openers   |
| `readability` |        3 |   yes   | syllables per word, polysyllabic words, LIX                       |
| `repetition`  |        3 |   yes   | deflate compression ratio, repeated bigrams and trigrams          |
| `morph`       |       26 |   yes   | part of speech, case, tense, person, unknown words, lemma variety |
| `formatting`  |        3 |   no    | line breaks, list items, Markdown symbols                         |
| `length`      |        4 |   no    | raw counts of characters, words, sentences, lines                 |

Every feature has a name and a one-line definition; `ru-stylometry list-features --format markdown`
prints the whole catalogue.

`formatting` and `length` are opt-in because they depend on how a corpus was collected more than on
style: texts of different classes often differ in length and layout for reasons that have nothing
to do with authorship.

## Evaluation, explanation and robustness

| module | what it offers |
|:--|:--|
| `ru_stylometry.evaluation` | accuracy, macro-F1, precision and recall of every class, confusion matrix, time per document, bootstrap intervals that resample groups of related texts |
| `ru_stylometry.explain` | effect of every feature and group on one decision; permutation importance of whole feature groups |
| `ru_stylometry.perturb` | text transformations for robustness checks: typography, Markdown, discourse markers, sentence order, typos |

Macro averaging counts every class equally, so a weak result on a rare source is not hidden by the
large classes.

## How well it works

Measured by the authors (the experiment scripts are not part of this repository) on the Russian
part of the LLMTrace corpus ([arXiv:2509.21269](https://arxiv.org/abs/2509.21269): 340,197 texts by
humans and 31 language models in eight genres). The official test split has 52,521 texts that share
no topic with the training split; the classifier is gradient boosting on the first 500 words of
every text:

| features | number | accuracy | macro-F1 | AUC |
|:--|--:|--:|--:|--:|
| length of the text only | 2 | 0.705 | 0.684 | 0.754 |
| the 8 features of `stylometric-ai-detector` | 8 | 0.751 | 0.735 | 0.820 |
| **`ru-stylometry`, default groups** | 66 | **0.883** | **0.879** | **0.954** |

The 95% interval of the accuracy of the 66 features is [0.881, 0.886]; it resamples whole topics,
because texts about the same subject are not independent. For the 15 classes (humans and 14 model
families) the macro-F1 is 0.446 against 0.195 for the 8 legacy features.

- **A fine-tuned language model is stronger.** `rubert-base-cased` fine-tuned on the same split
  reaches an accuracy of 0.953, about 7 points more. The stylometric classifier is a 1.2 MB file, needs
  no GPU (about 3.4 ms per text on one CPU core with a warm cache, around 290 texts per second) and
  explains every decision. Adding the 66 features to the outputs of the language model gave 1.4-2.6
  points of macro-F1 over the model families and about 0.1 point of accuracy for human vs machine.
- **The decision threshold does not carry over to another genre.** Trained on LLMTrace and tested on
  scientific abstracts (AINL-Eval 2025, [arXiv:2508.09622](https://arxiv.org/abs/2508.09622)), the
  ranking of the texts partly transfers (AUC 0.769) but only 36% of the human abstracts are
  recognised: formal human text is taken for machine text. Calibrate the threshold on the genre you
  are going to classify.

## Design notes

- **No raw counts by default.** Features are shares, densities per 100 words, means, or indices
  designed to be comparable between texts of different length (MATTR and MTLD instead of the plain
  type-token ratio).
- **Morphology comes from pymorphy3**, using the most probable parse of each word without context.
  Analyses are cached per word, so a word is parsed once per process. Importing the package does
  not load the dictionaries; the first use of the `morph` group does.
- **Sentence splitting is rule-based**: it knows common Russian abbreviations (т.д., г., ул.),
  initials, ellipses, closing quotes and list items.
- **Lexicons are heuristics.** The lists of function words and discourse markers are hand-made and
  small; treat the features built on them as interpretable markers rather than measurements.
- Flesch-type readability formulas are left out on purpose: they are linear combinations of
  features that are already in the vector, and published Russian coefficients differ.

## Limitations

- Short texts (below roughly 50 words) give noisy lexical-diversity and sentence-rhythm values.
- Homonymous word forms are resolved by frequency, not by context, so part-of-speech shares are
  approximate.
- Stylometric features describe style, not truth or authorship in a legal sense: a classifier built
  on them gives a probability that depends on the corpus it was trained on, and it should not be
  used as the only evidence against a person.

## Roadmap

- a stacking classifier as a class of the library: boosting on the outputs of a fine-tuned language
  model and the 66 features. In the authors' experiments it did as well as or better than training
  the two jointly inside one network, so it is the design to implement;
- an open-set decision for texts of unseen generators.

## Development

```bash
pip install -e .[dev]
isort . && black .       # style: black and isort, line length 100 (see pyproject.toml)
python -m build          # sdist and wheel in dist/
```

Pushing a version tag (`git tag v0.1.0 && git push origin v0.1.0`) starts the GitHub Actions
workflow: it checks the style, installs the package on Python 3.9-3.12, builds it and uploads it to
TestPyPI with the repository secret `TESTPYPI_TOKEN`. TestPyPI accepts a version only once, so raise
`__version__` in `src/ru_stylometry/__init__.py` before tagging.

## License

MIT, see the `LICENSE` file.
