# Modelling

Eleven simple lexical baselines use the prepared corpus and queries, require no
training, and depend only on the Python standard library. See the
[root README](../README.md#setup) for the project environment.

## Versions

Version numbers come first in script names, TREC filenames, and run tags.
The first two versions retain their original scoring behavior. New scoring
variants use successive minor numbers within the simple lexical-model family.

In the formulas below, `o` is the number of distinct shared words, `q` is the
number of distinct query words, `d` is the number of distinct document words,
and `tf(t)` is the number of occurrences of word `t` in the document. For IDF,
`N` is the number of corpus documents (including empty documents), and `df(t)`
is the number of documents containing word `t`.

| Version | Script | Score |
| --- | --- | --- |
| `1.1.1` | [term overlap](scripts/1.1.1-term-overlap.py) | `o` |
| `1.1.2` | [query coverage](scripts/1.1.2-term-overlap-percentage.py) | `o / q` |
| `1.2.1` | [document coverage](scripts/1.2.1-document-coverage.py) | `o / d` |
| `1.3.1` | [Jaccard](scripts/1.3.1-jaccard.py) | `o / (q + d - o)` |
| `1.4.1` | [binary cosine](scripts/1.4.1-binary-cosine.py) | `o / sqrt(q * d)` |
| `1.5.1` | [term frequency](scripts/1.5.1-term-frequency.py) | Sum of `tf(t)` over distinct query words |
| `1.5.2` | [log term frequency](scripts/1.5.2-log-term-frequency.py) | Sum of `ln(1 + tf(t))` over distinct query words |
| `1.6.1` | [bigram overlap](scripts/1.6.1-bigram-overlap.py) | Number of distinct adjacent word pairs shared by query and document |
| `1.7.1` | [stopword-filtered overlap](scripts/1.7.1-stopword-overlap.py) | `o` after removing a small explicit list of common words |
| `1.8.1` | [IDF-only overlap](scripts/1.8.1-idf-overlap.py) | Sum of `ln((N + 1) / (df(t) + 1)) + 1` over distinct shared words |
| `1.9.1` | [unigram + bigram overlap](scripts/1.9.1-unigram-bigram-overlap.py) | Distinct shared words plus distinct shared adjacent word pairs |

Each version is a complete, standalone script with its own tokenization,
indexing, scoring, input validation, and output handling. Code is intentionally
repeated so versions can be read and changed independently. There is no shared
modelling helper. Tests in `modelling/tests/` cover retrieval sidecar output.

Version 1.1.2 divides all scores for one query by the same constant, preserving
rankings and evaluation metrics. The document-length variants can change the
ranking: they calculate the final score before selecting the top results.

## Inputs and matching

For `--dataset scifact --split test`, the inputs are:

- `data-preparation/data/scifact/corpus.jsonl`
- `data-preparation/data/scifact/queries/test.jsonl`

Other prepared datasets work through the same `--dataset` argument. Query and
document IDs are preserved exactly, including leading zeros. Retrieval does
not read relevance judgments or use them to select candidates.

All versions lowercase text and extract Unicode alphanumeric sequences.
Punctuation and underscores separate words; accents and numbers are retained.
There is no stemming. Only version 1.8.1 uses IDF weighting.

- Set-based methods count repeated words once in both query and document.
  Query words absent from the corpus still contribute to query length.
- Frequency methods count repetitions in documents, but count each distinct
  query word once. Log frequency uses natural logarithms and accurate summation
  to keep equivalent totals tied consistently.
- Bigram overlap forms adjacent pairs after tokenization. Punctuation and sentence
  boundaries do not interrupt the resulting word sequence. Repeated pairs count
  once. A query with fewer than two words returns no results.
- IDF-only overlap calculates smoothed weights from the corpus alone. Repeated
  words contribute once; neither term frequency nor document length changes a
  matched word's contribution. A word in every document has weight 1, and rarer
  words have higher weights. Words absent from the corpus contribute nothing.
- Combined unigram-bigram overlap gives one point to each distinct shared word
  and one to each distinct shared adjacent word pair. There is no tuned bonus
  parameter. A single-word query still matches through its unigram. Pairs follow
  the same tokenization and punctuation rules as the bigram-only model.
- Stopword overlap removes words from both queries and documents using the small
  English or Italian lists embedded in its script. These are deliberately simple
  lists, not comprehensive linguistic resources. Negations such as `not` and
  `non` are retained. The language is inferred as English for SciFact/MS MARCO
  and Italian for JuriFindIT/mMARCO Italian. Use `--language en` or `--language it`
  to override it; other dataset names require an explicit language.

Each model builds an in-memory inverted index. Only documents with positive
scores are returned. Empty, punctuation-only, and unmatched queries produce no
rows; evaluation counts missing judged queries as zero. Ties use descending
document ID as strings, matching evaluation.

## Run

The normal workflow runs retrieval and evaluation together from the repository root:

```bash
python run_experiments.py --dataset scifact
```

See the [root workflow](../README.md#workflow) for method selection, dataset
selection, listing, and language overrides. The commands below are for running
retrieval separately.

Run these Linux commands from the repository root. For example, run Jaccard:

```bash
./.venv/bin/python modelling/scripts/1.3.1-jaccard.py --dataset scifact
```

To generate all eleven runs:

```bash
for script in modelling/scripts/*.py; do
    ./.venv/bin/python "$script" --dataset scifact || exit 1
done
```

Defaults are `--dataset scifact`, `--split test`, and `--top-k 1000`.
Use `--split train` for training queries, `--top-k N` to change retrieval depth,
and `--output-dir PATH` for a custom output directory. Explicit output paths are
relative to the working directory; default locations are relative to the scripts.
Use distinct output directories when comparing different settings of one version.

Each run rebuilds its index. The larger MARCO datasets require substantially more
RAM and processing time with these deliberately simple indexes. Full initial
runs and evaluation were performed on SciFact and JuriFindIT; the MARCO datasets
have not been run end to end with these methods.

## Outputs and evaluation

Runs are saved as `modelling/data/<dataset>/<split>/<version>-<method>.trec`.
For example:

```text
modelling/data/scifact/test/1.1.1-term-overlap.trec
modelling/data/scifact/test/1.1.2-term-overlap-percentage.trec
modelling/data/scifact/test/1.3.1-jaccard.trec
```

Each line uses the standard TREC format:

```text
query_id Q0 doc_id rank score version-method
```

Ranks start at 1. Each query has at most `--top-k` rows. Existing runs require
`--overwrite` to replace them; failed input validation leaves previous runs
intact. A sibling `<version>-<method>.metadata.json` records dataset, split,
method/version, configured `top_k`, and resolved language for the stopword model.
Keep this file with its ranking: evaluation uses it to log configuration to MLflow.
Both files are staged after input validation and replaced on successful retrieval.
They are not a filesystem transaction; an interrupted replacement may require
rerunning retrieval. Generated runs and sidecars under `modelling/data/` are ignored
by Git. Sidecars are descriptive metadata, not content fingerprints.

Retrieval stays independent of MLflow. See [Tracking](../tracking/README.md) for
storage, comparison, and the sidecar format for external methods.

Run and compare all current versions with:

```bash
python run_experiments.py --dataset scifact
```

This selects the scripts currently on disk and excludes retired ranking files.
For separately generated rankings, pass their exact paths to
[Evaluation](../evaluation/README.md).

The reports include aggregate and per-query metrics. Versions 1.1.1 and 1.1.2
should match exactly; the other variants can change rankings. One shared report
set is saved in `evaluation/data/scifact/test/`. The command replaces that report
set with a comparison of every supplied run. Each file also becomes a separate
MLflow run in the `scifact` experiment. Methods evaluated in separate invocations
can be compared there without rebuilding a combined report.
These initial comparisons describe one dataset, rather than establishing which
method will be best across datasets. Keep parameter tuning separate from final
evaluation.

## JuriFindIT

All eleven models were run against 23,617 prepared documents and 179 queries
in `--split test` (the original validation split), using `--top-k 1000`.
The stopword model automatically selects Italian. No training or parameter
tuning is required. See the [comparison results](../evaluation/README.md#jurifindit-results).

To reproduce in Windows PowerShell from the repository root:

```powershell
& ./.venv/Scripts/python.exe -B run_experiments.py --dataset jurifindit
```

This refreshes the selected methods and local reports and logs new MLflow runs.
Historical reports may use the earlier names for the renamed methods.

## Adding a method

Add a public Python script under `modelling/scripts/` with the common options
`--dataset`, `--split`, `--top-k`, `--output-dir`, and `--overwrite`. It must write
`<script-stem>.trec` and `<script-stem>.metadata.json` into `--output-dir`, exit
nonzero on failure, and keep execution behind its `__main__` guard. `MODEL_NAME`
must match the script stem. Empty rankings are valid; metadata must be present.
Method-specific scalar options can be passed through the orchestrator's repeated
`--method-option METHOD:OPTION=VALUE`. Common orchestration options cannot be
overridden this way. Keep MLflow integration in the evaluator.

## Verification

```powershell
& ./.venv/Scripts/python.exe -B -m unittest discover -s modelling/tests -p 'test_*.py' -v
```
