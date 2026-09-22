# Evaluation

Evaluate saved retrieval results against relevance judgments, independently of
how a model retrieves or scores documents. The evaluator uses `ir-measures` and
its `pytrec_eval`, MS MARCO reciprocal-rank, and judgment-coverage backends.
Metric formulas are supplied by these established implementations.

## Setup

Follow the [root setup](../README.md#setup). All project dependencies, including
evaluation, are pinned in the root [requirements.txt](../requirements.txt).
Commands in this README use Linux and run from the repository root.

## Inputs

Supply one or more run files for the **same dataset and split**. A run contains
document scores for each query. Larger scores indicate better results. The
model should return unique document IDs per query and preserve the prepared
query/document IDs exactly, including leading zeros.

For a prepared dataset, `--dataset scifact` selects
`data-preparation/data/scifact/qrels/test.jsonl`. Other dataset directory names
are supported, and `--split train` selects training judgments. Alternatively,
`--qrels PATH` accepts an external judgment file.

Two formats are supported:

| Input | JSONL fields | TREC columns, separated by whitespace |
| --- | --- | --- |
| Judgments | `query_id`, `doc_id`, `relevance` | `query_id iteration doc_id relevance` |
| Run | `query_id`, `doc_id`, `score` | `query_id Q0 doc_id rank score run_tag` |

Example TREC run (illustrative IDs and scores):

```text
q1 Q0 d7 1 12.5 bm25
q1 Q0 d2 2 9.8 bm25
q2 Q0 d3 1 11.0 bm25
```

Equivalent JSONL:

```jsonl
{"query_id": "q1", "doc_id": "d7", "score": 12.5}
{"query_id": "q1", "doc_id": "d2", "score": 9.8}
{"query_id": "q2", "doc_id": "d3", "score": 11.0}
```

`.jsonl` and `.jsonl.gz` select JSONL automatically; other extensions select
TREC. Use `--run-format` or `--qrels-format` to override detection. Both formats
support gzip compression with a `.gz` suffix, UTF-8 BOMs, and blank lines.

IDs must be nonempty strings without whitespace. Scores must be finite numbers;
judgments must be integer grades from `0` to `2147483647`, or `-1` for unjudged.
The evaluator excludes `-1` judgments. It rejects duplicate query/document
pairs, malformed rows, nonfinite scores, and mixed TREC run tags in one file.
TREC ranks must be positive integers, but scores determine the evaluated order.

## Run

Replace the example run paths below with outputs from your retrieval models.
The [Modelling phase](../modelling/README.md) provides eleven simple lexical baselines
and commands for generating compatible runs.

```bash
./.venv/bin/python evaluation/scripts/evaluate.py \
    --dataset scifact \
    --run path/to/bm25.trec
```

Evaluate several runs together for a shared comparison table:

```bash
./.venv/bin/python evaluation/scripts/evaluate.py \
    --dataset scifact \
    --run path/to/bm25.trec path/to/dense.jsonl \
    --overwrite
```

Run filenames serve as report labels and must be unique within an invocation.
Include every run you want to compare in the same invocation.
Use `--overwrite` to replace the complete existing comparison; reports are not
appended or merged. All inputs are evaluated before
staged reports replace existing outputs, so invalid run files leave previous
reports intact. Input files cannot be overwritten by report paths.

To evaluate external judgments or select particular metrics:

```bash
./.venv/bin/python evaluation/scripts/evaluate.py \
    --qrels path/to/qrels.txt \
    --run path/to/model.trec \
    --metrics nDCG@10 MRR@10 MAP Recall@100 Recall@1000 Precision@10 \
    --output-dir evaluation/data/custom
```

`--cutoffs 5 10 100` changes the cutoffs of the default suite. `--metrics` replaces
the suite and takes precedence over `--cutoffs`. `--relevance-level 2` changes
the inclusive relevance threshold for binary metrics; graded nDCG continues to
use the original relevance values. Use `--help` for all arguments. Explicit input and output paths are relative to
the current working directory; default dataset and output locations are relative
to the repository.

## Metrics

The default suite reports **57 measurements**: eight full-ranking metrics plus
seven metric families at cutoffs **1, 3, 5, 10, 20, 100, and 1000**.

| Metric | Meaning | Default scope |
| --- | --- | --- |
| `nDCG` | Normalized discounted cumulative gain; rewards relevant documents near the top. | Full ranking and each cutoff |
| `AP` / `MAP` | Average precision per query; its aggregate is mean average precision. | Full ranking and each cutoff |
| `RR` / `MRR` | Reciprocal rank of the first relevant result; its aggregate is mean reciprocal rank. | Full ranking and each cutoff |
| `P` / `Precision` | Relevant results divided by the requested cutoff. | Each cutoff |
| `R` / `Recall` | Retrieved relevant documents divided by all known relevant documents. | Each cutoff |
| `Success` | Whether at least one relevant document is returned. | Each cutoff |
| `Judged` | Fraction of returned documents with judgments, including grade zero. | Each cutoff |
| `Rprec` | Precision at the number of known relevant documents for the query. | Full ranking |
| `Bpref` | Preference for judged relevant documents over judged nonrelevant documents. | Full ranking |
| `SetP`, `SetR`, `SetF` | Precision, recall, and F1 of all retrieved documents, ignoring order. | Full ranking |

Aliases are accepted on input; reports use canonical names such as `AP`, `RR`,
`P`, and `R`. Select metrics with `NAME` or `NAME@K`; arbitrary Python expressions
are not accepted. `Rprec`, `Bpref`, and the set metrics have no cutoff parameter.
`P`, `R`, and `Success` require a cutoff. `Judged` also supports the full ranking
when explicitly requested.

These metrics cover standard document-ranking evaluation with query/document
judgments. Intent-aware diversity, click-based evaluation, retrieval latency,
and memory usage require additional inputs or measurements and are outside
this evaluator's scope.

## Evaluation policy

- **Query population:** all queries with at least one nonnegative judgment.
  Queries missing from the run score zero. Queries appearing only in the run
  are ignored and counted in the coverage report. Query text files are not used
  to infer relevance for queries without judgments.
- **Aggregation:** every evaluated query has equal weight. Queries with no
  relevant documents remain in the population and score zero on effectiveness
  metrics; judgment coverage can still be nonzero.
- **Relevance:** grades of at least `1` are relevant by default. Unjudged documents
  occupy ranking positions and count as nonrelevant for conventional metrics.
  Bpref considers judged documents only.
- **Graded gain:** nDCG uses linear relevance grades and logarithmic discount,
  matching the `trec_eval` default; it does not use exponential gains.
- **Ranking and ties:** descending score, then descending document ID as strings,
  matching `trec_eval`. The supplied TREC rank column is not used for ordering.
  Internally, unique ordinal scores ensure every metric backend uses that order.
- **Short runs:** precision still divides by the requested cutoff. `Judged@K`
  divides by the number of documents actually returned up to K. Empty runs score
  zero. Full-ranking metrics use all submitted documents; nothing is silently
  truncated at 1000.

Bpref is most informative when both relevant and nonrelevant judgments exist.
The currently prepared test qrels contain only positive judgments, so Bpref
adds limited information and judgment coverage is sparse. These judgments do
not prove that every unjudged document is irrelevant.

The prepared MARCO test sets each contain 101,093 query texts but judgments for
55,578 queries. The evaluator uses the latter population. Also, `test` here is
the prepared name for MARCO's dev split and JuriFindIT's validation split; scores
refer to these prepared splits and are not automatically leaderboard scores.
Use a separate validation protocol for tuning models or selecting parameters.

## Outputs

For prepared datasets, one comparison report set lives directly under
`evaluation/data/<dataset>/<split>/`, which is ignored by Git. There are no
experiment-group subfolders. Train and test remain separate to prevent mixing
results from different query sets.

```text
evaluation/data/scifact/test/
    per-query.csv
    report.json
    summary.csv
```

The three files contain all runs supplied for that dataset and split.
`--output-dir` overrides the destination. With external `--qrels`, no dataset
name is available: the fallback remains `evaluation/data/`; specify
`--output-dir evaluation/data/<dataset>/<split>` to keep the same organization.

| File | Contents |
| --- | --- |
| `summary.csv` | One row per run, with aggregate metrics and query/depth coverage. |
| `per-query.csv` | One row per run and judged query, with all metric values. |
| `report.json` | Aggregate results, coverage, source paths, metric configuration, policies, UTC timestamp, and software versions. |

Coverage includes judged queries, queries with relevant documents, missing judged
queries, ignored unjudged queries, retrieved-document counts, and minimum/maximum
returned depth across judged queries. An empty run file is valid and gets
zero scores rather than an inflated average over a subset.

The evaluator loads judgments and one run at a time into memory and holds that
run's per-query scores while reporting. Large MARCO runs therefore require
substantial RAM. Evaluate datasets separately. Source files are never modified.
The evaluator checks file structure, but does not load the corpus to validate
that every returned document ID exists in it.

## JuriFindIT results

All eleven models were evaluated on 2026-09-22 using 23,617 corpus documents,
179 judged validation queries (prepared as `test`), and up to 1,000 results per
query. Every model returned results for all 179 queries; no unjudged queries
were ignored. All 57 default metrics are included in the generated reports.
The following aggregate values are rounded to six decimals; MAP is reported
as `AP` and MRR as `RR` in the output files.

| Model | nDCG@10 | MRR@10 | MAP | Recall@1000 |
| --- | --- | --- | --- | --- |
| 1.1.1-term-overlap | 0.028016 | 0.028383 | 0.026819 | 0.494743 |
| 1.1.2-term-overlap | 0.028016 | 0.028383 | 0.026819 | 0.494743 |
| 1.2.1-document-coverage | 0.004268 | 0.002195 | 0.005195 | 0.359083 |
| 1.3.1-jaccard | 0.137772 | 0.127208 | 0.125321 | 0.556488 |
| 1.4.1-binary-cosine | 0.196051 | 0.182992 | 0.174172 | 0.640594 |
| 1.4.2-term-frequency | 0.001230 | 0.005587 | 0.001027 | 0.090809 |
| 1.5.1-log-term-frequency | 0.002061 | 0.006145 | 0.002179 | 0.190428 |
| 1.6.1-bigram-overlap | 0.130471 | 0.130901 | 0.113274 | 0.603724 |
| 1.7.1-stopword-overlap | 0.047787 | 0.044808 | 0.046433 | 0.704684 |
| 1.8.1-idf-overlap | 0.070850 | 0.065647 | 0.064153 | 0.697028 |
| 1.9.1-unigram-bigram-overlap | 0.074089 | 0.074370 | 0.064486 | 0.615360 |

Binary cosine leads these baselines on nDCG@10, MRR@10, and MAP;
stopword-filtered overlap has the highest Recall@1000. Versions 1.1.1 and 1.1.2
match on all metrics, as expected from their identical rankings. These are
validation results without parameter tuning, not held-out leaderboard scores.
MAP uses the submitted ranking, which is limited to 1,000 results per query.

Local generated reports (ignored by Git):

- [summary.csv](data/jurifindit/test/summary.csv): all aggregate metrics.
- [per-query.csv](data/jurifindit/test/per-query.csv): each query's metrics.
- [report.json](data/jurifindit/test/report.json): metrics, coverage, and provenance.

See [JuriFindIT modelling commands](../modelling/README.md#jurifindit) to reproduce
the runs and comparison on Windows PowerShell.

## Verification

Run the regression suite without downloading data or running a retrieval model:

```bash
./.venv/bin/python -B -m unittest discover -s evaluation/tests -p evaluate.py -v
```

Tests cover known metric values, agreement with `trec_eval`, graded judgments,
missing queries, empty runs, tied scores, invalid inputs, format equivalence,
and multi-run report generation.

## References

- [ir-measures metric definitions](https://ir-measur.es/en/latest/measures.html)
- [ir-measures usage and providers](https://ir-measur.es/en/latest/getting-started.html)
- [NIST trec_eval input formats and ranking conventions](https://github.com/usnistgov/trec_eval/blob/main/formats.c)
