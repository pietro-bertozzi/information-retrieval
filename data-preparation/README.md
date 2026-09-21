# Data Preparation

Convert the four downloaded datasets into a common UTF-8 JSONL format for
corpora, queries, and relevance judgments.

See the [root README](../README.md#setup) for environment setup.

## Inputs

[prepare.py](scripts/prepare.py) reads the exports in
`data-understanding/data/`. Complete all four downloads described in
[Data Understanding](../data-understanding/README.md#run) first.

## Run

Run from the repository root:

```bash
./.venv/bin/python data-preparation/scripts/prepare.py
```

The script prepares SciFact, JuriFindIT, mMARCO Italian, and MS MARCO in that
order. It creates output directories as needed and overwrites prepared files
under `data-preparation/data/`, leaving the downloaded inputs unchanged.
It uses only the Python standard library and requires no network access.

## Outputs

Each dataset directory (`scifact`, `jurifindit`, `mmarco-it`, and `msmarco`) has
the same structure:

```text
<dataset>/
    corpus.jsonl
    queries/
        train.jsonl
        test.jsonl
    qrels/
        train.jsonl
        test.jsonl
```

Each line contains one JSON object:

| File | Fields |
| --- | --- |
| `corpus.jsonl` | `doc_id` (string), `text` (string) |
| `queries/*.jsonl` | `query_id` (string), `text` (string) |
| `qrels/*.jsonl` | `query_id` (string), `doc_id` (string), `relevance` (numeric) |

A qrel links a query to a document and records its relevance value.

## Processing choices

| Dataset | Source splits | Prepared splits |
| --- | --- | --- |
| SciFact | `train`, `test` | `train`, `test` |
| JuriFindIT | `train`, `validation` | `train`, `test` |
| mMARCO Italian | `train`, `dev` | `train`, `test` |
| MS MARCO | `train`, `dev` | `train`, `test` |

Here, `test` is the project's common output name for the second split. For
JuriFindIT and the MARCO datasets, it refers to the original validation or dev
split; preparation does not create a new split or resample records.

- **SciFact:** joins nonempty title and text with a blank line, converts IDs to
  strings, and selects each split's queries using the query IDs in its qrels.
  Source relevance scores are preserved.
- **JuriFindIT:** uses document `content` as text and converts document IDs to
  strings. Query IDs are stringified row numbers starting at `0` independently
  in each split, so they are unique within a split and depend on source row
  order. Each listed relevant document produces a qrel with relevance `1`.
- **MARCO datasets:** retain the downloaded records and field values, organizing
  queries and qrels into the shared directory layout and mapping `dev` to `test`.

Preparation does not tokenize, lowercase, stem, or otherwise normalize text.
