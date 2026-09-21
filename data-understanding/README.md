# Data Understanding

Download the four datasets used by the project and export them as UTF-8 JSONL
(one JSON object per line) under `data-understanding/data/`. The current scripts
cover data acquisition; they do not perform exploratory analysis.

See the [root README](../README.md#setup) for environment setup.

## Inputs

The scripts use the following dataset identifiers:

| Dataset | Script | Source |
| --- | --- | --- |
| SciFact | [scifact.py](scripts/scifact.py) | Hugging Face `BeIR/scifact` (corpus and queries) and `BeIR/scifact-qrels`. |
| JuriFindIT | [jurifindit.py](scripts/jurifindit.py) | Hugging Face `jurifindit/JuriFindIT` (corpus and questions). |
| mMARCO Italian | [mmarco_it.py](scripts/mmarco_it.py) | `ir_datasets`: `mmarco/it`, `mmarco/it/train`, `mmarco/it/dev`. |
| MS MARCO | [msmarco.py](scripts/msmarco.py) | `ir_datasets`: `msmarco-passage`, `msmarco-passage/train`, `msmarco-passage/dev`. |

## Run

Run each command from the repository root using the environment created during
setup. The download scripts are independent; all four must complete before
running preparation.

```bash
./.venv/bin/python data-understanding/scripts/scifact.py
./.venv/bin/python data-understanding/scripts/jurifindit.py
./.venv/bin/python data-understanding/scripts/mmarco_it.py
./.venv/bin/python data-understanding/scripts/msmarco.py
```

Each script creates its output directories and overwrites its dataset files.

## Outputs

Paths below are relative to `data-understanding/data/`. A *qrel* is a relevance
judgment linking a query to a document.

| Dataset directory | Files |
| --- | --- |
| `scifact/` | `corpus.jsonl`, `queries.jsonl`, `qrels/train.jsonl`, `qrels/test.jsonl` |
| `jurifindit/` | `corpus.jsonl`, `questions-train.jsonl`, `questions-validation.jsonl` |
| `mmarco-it/` | `corpus.jsonl`, `queries-train.jsonl`, `queries-dev.jsonl`, `qrels/train.jsonl`, `qrels/dev.jsonl` |
| `msmarco/` | `corpus.jsonl`, `queries-train.jsonl`, `queries-dev.jsonl`, `qrels/train.jsonl`, `qrels/dev.jsonl` |

SciFact and JuriFindIT retain the fields returned by Hugging Face. The fields
consumed by preparation are:

| Dataset | Corpus fields | Query or question fields | Relevance fields |
| --- | --- | --- | --- |
| SciFact | `_id`, `title`, `text` | `_id`, `text` | `query-id`, `corpus-id`, `score` in qrels |
| JuriFindIT | `id`, `content` | `question` | `relevant_doc_ids` in each question |

The MARCO scripts export selected fields from `ir_datasets`: `doc_id` and `text`
for the corpus; `query_id` and `text` for queries; and `query_id`, `doc_id`, and
`relevance` for qrels. They retain the source `train` and `dev` split names.

## Next step

Run [Data Preparation](../data-preparation/README.md) to standardize the formats
and split names. Keep these downloaded exports as the inputs to that phase.
