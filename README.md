# Information Retrieval

A project organized around CRISP-DM phases. The current pipeline downloads
SciFact, JuriFindIT, mMARCO Italian, and MS MARCO, then prepares a common format
for information retrieval experiments. Versioned simple lexical baselines generate
ranked results, and a separate evaluator scores them against prepared judgments.

## Repository structure

| Location | Purpose |
| --- | --- |
| [run_experiments.py](run_experiments.py) | Discover and run retrieval experiments, then evaluate each dataset. |
| [data-understanding/](data-understanding/README.md) | Download datasets and document their source formats. |
| [data-preparation/](data-preparation/README.md) | Convert downloaded data into a common corpus, query, and relevance format. |
| [modelling/](modelling/README.md) | Generate versioned retrieval runs with simple lexical baselines. |
| [evaluation/](evaluation/README.md) | Score and compare saved retrieval runs using standard IR metrics. |
| [tracking/](tracking/README.md) | Compare dataset experiments and evaluation runs in MLflow. |
| [theory/](theory/) | Reference reading. |

Processing phases contain `scripts/` for code and `data/` for generated
outputs. The `tracking/` directory documents MLflow and holds its local data. Dataset directories and `.venv/` are excluded from Git.

## Setup

The commands below use a Linux shell and run from the repository root.
Create the environment once; reuse it for subsequent runs.

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
```

The root [requirements.txt](requirements.txt) covers all current scripts and
tests, pinned to the direct dependency versions in the working environment
(Python 3.14.7). Pip installs their transitive dependencies automatically; those
versions are not pinned. Preparation and the current modelling baselines use
only the Python standard library.

## Workflow

1. Run the four download scripts in [Data Understanding](data-understanding/README.md#run).
2. After all downloads succeed, run [Data Preparation](data-preparation/README.md#run).
3. Run retrieval and evaluation together with `run_experiments.py` (examples below).
4. Open [MLflow Tracking](tracking/README.md) to compare runs and inspect artifacts.

With the project virtual environment active:

```bash
# Run every retrieval method on one dataset
python run_experiments.py --dataset scifact

# Run one retrieval method on every runnable dataset
python run_experiments.py --method 1.1.1-term-overlap

# Run one dataset/method combination
python run_experiments.py --dataset scifact --method 1.1.1-term-overlap

# Run every dataset/method combination
python run_experiments.py --all

# Show discovered datasets and methods without running them
python run_experiments.py --list
```

Without activation on Windows, replace `python` with
`& ./.venv/Scripts/python.exe`; on Linux use `./.venv/bin/python`.
Defaults are `--split test --top-k 1000`. Use `--split train` or `--top-k N`
when needed. Datasets are discovered from `data-preparation/data/` when nonempty
`corpus.jsonl`, `queries/<split>.jsonl`, and `qrels/<split>.jsonl` files exist.
Methods are public `.py` entry points in `modelling/scripts/`; helper files should
start with `_`. New datasets and methods need no registry edits.

The orchestrator runs sequentially using the same Python interpreter. It creates
fresh rankings and metadata, replacing only selected methods' outputs under
`modelling/data/<dataset>/<split>/`. After all selected methods succeed for a
dataset, it invokes the existing evaluator once with exactly those rankings,
replacing the combined reports under `evaluation/data/<dataset>/<split>/` and
logging new MLflow runs. Old, unselected ranking files are preserved and excluded.
A retrieval failure stops the invocation before publishing that dataset's staged
outputs or evaluating it; the summary identifies failures and skipped work.
Earlier completed datasets remain available. Evaluation or tracking failures
also stop the invocation and are reported as failures.

Use `--no-tracking` for local reports only. `--tracking-uri URL` and
`--log-rankings` are forwarded to the evaluator. Method-specific options can be
passed without changing the orchestrator, for example for a new Italian dataset:

```bash
python run_experiments.py --dataset my-dataset --method-option 1.7.1-stopword-overlap:language=it
```

The stopword method infers language for the existing datasets; new dataset names
need its explicit `language=en` or `language=it` option. Other methods receive no
language override. Individual [modelling](modelling/README.md#run) and
[evaluation](evaluation/README.md#run) commands remain available.

The scripts resolve dataset paths relative to their own locations. Commands in
these READMEs assume the repository root only to locate the scripts and Python.
Download and preparation scripts overwrite their output files when rerun.
Direct modelling and evaluation commands require `--overwrite` to replace local
outputs; the orchestrator supplies it for the selected experiments.
Evaluation also logs a new MLflow run for each supplied retrieval file. Tracking
defaults to local SQLite and artifact storage under `tracking/data/`; use
`--no-tracking` for local reports only.

Downloads require network access when source data is not already cached. The
MARCO datasets are large: allow disk space for downloaded exports, prepared
copies, and library caches. Preparation reads local files and does not download
data.

## Verification

Run tests against temporary fixtures, without running the full datasets:

```bash
python -B -m unittest discover -s tests -p 'test_*.py' -v
python -B -m unittest discover -s modelling/tests -p 'test_*.py' -v
python -B -m unittest discover -s evaluation/tests -p '*.py' -v
```

## Documentation

Keep shared setup and the phase overview here. Each phase README documents its
inputs, commands, outputs, and processing choices. Update it when the scripts or
data formats change, and add documentation for future phases when implemented.
