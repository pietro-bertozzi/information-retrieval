# Information Retrieval

A project organized around CRISP-DM phases. The current pipeline downloads
SciFact, JuriFindIT, mMARCO Italian, and MS MARCO, then prepares a common format
for information retrieval experiments. Versioned lexical and vector methods generate
ranked results, and a separate evaluator scores them against prepared judgments.

## Repository structure

| Location | Purpose |
| --- | --- |
| [experiments/](experiments/README.md) | Discover and run retrieval experiments, then evaluate each dataset. |
| [infrastructure/qdrant/](infrastructure/qdrant/README.md) | Shared Qdrant service configuration and storage mapping. |
| [data-understanding/](data-understanding/README.md) | Download datasets and document their source formats. |
| [data-preparation/](data-preparation/README.md) | Convert downloaded data into a common corpus, query, and relevance format. |
| [modelling/](modelling/README.md) | Generate versioned lexical and vector retrieval runs. |
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
versions are not pinned. Preparation and lexical modelling use only the Python
standard library.
Vector retrieval uses FastEmbed and a local Qdrant server.

## Workflow

1. Run the four download scripts in [Data Understanding](data-understanding/README.md#run).
2. After all downloads succeed, run [Data Preparation](data-preparation/README.md#run).
3. Run retrieval and evaluation together with `python -m experiments.run_experiments` (examples below).
4. Open [MLflow Tracking](tracking/README.md) to compare runs and inspect artifacts.

With the project virtual environment active:

```bash
# Run every retrieval method on one dataset
python -m experiments.run_experiments --dataset scifact

# Run one retrieval method on every runnable dataset
python -m experiments.run_experiments --method 1.1.1-term-overlap

# Run one dataset/method combination
python -m experiments.run_experiments --dataset scifact --method 1.1.1-term-overlap

# Run every dataset/method combination
python -m experiments.run_experiments --all

# Show discovered datasets and methods without running them
python -m experiments.run_experiments --list

# Run configured variants of benchmark-capable methods on one dataset
python -m experiments.run_experiments --dataset scifact --benchmark

# Run those benchmark variants on every runnable dataset
python -m experiments.run_experiments --all --benchmark
```

Without activation on Windows, replace `python` with
`& ./.venv/Scripts/python.exe`; on Linux use `./.venv/bin/python`.
Defaults are `--split test --top-k 1000`. Use `--split train` or `--top-k N`
when needed. Datasets are discovered from `data-preparation/data/` when nonempty
`corpus.jsonl`, `queries/<split>.jsonl`, and `qrels/<split>.jsonl` files exist.
Methods are public `.py` entry points in `modelling/scripts/`; helper files should
start with `_`. New datasets and methods need no registry edits.

The orchestrator runs sequentially using the same Python interpreter. In normal
mode it stages the selected methods for one dataset together, invokes the evaluator
with exactly their fresh rankings, and preserves old unselected files. A retrieval
failure leaves that dataset's previous outputs untouched. Benchmark mode instead
uses one transaction per configuration: each validated ranking and metadata pair
is published immediately, and all successful current outputs are evaluated even
if a later independent configuration fails. Failed or stale outputs are excluded,
and any retrieval, evaluation, or tracking failure still makes the command return
nonzero.

Use `--no-tracking` for local reports only. `--tracking-uri URL` and
`--log-rankings` are forwarded to the evaluator. Method-specific options can be
passed without changing the orchestrator, for example for a new Italian dataset:

```bash
python -m experiments.run_experiments --dataset my-dataset --method-option 1.7.1-stopword-overlap:language=it
```

The stopword method infers language for the existing datasets; new dataset names
need its explicit `language=en` or `language=it` option. Other methods receive no
language override. Individual [modelling](modelling/README.md#run) and
[evaluation](evaluation/README.md#run) commands remain available.

Manual mode treats each discovered script as one method. This remains the right
workflow for lexical methods and one-off vector runs. For example:

```bash
python -m experiments.run_experiments --dataset scifact --method 1.1.1-term-overlap
python -m experiments.run_experiments --dataset scifact --method 2.1.1-dense-retrieval
python -m experiments.run_experiments --dataset scifact --method 2.1.1-dense-retrieval --method-option 2.1.1-dense-retrieval:embedding-model=minishlab/potion-multilingual-128M
```

Benchmark mode reads [benchmark-variants.json](modelling/configs/benchmark-variants.json), which
contains selected values only—not an inventory of retrieval methods. Public
scripts remain the source of truth for which methods exist. A configurable script
opts in with a small literal capability declaration that maps its experiment
dimensions to CLI options. The orchestrator reads that declaration without
importing the script, expands one-dimensional variants or multi-dimensional
Cartesian products, and ignores ordinary methods such as the 1.x lexical scripts.
Stable short IDs produce distinct ranking filenames and MLflow run names.
`--method` and `--method-option` are intentionally unavailable in benchmark mode;
edit the configuration or use manual mode. `--list` shows method discovery,
capabilities, and the expanded benchmark configurations.

The catalog lives under `modelling/configs/`; the default plan contains exactly
five vector configurations: dense MiniLM, dense Potion, sparse BM42, and each
dense model combined with BM42. MPNet remains available through explicit model
options or a custom catalog.

Use `python -m experiments.run_experiments --dataset scifact --benchmark --resume`
to reuse compatible completed ranking/metadata pairs. Reuse checks model values,
top-k and declared search settings as well as identity and file validity.
Reused rankings are evaluated again and receive new MLflow execution runs.
See [experiment ownership and resume limitations](experiments/README.md).

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

## Vector retrieval

Start local Qdrant, then run the configurable dense baseline:

```text
docker compose -f infrastructure/qdrant/compose.yaml up -d
python -m experiments.run_experiments --dataset scifact --method 2.1.1-dense-retrieval
```

The default dense model is multilingual MiniLM, run locally through FastEmbed. Set
`--method-option 2.1.1-dense-retrieval:embedding-model=MODEL` to select another
supported dense model. Compatible document indexes are reused automatically.
The relocated Compose file retains the existing project and named volume;
see [Qdrant storage preservation](infrastructure/qdrant/README.md).
See [dense retrieval setup and lifecycle](modelling/README.md#dense-retrieval).

Selecting every method includes one default dense, sparse, and hybrid
configuration, so those commands require Qdrant and may download models on first
use. `--all` indexes every runnable corpus, including large MARCO datasets; use a
specific dataset while developing. Existing lexical-only selections need no
Qdrant service.

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
