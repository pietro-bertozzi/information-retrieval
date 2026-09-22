# Information Retrieval

A project organized around CRISP-DM phases. The current pipeline downloads
SciFact, JuriFindIT, mMARCO Italian, and MS MARCO, then prepares a common format
for information retrieval experiments. Versioned simple lexical baselines generate
ranked results, and a separate evaluator scores them against prepared judgments.

## Repository structure

| Location | Purpose |
| --- | --- |
| [data-understanding/](data-understanding/README.md) | Download datasets and document their source formats. |
| [data-preparation/](data-preparation/README.md) | Convert downloaded data into a common corpus, query, and relevance format. |
| [modelling/](modelling/README.md) | Generate versioned retrieval runs with simple lexical baselines. |
| [evaluation/](evaluation/README.md) | Score and compare saved retrieval runs using standard IR metrics. |
| [theory/](theory/) | Reference reading. |

Each implemented phase contains `scripts/` for code and `data/` for generated
outputs. Dataset directories and `.venv/` are excluded from Git.

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
3. Run [Modelling](modelling/README.md#run) to generate versioned retrieval results
   from the standardized files in `data-preparation/data/`.
4. Run [Evaluation](evaluation/README.md#run) on those saved results.

The scripts resolve dataset paths relative to their own locations. Commands in
these READMEs assume the repository root only to locate the scripts and Python.
Download and preparation scripts overwrite their output files when rerun.
Modelling and evaluation require `--overwrite` to replace existing outputs.

Downloads require network access when source data is not already cached. The
MARCO datasets are large: allow disk space for downloaded exports, prepared
copies, and library caches. Preparation reads local files and does not download
data.

## Documentation

Keep shared setup and the phase overview here. Each phase README documents its
inputs, commands, outputs, and processing choices. Update it when the scripts or
data formats change, and add documentation for future phases when implemented.
