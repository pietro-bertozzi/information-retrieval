# Information Retrieval

A project organized around CRISP-DM phases. The current pipeline downloads
SciFact, JuriFindIT, mMARCO Italian, and MS MARCO, then prepares a common format
for information retrieval experiments.

## Repository structure

| Location | Purpose |
| --- | --- |
| [data-understanding/](data-understanding/README.md) | Download datasets and document their source formats. |
| [data-preparation/](data-preparation/README.md) | Convert downloaded data into a common corpus, query, and relevance format. |
| [theory/](theory/) | Reference reading. |

Each implemented phase contains `scripts/` for code and `data/` for generated
outputs. Dataset directories and `.venv/` are excluded from Git.

## Setup

The commands below use a Linux shell and run from the repository root.
Create the environment once; reuse it for subsequent runs.

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install datasets==5.0.1 ir_datasets==0.6.3
```

These are the direct dependency versions in the working environment, which uses
Python 3.14.7. Transitive dependencies are not pinned. Preparation itself uses
only the Python standard library.

## Workflow

1. Run the four download scripts in [Data Understanding](data-understanding/README.md#run).
2. After all downloads succeed, run [Data Preparation](data-preparation/README.md#run).
3. Use the standardized files in `data-preparation/data/` for subsequent work.

The scripts resolve dataset paths relative to their own locations. Commands in
these READMEs assume the repository root only to locate the scripts and Python.
Rerunning a script overwrites its output files.

Downloads require network access when source data is not already cached. The
MARCO datasets are large: allow disk space for downloaded exports, prepared
copies, and library caches. Preparation reads local files and does not download
data.

## Documentation

Keep shared setup and the phase overview here. Each phase README documents its
inputs, commands, outputs, and processing choices. Update it when the scripts or
data formats change, and add documentation for future phases when implemented.
