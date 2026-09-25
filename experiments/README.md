# Experiments

Run from the repository root with the project environment active:

```text
python -m experiments.run_experiments --list
python -m experiments.run_experiments --dataset scifact --benchmark
python -m experiments.run_experiments --dataset scifact --benchmark --resume
python -m experiments.run_experiments --dataset scifact --method 1.1.1-term-overlap
```

Module invocation is the supported interface. There is no root script or direct
script entry point. Existing selection, split, top-k, method-option, and evaluator
tracking flags are retained. `--resume` requires `--benchmark`.

## Ownership

| Module | Responsibility |
| --- | --- |
| `run_experiments.py` | Minimal module dispatch to the CLI. |
| `cli.py` | Argument parsing, selection validation, discovery display, dispatch and exit status. |
| `discovery.py` | Prepared datasets, public retrieval scripts and literal capabilities. |
| `configuration.py` | Catalog schema and repository-relative default/custom paths. |
| `planning.py` | Planned configurations, dimension combinations, stable IDs and expected metadata. |
| `runner.py` | Retrieval subprocesses, batch coordination, progress, failures and summaries. |
| `artifacts.py` | Ranking/metadata paths, staging, validation, publication and reuse eligibility. |
| `evaluation.py` | Evaluator command construction and invocation. |

The CLI combines discovery, configuration and planning, then dispatches execution.
The runner uses artifacts and evaluation; neither calls back into the CLI.
All modules use the standard library. No module imports retrieval implementations,
computes metrics, implements Qdrant compatibility, or maintains tracking history.

## Planning

Public Python scripts in `modelling/scripts/` are the method inventory.
Underscore-prefixed helpers are excluded. Ordinary methods need no catalog entry.
Configurable methods declare `EXPERIMENT_CAPABILITY`; discovery reads this literal
with AST parsing without executing imports.

[modelling/configs/benchmark-variants.json](../modelling/configs/benchmark-variants.json)
contains the selected values. Its path is resolved from the package location,
independently of the current working directory. `--benchmark-config PATH` uses
the same repository-relative rule for relative paths.

The default plan contains dense MiniLM, dense Potion, sparse BM42, hybrid
MiniLM + BM42, and hybrid Potion + BM42. The planner derives all five from
capabilities and the catalog. MPNet remains supported by explicit method options
or a custom catalog; it is excluded only from the default selection.

Capability version 1 requires `version` and `dimensions`. Existing declarations
remain supported. Two optional fields describe how existing metadata is compared
for resume:

```python
EXPERIMENT_CAPABILITY = {
    "version": 1,
    "dimensions": {"encoders": "encoder"},
    "parameter_names": {"encoder": "actual_encoder"},
    "parameters": {"search_depth": 64},
}
```

By default, CLI hyphens become metadata underscores. `parameter_names` overrides
that mapping when a method's existing sidecar uses a different field.
`parameters` declares fixed benchmark search/fusion settings expected in the
sidecar. Methods must keep these declarations aligned with their emitted metadata.
Selected dimension values override fixed values for the same metadata field.

## Artifacts, resume and identities

A configuration ID is `<script-stem>__<variant-id>...`, in declaration order.
Existing ranking filenames and scalar sidecars are preserved. Changing a model
value under the same variant ID invalidates resume even though the filename stays
the same. Use new IDs to retain both local configurations.

Each benchmark configuration stages its pair independently and validates TREC
structure and the evaluator's scalar metadata contract before publication.
A handled publication failure restores the previous pair. Successful independent
configurations survive later failures and are submitted to evaluation; failures
return nonzero. Ctrl+C stops new retrieval work, attempts evaluation of completed
configurations, and returns 130. Normal mode retains its dataset-level retrieval
staging and stop-on-failure behavior.

Resume validates both nonempty files, TREC structure, metadata validity, dataset,
split, method, configuration ID, top-k, actual configured model values, and declared
search/fusion settings. Eligible files are read without rewriting them.
Reused files are evaluated again along with fresh successes, preserving existing
resume behavior. Evaluation may replace local reports and creates new MLflow runs.

Three identities remain distinct:

- Qdrant collection identity and compatibility belong to modelling and depend on
  corpus/index configuration. Resume never contacts Qdrant or repairs collections.
- Ranking identity describes a configuration and query workload; its stable filename
  is separate from an individual execution.
- MLflow run IDs describe execution history and can differ for repeated evaluation
  of the same ranking.

Limitations: legacy sidecars have no query fingerprint or artifact checksum.
Resume cannot prove that queries, corpus bytes, retrieval code, or cached weights
are unchanged, nor detect a structurally valid truncated ranking. It compares
recorded fields against the plan, not the live index. It conservatively reruns
empty rankings, although fresh empty rankings remain valid evaluator input.
After changing inputs/code, rerun without resume. Two file renames are not a
crash-safe filesystem transaction; avoid concurrent writers for the same outputs.
