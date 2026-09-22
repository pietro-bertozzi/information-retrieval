# MLflow Tracking

Evaluation logs standard IR results computed by `ir_measures` to MLflow. There is
one experiment per dataset and one run per retrieval file evaluated on one split.
The MLflow UI replaces the former custom evaluation dashboard.

## Local workflow

Install the root `requirements.txt`, then run
`python run_experiments.py --dataset scifact` from the repository root. It invokes
the evaluator, which owns all MLflow logging. No server is required
to log locally. The evaluator uses an absolute SQLite URI rooted at this repository
and stores everything under `tracking/data/`, which is ignored by Git:

```text
tracking/data/
    mlflow.db
    artifacts/
```

From the repository root, start the UI in Windows PowerShell:

```powershell
New-Item -ItemType Directory -Force tracking/data | Out-Null
& ./.venv/Scripts/mlflow.exe server --backend-store-uri sqlite:///tracking/data/mlflow.db --host 127.0.0.1 --port 5000
```

On Linux:

```bash
mkdir -p tracking/data
./.venv/bin/mlflow server --backend-store-uri sqlite:///tracking/data/mlflow.db --host 127.0.0.1 --port 5000
```

Open <http://127.0.0.1:5000>, choose an experiment, and open its Runs view.
The server command stays in the foreground; stop it with Ctrl+C. Artifact paths
for locally created experiments are absolute, so moving a tracking directory is
not a portable HTML export. Back up the database and artifacts together.

For a shared server, set `MLFLOW_TRACKING_URI` or pass `--tracking-uri URL` to the
evaluator. Explicit CLI configuration takes precedence over the environment;
otherwise the repository-local SQLite store is used. For an override, experiment
artifact placement is managed by MLflow/the server. Keep credentials outside code.

## Compare runs

Filter by `tags.split` before comparing metrics. The source split is also recorded:
JuriFindIT `test` means `validation`, and MARCO `test` means `dev`. Compare the same
data revision and evaluation settings; the minimal metadata does not fingerprint
corpora or detect dataset changes.

Select runs to compare aggregate metrics and retrieval parameters. Useful columns
include `ir.nDCG_at_10`, `ir.RR_at_10`, `ir.AP`, and `ir.R_at_1000`. AP is MAP when
aggregated; RR is MRR. `Judged` measures judgment coverage, not effectiveness.

Every evaluation execution creates new runs. Local `--overwrite` does not delete
MLflow history. Run duration covers logging, not retrieval latency. The UI does
not recreate the former custom query win/loss controls or cutoff curves; download
per-query CSVs for those analyses. No custom dashboard code is maintained.

## Logged data

| Kind | Contents |
| --- | --- |
| Parameters | Available retrieval method/version, configured `top_k`, applicable language, evaluation metrics, relevance threshold, and evaluation policies. |
| Metrics | Aggregate IR scores and existing query/depth coverage statistics. |
| Tags | Dataset, prepared/source split, evaluation Git revision when available, and whether retrieval metadata is available. |
| Artifacts | That run's `report.json`, `summary.csv`, `per-query.csv`, metric-name mapping, and retrieval metadata when supplied. |

Metric names use MLflow-compatible keys: `nDCG@10` becomes `ir.nDCG_at_10`, and
`AP(rel=2)@10` becomes `ir.AP_rel_2_at_10`. Original canonical names and full
precision remain in the JSON/CSV artifacts. Per-query scores are artifacts, not
thousands of separate metrics. Existing report software versions are retained.

`--log-rankings` also uploads the submitted TREC/JSONL file, preserving compression
when present. Corpora and qrels are not uploaded. No model registry, automatic
metric computation, environment snapshots, or dataset fingerprint system is used.
The evaluation Git tag describes the checkout used for evaluation, not necessarily
the code that produced an old ranking; it does not capture uncommitted changes.

## Retrieval metadata

New lexical runs have a sibling `.metadata.json` file. For example,
`1.7.1-stopword-overlap.trec` has `1.7.1-stopword-overlap.metadata.json`:

```json
{
  "dataset": "jurifindit",
  "split": "test",
  "parameters": {
    "method": "stopword-overlap",
    "version": "1.7.1",
    "top_k": 1000,
    "language": "it"
  }
}
```

External methods may use the same format and add actual model settings as scalar
parameters (for example `model`, `revision`, or `k1`). Parameter keys accept letters,
numbers, underscores, periods, and dashes. For `model.jsonl.gz`, use
`model.metadata.json`. Evaluation rejects malformed sidecars and mismatched
metadata dataset/split labels before replacing reports or creating MLflow runs.
Keep sidecars paired with their rankings when renaming or moving files.

Older rankings without sidecars still work: their filename identifies the run,
and unavailable retrieval parameters remain absent. Do not reconstruct `top_k`
from observed ranking depth or assume defaults were used.

## Existing results and failures

To bring existing rankings into MLflow, evaluate them with a fresh `--output-dir`
(or `--overwrite` if intentionally replacing local reports). This performs a new
evaluation; historical timestamps and missing settings are not invented. Existing
local reports and generated legacy dashboard files are not deleted automatically.

All inputs must evaluate successfully before logging starts. If an upload fails,
the command exits nonzero and prints the created run ID. It attempts to mark that
run failed; a disconnected server can leave it running. Earlier uploads remain
available. Local reports survive for inspection. Re-running evaluation creates
new runs rather than resuming an incomplete upload; filter by finished status and
use timestamps/run IDs to distinguish repetitions. Use `--no-tracking` when only
local reports are wanted; tracking errors never silently switch to local-only mode.

## Verification

```powershell
& ./.venv/Scripts/python.exe -B -m unittest discover -s evaluation/tests -p '*.py' -v
& ./.venv/Scripts/python.exe -B -m unittest discover -s modelling/tests -p 'test_*.py' -v
& ./.venv/Scripts/python.exe -B -m pip check
```

Tests use small fixtures and temporary tracking storage, not project experiments.
