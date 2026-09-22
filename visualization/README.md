# Visualization

Explore saved evaluation reports in an interactive, offline browser dashboard.
This phase reads evaluation outputs independently and does not run models or
recompute metrics. The builder uses only the Python standard library; the
browser uses bundled HTML, CSS, JavaScript, and SVG with no external services.

## Build and open

Run from the repository root in Windows PowerShell:

```powershell
& ./.venv/Scripts/python.exe -B visualization/scripts/build.py
```

Or on Linux:

```bash
./.venv/bin/python -B visualization/scripts/build.py
```

Open **`visualization/data/index.html`** in a full browser such as Edge, Chrome,
or Firefox. Opening the file in an editor shows its source; some editor previews
also disable the JavaScript needed by the dashboard. No web server is required.

Do not open `visualization/assets/index.html`: it is the source template and has
no embedded styles or evaluation data. If opened accidentally, it displays a
link to the generated dashboard.

On Windows, open the generated dashboard in Edge from the repository root:

```powershell
Start-Process msedge (Resolve-Path visualization/data/index.html).Path
```
Use `--overwrite` to rebuild after running new evaluations. Share the **entire
`visualization/data/` folder**, including the query payload files, for a portable
report. Generated files are ignored by Git under the existing `*/data` rule.

Optional paths:

```powershell
& ./.venv/Scripts/python.exe -B visualization/scripts/build.py --input-dir evaluation/data --output-dir visualization/data --overwrite
```

Defaults are relative to the repository. Explicit paths are relative to the
current directory. The output must be outside the input directory.

## Explore

- **Overview:** select a dataset, report/split, metric, baseline, and visible
  models. Compare ranked bars, sort the table, and export selected model scores
  and absolute baseline differences as CSV.
- **Retrieval depth:** compare a metric family across its available saved
  cutoffs. The horizontal axis is logarithmic; the vertical axis starts at zero
  and scales to the selected values. Exact values are available in a table.
- **Query analysis:** select a baseline and candidate, then filter wins, ties,
  or losses, search query IDs, and sort by change. Tables show 50 rows per page.
  Differences smaller than or equal to `1e-12` count as ties. Model checkboxes
  affect aggregate charts; query comparisons use the two explicit selectors.
- **Evaluation details:** inspect timestamps, query and tie policies, relevance
  threshold, software versions, source report, and baseline coverage.

All displayed scores come from the evaluator, with six decimals in tables.
Exported scores retain their stored precision. Dataset comparisons remain
separate because their query populations differ. JuriFindIT's prepared `test`
label is shown as its original validation split. `Judged` measures judgment
coverage, not effectiveness. Query analysis uses IDs, not query/document text.

## Inputs and structure

The builder discovers every `report.json` recursively under `evaluation/data`.
Each report must use schema version 1 and have a sibling `per-query.csv` with
matching metrics, run names, judged-query counts, and query populations.
Unsupported schemas, nonfinite values, duplicate query IDs, and incomplete
reports fail the build. All discovered reports are validated before existing
outputs are changed. Absolute source paths are omitted from the dashboard.

```text
visualization/
    scripts/build.py
    assets/index.html
    assets/dashboard.css
    assets/dashboard.js
    tests/test_build.py
    data/                 # generated, ignored by Git
        index.html
        queries-<hash>.js
```

Aggregate reports are embedded in the HTML. Per-query CSVs are streamed into
separate content-addressed payloads for each model. The browser loads only the
selected comparison and caches up to two model payloads; changing the metric
reuses those values. A model's complete per-query payload still must fit in
browser memory. Very large evaluations may eventually need query pagination
on a server; this first version avoids a database or service dependency.

Rebuilds reuse unchanged query payloads and replace the HTML last. Old payloads
are retained so already-open reports remain usable, and unrelated files in the
output directory are preserved. For a minimal shareable export after many
rebuilds, build into a fresh output directory. Generated payloads are trusted
JavaScript produced from the saved CSV; do not replace them with untrusted code.

## Verification

```powershell
& ./.venv/Scripts/python.exe -B -m unittest discover -s visualization/tests -p 'test_*.py' -v
node --check visualization/assets/dashboard.js
```

Tests cover value/ID preservation, dataset discovery, schema and CSV validation,
HTML data escaping, missing queries, and safe rebuild behavior.

An optional browser check exercises the generated SciFact and JuriFindIT dashboard
using Node.js 22+ and a local Chromium browser. It adds no runtime dependencies:

```powershell
node visualization/tests/browser.mjs
```

The default browser path is Microsoft Edge on Windows. Set `BROWSER_PATH` to a
Chromium executable on another installation or operating system. Both datasets
with the current eleven models must be included in the generated dashboard for
this integration check. It verifies interactions, offline per-query loading,
pagination, matching model scores, and mobile layout, and writes a screenshot
to the system temporary directory.
