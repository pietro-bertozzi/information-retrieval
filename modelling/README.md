# Modelling

Eleven simple lexical baselines use the prepared corpus and queries, require no
training, and depend only on the Python standard library. See the
[root README](../README.md#setup) for the project environment.

## Versions

Version numbers come first in script names, TREC filenames, and run tags.
The first two versions retain their original scoring behavior. New scoring
variants use successive minor numbers within the simple lexical-model family.

In the formulas below, `o` is the number of distinct shared words, `q` is the
number of distinct query words, `d` is the number of distinct document words,
and `tf(t)` is the number of occurrences of word `t` in the document. For IDF,
`N` is the number of corpus documents (including empty documents), and `df(t)`
is the number of documents containing word `t`.

| Version | Script | Score |
| --- | --- | --- |
| `1.1.1` | [term overlap](scripts/1.1.1-term-overlap.py) | `o` |
| `1.1.2` | [query coverage](scripts/1.1.2-term-overlap-percentage.py) | `o / q` |
| `1.2.1` | [document coverage](scripts/1.2.1-document-coverage.py) | `o / d` |
| `1.3.1` | [Jaccard](scripts/1.3.1-jaccard.py) | `o / (q + d - o)` |
| `1.4.1` | [binary cosine](scripts/1.4.1-binary-cosine.py) | `o / sqrt(q * d)` |
| `1.5.1` | [term frequency](scripts/1.5.1-term-frequency.py) | Sum of `tf(t)` over distinct query words |
| `1.5.2` | [log term frequency](scripts/1.5.2-log-term-frequency.py) | Sum of `ln(1 + tf(t))` over distinct query words |
| `1.6.1` | [bigram overlap](scripts/1.6.1-bigram-overlap.py) | Number of distinct adjacent word pairs shared by query and document |
| `1.7.1` | [stopword-filtered overlap](scripts/1.7.1-stopword-overlap.py) | `o` after removing a small explicit list of common words |
| `1.8.1` | [IDF-only overlap](scripts/1.8.1-idf-overlap.py) | Sum of `ln((N + 1) / (df(t) + 1)) + 1` over distinct shared words |
| `1.9.1` | [unigram + bigram overlap](scripts/1.9.1-unigram-bigram-overlap.py) | Distinct shared words plus distinct shared adjacent word pairs |

Each lexical version is a complete, standalone script with its own tokenization,
indexing, scoring, input validation, and output handling. Code is intentionally
repeated so versions can be read and changed independently. Vector methods share
only narrow validation/model-loading helpers. Tests in `modelling/tests/` cover
retrieval sidecars and vector-index lifecycle.

Version 1.1.2 divides all scores for one query by the same constant, preserving
rankings and evaluation metrics. The document-length variants can change the
ranking: they calculate the final score before selecting the top results.

## Inputs and matching

For `--dataset scifact --split test`, the inputs are:

- `data-preparation/data/scifact/corpus.jsonl`
- `data-preparation/data/scifact/queries/test.jsonl`

Other prepared datasets work through the same `--dataset` argument. Query and
document IDs are preserved exactly, including leading zeros. Retrieval does
not read relevance judgments or use them to select candidates.

All lexical versions lowercase text and extract Unicode alphanumeric sequences.
Punctuation and underscores separate words; accents and numbers are retained.
There is no stemming. Only version 1.8.1 uses IDF weighting.

- Set-based methods count repeated words once in both query and document.
  Query words absent from the corpus still contribute to query length.
- Frequency methods count repetitions in documents, but count each distinct
  query word once. Log frequency uses natural logarithms and accurate summation
  to keep equivalent totals tied consistently.
- Bigram overlap forms adjacent pairs after tokenization. Punctuation and sentence
  boundaries do not interrupt the resulting word sequence. Repeated pairs count
  once. A query with fewer than two words returns no results.
- IDF-only overlap calculates smoothed weights from the corpus alone. Repeated
  words contribute once; neither term frequency nor document length changes a
  matched word's contribution. A word in every document has weight 1, and rarer
  words have higher weights. Words absent from the corpus contribute nothing.
- Combined unigram-bigram overlap gives one point to each distinct shared word
  and one to each distinct shared adjacent word pair. There is no tuned bonus
  parameter. A single-word query still matches through its unigram. Pairs follow
  the same tokenization and punctuation rules as the bigram-only model.
- Stopword overlap removes words from both queries and documents using the small
  English or Italian lists embedded in its script. These are deliberately simple
  lists, not comprehensive linguistic resources. Negations such as `not` and
  `non` are retained. The language is inferred as English for SciFact/MS MARCO
  and Italian for JuriFindIT/mMARCO Italian. Use `--language en` or `--language it`
  to override it; other dataset names require an explicit language.

Each model builds an in-memory inverted index. Only documents with positive
scores are returned. Empty, punctuation-only, and unmatched queries produce no
rows; evaluation counts missing judged queries as zero. Ties use descending
document ID as strings, matching evaluation.

## Run

The normal workflow runs retrieval and evaluation together from the repository root:

```bash
python -m experiments.run_experiments --dataset scifact
```

See the [root workflow](../README.md#workflow) for method selection, dataset
selection, listing, and language overrides. The commands below are for running
retrieval separately.

Run these Linux commands from the repository root. For example, run Jaccard:

```bash
./.venv/bin/python modelling/scripts/1.3.1-jaccard.py --dataset scifact
```

To generate all eleven runs:

```bash
for script in modelling/scripts/1.*.py; do
    ./.venv/bin/python "$script" --dataset scifact || exit 1
done
```

Defaults are `--dataset scifact`, `--split test`, and `--top-k 1000`.
Use `--split train` for training queries, `--top-k N` to change retrieval depth,
and `--output-dir PATH` for a custom output directory. Explicit output paths are
relative to the working directory; default locations are relative to the scripts.
Use distinct output directories when comparing different settings of one version.

Each run rebuilds its index. The larger MARCO datasets require substantially more
RAM and processing time with these deliberately simple indexes. Full initial
runs and evaluation were performed on SciFact and JuriFindIT; the MARCO datasets
have not been run end to end with these methods.

## Configured benchmark

A retrieval **method** is a script, such as `1.1.1-term-overlap` or
`2.1.1-dense-retrieval`. An experiment **configuration** is one invocation of
that method with concrete settings. `modelling/scripts/` is the source of truth
for methods: every public Python script is discovered automatically. Ordinary
methods, including all 1.x lexical methods, need no benchmark configuration entry.

[benchmark-variants.json](configs/benchmark-variants.json) is the researcher-editable
selection of benchmark values. It contains no retrieval-script inventory, cache
paths, generated model metadata, or fusion defaults:

```json
{
  "version": 1,
  "variants": {
    "dense_models": [
      {"id": "multilingual-minilm", "value": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"},
      {"id": "potion-multilingual", "value": "minishlab/potion-multilingual-128M"}
    ],
    "sparse_models": [
      {"id": "bm42", "value": "Qdrant/bm42-all-minilm-l6-v2-attentions"}
    ]
  }
}
```

Add or remove objects under `dense_models` and `sparse_models` to change the model
selection. `id` is a unique, stable lowercase filename slug; `value` is the exact
value passed to the method's declared CLI option. The default catalog selects
MiniLM and Potion plus BM42, producing exactly five vector configurations.
MPNet is excluded only from this default catalog; explicit model overrides remain
available.

Configurable scripts opt in by defining a literal module-level declaration:

```python
EXPERIMENT_CAPABILITY = {
    "version": 1,
    "dimensions": {"dense_models": "embedding-model"},
}
```

The orchestrator parses this declaration as data without importing the module.
Optional `parameter_names` maps CLI names to existing metadata fields, and
`parameters` declares fixed search/fusion settings for resume validation.
See [the capability and artifact contract](../experiments/README.md).
Dense consumes `dense_models`, sparse consumes `sparse_models`, and hybrid declares
both dimensions with its two corresponding CLI options. Consequently dense and
sparse expand independently, while hybrid receives the dense × sparse Cartesian
product automatically. A future configurable method participates by declaring
its dimensions and adding their selected values to the configuration; production
orchestrator code does not change. A future ordinary script needs only to exist
under `modelling/scripts/` and remains available in normal mode.

Run the configured benchmark on SciFact or all runnable datasets:

```text
python -m experiments.run_experiments --dataset scifact --benchmark
python -m experiments.run_experiments --all --benchmark
python -m experiments.run_experiments --dataset scifact --benchmark --resume
```

Use `python -m experiments.run_experiments --list` to inspect discovery and the expanded
benchmark summary. `--benchmark-config PATH` selects another validated configuration
with `--benchmark` or `--list`. Benchmark mode runs only scripts that expose the
capability declaration; it never reruns ordinary lexical methods. It rejects
`--method` and `--method-option` so configuration and command-line settings cannot
silently override each other.
Manual runs remain available:

```text
python -m experiments.run_experiments --dataset scifact --method 1.1.1-term-overlap
python -m experiments.run_experiments --dataset scifact --method 2.1.1-dense-retrieval
python -m experiments.run_experiments --dataset scifact --method 2.1.1-dense-retrieval --method-option 2.1.1-dense-retrieval:embedding-model=minishlab/potion-multilingual-128M
```

Benchmark rankings use deterministic names such as:

```text
2.1.1-dense-retrieval__multilingual-minilm.trec
2.2.1-sparse-retrieval__bm42.trec
2.3.1-hybrid-retrieval__multilingual-minilm__bm42.trec
```

Each sidecar retains the retrieval script's model and index metadata and adds
`retrieval_method` plus `experiment_id`. The evaluator therefore creates a
distinct MLflow run name from each filename and logs the stable identity and
embedding/fusion settings as `retrieval.*` parameters. Rankings remain under
`modelling/data/<dataset>/<split>/`; the combined report remains under
`evaluation/data/<dataset>/<split>/`.

Each configuration stages and validates its own ranking and metadata, then publishes
the pair under its experiment ID. If a later configuration fails, earlier successes
remain published and are still evaluated/logged. The evaluator receives only
successful outputs produced by the current invocation, never a stale file left by
a failed configuration. Failures are reported and the overall command returns
nonzero. Compatible Qdrant indexes are still reused by the retrieval scripts;
changing a model selects its compatible model/corpus-specific collection or creates
a new one. `--overwrite` replaces ranking files only and never rebuilds indexes.

Resume reuses only valid pairs matching dataset, split, method, configuration ID,
actual model values, top-k and declared search/fusion settings. Existing completed
SciFact filenames are preserved. Reused rankings are evaluated again; each
evaluation creates a new MLflow run. Resume does not repair interrupted Qdrant
collections. Legacy metadata cannot establish unchanged query/corpus bytes or
detect every structurally valid truncation; see the experiments documentation.

## Outputs and evaluation

Runs are saved as `modelling/data/<dataset>/<split>/<version>-<method>.trec`.
For example:

```text
modelling/data/scifact/test/1.1.1-term-overlap.trec
modelling/data/scifact/test/1.1.2-term-overlap-percentage.trec
modelling/data/scifact/test/1.3.1-jaccard.trec
```

Each line uses the standard TREC format:

```text
query_id Q0 doc_id rank score version-method
```

Ranks start at 1. Each query has at most `--top-k` rows. Existing runs require
`--overwrite` to replace them; failed input validation leaves previous runs
intact. A sibling `<version>-<method>.metadata.json` records dataset, split,
method/version, configured `top_k`, and resolved language for the stopword model.
Keep this file with its ranking: evaluation uses it to log configuration to MLflow.
Both files are staged after input validation and replaced on successful retrieval.
They are not a filesystem transaction; an interrupted replacement may require
rerunning retrieval. Generated runs and sidecars under `modelling/data/` are ignored
by Git. Sidecars are descriptive metadata, not content fingerprints.

Retrieval stays independent of MLflow. See [Tracking](../tracking/README.md) for
storage, comparison, and the sidecar format for external methods.

Run and compare all current versions with:

```bash
python -m experiments.run_experiments --dataset scifact
```

This selects the scripts currently on disk and excludes retired ranking files.
For separately generated rankings, pass their exact paths to
[Evaluation](../evaluation/README.md).

The reports include aggregate and per-query metrics. Versions 1.1.1 and 1.1.2
should match exactly; the other variants can change rankings. One shared report
set is saved in `evaluation/data/scifact/test/`. The command replaces that report
set with a comparison of every supplied run. Each file also becomes a separate
MLflow run in the `scifact` experiment. Methods evaluated in separate invocations
can be compared there without rebuilding a combined report.
These initial comparisons describe one dataset, rather than establishing which
method will be best across datasets. Keep parameter tuning separate from final
evaluation.

## JuriFindIT

All eleven lexical models were run against 23,617 prepared documents and 179 queries
in `--split test` (the original validation split), using `--top-k 1000`.
The stopword model automatically selects Italian. No training or parameter
tuning is required. See the [comparison results](../evaluation/README.md#jurifindit-results).

To reproduce in Windows PowerShell from the repository root:

```powershell
& ./.venv/Scripts/python.exe -B -m experiments.run_experiments --dataset jurifindit
```

This refreshes the selected methods and local reports and logs new MLflow runs.
Historical reports may use the earlier names for the renamed methods.

## Adding a method

Add a public Python script under `modelling/scripts/` with the common options
`--dataset`, `--split`, `--top-k`, `--output-dir`, and `--overwrite`. It must write
`<script-stem>.trec` and `<script-stem>.metadata.json` into `--output-dir`, exit
nonzero on failure, and keep execution behind its `__main__` guard. `MODEL_NAME`
must match the script stem. Empty rankings are valid; metadata must be present.
Method-specific scalar options can be passed through the orchestrator's repeated
`--method-option METHOD:OPTION=VALUE`. Common orchestration options cannot be
overridden this way. Keep MLflow integration in the evaluator.

## Verification

```powershell
& ./.venv/Scripts/python.exe -B -m unittest discover -s modelling/tests -p 'test_*.py' -v
```

## Dense retrieval

`2.1.1-dense-retrieval.py` is one configurable dense bi-encoder baseline. FastEmbed
is the CPU embedding runtime; the embedding model is an experiment parameter;
Qdrant stores document vectors and returns their cosine-similarity ranking.
The method reads the same normalized corpus/query JSONL as the lexical methods.
It embeds original text without lexical tokenization and never reads qrels.

### Model configurations

These are embedding-model configurations of the same
`2.1.1-dense-retrieval` method. MiniLM and Potion are the default benchmark choices;
MPNet is an optional explicit model. The identifiers and
dimensions below come from the installed FastEmbed 0.8.1 registry.

| FastEmbed model identifier | Family | Languages | Dimension | Why include it | Query/document encoding |
| --- | --- | --- | ---: | --- | --- |
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | multilingual MiniLM sentence transformer | 50+ languages, including English and Italian | 384 | Smallest transformer baseline in the set; approximately 0.22 GB in FastEmbed. This remains the deterministic default. | No prefixes required. The method still calls `passage_embed` and `query_embed` explicitly. |
| `minishlab/potion-multilingual-128M` | POTION/Model2Vec static model distilled from BGE-M3 | 101 languages, including English and Italian | 256 | A meaningfully different static/distilled approach with fast CPU inference; approximately 0.51 GB in FastEmbed. | No prefixes required. The method still calls `passage_embed` and `query_embed` explicitly. |
| `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` | multilingual MPNet sentence transformer | 50+ languages, including English and Italian | 768 | A larger, higher-capacity transformer comparison while remaining practical locally; approximately 1.0 GB in FastEmbed. | No prefixes required. The method still calls `passage_embed` and `query_embed` explicitly. |

The `sentence-transformers/` prefix is part of two model identifiers; the
Sentence Transformers Python framework is not used. These are initial
representatives, not a claim that one will win on every dataset. The CLI remains
open to any dense model supported by the installed FastEmbed version; no script
or orchestrator registry change is needed. See [FastEmbed's model catalog](https://qdrant.github.io/fastembed/examples/Supported_Models/),
the [POTION model card](https://huggingface.co/minishlab/potion-multilingual-128M),
and the Sentence Transformers [multilingual model documentation](https://www.sbert.net/docs/sentence_transformer/pretrained_models.html).

### Setup and commands

From the repository root, with the project environment active:

```text
python -m pip install -r requirements.txt
docker compose -f infrastructure/qdrant/compose.yaml up -d
python -m experiments.run_experiments --dataset scifact --method 2.1.1-dense-retrieval
```

This reuses the local Docker server configuration (Qdrant 1.19.1) without running
or importing the learning demo. Collection metadata requires a recent server;
use the pinned Compose image. Open <http://localhost:6333/dashboard> to inspect
collections. HTTP uses localhost port 6333; 6334 is reserved for gRPC.
On Windows without environment activation, replace `python` with
`& ./.venv/Scripts/python.exe` in PowerShell.

Run each configuration explicitly using the existing generic method-option interface:

```text
python -m experiments.run_experiments --dataset scifact --method 2.1.1-dense-retrieval --method-option 2.1.1-dense-retrieval:embedding-model=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
python -m experiments.run_experiments --dataset scifact --method 2.1.1-dense-retrieval --method-option 2.1.1-dense-retrieval:embedding-model=minishlab/potion-multilingual-128M
python -m experiments.run_experiments --dataset scifact --method 2.1.1-dense-retrieval --method-option 2.1.1-dense-retrieval:embedding-model=sentence-transformers/paraphrase-multilingual-mpnet-base-v2
```

These are three separate manual invocations. Normal `--all` still runs one dense
configuration using the default unless an explicit `embedding-model` option is
supplied; `--benchmark` expands the configured variants. Each invocation logs its model
as `retrieval.embedding_model` in the dataset's MLflow experiment. Add
`--log-rankings` when the ranking file itself should remain attached to each run;
the next invocation replaces this method's local TREC and metadata files.

The script also works directly, producing retrieval outputs only:

```text
python modelling/scripts/2.1.1-dense-retrieval.py --dataset scifact --split test --top-k 1000 --overwrite
```

Additional method options are `qdrant-url`, `batch-size` (default 32), `threads`
(default 2), and `rebuild-index`. For example, explicitly rebuild the selected index:

```text
python -m experiments.run_experiments --dataset scifact --method 2.1.1-dense-retrieval --method-option 2.1.1-dense-retrieval:rebuild-index=true
```

Direct invocation also accepts bare `--rebuild-index`. **`--overwrite` replaces
ranking/metadata files only; it never authorizes index deletion.**

### Indexing and reuse

1. Validate normalized records and fingerprint the corpus bytes with SHA-256.
2. Derive `ir_dense_<dataset>_<configuration-hash>` from dataset, corpus hash,
   count, embedding model/definition, resolved weight/tokenizer file hashes,
   FastEmbed/ONNX Runtime versions, dimension, CPU provider,
   passage/query encoding convention, and cosine distance. Splits share document
   vectors; split and top-k do not change the collection identity.
3. Reuse an existing collection only if its stored manifest, completion flag,
   vector dimension/distance, and exact point count match. Incompatibility is an
   error, not an automatic delete/rebuild.
4. Otherwise embed documents in bounded batches and upload points, waiting for
   each write. Mark the collection complete only after checking corpus stability
   and stored point count. Interrupted indexes require explicit rebuilding.
5. Embed queries with the same model's `query_embed`, search Qdrant, and write
   finite scores and original IDs to TREC. Empty/incomplete unexpected results
   fail the run without replacing previous ranking files.

Qdrant point IDs are sequential integers; payload `doc_id` preserves the exact
normalized identifier, including leading zeros. Only vectors and this ID mapping
are uploaded; text remains in the normalized corpus. Corpus vectors use
`passage_embed`. Query vectors are searched, not stored. Qdrant handles cosine
normalization. Search uses `hnsw_ef=128`, `exact=False`; Qdrant may scan small
collections directly. Scores are sorted descending, with reverse document-ID
tie ordering matching the evaluator.

A different model, corpus revision, dimension, or recorded embedding configuration
gets a different collection. Old collections remain until explicitly removed.
Rebuild deletes only the collection matching the current identity. Run one writer
per collection; concurrent indexing/rebuilds are not supported. Do not manually
modify managed collection points or metadata.

### Outputs and tracking

Defaults produce:

- `modelling/data/<dataset>/<split>/2.1.1-dense-retrieval.trec`
- `modelling/data/<dataset>/<split>/2.1.1-dense-retrieval.metadata.json`
- Existing evaluator reports under `evaluation/data/<dataset>/<split>/`.

The sidecar follows the existing `dataset`, `split`, scalar `parameters` schema.
It records the model/runtime, dimension, collection, corpus hash, distance,
model-file fingerprint, index reuse, batching/thread settings, search settings,
and software versions.
Evaluation logs these as `retrieval.*` parameters, including
**`retrieval.embedding_model`**, inside the **dataset's MLflow experiment**.
Modelling has no MLflow dependency or logging code.

Each manual orchestrator invocation creates a new MLflow run but replaces this method's
local ranking files. To retain rankings for model comparisons, use the
orchestrator's `--log-rankings` option or copy each ranking and its sidecar before
the next invocation. Selecting all methods includes exactly one dense run with
the default (or explicitly supplied) model. It does not enumerate FastEmbed models.
Lexical-only selections do not contact Qdrant or load embedding models.

### Operational limits

The first run downloads public model files; later runs reuse
`modelling/data/.fastembed-cache/`. Inference is local and requires no API key.
Inputs are fingerprinted/validated on reuse, but documents are not re-embedded.
Embeddings are batched; duplicate-ID validation still keeps document IDs in memory.
Long text is truncated according to the selected model; there is no chunking.
Resolved model files are fingerprinted on each run, so changed weights/tokenizers
under the same model name receive a separate index. Preserve the model cache to
reproduce the exact weights later; the CLI does not select upstream revisions.

Before MARCO-scale runs, review RAM/disk requirements, indexing time, model
truncation, and approximate-search recall. No large-corpus benchmark or distributed
indexing is provided. The development smoke test used five documents and two
English/Italian queries, with real FastEmbed, Docker Qdrant, evaluation and MLflow.

The reused Docker volume `ir-qdrant-learning-data` now holds dense indexes as well
as the demo. `docker compose -f infrastructure/qdrant/compose.yaml down` preserves
it; **`down --volumes` deletes all these indexes too**. The demo itself recreates
only its separate `qdrant_learning_documents` collection.

## Vector retrieval taxonomy

The vector methods use the same prepared inputs and TREC/metadata output contract:

| Family | Method | Representation and search |
| --- | --- | --- |
| 2.1 Dense | `2.1.1-dense-retrieval` | One full vector per text; Qdrant cosine search captures semantic similarity. |
| 2.2 Sparse neural | `2.2.1-sparse-retrieval` | Learned nonzero token indices and weights; Qdrant sparse search keeps the representation sparse. |
| 2.3 Hybrid | `2.3.1-hybrid-retrieval` | Dense and sparse searches feed Qdrant-native Reciprocal Rank Fusion (RRF). |

Filesystem discovery means `--all` runs one default configuration of each vector
method. Only `--benchmark` expands the dimensions declared by configurable methods.

FastEmbed computes both kinds of embeddings. Qdrant stores and indexes them;
it does not create embeddings. Corpus text uses `passage_embed`, while queries
use `query_embed`, so models with distinct document/query behavior retain it.

### Sparse neural retrieval

The default sparse model is
`Qdrant/bm42-all-minilm-l6-v2-attentions`, the installed FastEmbed 0.8.1
registry's small BM42 neural sparse model. Document weights come from MiniLM
attention, queries follow BM42's query encoder, and Qdrant applies its required
IDF modifier. Sparse vectors remain `indices + values`; they are never expanded
to dense arrays. BM42 is English-oriented, so Italian sparse and hybrid results
should be treated as exploratory until a suitable multilingual sparse model is
selected and tested.

Run the default sparse configuration:

```text
python -m experiments.run_experiments --dataset scifact --method 2.2.1-sparse-retrieval
```

Override it with any sparse model supported by the installed FastEmbed version:

```text
python -m experiments.run_experiments --dataset scifact --method 2.2.1-sparse-retrieval --method-option 2.2.1-sparse-retrieval:embedding-model=<FASTEMBED_SPARSE_MODEL>
```

### Hybrid retrieval

Hybrid retrieval uses the dense default
`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` and the sparse
BM42 default above. Each Qdrant point has named `dense` and `sparse` vectors.
Qdrant retrieves `top_k` candidates independently from both indexes and combines
their ranks with native RRF (`k=60`, equal influence). Rank fusion avoids treating
dense cosine scores and sparse scores as directly comparable.

```text
python -m experiments.run_experiments --dataset scifact --method 2.3.1-hybrid-retrieval
```

Override either model independently through generic method options:

```text
python -m experiments.run_experiments --dataset scifact --method 2.3.1-hybrid-retrieval --method-option 2.3.1-hybrid-retrieval:dense-embedding-model=<FASTEMBED_DENSE_MODEL> --method-option 2.3.1-hybrid-retrieval:sparse-embedding-model=<FASTEMBED_SPARSE_MODEL>
```

### Sparse and hybrid index lifecycle

Sparse uses a dedicated sparse-only collection. Hybrid uses a separate collection
with named dense and sparse vectors; it does not alter or silently reuse the
dense-only collection. Collection identities include the dataset, corpus hash and
count, model definitions and downloaded-file fingerprints, runtime versions,
vector configuration, dense dimension where applicable, and sparse IDF behavior.
Completion metadata, collection configuration, and exact point count are checked
before reuse. An incompatible or incomplete collection fails clearly.

`--overwrite` replaces local TREC and metadata files only. Rebuild the selected
method's exact collection explicitly when needed:

```text
python -m experiments.run_experiments --dataset scifact --method 2.2.1-sparse-retrieval --method-option 2.2.1-sparse-retrieval:rebuild-index=true
python -m experiments.run_experiments --dataset scifact --method 2.3.1-hybrid-retrieval --method-option 2.3.1-hybrid-retrieval:rebuild-index=true
```

Sparse metadata records `sparse_embedding_model`, vector type/configuration, and
the index fingerprint. Hybrid metadata records both embedding models, named-vector
configuration, and RRF settings. The existing evaluator logs these scalar fields
as `retrieval.*` MLflow parameters in the dataset experiment; retrieval scripts
contain no MLflow logic.
