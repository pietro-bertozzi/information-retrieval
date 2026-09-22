"""Rank documents by the number of shared distinct adjacent word pairs."""

import argparse
import heapq
import json
import re
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

MODEL_NAME = "1.6.1-bigram-overlap"
DATA_DIR = Path(__file__).resolve().parents[2] / "data-preparation" / "data"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data"
TOKEN_PATTERN = re.compile(r"[^\W_]+")


def tokenize(text):
    """Return distinct adjacent pairs in the lowercase word sequence."""
    words = TOKEN_PATTERN.findall(text.lower())
    return set(zip(words, words[1:]))


def read_records(path, id_field):
    seen = set()
    with path.open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("each record must be a JSON object")
                identifier = row[id_field]
                if (
                    not isinstance(identifier, str)
                    or not identifier
                    or any(character.isspace() for character in identifier)
                ):
                    raise ValueError(f"{id_field} must be a nonempty, whitespace-free string")
                if identifier in seen:
                    raise ValueError(f"duplicate {id_field}: {identifier}")
                if not isinstance(row["text"], str):
                    raise ValueError("text must be a string")
                seen.add(identifier)
                yield identifier, row["text"]
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error


def build_index(corpus_path):
    postings = defaultdict(list)
    document_count = 0
    for doc_id, text in read_records(corpus_path, "doc_id"):
        for term in tokenize(text):
            postings[term].append(doc_id)
        document_count += 1
    if document_count == 0:
        raise ValueError(f"{corpus_path}: corpus is empty")
    return postings, document_count


def score(overlap, query_length):
    return overlap


def retrieve(query, postings, top_k=1000):
    if top_k < 1:
        raise ValueError("top-k must be positive")
    terms = tokenize(query)
    if not terms:
        return []
    overlaps = Counter()
    for term in terms:
        overlaps.update(postings.get(term, ()))
    # Ties use descending document IDs, matching evaluation.
    ranked = heapq.nlargest(
        top_k, overlaps.items(), key=lambda item: (item[1], item[0])
    )
    return [(doc_id, score(count, len(terms))) for doc_id, count in ranked]


def main(argv=None):
    parser = argparse.ArgumentParser(description=f"Generate a {MODEL_NAME} TREC run.")
    parser.add_argument("--dataset", default="scifact")
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--top-k", type=int, default=1000)
    parser.add_argument(
        "--output-dir", type=Path,
        help="default: modelling/data/<dataset>/<split>",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="replace this version's existing run"
    )
    args = parser.parse_args(argv)
    try:
        if Path(args.dataset).name != args.dataset or args.dataset in (".", ".."):
            raise ValueError("--dataset must be a directory name, not a path")
        if args.top_k < 1:
            raise ValueError("--top-k must be positive")
        dataset_dir = DATA_DIR / args.dataset
        corpus_path = dataset_dir / "corpus.jsonl"
        queries_path = dataset_dir / "queries" / f"{args.split}.jsonl"
        if not queries_path.is_file():
            raise ValueError(f"query file does not exist: {queries_path}")
        output_dir = args.output_dir or OUTPUT_DIR / args.dataset / args.split
        output_path = output_dir / f"{MODEL_NAME}.trec"
        if output_path.exists() and not args.overwrite:
            raise ValueError(f"run exists: {output_path}; use --overwrite to replace it")

        print(f"Indexing {corpus_path}...", flush=True)
        postings, document_count = build_index(corpus_path)
        print(f"Indexed {document_count} documents and {len(postings)} terms.", flush=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        query_count = 0
        matched_queries = 0
        result_count = 0
        with tempfile.TemporaryDirectory(prefix=".retrieval-", dir=output_dir) as temp:
            temporary_path = Path(temp) / output_path.name
            with temporary_path.open("w", encoding="utf-8") as stream:
                for query_id, text in read_records(queries_path, "query_id"):
                    results = retrieve(text, postings, args.top_k)
                    for rank, (doc_id, value) in enumerate(results, 1):
                        stream.write(
                            f"{query_id} Q0 {doc_id} {rank} {value} {MODEL_NAME}\n"
                        )
                    query_count += 1
                    matched_queries += bool(results)
                    result_count += len(results)
                    if query_count % 1000 == 0:
                        print(f"Processed {query_count} queries...", flush=True)
            if query_count == 0:
                raise ValueError(f"{queries_path}: query file is empty")
            temporary_path.replace(output_path)
        print(f"{matched_queries}/{query_count} queries matched; {result_count} results.")
        print(f"Run saved to {output_path.resolve()}")
    except (OSError, ValueError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()
