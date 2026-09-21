import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data-understanding" / "data"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data"


def read_jsonl(path):
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            yield json.loads(line)


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            json.dump(row, file, ensure_ascii=False)
            file.write("\n")


def prepare_scifact():
    print("\n=== Preparing SciFact ===")

    source_dir = DATA_DIR / "scifact"
    output_dir = OUTPUT_DIR / "scifact"

    write_jsonl(
        output_dir / "corpus.jsonl",
        (
            {
                "doc_id": str(row["_id"]),
                "text": "\n\n".join(
                    part for part in (row["title"], row["text"]) if part
                ),
            }
            for row in read_jsonl(source_dir / "corpus.jsonl")
        ),
    )

    queries = list(read_jsonl(source_dir / "queries.jsonl"))

    for split_name in ("train", "test"):
        qrels = list(read_jsonl(source_dir / "qrels" / f"{split_name}.jsonl"))
        query_ids = {str(row["query-id"]) for row in qrels}

        write_jsonl(
            output_dir / "queries" / f"{split_name}.jsonl",
            (
                {
                    "query_id": str(row["_id"]),
                    "text": row["text"],
                }
                for row in queries
                if str(row["_id"]) in query_ids
            ),
        )

        write_jsonl(
            output_dir / "qrels" / f"{split_name}.jsonl",
            (
                {
                    "query_id": str(row["query-id"]),
                    "doc_id": str(row["corpus-id"]),
                    "relevance": row["score"],
                }
                for row in qrels
            ),
        )


def prepare_jurifindit():
    print("\n=== Preparing JuriFindIT ===")

    source_dir = DATA_DIR / "jurifindit"
    output_dir = OUTPUT_DIR / "jurifindit"

    write_jsonl(
        output_dir / "corpus.jsonl",
        (
            {
                "doc_id": str(row["id"]),
                "text": row["content"],
            }
            for row in read_jsonl(source_dir / "corpus.jsonl")
        ),
    )

    split_mapping = {
        "train": "train",
        "validation": "test",
    }

    for source_split, target_split in split_mapping.items():
        questions = read_jsonl(source_dir / f"questions-{source_split}.jsonl")

        query_rows = []
        qrel_rows = []

        for query_id, row in enumerate(questions):
            query_id = str(query_id)

            query_rows.append(
                {
                    "query_id": query_id,
                    "text": row["question"],
                }
            )

            for doc_id in row["relevant_doc_ids"]:
                qrel_rows.append(
                    {
                        "query_id": query_id,
                        "doc_id": str(doc_id),
                        "relevance": 1,
                    }
                )

        write_jsonl(
            output_dir / "queries" / f"{target_split}.jsonl",
            query_rows,
        )

        write_jsonl(
            output_dir / "qrels" / f"{target_split}.jsonl",
            qrel_rows,
        )


def prepare_marco(dataset_name):
    print(f"\n=== Preparing {dataset_name} ===")

    source_dir = DATA_DIR / dataset_name
    output_dir = OUTPUT_DIR / dataset_name

    write_jsonl(
        output_dir / "corpus.jsonl",
        read_jsonl(source_dir / "corpus.jsonl"),
    )

    split_mapping = {
        "train": "train",
        "dev": "test",
    }

    for source_split, target_split in split_mapping.items():
        write_jsonl(
            output_dir / "queries" / f"{target_split}.jsonl",
            read_jsonl(source_dir / f"queries-{source_split}.jsonl"),
        )

        write_jsonl(
            output_dir / "qrels" / f"{target_split}.jsonl",
            read_jsonl(source_dir / "qrels" / f"{source_split}.jsonl"),
        )


def main():
    prepare_scifact()
    prepare_jurifindit()
    prepare_marco("mmarco-it")
    prepare_marco("msmarco")

    print("\n=== Data preparation complete ===")


if __name__ == "__main__":
    main()