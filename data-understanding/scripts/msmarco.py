import json
from pathlib import Path

import ir_datasets

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            json.dump(row, file, ensure_ascii=False)
            file.write("\n")


def download_msmarco():
    output_dir = DATA_DIR / "msmarco"
    qrels_dir = output_dir / "qrels"

    output_dir.mkdir(parents=True, exist_ok=True)
    qrels_dir.mkdir(exist_ok=True)

    print("\n=== MS MARCO ===")

    corpus_dataset = ir_datasets.load("msmarco-passage")
    train_dataset = ir_datasets.load("msmarco-passage/train")
    dev_dataset = ir_datasets.load("msmarco-passage/dev")

    print("Downloading and saving corpus...")
    write_jsonl(
        output_dir / "corpus.jsonl",
        (
            {
                "doc_id": doc.doc_id,
                "text": doc.text,
            }
            for doc in corpus_dataset.docs_iter()
        ),
    )

    for split_name, dataset in (("train", train_dataset), ("dev", dev_dataset)):
        print(f"Downloading and saving {split_name} queries...")
        write_jsonl(
            output_dir / f"queries-{split_name}.jsonl",
            (
                {
                    "query_id": query.query_id,
                    "text": query.text,
                }
                for query in dataset.queries_iter()
            ),
        )

        print(f"Downloading and saving {split_name} qrels...")
        write_jsonl(
            qrels_dir / f"{split_name}.jsonl",
            (
                {
                    "query_id": qrel.query_id,
                    "doc_id": qrel.doc_id,
                    "relevance": qrel.relevance,
                }
                for qrel in dataset.qrels_iter()
            ),
        )

    print("MS MARCO ready.")


if __name__ == "__main__":
    download_msmarco()