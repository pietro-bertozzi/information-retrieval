import json
import os
import sys
from pathlib import Path

import ir_datasets

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def ensure_utf8_mode():
    """Restart Python in UTF-8 mode on Windows if necessary."""
    if sys.platform == "win32" and sys.flags.utf8_mode == 0:
        print("Restarting Python in UTF-8 mode...")

        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"

        os.execve(
            sys.executable,
            [sys.executable, *sys.argv],
            env,
        )


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            json.dump(row, file, ensure_ascii=False)
            file.write("\n")


def download_mmarco_it():
    output_dir = DATA_DIR / "mmarco-it"
    qrels_dir = output_dir / "qrels"

    output_dir.mkdir(parents=True, exist_ok=True)
    qrels_dir.mkdir(exist_ok=True)

    print("\n=== mMARCO Italian ===")

    corpus_dataset = ir_datasets.load("mmarco/it")
    train_dataset = ir_datasets.load("mmarco/it/train")
    dev_dataset = ir_datasets.load("mmarco/it/dev")

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

    print("mMARCO Italian ready.")


if __name__ == "__main__":
    ensure_utf8_mode()
    download_mmarco_it()