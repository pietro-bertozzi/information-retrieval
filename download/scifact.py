from pathlib import Path

from datasets import load_dataset


DATA_DIR = Path("data")


def download_scifact():
    output_dir = DATA_DIR / "scifact"
    qrels_dir = output_dir / "qrels"

    output_dir.mkdir(parents=True, exist_ok=True)
    qrels_dir.mkdir(exist_ok=True)

    print("\n=== SciFact ===")

    print("Downloading corpus...")
    corpus = load_dataset(
        "BeIR/scifact",
        "corpus",
        split="corpus",
    )

    print("Downloading queries...")
    queries = load_dataset(
        "BeIR/scifact",
        "queries",
        split="queries",
    )

    print("Downloading qrels...")
    qrels = load_dataset("BeIR/scifact-qrels")

    print("Saving...")
    corpus.to_json(output_dir / "corpus.jsonl")
    queries.to_json(output_dir / "queries.jsonl")

    for split_name, split in qrels.items():
        split.to_json(qrels_dir / f"{split_name}.jsonl")

    print("SciFact ready.")


if __name__ == "__main__":
    download_scifact()