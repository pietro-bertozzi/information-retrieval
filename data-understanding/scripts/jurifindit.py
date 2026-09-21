from pathlib import Path

from datasets import load_dataset

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def download_jurifindit():
    output_dir = DATA_DIR / "jurifindit"

    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== JuriFindIT ===")

    print("Downloading corpus...")
    corpus = load_dataset(
        "jurifindit/JuriFindIT",
        "corpus",
        split="corpus",
    )

    print("Downloading expert questions...")
    questions = load_dataset(
        "jurifindit/JuriFindIT",
        "questions",
    )

    print("Saving...")
    corpus.to_json(output_dir / "corpus.jsonl")

    for split_name, split in questions.items():
        split.to_json(output_dir / f"questions-{split_name}.jsonl")

    print("JuriFindIT ready.")


if __name__ == "__main__":
    download_jurifindit()