"""Build searchable chunks from the demonstration documents."""

from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path

from src.ingestion.chunker import chunk_documents
from src.ingestion.text_loader import load_text_documents


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_DIRECTORY = PROJECT_ROOT / "data" / "sample_docs"

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sample_chunks.jsonl"
)


def main() -> None:
    """Load documents, create chunks and save them as JSON Lines."""

    documents = load_text_documents(INPUT_DIRECTORY)
    chunks = chunk_documents(documents)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as output_stream:
        for chunk in chunks:
            json_record = json.dumps(
                asdict(chunk),
                ensure_ascii=False,
            )
            output_stream.write(json_record + "\n")

    language_counts = Counter(
        chunk.language for chunk in chunks
    )

    print(f"Loaded documents: {len(documents)}")
    print(f"Created chunks: {len(chunks)}")
    print(f"Language counts: {dict(language_counts)}")
    print(f"Output file: {OUTPUT_FILE}")
    print("-" * 80)

    for chunk in chunks:
        print(
            f"{chunk.chunk_id} | "
            f"language={chunk.language} | "
            f"section={chunk.section} | "
            f"characters={chunk.character_count}"
        )


if __name__ == "__main__":
    main()