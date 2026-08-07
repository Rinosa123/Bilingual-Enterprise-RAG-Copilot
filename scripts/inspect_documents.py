"""Inspect the English and Arabic demonstration documents."""

from pathlib import Path

from src.ingestion.text_loader import load_text_documents


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DOCUMENT_DIRECTORY = PROJECT_ROOT / "data" / "sample_docs"


def main() -> None:
    """Load the documents and display their metadata."""

    documents = load_text_documents(SAMPLE_DOCUMENT_DIRECTORY)

    print(f"Loaded documents: {len(documents)}")
    print("-" * 70)

    for document in documents:
        print(
            f"{document.document_id} | "
            f"language={document.language} | "
            f"source={document.source} | "
            f"characters={len(document.text)}"
        )


if __name__ == "__main__":
    main()