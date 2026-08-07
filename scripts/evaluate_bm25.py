"""Evaluate the BM25 baseline using bilingual questions."""

from pathlib import Path

from src.ingestion.chunker import chunk_documents
from src.ingestion.text_loader import load_text_documents
from src.retrieval.bm25_retriever import BM25Retriever


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCUMENT_DIRECTORY = PROJECT_ROOT / "data" / "sample_docs"


# label, question, expected chunk
DEMONSTRATION_QUERIES = (
    (
        "English question -> English document",
        "How many annual leave days do full-time employees receive?",
        "HR-EN-001-CH-003",
    ),
    (
        "Arabic question -> Arabic document",
        "ما الحد الأقصى لتكلفة الفندق؟",
        "HR-AR-001-CH-003",
    ),
    (
        "Arabic question -> English document",
        "كم عدد أيام الإجازة السنوية للموظف؟",
        "HR-EN-001-CH-003",
    ),
    (
        "English question -> Arabic document",
        "When must an expense claim be submitted?",
        "HR-AR-001-CH-002",
    ),
)


def main() -> None:
    """Build the retriever and evaluate all demonstration questions."""

    documents = load_text_documents(DOCUMENT_DIRECTORY)
    chunks = chunk_documents(documents)
    retriever = BM25Retriever(chunks)

    successful_queries = 0

    for label, query, expected_chunk_id in DEMONSTRATION_QUERIES:
        results = retriever.search(query, top_k=3)

        retrieved_chunk_id = (
            results[0].chunk.chunk_id
            if results
            else "NO_MATCH"
        )

        passed = retrieved_chunk_id == expected_chunk_id

        if passed:
            successful_queries += 1

        outcome = "PASS" if passed else "MISS"

        print("=" * 80)
        print(f"Test: {label}")
        print(f"Question: {query}")
        print(f"Expected: {expected_chunk_id}")
        print(f"Retrieved: {retrieved_chunk_id}")
        print(f"Result: {outcome}")

        if results:
            print(f"BM25 score: {results[0].score:.4f}")
            print(f"Section: {results[0].chunk.section}")

    total_queries = len(DEMONSTRATION_QUERIES)
    accuracy = successful_queries / total_queries

    print("=" * 80)
    print(
        f"Baseline accuracy: "
        f"{successful_queries}/{total_queries} "
        f"({accuracy:.0%})"
    )


if __name__ == "__main__":
    main()