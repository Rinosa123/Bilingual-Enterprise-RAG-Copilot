"""Load UTF-8 English and Arabic enterprise documents."""

from dataclasses import dataclass
from pathlib import Path
import re


# Our demonstration IDs follow formats such as HR-EN-001 and HR-AR-001.
DOCUMENT_ID_PATTERN = re.compile(r"\bHR-[A-Z]{2}-\d{3}\b")


@dataclass(frozen=True)
class LoadedDocument:
    """A document and its essential metadata."""

    document_id: str
    language: str
    source: str
    text: str


def detect_language(text: str) -> str:
    """
    Detect whether a document is mainly Arabic or English.

    This lightweight method counts alphabetic Arabic characters.
    A multilingual model will be used later for semantic retrieval.
    """

    alphabetic_count = sum(character.isalpha() for character in text)

    if alphabetic_count == 0:
        return "unknown"

    arabic_count = sum(
        character.isalpha() and "\u0600" <= character <= "\u06ff"
        for character in text
    )

    arabic_ratio = arabic_count / alphabetic_count

    return "ar" if arabic_ratio >= 0.20 else "en"


def extract_document_id(text: str, source_path: Path) -> str:
    """Extract the document ID or use the filename as a fallback."""

    match = DOCUMENT_ID_PATTERN.search(text)

    if match:
        return match.group(0)

    return source_path.stem


def load_text_documents(input_directory: Path) -> list[LoadedDocument]:
    """Load every non-empty UTF-8 text document from a directory."""

    if not input_directory.exists():
        raise FileNotFoundError(
            f"Input directory does not exist: {input_directory}"
        )

    text_files = sorted(input_directory.glob("*.txt"))

    if not text_files:
        raise FileNotFoundError(
            f"No .txt documents were found in: {input_directory}"
        )

    documents: list[LoadedDocument] = []

    for source_path in text_files:
        text = source_path.read_text(encoding="utf-8").strip()

        if not text:
            continue

        document = LoadedDocument(
            document_id=extract_document_id(text, source_path),
            language=detect_language(text),
            source=source_path.name,
            text=text,
        )

        documents.append(document)

    return documents