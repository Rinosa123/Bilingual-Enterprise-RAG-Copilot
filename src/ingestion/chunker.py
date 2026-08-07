"""Split enterprise documents into section-aware searchable chunks."""

from collections.abc import Iterable
from dataclasses import dataclass
import re

from src.ingestion.text_loader import LoadedDocument


# Detect numbered headings such as:
# 1. Annual Leave
# 1. مطالبات المصروفات
SECTION_HEADING_PATTERN = re.compile(r"^\d+\.\s+\S.*$")


@dataclass(frozen=True)
class TextChunk:
    """A searchable piece of an enterprise document."""

    chunk_id: str
    document_id: str
    language: str
    source: str
    section: str
    section_index: int
    text: str
    character_count: int


def split_document_into_sections(
    document: LoadedDocument,
) -> list[tuple[str, str]]:
    """Split a document using its numbered section headings."""

    default_section = (
        "Document information"
        if document.language == "en"
        else "معلومات الوثيقة"
    )

    sections: list[tuple[str, str]] = []
    current_section = default_section
    current_lines: list[str] = []

    for raw_line in document.text.splitlines():
        line = raw_line.strip()

        if not line:
            continue

        if SECTION_HEADING_PATTERN.match(line):
            if current_lines:
                sections.append(
                    (current_section, "\n".join(current_lines))
                )

            current_section = line
            current_lines = [line]
        else:
            current_lines.append(line)

    if current_lines:
        sections.append(
            (current_section, "\n".join(current_lines))
        )

    return sections


def chunk_document(document: LoadedDocument) -> list[TextChunk]:
    """Convert one document into searchable section chunks."""

    sections = split_document_into_sections(document)
    chunks: list[TextChunk] = []

    for section_index, (section, section_text) in enumerate(
        sections,
        start=1,
    ):
        chunk = TextChunk(
            chunk_id=(
                f"{document.document_id}-CH-{section_index:03d}"
            ),
            document_id=document.document_id,
            language=document.language,
            source=document.source,
            section=section,
            section_index=section_index,
            text=section_text,
            character_count=len(section_text),
        )

        chunks.append(chunk)

    return chunks


def chunk_documents(
    documents: Iterable[LoadedDocument],
) -> list[TextChunk]:
    """Convert several documents into one collection of chunks."""

    chunks: list[TextChunk] = []

    for document in documents:
        chunks.extend(chunk_document(document))

    return chunks