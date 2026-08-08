"""Validate generated citations against supplied RAG evidence."""

from collections.abc import Sequence
from dataclasses import dataclass
import re
from typing import Protocol


CITATION_PATTERN = re.compile(
    r"\[([A-Z]{2}-[A-Z]{2}-\d{3}-CH-\d{3})\]"
)


class EvidenceChunk(Protocol):
    """Minimum evidence interface required for validation."""

    chunk_id: str


@dataclass(frozen=True, slots=True)
class CitationValidation:
    """Structured result returned by citation validation."""

    citations: tuple[str, ...]
    allowed_chunk_ids: tuple[str, ...]
    unsupported_citations: tuple[str, ...]
    citations_valid: bool

    @property
    def has_citations(self) -> bool:
        """Return whether the answer contains citations."""
        return bool(self.citations)


def extract_citations(answer: str) -> tuple[str, ...]:
    """Extract unique citation IDs while preserving their order."""
    matches = CITATION_PATTERN.findall(answer)

    return tuple(
        dict.fromkeys(matches)
    )


def validate_citations(
    answer: str,
    evidence_chunks: Sequence[EvidenceChunk],
    *,
    require_citation: bool = True,
) -> CitationValidation:
    """Validate citations against the supplied evidence chunks."""
    citations = extract_citations(answer)

    allowed_chunk_ids = tuple(
        dict.fromkeys(
            chunk.chunk_id
            for chunk in evidence_chunks
        )
    )

    allowed_chunk_id_set = set(allowed_chunk_ids)

    unsupported_citations = tuple(
        citation
        for citation in citations
        if citation not in allowed_chunk_id_set
    )

    citations_valid = (
        not unsupported_citations
        and (
            bool(citations)
            or not require_citation
        )
    )

    return CitationValidation(
        citations=citations,
        allowed_chunk_ids=allowed_chunk_ids,
        unsupported_citations=unsupported_citations,
        citations_valid=citations_valid,
    )