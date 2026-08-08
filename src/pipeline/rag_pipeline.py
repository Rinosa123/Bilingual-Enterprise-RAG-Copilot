"""End-to-end orchestration for the bilingual enterprise RAG system."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from src.generation.citation_validator import validate_citations
from src.generation.prompt_builder import (
    REFUSAL_RESPONSES,
    detect_question_language,
)


class PipelineChunk(Protocol):
    """Minimum chunk interface required by the pipeline."""

    chunk_id: str


class Retriever(Protocol):
    """Interface implemented by a retrieval component."""

    def __call__(
        self,
        question: str,
        top_k: int,
    ) -> Sequence[PipelineChunk]:
        """Return candidate chunks for a question."""
        ...


class Reranker(Protocol):
    """Interface implemented by a reranking component."""

    def __call__(
        self,
        question: str,
        candidates: Sequence[PipelineChunk],
        top_k: int,
    ) -> Sequence[PipelineChunk]:
        """Reorder candidate chunks according to relevance."""
        ...


class Generator(Protocol):
    """Interface implemented by a grounded answer generator."""

    def __call__(
        self,
        question: str,
        evidence_chunks: Sequence[PipelineChunk],
    ) -> str:
        """Generate an evidence-grounded answer."""
        ...


@dataclass(frozen=True, slots=True)
class RAGResponse:
    """Structured response returned by the RAG pipeline."""

    question: str
    answer: str
    language: str
    candidate_chunk_ids: tuple[str, ...]
    evidence_chunk_ids: tuple[str, ...]
    citations: tuple[str, ...]
    unsupported_citations: tuple[str, ...]
    citations_valid: bool
    refused: bool
    safety_blocked: bool


def _unique_chunks(
    chunks: Sequence[PipelineChunk],
) -> tuple[PipelineChunk, ...]:
    """Remove duplicate chunks while preserving their order."""
    unique_chunks: list[PipelineChunk] = []
    seen_chunk_ids: set[str] = set()

    for chunk in chunks:
        chunk_id = chunk.chunk_id.strip()

        if not chunk_id:
            raise ValueError(
                "Every pipeline chunk must have a chunk ID."
            )

        if chunk_id in seen_chunk_ids:
            continue

        seen_chunk_ids.add(chunk_id)
        unique_chunks.append(chunk)

    return tuple(unique_chunks)


class EnterpriseRAGPipeline:
    """Coordinate retrieval, reranking, generation and validation."""

    def __init__(
        self,
        *,
        retriever: Retriever,
        generator: Generator,
        reranker: Reranker | None = None,
        candidate_k: int = 8,
        evidence_k: int = 3,
    ) -> None:
        if candidate_k < 1:
            raise ValueError(
                "candidate_k must be at least 1."
            )

        if evidence_k < 1:
            raise ValueError(
                "evidence_k must be at least 1."
            )

        if evidence_k > candidate_k:
            raise ValueError(
                "evidence_k cannot exceed candidate_k."
            )

        self._retriever = retriever
        self._reranker = reranker
        self._generator = generator
        self._candidate_k = candidate_k
        self._evidence_k = evidence_k

    def answer(self, question: str) -> RAGResponse:
        """Run the complete RAG pipeline for one question."""
        normalized_question = question.strip()

        if not normalized_question:
            raise ValueError(
                "Question cannot be empty."
            )

        language = detect_question_language(
            normalized_question
        )
        refusal_response = REFUSAL_RESPONSES[language]

        retrieved_chunks = self._retriever(
            normalized_question,
            self._candidate_k,
        )

        candidates = _unique_chunks(
            retrieved_chunks
        )[: self._candidate_k]

        candidate_chunk_ids = tuple(
            chunk.chunk_id
            for chunk in candidates
        )

        if not candidates:
            return self._build_refusal(
                question=normalized_question,
                language=language,
                candidate_chunk_ids=(),
                evidence_chunks=(),
                safety_blocked=False,
            )

        ranked_chunks = candidates

        if self._reranker is not None:
            reranked_chunks = _unique_chunks(
                self._reranker(
                    normalized_question,
                    candidates,
                    self._evidence_k,
                )
            )

            allowed_candidate_ids = {
                chunk.chunk_id
                for chunk in candidates
            }

            ranked_chunks = tuple(
                chunk
                for chunk in reranked_chunks
                if chunk.chunk_id in allowed_candidate_ids
            )

        evidence_chunks = ranked_chunks[
            : self._evidence_k
        ]

        if not evidence_chunks:
            return self._build_refusal(
                question=normalized_question,
                language=language,
                candidate_chunk_ids=candidate_chunk_ids,
                evidence_chunks=(),
                safety_blocked=False,
            )

        generated_answer = self._generator(
            normalized_question,
            evidence_chunks,
        )

        if not isinstance(generated_answer, str):
            generated_answer = ""

        generated_answer = generated_answer.strip()

        if not generated_answer:
            return self._build_refusal(
                question=normalized_question,
                language=language,
                candidate_chunk_ids=candidate_chunk_ids,
                evidence_chunks=evidence_chunks,
                safety_blocked=True,
                citations_valid=False,
            )

        refused = generated_answer == refusal_response

        validation = validate_citations(
            generated_answer,
            evidence_chunks,
            require_citation=not refused,
        )

        if refused:
            return RAGResponse(
                question=normalized_question,
                answer=generated_answer,
                language=language,
                candidate_chunk_ids=candidate_chunk_ids,
                evidence_chunk_ids=tuple(
                    chunk.chunk_id
                    for chunk in evidence_chunks
                ),
                citations=validation.citations,
                unsupported_citations=(
                    validation.unsupported_citations
                ),
                citations_valid=validation.citations_valid,
                refused=True,
                safety_blocked=False,
            )

        if not validation.citations_valid:
            return RAGResponse(
                question=normalized_question,
                answer=refusal_response,
                language=language,
                candidate_chunk_ids=candidate_chunk_ids,
                evidence_chunk_ids=tuple(
                    chunk.chunk_id
                    for chunk in evidence_chunks
                ),
                citations=validation.citations,
                unsupported_citations=(
                    validation.unsupported_citations
                ),
                citations_valid=False,
                refused=True,
                safety_blocked=True,
            )

        return RAGResponse(
            question=normalized_question,
            answer=generated_answer,
            language=language,
            candidate_chunk_ids=candidate_chunk_ids,
            evidence_chunk_ids=tuple(
                chunk.chunk_id
                for chunk in evidence_chunks
            ),
            citations=validation.citations,
            unsupported_citations=(
                validation.unsupported_citations
            ),
            citations_valid=True,
            refused=False,
            safety_blocked=False,
        )

    def _build_refusal(
        self,
        *,
        question: str,
        language: str,
        candidate_chunk_ids: tuple[str, ...],
        evidence_chunks: Sequence[PipelineChunk],
        safety_blocked: bool,
        citations_valid: bool = True,
    ) -> RAGResponse:
        """Create a localized safe-refusal response."""
        refusal_response = REFUSAL_RESPONSES[language]

        return RAGResponse(
            question=question,
            answer=refusal_response,
            language=language,
            candidate_chunk_ids=candidate_chunk_ids,
            evidence_chunk_ids=tuple(
                chunk.chunk_id
                for chunk in evidence_chunks
            ),
            citations=(),
            unsupported_citations=(),
            citations_valid=citations_valid,
            refused=True,
            safety_blocked=safety_blocked,
        )