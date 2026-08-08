"""Dependency-free tests for the end-to-end RAG pipeline."""

import unittest
from collections.abc import Sequence
from dataclasses import dataclass

from src.generation.prompt_builder import REFUSAL_RESPONSES
from src.pipeline.rag_pipeline import EnterpriseRAGPipeline


@dataclass(frozen=True)
class FakeChunk:
    """Small chunk object used without loading ML models."""

    chunk_id: str


class FakeRetriever:
    """Return predefined retrieval candidates."""

    def __init__(
        self,
        chunks: Sequence[FakeChunk],
    ) -> None:
        self.chunks = tuple(chunks)
        self.calls: list[tuple[str, int]] = []

    def __call__(
        self,
        question: str,
        top_k: int,
    ) -> Sequence[FakeChunk]:
        self.calls.append(
            (question, top_k)
        )
        return self.chunks[:top_k]


class FakeReranker:
    """Return candidates in a predefined reranked order."""

    def __init__(
        self,
        chunks: Sequence[FakeChunk],
    ) -> None:
        self.chunks = tuple(chunks)
        self.calls: list[
            tuple[str, tuple[str, ...], int]
        ] = []

    def __call__(
        self,
        question: str,
        candidates: Sequence[FakeChunk],
        top_k: int,
    ) -> Sequence[FakeChunk]:
        self.calls.append(
            (
                question,
                tuple(
                    chunk.chunk_id
                    for chunk in candidates
                ),
                top_k,
            )
        )

        return self.chunks[:top_k]


class FakeGenerator:
    """Return a predefined generated answer."""

    def __init__(
        self,
        answer: str,
    ) -> None:
        self.answer = answer
        self.calls: list[
            tuple[str, tuple[str, ...]]
        ] = []

    def __call__(
        self,
        question: str,
        evidence_chunks: Sequence[FakeChunk],
    ) -> str:
        self.calls.append(
            (
                question,
                tuple(
                    chunk.chunk_id
                    for chunk in evidence_chunks
                ),
            )
        )

        return self.answer


class TestEnterpriseRAGPipeline(unittest.TestCase):
    """Test orchestration and safety behavior."""

    def setUp(self) -> None:
        self.english_leave = FakeChunk(
            "HR-EN-001-CH-003"
        )
        self.english_hours = FakeChunk(
            "HR-EN-001-CH-002"
        )
        self.arabic_travel = FakeChunk(
            "HR-AR-001-CH-003"
        )

    def test_returns_supported_grounded_answer(
        self,
    ) -> None:
        retriever = FakeRetriever(
            [
                self.english_leave,
                self.english_hours,
            ]
        )
        generator = FakeGenerator(
            "Employees receive 24 days "
            "[HR-EN-001-CH-003]."
        )

        pipeline = EnterpriseRAGPipeline(
            retriever=retriever,
            generator=generator,
            candidate_k=2,
            evidence_k=1,
        )

        result = pipeline.answer(
            "How many annual leave days are provided?"
        )

        self.assertEqual(
            result.answer,
            "Employees receive 24 days "
            "[HR-EN-001-CH-003].",
        )
        self.assertEqual(result.language, "en")
        self.assertEqual(
            result.candidate_chunk_ids,
            (
                "HR-EN-001-CH-003",
                "HR-EN-001-CH-002",
            ),
        )
        self.assertEqual(
            result.evidence_chunk_ids,
            ("HR-EN-001-CH-003",),
        )
        self.assertEqual(
            result.citations,
            ("HR-EN-001-CH-003",),
        )
        self.assertTrue(result.citations_valid)
        self.assertFalse(result.refused)
        self.assertFalse(result.safety_blocked)

    def test_refuses_when_retrieval_is_empty(
        self,
    ) -> None:
        retriever = FakeRetriever([])
        generator = FakeGenerator(
            "This generator must not be called."
        )

        pipeline = EnterpriseRAGPipeline(
            retriever=retriever,
            generator=generator,
        )

        result = pipeline.answer(
            "What is the maternity leave policy?"
        )

        self.assertEqual(
            result.answer,
            REFUSAL_RESPONSES["en"],
        )
        self.assertTrue(result.refused)
        self.assertFalse(result.safety_blocked)
        self.assertTrue(result.citations_valid)
        self.assertEqual(generator.calls, [])

    def test_returns_localized_arabic_refusal(
        self,
    ) -> None:
        pipeline = EnterpriseRAGPipeline(
            retriever=FakeRetriever([]),
            generator=FakeGenerator(
                "This generator must not be called."
            ),
        )

        result = pipeline.answer(
            "ما هي سياسة إجازة الأمومة؟"
        )

        self.assertEqual(result.language, "ar")
        self.assertEqual(
            result.answer,
            REFUSAL_RESPONSES["ar"],
        )
        self.assertTrue(result.refused)
        self.assertFalse(result.safety_blocked)

    def test_blocks_unsupported_citation(
        self,
    ) -> None:
        pipeline = EnterpriseRAGPipeline(
            retriever=FakeRetriever(
                [self.english_leave]
            ),
            generator=FakeGenerator(
                "Incorrect answer "
                "[HR-AR-001-CH-003]."
            ),
            candidate_k=1,
            evidence_k=1,
        )

        result = pipeline.answer(
            "How many annual leave days are provided?"
        )

        self.assertEqual(
            result.answer,
            REFUSAL_RESPONSES["en"],
        )
        self.assertEqual(
            result.unsupported_citations,
            ("HR-AR-001-CH-003",),
        )
        self.assertFalse(result.citations_valid)
        self.assertTrue(result.refused)
        self.assertTrue(result.safety_blocked)

    def test_blocks_answer_without_citation(
        self,
    ) -> None:
        pipeline = EnterpriseRAGPipeline(
            retriever=FakeRetriever(
                [self.english_leave]
            ),
            generator=FakeGenerator(
                "Employees receive 24 days."
            ),
            candidate_k=1,
            evidence_k=1,
        )

        result = pipeline.answer(
            "How many annual leave days are provided?"
        )

        self.assertEqual(
            result.answer,
            REFUSAL_RESPONSES["en"],
        )
        self.assertEqual(result.citations, ())
        self.assertEqual(
            result.unsupported_citations,
            (),
        )
        self.assertFalse(result.citations_valid)
        self.assertTrue(result.safety_blocked)

    def test_accepts_exact_model_refusal(
        self,
    ) -> None:
        pipeline = EnterpriseRAGPipeline(
            retriever=FakeRetriever(
                [self.english_leave]
            ),
            generator=FakeGenerator(
                REFUSAL_RESPONSES["en"]
            ),
            candidate_k=1,
            evidence_k=1,
        )

        result = pipeline.answer(
            "What is the maternity leave policy?"
        )

        self.assertEqual(
            result.answer,
            REFUSAL_RESPONSES["en"],
        )
        self.assertTrue(result.refused)
        self.assertTrue(result.citations_valid)
        self.assertFalse(result.safety_blocked)

    def test_uses_reranked_evidence_order(
        self,
    ) -> None:
        retriever = FakeRetriever(
            [
                self.english_hours,
                self.english_leave,
            ]
        )
        reranker = FakeReranker(
            [
                self.english_leave,
                self.english_hours,
            ]
        )
        generator = FakeGenerator(
            "Employees receive 24 days "
            "[HR-EN-001-CH-003]."
        )

        pipeline = EnterpriseRAGPipeline(
            retriever=retriever,
            reranker=reranker,
            generator=generator,
            candidate_k=2,
            evidence_k=1,
        )

        result = pipeline.answer(
            "How many annual leave days are provided?"
        )

        self.assertEqual(
            result.evidence_chunk_ids,
            ("HR-EN-001-CH-003",),
        )
        self.assertEqual(
            generator.calls[0][1],
            ("HR-EN-001-CH-003",),
        )
        self.assertFalse(result.refused)

    def test_reranker_cannot_inject_unknown_chunk(
        self,
    ) -> None:
        unknown_chunk = FakeChunk(
            "HR-EN-999-CH-999"
        )
        generator = FakeGenerator(
            "This generator must not be called."
        )

        pipeline = EnterpriseRAGPipeline(
            retriever=FakeRetriever(
                [self.english_leave]
            ),
            reranker=FakeReranker(
                [unknown_chunk]
            ),
            generator=generator,
            candidate_k=1,
            evidence_k=1,
        )

        result = pipeline.answer(
            "How many annual leave days are provided?"
        )

        self.assertEqual(
            result.candidate_chunk_ids,
            ("HR-EN-001-CH-003",),
        )
        self.assertEqual(
            result.evidence_chunk_ids,
            (),
        )
        self.assertTrue(result.refused)
        self.assertEqual(generator.calls, [])

    def test_rejects_empty_question(
        self,
    ) -> None:
        pipeline = EnterpriseRAGPipeline(
            retriever=FakeRetriever([]),
            generator=FakeGenerator("Answer"),
        )

        with self.assertRaises(ValueError):
            pipeline.answer("   ")

    def test_rejects_invalid_pipeline_limits(
        self,
    ) -> None:
        retriever = FakeRetriever([])
        generator = FakeGenerator("Answer")

        invalid_limits = (
            {
                "candidate_k": 0,
                "evidence_k": 1,
            },
            {
                "candidate_k": 2,
                "evidence_k": 0,
            },
            {
                "candidate_k": 1,
                "evidence_k": 2,
            },
        )

        for limits in invalid_limits:
            with self.subTest(limits=limits):
                with self.assertRaises(ValueError):
                    EnterpriseRAGPipeline(
                        retriever=retriever,
                        generator=generator,
                        **limits,
                    )


if __name__ == "__main__":
    unittest.main()