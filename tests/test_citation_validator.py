"""Tests for generated citation validation."""

import unittest
from dataclasses import dataclass

from src.generation.citation_validator import (
    extract_citations,
    validate_citations,
)


@dataclass(frozen=True)
class FakeEvidenceChunk:
    """Minimal evidence object used by the tests."""

    chunk_id: str


class TestCitationValidator(unittest.TestCase):
    """Test supported, missing and fabricated citations."""

    def setUp(self) -> None:
        self.english_chunk = FakeEvidenceChunk(
            "HR-EN-001-CH-003"
        )
        self.arabic_chunk = FakeEvidenceChunk(
            "HR-AR-001-CH-003"
        )

    def test_extracts_unique_citations_in_order(self) -> None:
        answer = (
            "First [HR-EN-001-CH-003]. "
            "Again [HR-EN-001-CH-003]. "
            "Arabic [HR-AR-001-CH-003]."
        )

        self.assertEqual(
            extract_citations(answer),
            (
                "HR-EN-001-CH-003",
                "HR-AR-001-CH-003",
            ),
        )

    def test_accepts_supported_citation(self) -> None:
        result = validate_citations(
            "Answer [HR-EN-001-CH-003].",
            [self.english_chunk],
        )

        self.assertTrue(result.has_citations)
        self.assertTrue(result.citations_valid)
        self.assertEqual(
            result.unsupported_citations,
            (),
        )

    def test_rejects_unsupported_citation(self) -> None:
        result = validate_citations(
            "Answer [HR-AR-001-CH-003].",
            [self.english_chunk],
        )

        self.assertFalse(result.citations_valid)
        self.assertEqual(
            result.unsupported_citations,
            ("HR-AR-001-CH-003",),
        )

    def test_requires_citation_by_default(self) -> None:
        result = validate_citations(
            "Answer without a citation.",
            [self.english_chunk],
        )

        self.assertFalse(result.has_citations)
        self.assertFalse(result.citations_valid)

    def test_allows_citation_free_refusal(self) -> None:
        result = validate_citations(
            "I could not find sufficient evidence.",
            [],
            require_citation=False,
        )

        self.assertFalse(result.has_citations)
        self.assertTrue(result.citations_valid)

    def test_rejects_unsupported_citation_when_optional(
        self,
    ) -> None:
        result = validate_citations(
            "No evidence [HR-AR-001-CH-003].",
            [],
            require_citation=False,
        )

        self.assertFalse(result.citations_valid)


if __name__ == "__main__":
    unittest.main()