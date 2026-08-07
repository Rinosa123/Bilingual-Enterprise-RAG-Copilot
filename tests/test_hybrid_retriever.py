"""Tests for the hybrid retrieval rank-fusion module."""

import unittest

from src.retrieval.hybrid_retriever import (
    reciprocal_rank_fusion,
)


class TestReciprocalRankFusion(unittest.TestCase):
    """Verify Reciprocal Rank Fusion behaviour."""

    def test_combines_overlapping_rankings(self) -> None:
        rankings = {
            "bm25": ["chunk-a", "chunk-b", "chunk-c"],
            "dense": ["chunk-b", "chunk-d", "chunk-a"],
        }

        results = reciprocal_rank_fusion(rankings)

        result_ids = [
            result.chunk_id
            for result in results
        ]

        self.assertEqual(
            result_ids,
            [
                "chunk-b",
                "chunk-a",
                "chunk-d",
                "chunk-c",
            ],
        )

        self.assertEqual(
            results[0].source_ranks,
            {
                "bm25": 2,
                "dense": 1,
            },
        )

    def test_supports_source_weights(self) -> None:
        rankings = {
            "bm25": ["chunk-a", "chunk-b"],
            "dense": ["chunk-b", "chunk-a"],
        }

        results = reciprocal_rank_fusion(
            rankings,
            weights={
                "bm25": 2.0,
                "dense": 1.0,
            },
        )

        self.assertEqual(
            results[0].chunk_id,
            "chunk-a",
        )

    def test_ignores_duplicate_chunk_ids(self) -> None:
        rankings = {
            "bm25": [
                "chunk-a",
                "chunk-a",
                "chunk-b",
            ],
        }

        results = reciprocal_rank_fusion(rankings)

        self.assertEqual(
            [result.chunk_id for result in results],
            ["chunk-a", "chunk-b"],
        )

        self.assertEqual(
            results[1].source_ranks["bm25"],
            2,
        )

    def test_rejects_negative_rank_constant(self) -> None:
        with self.assertRaises(ValueError):
            reciprocal_rank_fusion(
                {"bm25": ["chunk-a"]},
                rank_constant=-1,
            )

    def test_rejects_negative_weight(self) -> None:
        with self.assertRaises(ValueError):
            reciprocal_rank_fusion(
                {"bm25": ["chunk-a"]},
                weights={"bm25": -1.0},
            )


if __name__ == "__main__":
    unittest.main()