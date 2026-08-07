"""Dependency-free BM25 keyword retrieval for English and Arabic."""

from collections import Counter
from dataclasses import dataclass
import math
import re
import unicodedata

from src.ingestion.chunker import TextChunk


TOKEN_PATTERN = re.compile(r"[^\W_]+", flags=re.UNICODE)

ARABIC_DIACRITICS_PATTERN = re.compile(
    r"[\u0617-\u061a\u064b-\u0652\u0670\u06d6-\u06ed]"
)

ARABIC_NORMALIZATION_TABLE = str.maketrans(
    {
        "أ": "ا",
        "إ": "ا",
        "آ": "ا",
        "ى": "ي",
    }
)


@dataclass(frozen=True)
class SearchResult:
    """One ranked retrieval result."""

    rank: int
    score: float
    chunk: TextChunk


def normalize_text(text: str) -> str:
    """Normalize English and Arabic text before tokenization."""

    normalized_text = unicodedata.normalize("NFKC", text).casefold()
    normalized_text = ARABIC_DIACRITICS_PATTERN.sub("", normalized_text)
    normalized_text = normalized_text.replace("ـ", "")
    normalized_text = normalized_text.translate(
        ARABIC_NORMALIZATION_TABLE
    )

    return normalized_text


def tokenize(text: str) -> list[str]:
    """Convert normalized text into searchable word tokens."""

    return TOKEN_PATTERN.findall(normalize_text(text))


class BM25Retriever:
    """Rank document chunks using the BM25 keyword algorithm."""

    def __init__(
        self,
        chunks: list[TextChunk],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if not chunks:
            raise ValueError("At least one chunk is required.")

        self.chunks = chunks
        self.k1 = k1
        self.b = b

        self.tokenized_chunks = [
            tokenize(chunk.text) for chunk in chunks
        ]

        self.chunk_lengths = [
            len(tokens) for tokens in self.tokenized_chunks
        ]

        self.average_chunk_length = (
            sum(self.chunk_lengths) / len(self.chunk_lengths)
        )

        document_frequency: Counter[str] = Counter()

        for tokens in self.tokenized_chunks:
            document_frequency.update(set(tokens))

        number_of_chunks = len(chunks)

        self.inverse_document_frequency = {
            term: math.log(
                1
                + (
                    number_of_chunks
                    - frequency
                    + 0.5
                )
                / (frequency + 0.5)
            )
            for term, frequency in document_frequency.items()
        }

    def _score_chunk(
        self,
        query_terms: set[str],
        chunk_index: int,
    ) -> float:
        """Calculate the BM25 score for one chunk."""

        tokens = self.tokenized_chunks[chunk_index]
        term_frequencies = Counter(tokens)
        chunk_length = self.chunk_lengths[chunk_index]

        score = 0.0

        for term in query_terms:
            term_frequency = term_frequencies.get(term, 0)

            if term_frequency == 0:
                continue

            inverse_document_frequency = (
                self.inverse_document_frequency.get(term, 0.0)
            )

            numerator = term_frequency * (self.k1 + 1)

            denominator = term_frequency + self.k1 * (
                1
                - self.b
                + self.b
                * chunk_length
                / self.average_chunk_length
            )

            score += (
                inverse_document_frequency
                * numerator
                / denominator
            )

        return score

    def search(
        self,
        query: str,
        top_k: int = 3,
    ) -> list[SearchResult]:
        """Return the highest-scoring chunks for a query."""

        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        query_terms = set(tokenize(query))

        if not query_terms:
            return []

        scored_chunks = []

        for chunk_index in range(len(self.chunks)):
            score = self._score_chunk(
                query_terms,
                chunk_index,
            )

            if score > 0:
                scored_chunks.append((chunk_index, score))

        scored_chunks.sort(
            key=lambda item: (
                -item[1],
                self.chunks[item[0]].chunk_id,
            )
        )

        results = []

        for rank, (chunk_index, score) in enumerate(
            scored_chunks[:top_k],
            start=1,
        ):
            results.append(
                SearchResult(
                    rank=rank,
                    score=score,
                    chunk=self.chunks[chunk_index],
                )
            )

        return results