"""Semantic search over the regulatory corpus.

Turns a question into the passages of the GDPR and the AI Act most likely to
answer it, each carrying the citation that lets a reader check it. Nothing here
generates text: an LLM downstream answers *from* these passages, so a passage
that cannot be cited precisely is of no use.

Search is an exact scan, not an approximate index. With a corpus this size the
scan is one matrix product over a few hundred rows -- microseconds -- so an
ANN structure such as HNSW would trade exactness for a speed-up that does not
exist yet. That changes around six figures of chunks, not three.
"""

import json
import logging
from pathlib import Path

import numpy as np

from src.config import CORPUS_INDEX_PATH, CORPUS_PATH, EMBEDDING_MODEL

logger = logging.getLogger(__name__)


def load_model():
    """Load the embedding model shared by ingestion and search.

    Imported here rather than at module scope: fastembed lives in the optional
    `genai` group, which CI does not install, and importing this module must
    not depend on it.
    """
    from fastembed import TextEmbedding

    return TextEmbedding(model_name=EMBEDDING_MODEL)


class CorpusRetriever:
    """Ranks corpus chunks against a question by cosine similarity."""

    def __init__(self, chunks: list[dict], vectors: np.ndarray, embed_query):
        """Bind a corpus to the vectors built from it.

        Args:
            chunks: Corpus records, in the order their vectors were built.
            vectors: One L2-normalised row per chunk.
            embed_query: Callable turning a question into one such row.

        Raises:
            ValueError: The vectors do not describe these chunks.
        """
        if len(chunks) != len(vectors):
            raise ValueError(
                f"{len(vectors)} vectors for {len(chunks)} chunks: the index is "
                "stale. Re-run `uv run python -m scripts.ingest_corpus`."
            )
        self.chunks = chunks
        self.vectors = vectors
        self.embed_query = embed_query

    @classmethod
    def from_files(
        cls, corpus_path: Path = CORPUS_PATH, index_path: Path = CORPUS_INDEX_PATH
    ) -> "CorpusRetriever":
        """Load the corpus, its vectors and the model that embeds questions.

        Args:
            corpus_path: JSONL written by scripts/ingest_corpus.py.
            index_path: Matching .npz written by the same run.

        Returns:
            A retriever ready to search.

        Raises:
            FileNotFoundError: Either artifact is missing.
            ValueError: The two do not describe the same corpus.
        """
        with corpus_path.open(encoding="utf-8") as handle:
            chunks = [json.loads(line) for line in handle]

        stored = np.load(index_path)
        # Counts alone would pass for a corpus re-chunked into the same number
        # of pieces, which is exactly the drift worth catching: the text would
        # then be served under someone else's citation.
        if list(stored["chunk_ids"]) != [chunk["chunk_id"] for chunk in chunks]:
            raise ValueError(
                f"{index_path.name} was built from a different corpus than "
                f"{corpus_path.name}. Re-run `uv run python -m scripts.ingest_corpus`."
            )

        model = load_model()
        logger.info("Corpus retriever ready over %d chunks", len(chunks))
        return cls(
            chunks,
            stored["vectors"],
            # BGE v1.5 needs no instruction prefix on the query -- fastembed's
            # query_embed returns the same vector as embed for the same text.
            # Earlier BGE and E5 releases do, and getting that wrong degrades
            # ranking silently, so it moves with the model.
            lambda query: next(iter(model.query_embed([query]))),
        )

    def search(self, query: str, k: int = 5) -> list[dict]:
        """Return the k passages closest to a question, best first.

        Args:
            query: A natural-language question.
            k: How many passages to return.

        Returns:
            One dict per passage with its citation, text and similarity.
        """
        # Both sides are L2-normalised, so the dot product is the cosine.
        scores = self.vectors @ self.embed_query(query)
        # argpartition finds the top k without ordering the rest; only those k
        # are then sorted.
        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        return [
            {
                "chunk_id": self.chunks[index]["chunk_id"],
                "citation": self.chunks[index]["citation"],
                "text": self.chunks[index]["text"],
                "source_url": self.chunks[index]["source_url"],
                "retrieved_on": self.chunks[index]["retrieved_on"],
                "score": round(float(scores[index]), 4),
            }
            for index in top[np.argsort(-scores[top])]
        ]
