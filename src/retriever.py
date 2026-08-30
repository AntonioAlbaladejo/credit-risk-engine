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

from src.config import (
    CORPUS_INDEX_PATH,
    CORPUS_PATH,
    DEONTIC_ANCHORS,
    EMBEDDING_MODEL,
    EVIDENTIAL_ANCHORS,
    MIN_ANCHOR_SCORE,
    MIN_SCORE,
    MIN_SCORE_WITH_PASSAGE,
    QUERY_INSTRUCTION,
    UNIT_WEIGHTS,
)

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

    def __init__(
        self,
        chunks: list[dict],
        vectors: np.ndarray,
        embed_query,
        unit_weights: dict[str, float] | None = None,
        embed_anchor=None,
    ):
        """Bind a corpus to the vectors built from it.

        Args:
            chunks: Corpus records, in the order their vectors were built.
            vectors: One L2-normalised row per chunk.
            embed_query: Callable turning a question into one such row.
            unit_weights: Multiplier per `unit`, defaulting to `UNIT_WEIGHTS`.
                A unit not listed is left alone.
            embed_anchor: Callable embedding the modality anchors. They are
                prototypes rather than searches, so they take no query
                instruction; defaults to `embed_query`.

        Raises:
            ValueError: The vectors do not describe these chunks.
        """
        if len(chunks) != len(vectors):
            raise ValueError(
                f"{len(vectors)} vectors for {len(chunks)} chunks: the index is "
                "stale. Re-run `uv run python -m scripts.ingest_corpus`."
            )
        weights = UNIT_WEIGHTS if unit_weights is None else unit_weights
        self.chunks = chunks
        self.vectors = vectors
        self.embed_query = embed_query
        self.embed_anchor = embed_anchor or embed_query
        self._anchor_delta: np.ndarray | None = None
        self.weights = np.array(
            [weights.get(chunk.get("unit"), 1.0) for chunk in chunks], dtype=np.float32
        )

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
            # The instruction is prepended here rather than relying on
            # query_embed, which returns a vector identical to embed for this
            # model: fastembed does not apply one. Without it the search is
            # symmetric while the task is not, and only the ranking suffers --
            # nothing errors.
            lambda query: next(iter(model.embed([QUERY_INSTRUCTION + query]))),
            embed_anchor=lambda text: next(iter(model.embed([text]))),
        )

    def deontic_score(self, query_vector: np.ndarray) -> float:
        """How far the question asks what the law requires, not what we did.

        Args:
            query_vector: The embedded question, as `embed_query` returns it.

        Returns:
            Similarity to the deontic anchors minus similarity to the
            evidential ones. Positive asks for a requirement, which the corpus
            can supply; negative asks for a fact about this organisation,
            which no provision holds however well it matches.
        """
        if self._anchor_delta is None:
            anchors = np.array(
                [
                    self.embed_anchor(text)
                    for text in DEONTIC_ANCHORS + EVIDENTIAL_ANCHORS
                ]
            )
            half = len(DEONTIC_ANCHORS)
            self._anchor_delta = anchors[:half].mean(axis=0) - anchors[half:].mean(
                axis=0
            )
        return float(query_vector @ self._anchor_delta)

    def search(
        self,
        query: str,
        k: int = 5,
        min_score: float | None = None,
        hypothetical_passage: str = "",
    ) -> list[dict]:
        """Return the k passages closest to a question, best first.

        Args:
            query: A natural-language question.
            k: How many passages to return.
            min_score: Passages scoring below this are dropped, and 0.0 lifts
                the abstention veto entirely so that the ranking itself can be
                measured. Defaults to the threshold fitted for whichever path
                is taken.
            hypothetical_passage: An invented passage answering `query`, in the
                register of the corpus. When given, it drives the ranking and
                `query` is used only to decide whether to answer at all.
                Optional: without it the search behaves exactly as before.

        Returns:
            One dict per passage with its citation, text and score, or an empty
            list when nothing clears `min_score` -- which is the retriever
            saying it has no answer, and must be passed on as that rather than
            filled in downstream. `score` is the ranking score, not a raw
            cosine: it carries the unit weight, so two passages with the same
            score are not equally similar unless they share a unit.
        """
        if min_score is None:
            min_score = MIN_SCORE_WITH_PASSAGE if hypothetical_passage else MIN_SCORE
        # Both sides are L2-normalised, so the dot product is the cosine. The
        # weight then demotes whole classes of passage that embed well and
        # answer badly.
        query_vector = self.embed_query(query)
        scores = (self.vectors @ query_vector) * self.weights
        cutoff = min_score
        if hypothetical_passage:
            # Passage orders, question decides. Vetoing on the passage's own
            # score was measured and lost: it separates groundable questions
            # slightly better (AUC 0.77 against 0.71) yet every fitted cut for
            # it serves more wrong citations, 23.6 against 19.1 per fold.
            #
            # Similarity is not the only arm: a question about what this
            # organisation did scores high against the provision governing the
            # matter, and no amount of ranking can fix an answer the corpus
            # does not hold. min_score of 0 disables both arms, which is how
            # the ranking itself is measured.
            if min_score > 0 and (
                float(scores.max()) < min_score
                or self.deontic_score(query_vector) < MIN_ANCHOR_SCORE
            ):
                return []
            scores = (
                self.vectors @ self.embed_query(hypothetical_passage)
            ) * self.weights
            # Gate already decided, and this scale is not the one min_score
            # was fitted against.
            cutoff = 0.0
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
            if scores[index] >= cutoff
        ]
