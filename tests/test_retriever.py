"""Ranking and staleness guards for the corpus retriever.

The ranking itself is exercised against a hand-built index with a stub
embedder, so the maths is covered without the `genai` group or a 130 MB model
download -- which is how CI installs. The cases that need the real model are
skipped there, the same way the MCP ones are.

What matters here is not that search returns something plausible but that what
it returns still belongs to the passage it names. A retriever that serves the
right text under the wrong citation is worse than one that finds nothing.
"""

import json

import numpy as np
import pytest

from src.config import CORPUS_PATH, GOLDEN_SET_PATH, QUERY_INSTRUCTION
from src.retriever import CorpusRetriever

# Three chunks placed on three orthogonal axes, so a query aimed at one axis
# has a known, exact ranking rather than an approximately right one.
CHUNKS = [
    {
        "chunk_id": "gdpr:art_22#1",
        "citation": "GDPR, Article 22(1-4)",
        "text": "automated individual decision-making",
        "source_url": "https://example.invalid/gdpr",
        "retrieved_on": "2026-08-17",
    },
    {
        "chunk_id": "ai_act:anx_III#2",
        "citation": "AI Act, ANNEX III",
        "text": "creditworthiness of natural persons",
        "source_url": "https://example.invalid/ai-act",
        "retrieved_on": "2026-08-17",
    },
    {
        "chunk_id": "gdpr:rct_71#1",
        "citation": "GDPR, Recital 71",
        "text": "right to obtain an explanation",
        "source_url": "https://example.invalid/gdpr",
        "retrieved_on": "2026-08-17",
    },
]
VECTORS = np.eye(3, dtype=np.float32)


def retriever(aim=0):
    """A retriever whose stub embedder always points at one known chunk."""
    return CorpusRetriever(CHUNKS, VECTORS, lambda query: VECTORS[aim])


def test_search_ranks_the_closest_passage_first():
    assert retriever(aim=1).search("anything", k=3)[0]["chunk_id"] == "ai_act:anx_III#2"


def test_search_returns_results_in_descending_order():
    scores = [hit["score"] for hit in retriever().search("anything", k=3)]
    assert scores == sorted(scores, reverse=True)


def test_score_is_the_cosine_between_query_and_passage():
    """Both sides are L2-normalised, so an exact hit scores 1 and the rest 0."""
    hits = retriever(aim=2).search("anything", k=3)
    assert hits[0]["score"] == 1.0
    assert all(hit["score"] == 0.0 for hit in hits[1:])


def test_asking_for_more_than_the_corpus_holds_returns_the_corpus():
    assert len(retriever().search("anything", k=50)) == len(CHUNKS)


def test_every_hit_carries_what_makes_it_checkable(chunks=CHUNKS):
    """A passage without its citation and consultation date cannot be verified."""
    hit = retriever().search("anything", k=1)[0]
    assert hit["citation"] == "GDPR, Article 22(1-4)"
    assert hit["source_url"] and hit["retrieved_on"]


def test_a_vector_count_that_does_not_match_the_corpus_is_rejected():
    """The index and the corpus are one artifact; half of it is not usable."""
    with pytest.raises(ValueError, match="stale"):
        CorpusRetriever(CHUNKS, VECTORS[:2], lambda query: VECTORS[0])


def test_an_index_built_from_a_different_corpus_is_rejected(tmp_path):
    """Equal counts are not enough: the ids have to be the same ids.

    A corpus re-chunked into the same number of pieces would pass a length
    check while every vector had drifted one passage along, and the retriever
    would then serve real text under someone else's citation.
    """
    corpus_path = tmp_path / "corpus.jsonl"
    corpus_path.write_text(
        "".join(json.dumps(chunk) + "\n" for chunk in CHUNKS), encoding="utf-8"
    )
    index_path = tmp_path / "index.npz"
    np.savez(
        index_path,
        vectors=VECTORS,
        chunk_ids=np.array(["gdpr:art_22#1", "gdpr:art_22#2", "gdpr:rct_71#1"]),
    )

    with pytest.raises(ValueError, match="different corpus"):
        CorpusRetriever.from_files(corpus_path, index_path)


def test_a_demoted_unit_loses_to_an_equally_similar_one():
    """The whole point: same similarity, different unit, different rank.

    The stub embedder points exactly between the article and the recital, so
    the cosine is identical and only the weight can break the tie.
    """
    chunks = [
        {**CHUNKS[0], "unit": "recital"},
        {**CHUNKS[1], "unit": "article"},
    ]
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    midpoint = np.array([0.5**0.5, 0.5**0.5], dtype=np.float32)

    hits = CorpusRetriever(
        chunks, vectors, lambda query: midpoint, {"recital": 0.9, "article": 1.0}
    ).search("anything", k=2)

    assert hits[0]["chunk_id"] == CHUNKS[1]["chunk_id"]
    assert hits[0]["score"] > hits[1]["score"]


def test_a_unit_with_no_weight_is_left_alone():
    """An unlisted unit must not be silently demoted to zero."""
    chunks = [{**CHUNKS[0], "unit": "something_new"}]
    vectors = np.array([[1.0]], dtype=np.float32)

    hits = CorpusRetriever(
        chunks,
        vectors,
        lambda query: np.array([1.0], dtype=np.float32),
        {"recital": 0.0},
    ).search("anything", k=1)

    assert hits[0]["score"] == 1.0


def test_the_query_carries_the_instruction_the_model_expects(tmp_path, monkeypatch):
    """Asymmetric retrieval needs the instruction on the query, and only there.

    Nothing fails when it is missing -- the search just quietly becomes
    symmetric and the ranking degrades -- so the only way to catch it is to
    assert on what actually reaches the encoder.
    """
    corpus_path = tmp_path / "corpus.jsonl"
    corpus_path.write_text(
        "".join(json.dumps(chunk) + "\n" for chunk in CHUNKS), encoding="utf-8"
    )
    index_path = tmp_path / "index.npz"
    np.savez(
        index_path,
        vectors=VECTORS,
        chunk_ids=np.array([chunk["chunk_id"] for chunk in CHUNKS]),
    )

    encoded = []

    class StubModel:
        def embed(self, texts):
            texts = list(texts)
            encoded.extend(texts)
            return iter(VECTORS[: len(texts)])

    monkeypatch.setattr("src.retriever.load_model", lambda: StubModel())
    CorpusRetriever.from_files(corpus_path, index_path).search("anything", k=1)

    assert encoded == [QUERY_INSTRUCTION + "anything"]


# --- The golden set is data, and wrong data fails silently ---


def test_every_expected_unit_exists_in_the_corpus():
    """A label naming a unit that is not there can never be hit.

    That is the failure mode worth a test: hit-rate would drop, the retriever
    would look worse than it is, and the cause would be a typo in a label
    rather than anything the retriever did.
    """
    with CORPUS_PATH.open(encoding="utf-8") as handle:
        units = {json.loads(line)["chunk_id"].split("#")[0] for line in handle}
    with GOLDEN_SET_PATH.open(encoding="utf-8") as handle:
        questions = [json.loads(line) for line in handle]

    unknown = {
        unit for record in questions for unit in record["expected"] if unit not in units
    }
    assert not unknown, f"golden set points at units absent from the corpus: {unknown}"


def test_unanswerable_questions_say_why_they_are_unanswerable():
    """An empty label is a claim about the corpus, so it carries its reasoning."""
    with GOLDEN_SET_PATH.open(encoding="utf-8") as handle:
        questions = [json.loads(line) for line in handle]

    assert any(record["expected"] for record in questions)
    assert all(record.get("note") for record in questions if not record["expected"]), (
        "a question labelled unanswerable needs a note explaining that judgement"
    )


# --- Cases below need the real model and the built index ---


@pytest.fixture(scope="module")
def real():
    pytest.importorskip(
        "fastembed", reason="optional 'genai' dependency group not installed"
    )
    from src.config import CORPUS_INDEX_PATH

    if not CORPUS_INDEX_PATH.exists():
        pytest.skip("index not built; run `uv run python -m scripts.ingest_corpus`")
    return CorpusRetriever.from_files()


def test_a_question_about_explanation_finds_the_article_granting_it(real):
    """AI Act Article 86 is titled 'Right to explanation of individual decision-making'."""
    found = {
        hit["chunk_id"]
        for hit in real.search(
            "Does the applicant have a right to an explanation of the decision?", k=5
        )
    }
    assert "ai_act:art_86#1" in found


def test_a_question_about_automated_decisions_finds_article_22(real):
    found = {
        hit["chunk_id"]
        for hit in real.search(
            "What information must be given about automated decisions?", k=5
        )
    }
    assert "gdpr:art_22#1" in found
