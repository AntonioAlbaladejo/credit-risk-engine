"""Chunking of the EU regulatory corpus.

The splitter is exercised against an inline fragment and through `fit` and
`pack` directly, so the parser is covered without network access or a
downloaded corpus. The cases that need `corpus/` skip when it is absent, the
same way the model tests skip without `models/`.

What these pin is the property the rest of the retrieval layer depends on: a
chunk must be one citable passage, must carry the reference that names it, and
must fit inside the embedding model's context window. A passage retrieved
without a precise citation is unusable here; one that overflows the window is
only half searchable, silently.
"""

import json

import pytest

from scripts import ingest_corpus
from scripts.ingest_corpus import (
    SOURCES,
    TOKEN_LIMIT,
    count_words,
    fit,
    pack,
    split_units,
)
from src.config import CORPUS_PATH

# Trimmed to the shape the splitter reads: flat ELI anchors, an article with a
# heading and two numbered paragraphs, and a recital, which has neither.
FRAGMENT = """
<div class="eli-subdivision" id="art_13">
  <p class="oj-ti-art">Article 13</p>
  <div class="eli-title" id="art_13.tit_1">
    <p class="oj-sti-art">Information to be provided</p>
  </div>
  <div id="013.001"><p class="oj-normal">1.&nbsp;&nbsp;The controller shall
  provide the data subject with the following information.<span
  class="oj-super oj-note-tag">(3)</span></p></div>
  <div id="013.002"><p class="oj-normal">2.&nbsp;&nbsp;The controller shall
  also provide information on automated decision-making.</p></div>
</div>
<div class="eli-subdivision" id="rct_71">
  <p class="oj-normal">(71)</p>
  <p class="oj-normal">Such processing should be subject to suitable safeguards.</p>
</div>
"""

DOC = {"doc_id": "gdpr", "celex": "32016R0679", "short": "GDPR", "title": "GDPR"}


@pytest.fixture(scope="module")
def parsed():
    return split_units(FRAGMENT, DOC, "2026-08-06", count_words)


def test_each_unit_becomes_at_least_one_numbered_part(parsed):
    assert [chunk["chunk_id"] for chunk in parsed] == ["gdpr:art_13#1", "gdpr:rct_71#1"]


def test_short_paragraphs_merge_and_the_citation_spans_them(parsed):
    """Both paragraphs fit in one chunk, so the citation must cover both."""
    article = parsed[0]
    assert article["unit"] == "article"
    assert article["paragraph"] == "1-2"
    assert article["ref"] == "Article 13(1-2)"
    assert article["citation"] == "GDPR, Article 13(1-2) - Information to be provided"
    # The citation leads the text so a retrieved passage names its own source.
    assert article["text"].startswith(article["citation"])


def test_recital_is_numbered_from_its_anchor(parsed):
    """Recitals have no title paragraph, so the number comes from the ELI id."""
    assert parsed[1]["unit"] == "recital"
    assert parsed[1]["ref"] == "Recital 71"
    assert parsed[1]["paragraph"] == ""


def test_units_do_not_bleed_into_each_other(parsed):
    """The failure that would silently poison every citation."""
    assert "safeguards" not in parsed[0]["text"]
    assert "automated decision-making" not in parsed[1]["text"]


def test_footnote_markers_are_dropped_from_the_body(parsed):
    assert "(3)" not in parsed[0]["text"]


def test_entities_and_non_breaking_spaces_are_resolved(parsed):
    assert "&nbsp;" not in parsed[0]["text"]
    assert "\xa0" not in parsed[0]["text"]


def test_fit_leaves_a_segment_that_already_fits():
    assert fit("Short enough.", budget=50, measure=count_words) == ["Short enough."]


def test_fit_splits_an_enumeration_at_its_semicolons():
    """Definitions and prohibition lists have no sentence ends to cut on.

    Their items close with ";" and open with "(n)", so a rule keyed only on
    full stops finds no boundary and leaves a 2,600-word article whole.
    """
    items = " ".join(
        f"({n}) the term number {n} means a defined thing;" for n in range(20)
    )
    pieces = fit(
        f"For the purposes of this Regulation: {items}", budget=40, measure=count_words
    )
    assert len(pieces) > 1
    assert all(len(piece.split()) <= 40 for piece in pieces)


def test_provenance_date_is_not_read_back_from_the_filesystem(tmp_path, monkeypatch):
    """A citation's consultation date must not move when a file is copied.

    The HTML written here has this moment as its mtime, so a fetch that dated
    the corpus from the filesystem would answer today instead of 2020.
    """
    monkeypatch.setattr(ingest_corpus, "CORPUS_RAW_DIR", tmp_path)
    (tmp_path / "gdpr.html").write_text(
        '<div class="eli-subdivision" id="art_1">x</div>'
    )
    (tmp_path / "gdpr.json").write_text('{"retrieved_on": "2020-01-01"}')

    _, retrieved_on = ingest_corpus.fetch(DOC)

    assert retrieved_on == "2020-01-01"


def test_pack_keeps_a_long_paragraph_out_of_its_neighbours_chunk():
    long_text = "word " * 60
    packed = pack(
        [("1", long_text.strip()), ("2", "short tail")], budget=50, measure=count_words
    )
    assert len(packed) > 1
    assert packed[-1] == ("2", "short tail")


# --- Cases below need the corpus that scripts/ingest_corpus.py writes ---

corpus = pytest.mark.skipif(
    not CORPUS_PATH.exists(),
    reason="corpus/ not built; run `uv run python -m scripts.ingest_corpus`",
)


@pytest.fixture(scope="module")
def chunks():
    with CORPUS_PATH.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def unit(chunks, anchor):
    """Every chunk cut from one article, recital or annex, in document order."""
    return [c for c in chunks if c["chunk_id"].startswith(f"{anchor}#")]


@corpus
def test_every_chunk_is_citable(chunks):
    for chunk in chunks:
        assert chunk["ref"], chunk["chunk_id"]
        assert chunk["citation"], chunk["chunk_id"]
        assert chunk["text"].strip(), chunk["chunk_id"]


@corpus
def test_chunk_ids_are_unique(chunks):
    ids = [chunk["chunk_id"] for chunk in chunks]
    assert len(ids) == len(set(ids))


@corpus
def test_every_source_is_present(chunks):
    assert {chunk["doc_id"] for chunk in chunks} == {doc["doc_id"] for doc in SOURCES}


@corpus
def test_no_chunk_overflows_the_embedding_window(chunks):
    """The reason the budget exists, checked with the tokenizer that will read it.

    A chunk past the window keeps its tail in the text but never gets it into
    the vector, so the passage is searchable only up to its cut and nothing
    reports it. Skipped without the `genai` group, which is how CI installs.
    """
    pytest.importorskip(
        "fastembed", reason="optional 'genai' dependency group not installed"
    )
    count = ingest_corpus.token_counter()
    oversized = [(c["chunk_id"], count(c["text"])) for c in chunks]
    assert [c for c in oversized if c[1] > TOKEN_LIMIT] == []


@corpus
def test_no_chunk_is_wildly_oversized(chunks):
    """Coarse guard that still runs where the tokenizer is unavailable.

    Words underestimate tokens badly on legal references -- 2.14 per word at
    worst against a 1.21 median -- so this only catches gross regressions. The
    exact check is the case above.
    """
    assert max(len(c["text"].split()) for c in chunks) <= TOKEN_LIMIT


@corpus
def test_gdpr_article_22_survives_whole(chunks):
    """The article the whole compliance angle hangs on, kept end to end."""
    parts = unit(chunks, "gdpr:art_22")
    assert len(parts) == 1, "Article 22 fits the budget and must not be split"
    text = parts[0]["text"]
    assert "solely on automated processing" in text  # paragraph 1
    assert "explicit consent" in text  # paragraph 2, last point
    assert "right to obtain human intervention" in text  # paragraph 3


@corpus
def test_recital_71_keeps_the_right_to_an_explanation(chunks):
    """Where the right to an explanation lives, and not in Article 22."""
    joined = " ".join(c["text"] for c in unit(chunks, "gdpr:rct_71"))
    assert "obtain an explanation of the decision" in joined


@corpus
def test_annex_iii_classifies_credit_scoring_as_high_risk(chunks):
    """The sentence that puts this very repository in scope of the AI Act.

    It sits ~590 tokens into the annex, so before the annex was split by
    budget no embedding of it could have matched a query about credit scoring.
    """
    hits = [
        c for c in unit(chunks, "ai_act:anx_III") if "creditworthiness" in c["text"]
    ]
    assert len(hits) == 1
    assert "ANNEX III" in hits[0]["citation"]


@corpus
def test_a_split_article_cites_the_paragraphs_it_covers(chunks):
    """Article 13 runs past the budget, so its parts must not both claim it."""
    parts = unit(chunks, "gdpr:art_13")
    assert len(parts) > 1
    assert [part["ref"] for part in parts] == ["Article 13(1)", "Article 13(2-4)"]
