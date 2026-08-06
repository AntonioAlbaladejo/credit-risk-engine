"""Chunking of the EU regulatory corpus.

`split_units` is exercised against an inline fragment, so the parser is covered
without network access or a downloaded corpus. The cases that need `corpus/`
skip when it is absent, the same way the model tests skip without `models/`.

What these pin is the one property the rest of the retrieval layer depends on:
a chunk must be exactly one citable unit and must carry the reference that
names it. A passage retrieved without a precise citation is unusable here.
"""

import json

import pytest

from scripts.ingest_corpus import SOURCES, split_units
from src.config import CORPUS_PATH

# Trimmed to the shape the splitter reads: two flat ELI anchors, an article
# with its heading, and a recital, which carries neither.
FRAGMENT = """
<div class="eli-subdivision" id="art_22">
  <p class="oj-ti-art">Article 22</p>
  <div class="eli-title" id="art_22.tit_1">
    <p class="oj-sti-art">Automated individual decision-making</p>
  </div>
  <div id="022.001"><p class="oj-normal">1.&nbsp;&nbsp;The data subject shall
  have the right not to be subject to a decision based solely on automated
  processing.<span class="oj-super oj-note-tag">(3)</span></p></div>
</div>
<div class="eli-subdivision" id="rct_71">
  <p class="oj-normal">(71)</p>
  <p class="oj-normal">Such processing should be subject to suitable safeguards.</p>
</div>
"""

DOC = {"doc_id": "gdpr", "celex": "32016R0679", "short": "GDPR", "title": "GDPR"}


@pytest.fixture(scope="module")
def parsed():
    return split_units(FRAGMENT, DOC, "2026-08-06")


def test_each_eli_anchor_becomes_one_chunk(parsed):
    assert [chunk["chunk_id"] for chunk in parsed] == ["gdpr:art_22", "gdpr:rct_71"]


def test_article_chunk_is_cited_by_number_and_heading(parsed):
    article = parsed[0]
    assert article["unit"] == "article"
    assert article["ref"] == "Article 22"
    assert (
        article["citation"] == "GDPR, Article 22 - Automated individual decision-making"
    )
    # The citation leads the text so a retrieved passage names its own source.
    assert article["text"].startswith(article["citation"])


def test_recital_is_numbered_from_its_anchor(parsed):
    """Recitals have no title paragraph, so the number comes from the ELI id."""
    assert parsed[1]["unit"] == "recital"
    assert parsed[1]["ref"] == "Recital 71"


def test_units_do_not_bleed_into_each_other(parsed):
    """The failure that would silently poison every citation."""
    assert "safeguards" not in parsed[0]["text"]
    assert "solely on automated processing" not in parsed[1]["text"]


def test_footnote_markers_are_dropped_from_the_body(parsed):
    assert "(3)" not in parsed[0]["text"]


def test_entities_and_non_breaking_spaces_are_resolved(parsed):
    assert "&nbsp;" not in parsed[0]["text"]
    assert "\xa0" not in parsed[0]["text"]


# --- Cases below need the corpus that scripts/ingest_corpus.py writes ---

corpus = pytest.mark.skipif(
    not CORPUS_PATH.exists(),
    reason="corpus/ not built; run `uv run python -m scripts.ingest_corpus`",
)


@pytest.fixture(scope="module")
def chunks():
    with CORPUS_PATH.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


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
def test_gdpr_article_22_survives_whole(chunks):
    """The article the whole compliance angle hangs on, kept end to end."""
    article = next(c for c in chunks if c["chunk_id"] == "gdpr:art_22")
    assert "solely on automated processing" in article["text"]  # paragraph 1
    assert "explicit consent" in article["text"]  # paragraph 2, last point
    assert "right to obtain human intervention" in article["text"]  # paragraph 3


@corpus
def test_recital_71_is_ingested(chunks):
    """Where the right to an explanation actually lives, not in Article 22."""
    recital = next(c for c in chunks if c["chunk_id"] == "gdpr:rct_71")
    assert "obtain an explanation of the decision" in recital["text"]


@corpus
def test_annex_iii_classifies_credit_scoring_as_high_risk(chunks):
    """The sentence that puts this very repository in scope of the AI Act."""
    annex = next(c for c in chunks if c["chunk_id"] == "ai_act:anx_III")
    assert "creditworthiness" in annex["text"]
