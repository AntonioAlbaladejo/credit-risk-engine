"""Fetch the EU regulatory corpus and split it into citable chunks.

Run it once before building the retrieval index::

    uv run python -m scripts.ingest_corpus

Downloads are cached under ``corpus/raw/`` so a re-run needs no network; delete
that directory to pick up an amended text.

EUR-Lex renders every act with ELI (European Legislation Identifier) anchors --
``id="art_22"``, ``id="rct_71"``, ``id="anx_III"`` -- which mark exactly the
units a legal citation refers to. The splitter follows those anchors instead of
a fixed token window, so no chunk ever straddles two articles and every chunk
can name its own source. A retrieved passage that cannot be cited precisely is
worse than useless in this domain.
"""

import html
import json
import logging
import re
import urllib.request
from datetime import date

from src.config import CORPUS_PATH, CORPUS_RAW_DIR, LOG_LEVEL

logger = logging.getLogger(__name__)

EURLEX_HTML = "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:{celex}"
USER_AGENT = "credit-risk-engine corpus ingest"

# `short` is what a citation shown to a user leads with.
SOURCES = (
    {
        "doc_id": "gdpr",
        "celex": "32016R0679",
        "short": "GDPR",
        "title": "Regulation (EU) 2016/679 (General Data Protection Regulation)",
    },
    {
        "doc_id": "ai_act",
        "celex": "32024R1689",
        "short": "AI Act",
        "title": "Regulation (EU) 2024/1689 (Artificial Intelligence Act)",
    },
)

# Articles and recitals are `eli-subdivision`, annexes are `eli-container`. The
# three are flat siblings -- never nested inside one another -- so the document
# splits cleanly on the opening tags.
UNIT_DIV = re.compile(
    r'<div class="eli-(?:subdivision|container)" id="(?P<anchor>(?:art|rct|anx)_[^"]+)"'
)
UNIT_NAMES = {"art": "article", "rct": "recital", "anx": "annex"}

# `oj-ti-art` holds "Article 22" and `oj-doc-ti` holds "ANNEX III"; the heading
# follows in `oj-sti-art` for articles and in a second `oj-doc-ti` for annexes.
TITLE_P = re.compile(r'<p[^>]*class="oj-(?:ti-art|doc-ti)"[^>]*>(.*?)</p>', re.S)
HEADING_P = re.compile(r'<p[^>]*class="oj-sti-art"[^>]*>(.*?)</p>', re.S)
# Footnote markers are superscripts sitting inside a sentence. Left in, they
# drop stray numbers into the middle of a legal clause.
FOOTNOTE = re.compile(r'<span[^>]*class="oj-super[^"]*"[^>]*>.*?</span>', re.S)
TAG = re.compile(r"<[^>]+>")


def to_text(fragment: str) -> str:
    """Strip markup from an HTML fragment and collapse its whitespace.

    Args:
        fragment: Raw HTML.

    Returns:
        Plain text on a single line, entities resolved.
    """
    stripped = TAG.sub(" ", FOOTNOTE.sub(" ", fragment))
    # EUR-Lex separates references with non-breaking spaces ("Article 6(2)").
    plain = html.unescape(stripped).replace("\xa0", " ")
    return re.sub(r"\s+", " ", plain).strip()


def split_units(page: str, doc: dict, retrieved_on: str) -> list[dict]:
    """Split one act into one chunk per article, recital and annex.

    Args:
        page: The act's EUR-Lex HTML.
        doc: An entry of SOURCES.
        retrieved_on: ISO date the HTML was downloaded, carried into each chunk.

    Returns:
        Chunks in document order, each self-describing enough to be cited.
    """
    matches = list(UNIT_DIV.finditer(page))
    chunks = []
    for position, match in enumerate(matches):
        # Units are flat, so one runs until the next one starts.
        following = (
            matches[position + 1].start() if position + 1 < len(matches) else len(page)
        )
        fragment = page[match.start() : following]

        anchor = match.group("anchor")
        kind, _, number = anchor.partition("_")

        # Articles and annexes state their own number; recitals only have the
        # anchor, their number being rendered as a table cell like every other
        # numbered paragraph.
        titles = [to_text(title) for title in TITLE_P.findall(fragment)]
        headings = [to_text(heading) for heading in HEADING_P.findall(fragment)]
        ref = titles[0] if titles else f"Recital {number}"
        heading = next(iter(titles[1:] + headings), "")
        citation = f"{doc['short']}, {ref}" + (f" - {heading}" if heading else "")

        # Drop the title paragraphs from the body: they are already in the
        # citation, and repeating them skews the embedding of short articles.
        body = to_text(HEADING_P.sub(" ", TITLE_P.sub(" ", fragment)))

        chunks.append(
            {
                "chunk_id": f"{doc['doc_id']}:{anchor}",
                "doc_id": doc["doc_id"],
                "doc_title": doc["title"],
                "jurisdiction": "EU",
                "unit": UNIT_NAMES[kind],
                "ref": ref,
                "heading": heading,
                "citation": citation,
                # The citation leads the text on purpose: the embedding then
                # carries the heading's wording, and a chunk handed to a model
                # names its source without depending on metadata travelling
                # alongside it.
                "text": f"{citation}\n\n{body}",
                "source_url": EURLEX_HTML.format(celex=doc["celex"]),
                "retrieved_on": retrieved_on,
            }
        )
    return chunks


def fetch(doc: dict) -> tuple[str, str]:
    """Return an act's HTML and the date it was downloaded, caching the file.

    Args:
        doc: An entry of SOURCES.

    Returns:
        The HTML and the download date as an ISO string.

    Raises:
        urllib.error.URLError: The act is not cached and EUR-Lex is unreachable.
    """
    raw_path = CORPUS_RAW_DIR / f"{doc['doc_id']}.html"
    if not raw_path.exists():
        url = EURLEX_HTML.format(celex=doc["celex"])
        logger.info("Downloading %s from %s", doc["short"], url)
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=60) as response:
            page = response.read()
        # Check before caching. EUR-Lex answers 200 with an empty body often
        # enough that a truncated response would otherwise be written to disk
        # and served silently to every later run.
        if b"eli-subdivision" not in page:
            raise ValueError(
                f"{doc['short']} came back without ELI anchors ({len(page)} bytes). "
                "Nothing cached; re-run to retry."
            )
        raw_path.write_bytes(page)
    downloaded = date.fromtimestamp(raw_path.stat().st_mtime).isoformat()
    return raw_path.read_text(encoding="utf-8"), downloaded


def main() -> None:
    """Ingest every source into a single JSONL corpus."""
    logging.basicConfig(level=LOG_LEVEL, format="%(levelname)s %(message)s")
    CORPUS_RAW_DIR.mkdir(parents=True, exist_ok=True)

    chunks: list[dict] = []
    for doc in SOURCES:
        page, retrieved_on = fetch(doc)
        parsed = split_units(page, doc, retrieved_on)
        if not parsed:
            raise ValueError(
                f"No ELI anchors found in {doc['short']}: EUR-Lex markup changed, "
                f"delete corpus/raw/{doc['doc_id']}.html and check the page."
            )
        logger.info("%s: %d chunks", doc["short"], len(parsed))
        chunks.extend(parsed)

    CORPUS_PATH.write_text(
        "".join(json.dumps(chunk, ensure_ascii=False) + "\n" for chunk in chunks),
        encoding="utf-8",
    )
    logger.info("Wrote %d chunks to %s", len(chunks), CORPUS_PATH)


if __name__ == "__main__":
    main()
