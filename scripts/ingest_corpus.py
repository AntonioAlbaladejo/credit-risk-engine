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

# Most retrieval embedding models are BERT-based and read at most 512 tokens;
# anything past that is dropped from the vector while still sitting in the text,
# so the passage stays searchable only up to its cut. 350 words is roughly 455
# tokens, which leaves margin for a tokenizer that splits legal terms finely.
WORD_BUDGET = 350

# Numbered paragraphs inside an article: `<div id="013.002">` is Article 13(2).
SUBPARA_DIV = re.compile(r'<div id="\d{3}\.(?P<paragraph>\d{3})">')
# Where a unit may be cut when it has no numbered paragraphs: after a period,
# semicolon or colon, before a capital or an opening bracket. Semicolons matter
# because definitions, prohibitions and task lists are enumerations -- their
# items end in ";" and open with "(1)" or "(a)", so a sentence-only rule finds
# no boundary at all and leaves them whole. Cross-references stay safe: nothing
# separates "Article 6" from "(2)". Crude, but a wrong break costs a chunk
# boundary in mid-sentence, never a lost or misattributed citation.
SEGMENT_BREAK = re.compile(r"(?<=[.;:])\s+(?=[A-Z(])")


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


def segment(fragment: str) -> list[tuple[str, str]]:
    """Split one unit into its smallest natural pieces.

    Numbered paragraphs are the act's own subdivision, so they win when they
    exist. Definitions articles, annexes and long recitals carry none; those
    fall back to whole-unit text, which `fit` then breaks by sentence.

    Args:
        fragment: The unit's HTML, titles already removed.

    Returns:
        (paragraph number, text) pairs; the number is "" when the text belongs
        to no numbered paragraph.
    """
    starts = list(SUBPARA_DIV.finditer(fragment))
    if not starts:
        return [("", to_text(fragment))]

    segments = []
    # Annexes and some articles open with text before the first numbered
    # point. It belongs to the unit, not to any one paragraph.
    lead = to_text(fragment[: starts[0].start()])
    if lead:
        segments.append(("", lead))
    for index, start in enumerate(starts):
        end = starts[index + 1].start() if index + 1 < len(starts) else len(fragment)
        number = str(int(start.group("paragraph")))
        segments.append((number, to_text(fragment[start.start() : end])))
    return [(number, text) for number, text in segments if text]


def fit(text: str, budget: int) -> list[str]:
    """Break one segment into sentence groups of at most `budget` words.

    Args:
        text: A single segment.
        budget: Word ceiling per group.

    Returns:
        The segment unchanged when it already fits, else its sentence groups.
    """
    if len(text.split()) <= budget:
        return [text]

    groups: list[str] = []
    current: list[str] = []
    size = 0
    for sentence in SEGMENT_BREAK.split(text):
        length = len(sentence.split())
        if current and size + length > budget:
            groups.append(" ".join(current))
            current, size = [], 0
        current.append(sentence)
        size += length
    if current:
        groups.append(" ".join(current))
    return groups


def pack(segments: list[tuple[str, str]], budget: int) -> list[tuple[str, str]]:
    """Group consecutive segments into chunks of at most `budget` words.

    Adjacent short paragraphs merge on purpose. A 12-word paragraph on its own
    embeds into a vector that matches almost nothing, and mixing 12-word with
    2,600-word chunks makes similarity scores incomparable across the corpus.

    Args:
        segments: Output of `segment`.
        budget: Word ceiling per chunk.

    Returns:
        (paragraph label, text) per chunk; the label spans the paragraphs the
        chunk covers ("2", "1-3", or "" when none are numbered).
    """
    chunks: list[list] = []
    for number, text in segments:
        for piece in fit(text, budget):
            length = len(piece.split())
            if chunks and chunks[-1][2] + length <= budget:
                chunks[-1][0].append(number)
                chunks[-1][1].append(piece)
                chunks[-1][2] += length
            else:
                chunks.append([[number], [piece], length])

    packed = []
    for numbers, texts, _ in chunks:
        numbered = [number for number in numbers if number]
        label = ""
        if numbered:
            first, last = numbered[0], numbered[-1]
            label = first if first == last else f"{first}-{last}"
        packed.append((label, " ".join(texts)))
    return packed


def split_units(page: str, doc: dict, retrieved_on: str) -> list[dict]:
    """Split one act into citable chunks of at most WORD_BUDGET words.

    Every article, recital and annex yields at least one chunk; long ones yield
    several, cut at their own paragraph boundaries.

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

        # Drop the title paragraphs before segmenting: they are already in the
        # citation, and repeating them skews the embedding of short articles.
        stripped = HEADING_P.sub(" ", TITLE_P.sub(" ", fragment))

        for part, (paragraph, body) in enumerate(
            pack(segment(stripped), WORD_BUDGET), start=1
        ):
            # A chunk cites the paragraphs it actually covers, so a passage
            # retrieved from Article 13 is quoted as Article 13(2), not as the
            # whole article the reader would then have to search by hand.
            cited = f"{ref}({paragraph})" if paragraph else ref
            citation = f"{doc['short']}, {cited}" + (f" - {heading}" if heading else "")
            chunks.append(
                {
                    # Parts are numbered even when a unit yields only one: the
                    # budget is a tuning knob, so today's single chunk may split
                    # tomorrow and ids must not silently change meaning.
                    "chunk_id": f"{doc['doc_id']}:{anchor}#{part}",
                    "doc_id": doc["doc_id"],
                    "doc_title": doc["title"],
                    "jurisdiction": "EU",
                    "unit": UNIT_NAMES[kind],
                    "ref": cited,
                    "paragraph": paragraph,
                    "heading": heading,
                    "citation": citation,
                    # The citation leads the text on purpose: the embedding
                    # then carries the heading's wording, and a chunk handed to
                    # a model names its source without depending on metadata
                    # travelling alongside it.
                    "text": f"{citation}\n\n{body}",
                    "source_url": EURLEX_HTML.format(celex=doc["celex"]),
                    "retrieved_on": retrieved_on,
                }
            )
    return chunks


def fetch(doc: dict) -> tuple[str, str]:
    """Return an act's HTML and the date it was downloaded, caching both.

    The date is written next to the HTML at download time rather than read back
    from the file's mtime. A copy, a checkout or a stray `touch` rewrites an
    mtime without changing a byte of text, and in a corpus whose whole purpose
    is citation the consultation date is part of the citation: acts get amended,
    so a citation with an unreliable date cannot be checked.

    A cached act whose date file is missing counts as a cache miss and is
    downloaded again -- half an entry is not an entry.

    Args:
        doc: An entry of SOURCES.

    Returns:
        The HTML and the download date as an ISO string.

    Raises:
        urllib.error.URLError: The act is not cached and EUR-Lex is unreachable.
        ValueError: EUR-Lex answered without the ELI anchors the splitter needs.
    """
    raw_path = CORPUS_RAW_DIR / f"{doc['doc_id']}.html"
    stamp_path = raw_path.with_suffix(".json")
    if not (raw_path.exists() and stamp_path.exists()):
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
        stamp_path.write_text(
            json.dumps({"retrieved_on": date.today().isoformat(), "source_url": url}),
            encoding="utf-8",
        )
    downloaded = json.loads(stamp_path.read_text(encoding="utf-8"))["retrieved_on"]
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
