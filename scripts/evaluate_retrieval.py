"""Measure the corpus retriever against the golden set.

Two numbers decide whether a retrieval change is an improvement:

* **hit-rate@k** -- of the questions the corpus can answer, how many get a
  correct unit inside the top k. This is the metric to move.
* **behaviour under the abstention threshold** -- what the retriever actually
  does with each question once `MIN_SCORE` is applied. Three outcomes, not two:

  1. the corpus cannot answer, and the retriever stays quiet: correct;
  2. the corpus can answer, the right unit was retrieved, and it is returned:
     correct;
  3. the corpus can answer but the right unit was *not* retrieved -- here
     staying quiet is **also correct**. Answering means citing the wrong
     provision confidently, which is the worst of the three outcomes and not
     the second worst.

Both are needed. A retriever that answers everything scores well on the first
and is unusable on the second. The baseline to beat on the second is not
"answer everything" but **abstain from everything**, which already handles
every unanswerable question correctly and is printed alongside for that reason.

Run::

    uv run python -m scripts.evaluate_retrieval

Expected units are unit-level ids (``gdpr:art_22``), not chunk ids: which
sub-split of an article carries the sentence is an implementation detail of the
chunker, and a golden set that depended on it would go stale on every re-chunk.
"""

import json
from pathlib import Path

from src.config import GOLDEN_SET_PATH, MIN_SCORE
from src.retriever import CorpusRetriever

# Deep enough to show whether a correct unit is being found at all but ranked
# badly -- a different problem from not being in the corpus.
CUTOFFS = (1, 3, 5)


def unit_of(chunk_id: str) -> str:
    """Strip the sub-split suffix: ``gdpr:art_22#1`` -> ``gdpr:art_22``."""
    return chunk_id.split("#")[0]


def load_golden_set(path: Path = GOLDEN_SET_PATH) -> list[dict]:
    """Read the labelled questions.

    Args:
        path: JSONL with `question`, `role` and `expected` per line.

    Returns:
        One dict per question, in file order.
    """
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def evaluate(retriever: CorpusRetriever, questions: list[dict]) -> dict:
    """Score the retriever over the golden set.

    Args:
        retriever: The retriever under test.
        questions: Golden set records; empty `expected` marks a question the
            corpus cannot answer, which must therefore rank low.

    Returns:
        Hit-rates per cutoff and the per-question detail, each record carrying
        the top score and whether the ranking found a correct unit at all.
    """
    hits = dict.fromkeys(CUTOFFS, 0)
    answerable = unanswerable = 0
    detail = []

    for record in questions:
        # min_score=0.0 on purpose: measuring the ranking means seeing all of
        # it. The threshold is applied in the accounting below, so that moving
        # it does not need the model run again.
        results = retriever.search(record["question"], k=max(CUTOFFS), min_score=0.0)
        units = [unit_of(hit["chunk_id"]) for hit in results]
        expected = set(record["expected"])

        if expected:
            answerable += 1
            # Rank of the first correct unit, or None if it never appears.
            rank = next(
                (i + 1 for i, unit in enumerate(units) if unit in expected), None
            )
            for k in CUTOFFS:
                if rank is not None and rank <= k:
                    hits[k] += 1
        else:
            unanswerable += 1
            rank = None

        detail.append(
            {
                **record,
                "units": units,
                "rank": rank,
                "top_score": results[0]["score"],
                "results": results,
            }
        )

    return {
        "answerable": answerable,
        "unanswerable": unanswerable,
        "hit_rate": {k: hits[k] / answerable for k in CUTOFFS} if answerable else {},
        "detail": detail,
    }


def outcomes(detail: list[dict], threshold: float) -> dict[str, int]:
    """Count what the retriever would do with each question at a threshold.

    Args:
        detail: Per-question records from `evaluate`.
        threshold: Scores below it mean the retriever answers nothing.

    Returns:
        Counts of `answered_right`, `answered_wrong`, `abstained_right` and
        `lost` (had the answer, threw it away). The first and third are the
        correct outcomes.
    """
    counted = dict.fromkeys(
        ("answered_right", "answered_wrong", "abstained_right", "lost"), 0
    )
    for record in detail:
        # Groundable = the corpus can answer it *and* the ranking found it.
        # A question whose answer exists but was not retrieved belongs with the
        # unanswerable ones: there is nothing right to say about it either.
        groundable = bool(record["expected"]) and record["rank"] is not None
        answered = record["top_score"] >= threshold
        if groundable:
            counted["answered_right" if answered else "lost"] += 1
        else:
            counted["answered_wrong" if answered else "abstained_right"] += 1
    return counted


def behaviour(detail: list[dict], threshold: float = MIN_SCORE) -> str:
    """Render the abstention trade-off, with the baseline worth beating."""
    total = len(detail)
    counted = outcomes(detail, threshold)
    correct = counted["answered_right"] + counted["abstained_right"]
    # Refusing every question already handles every ungroundable one. Any
    # threshold has to buy answers on top of that to be worth having.
    silent = outcomes(detail, float("inf"))["abstained_right"]

    lines = [
        f"\nBehaviour at MIN_SCORE = {threshold}:",
        f"  answered, right passage : {counted['answered_right']:>3}",
        f"  answered, WRONG passage : {counted['answered_wrong']:>3}   <- the costly one",
        f"  abstained, nothing to say: {counted['abstained_right']:>3}",
        f"  abstained, had it       : {counted['lost']:>3}",
        f"  handled correctly       : {correct:>3}/{total} ({correct / total:.1%})",
        f"  always abstaining would : {silent:>3}/{total} ({silent / total:.1%}), "
        f"so the threshold buys {counted['answered_right']} answers "
        f"for {counted['answered_wrong']} wrong citations",
        "\nThreshold sweep (re-fit here when the corpus or the model changes):",
    ]
    for candidate in (0.60, 0.62, 0.64, 0.66, 0.68, 0.70):
        swept = outcomes(detail, candidate)
        lines.append(
            f"  {candidate:.2f}  correct {swept['answered_right'] + swept['abstained_right']:>3}"
            f"/{total}   right {swept['answered_right']:>3}"
            f"   wrong {swept['answered_wrong']:>3}   lost {swept['lost']:>3}"
        )
    return "\n".join(lines)


def report(outcome: dict) -> None:
    """Print the evaluation in the form the decision needs."""
    print(f"\nAnswerable questions: {outcome['answerable']}")
    for k, rate in outcome["hit_rate"].items():
        found = round(rate * outcome["answerable"])
        print(f"  hit-rate@{k}: {rate:6.1%}  ({found}/{outcome['answerable']})")

    print(f"\nMissed entirely (no correct unit in top {max(CUTOFFS)}):")
    for record in outcome["detail"]:
        if record["expected"] and record["rank"] is None:
            print(f"  [{record['role']}] {record['question']}")
            print(
                f"      got {record['units'][0]}, wanted {'/'.join(record['expected'])}"
            )

    print(behaviour(outcome["detail"]))


def main() -> None:
    report(evaluate(CorpusRetriever.from_files(), load_golden_set()))


if __name__ == "__main__":
    main()
