"""Measure the corpus retriever against the golden set.

Two numbers decide whether a retrieval change is an improvement:

* **hit-rate@k** -- of the questions the corpus can answer, how many get a
  correct unit inside the top k. This is the metric to move.
* **score separation** -- whether the questions the corpus *cannot* answer
  score lower than the ones it can. If they do not, no similarity threshold can
  make the retriever say "I don't know", and every answer arrives equally
  confident whether or not it is grounded.

Both are needed. A retriever that answers everything scores well on the first
and is unusable on the second.

Run::

    uv run python -m scripts.evaluate_retrieval

Expected units are unit-level ids (``gdpr:art_22``), not chunk ids: which
sub-split of an article carries the sentence is an implementation detail of the
chunker, and a golden set that depended on it would go stale on every re-chunk.
"""

import json
import statistics
from pathlib import Path

from src.config import GOLDEN_SET_PATH
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
        Hit-rates per cutoff, the two top-score distributions, and the per
        question detail for inspection.
    """
    hits = dict.fromkeys(CUTOFFS, 0)
    answerable = unanswerable = 0
    top_scores = {"answerable": [], "unanswerable": []}
    detail = []

    for record in questions:
        results = retriever.search(record["question"], k=max(CUTOFFS))
        units = [unit_of(hit["chunk_id"]) for hit in results]
        expected = set(record["expected"])

        if expected:
            answerable += 1
            top_scores["answerable"].append(results[0]["score"])
            # Rank of the first correct unit, or None if it never appears.
            rank = next(
                (i + 1 for i, unit in enumerate(units) if unit in expected), None
            )
            for k in CUTOFFS:
                if rank is not None and rank <= k:
                    hits[k] += 1
        else:
            unanswerable += 1
            top_scores["unanswerable"].append(results[0]["score"])
            rank = None

        detail.append({**record, "units": units, "rank": rank, "results": results})

    return {
        "answerable": answerable,
        "unanswerable": unanswerable,
        "hit_rate": {k: hits[k] / answerable for k in CUTOFFS} if answerable else {},
        "top_scores": top_scores,
        "detail": detail,
    }


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

    answerable = outcome["top_scores"]["answerable"]
    unanswerable = outcome["top_scores"]["unanswerable"]
    print(
        f"\nTop-1 score, answerable   : {statistics.mean(answerable):.3f} mean, "
        f"{min(answerable):.3f} min, {max(answerable):.3f} max"
    )
    print(
        f"Top-1 score, unanswerable : {statistics.mean(unanswerable):.3f} mean, "
        f"{min(unanswerable):.3f} min, {max(unanswerable):.3f} max"
    )

    # A threshold can only exist if the worst answerable question still outscores
    # the best unanswerable one. Overlap is the size of the problem.
    overlap = sum(1 for score in unanswerable if score >= min(answerable))
    if overlap:
        print(
            f"\nNo abstention threshold exists: {overlap}/{len(unanswerable)} "
            f"unanswerable questions score at or above the weakest answerable one "
            f"({min(answerable):.3f})."
        )
    else:
        print(
            f"\nAbstention threshold available between {max(unanswerable):.3f} "
            f"and {min(answerable):.3f}."
        )


def main() -> None:
    report(evaluate(CorpusRetriever.from_files(), load_golden_set()))


if __name__ == "__main__":
    main()
