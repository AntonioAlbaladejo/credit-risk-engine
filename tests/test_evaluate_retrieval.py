"""The abstention accounting, which is where a flattering bug would hide.

`outcomes` decides what counts as the retriever behaving correctly, so an error
here does not break anything visible -- it just reports a better number than
the retriever earned. The case worth pinning down is the third one: a question
the corpus *can* answer but the ranking missed. Staying quiet there is correct,
and an accounting that called it a failure would push the threshold down until
the retriever answered everything.
"""

from scripts.evaluate_retrieval import behaviour, outcomes


def record(expected: list[str], rank: int | None, top_score: float) -> dict:
    return {"expected": expected, "rank": rank, "top_score": top_score}


def test_a_question_with_no_answer_in_the_corpus_should_be_met_with_silence():
    quiet = outcomes([record([], None, 0.50)], threshold=0.66)
    assert quiet["abstained_right"] == 1

    loud = outcomes([record([], None, 0.90)], threshold=0.66)
    assert loud["answered_wrong"] == 1


def test_a_retrieved_answer_above_the_threshold_is_the_only_win():
    counted = outcomes([record(["gdpr:art_22"], 1, 0.90)], threshold=0.66)
    assert counted["answered_right"] == 1


def test_answering_when_the_ranking_missed_counts_as_wrong_not_as_a_hit():
    """The answer exists, the retriever did not find it, and it answered anyway."""
    counted = outcomes([record(["gdpr:art_22"], None, 0.90)], threshold=0.66)
    assert counted["answered_wrong"] == 1
    assert counted["answered_right"] == 0


def test_staying_quiet_when_the_ranking_missed_counts_as_correct():
    counted = outcomes([record(["gdpr:art_22"], None, 0.50)], threshold=0.66)
    assert counted["abstained_right"] == 1


def test_throwing_away_an_answer_it_had_is_counted_separately():
    """Not a wrong citation, but not free either: it is the cost of the threshold."""
    counted = outcomes([record(["gdpr:art_22"], 1, 0.50)], threshold=0.66)
    assert counted["lost"] == 1
    assert counted["answered_right"] == 0


def test_the_report_names_the_baseline_the_threshold_has_to_beat():
    """Abstaining from everything already scores well; the report must say so.

    Without that line a threshold could be sold as an improvement while doing
    nothing but refusing more questions than before.
    """
    detail = [record([], None, 0.90), record(["gdpr:art_22"], 1, 0.90)]
    assert "always abstaining" in behaviour(detail, threshold=0.66)
