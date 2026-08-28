"""MCP tool surface checked against the real artifacts in models/.

Skipped unless the optional `genai` dependency group is installed
(`uv sync --all-groups`), so CI -- which runs a plain `uv sync --frozen` --
stays green without pulling the MCP SDK into the image's dependency tree.

The tools are thin wrappers over RiskExplainer, CreditRiskPredictor and
CorpusRetriever, all covered elsewhere. What these cases pin is the part only
reachable through the protocol: the schema the calling model is handed, and the
shape of what comes back.
"""

import asyncio
import importlib.util
import json

import pytest

from src.config import (
    CORPUS_INDEX_PATH,
    FEATURE_NAMES_PATH,
    MAX_LOAN_INT_RATE,
    MIN_LOAN_INT_RATE,
    MODEL_PATH,
    PREPROCESSOR_PATH,
    THRESHOLD_PATH,
)

pytest.importorskip("mcp", reason="optional 'genai' dependency group not installed")

from src.mcp_server import server  # noqa: E402

ARTIFACTS = [MODEL_PATH, PREPROCESSOR_PATH, FEATURE_NAMES_PATH, THRESHOLD_PATH]

# The regulatory tool needs the embedding model and a built index; the scoring
# tools need neither, which is the point of loading them separately.
needs_corpus = pytest.mark.skipif(
    importlib.util.find_spec("fastembed") is None or not CORPUS_INDEX_PATH.exists(),
    reason="corpus index not built, or fastembed not installed",
)

pytestmark = [
    pytest.mark.real_artifacts,
    pytest.mark.skipif(
        not all(p.exists() for p in ARTIFACTS),
        reason="models/ artifacts not present in this checkout",
    ),
]

APPLICATION = {
    "person_age": 32,
    "person_income": 33600,
    "person_home_ownership": "RENT",
    "person_emp_length": 0.0,
    "loan_intent": "EDUCATION",
    "loan_grade": "D",
    "loan_amnt": 8000,
    "loan_int_rate": 16.02,
    "loan_percent_income": 0.24,
    "cb_person_default_on_file": 0,
}


def call(tool: str, arguments: dict | None = None) -> dict:
    """Invoke a tool the way a client would and parse its JSON payload."""
    result = asyncio.run(server.call_tool(tool, arguments or {}))
    return json.loads(result.content[0].text)


def test_every_tool_is_exposed():
    names = {tool.name for tool in asyncio.run(server.list_tools())}
    assert names == {
        "assess_loan_application",
        "get_model_info",
        "search_regulation",
    }


def test_every_tool_carries_a_description():
    """Descriptions are how the calling model decides which tool applies."""
    for tool in asyncio.run(server.list_tools()):
        assert tool.description, f"{tool.name} has no description"


def test_published_schema_enforces_the_configured_interest_rate_scale():
    """The bounds in src/config.py must reach the model, not just the API.

    loan_int_rate is in percent units. An LLM that sends 0.16 for 16% would be
    scored ~3.5 sigma outside the training range, so the floor has to travel
    with the schema.
    """
    tool = next(
        t
        for t in asyncio.run(server.list_tools())
        if t.name == "assess_loan_application"
    )
    rate = tool.input_schema["$defs"]["LoanApplication"]["properties"]["loan_int_rate"]
    assert rate["minimum"] == MIN_LOAN_INT_RATE
    assert rate["maximum"] == MAX_LOAN_INT_RATE


def test_fractional_interest_rate_is_rejected():
    """Same regression as tests/test_schemas.py, reached through the protocol."""
    with pytest.raises(Exception, match="loan_int_rate"):
        call(
            "assess_loan_application",
            {"application": {**APPLICATION, "loan_int_rate": 0.16}},
        )


def test_assessment_decision_agrees_with_the_threshold():
    result = call("assess_loan_application", {"application": APPLICATION})
    expected = (
        "reject" if result["probability_default"] >= result["threshold"] else "approve"
    )
    assert result["decision"] == expected


def test_assessment_returns_ranked_reason_codes():
    result = call("assess_loan_application", {"application": APPLICATION})
    magnitudes = [abs(r["contribution"]) for r in result["reason_codes"]]
    assert magnitudes == sorted(magnitudes, reverse=True)
    assert all(
        r["direction"] in {"increases_risk", "decreases_risk"}
        for r in result["reason_codes"]
    )


def test_assessment_omits_the_per_feature_breakdown():
    """Scaled feature names are internals; the model gets reason codes only."""
    result = call("assess_loan_application", {"application": APPLICATION})
    assert "feature_contributions" not in result
    assert "margin" not in result


def test_model_info_reports_the_promoted_bundle():
    import joblib

    info = call("get_model_info")
    assert info["decision_threshold"] == joblib.load(THRESHOLD_PATH)
    assert (
        info["num_features"]
        == len(info["features"])
        == len(joblib.load(FEATURE_NAMES_PATH))
    )


# --- The regulatory tool: what it says when it has nothing to say ---


@needs_corpus
def test_a_question_the_corpus_cannot_answer_returns_no_passages():
    """The empty list is the contract this whole phase was built around.

    Returning a best guess instead is the failure the threshold exists to
    prevent: a citation that reads as grounding and is not. Measured, this
    question tops out at 0.61 against a threshold of 0.66.
    """
    result = call(
        "search_regulation", {"question": "Who signed off our latest model validation?"}
    )
    assert result["passages"] == []
    # The docstring tells the model what an empty list means; the payload has
    # to say it too, because the result is what it rereads while answering.
    assert result["note"]


@needs_corpus
def test_an_answerable_question_returns_checkable_passages():
    """A passage without its citation and consultation date cannot be verified."""
    result = call(
        "search_regulation",
        {
            "question": "Does the applicant have a right to an explanation of the decision?"
        },
    )
    assert result["passages"]
    for passage in result["passages"]:
        assert passage["citation"] and passage["text"]
        assert passage["source_url"] and passage["retrieved_on"]


def test_the_hypothetical_passage_is_offered_but_never_demanded():
    """A client that ignores it must still get the tool it had before.

    The passage is what makes the search find more, but it is written by the
    calling model and nothing can force it to arrive. Making it required would
    turn a client that does not read the docstring into a broken tool.
    """
    tool = next(
        t for t in asyncio.run(server.list_tools()) if t.name == "search_regulation"
    )
    assert "hypothetical_passage" in tool.input_schema["properties"]
    assert "hypothetical_passage" not in tool.input_schema.get("required", [])


@needs_corpus
def test_a_convincing_passage_cannot_make_the_corpus_answer():
    """Against the real index: a passage that ranks well must still change nothing.

    The question scores 0.50, under both thresholds. One sitting between them
    would be testing the lower threshold, not the veto.
    """
    result = call(
        "search_regulation",
        {
            "question": "Which hyperparameters gave the best score in the last sweep?",
            "hypothetical_passage": (
                "The provider shall put in place a quality management system "
                "including an accountability framework setting out the "
                "responsibilities of the management and other staff with "
                "regard to the approval and verification of changes."
            ),
        },
    )
    assert result["passages"] == []
    assert result["note"]


@needs_corpus
def test_passages_omit_the_ranking_internals():
    """The score carries the unit weight, so it is not a similarity to report.

    A model handed 0.67 starts describing its own confidence, and the
    threshold has already consumed that number on this side.
    """
    result = call(
        "search_regulation",
        {"question": "What information must be given about automated decisions?"},
    )
    assert all("score" not in p and "chunk_id" not in p for p in result["passages"])
