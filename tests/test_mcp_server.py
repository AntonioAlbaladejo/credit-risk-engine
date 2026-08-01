"""MCP tool surface checked against the real artifacts in models/.

Skipped unless the optional `genai` dependency group is installed
(`uv sync --all-groups`), so CI -- which runs a plain `uv sync --frozen` --
stays green without pulling the MCP SDK into the image's dependency tree.

The tools are thin wrappers over RiskExplainer and CreditRiskPredictor, both
covered elsewhere. What these cases pin is the part only reachable through the
protocol: the schema the calling model is handed, and the shape of what comes
back.
"""

import asyncio
import json

import pytest

from src.config import (
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
    "cb_person_cred_hist_length": 7,
}


def call(tool: str, arguments: dict | None = None) -> dict:
    """Invoke a tool the way a client would and parse its JSON payload."""
    result = asyncio.run(server.call_tool(tool, arguments or {}))
    return json.loads(result.content[0].text)


def test_both_tools_are_exposed():
    names = {tool.name for tool in asyncio.run(server.list_tools())}
    assert names == {"assess_loan_application", "get_model_info"}


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
