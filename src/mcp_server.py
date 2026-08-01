"""MCP server exposing the credit risk model to LLM clients.

Model Context Protocol clients (Claude Desktop, Claude Code, ...) launch this
module as a subprocess and speak JSON-RPC over stdin/stdout. The client asks
which tools exist, receives their names, descriptions and input schemas, and
lets its own model decide when to call them -- there is no LLM on this side.

Run it directly for a manual smoke test::

    uv run python -m src.mcp_server

It will appear to hang: that is a server waiting for JSON-RPC frames on stdin,
not a failure.

The tool docstrings below are prompt surface, not developer documentation --
they are what the calling model reads to decide whether a tool applies and how
to read its result. Changing them changes model behaviour.
"""

import logging

from mcp.server.mcpserver import MCPServer

from src.api.schemas import LoanApplication
from src.config import (
    API_VERSION,
    FEATURE_NAMES_PATH,
    LOG_LEVEL,
    MODEL_PATH,
    PREPROCESSOR_PATH,
    THRESHOLD_PATH,
)
from src.explainer import RiskExplainer
from src.predictor import CreditRiskPredictor

# Logs go to stderr on purpose. stdout carries the JSON-RPC frames, so anything
# printed there corrupts the protocol.
logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)

server = MCPServer(
    name="credit-risk-engine",
    version=API_VERSION,
    instructions=(
        "Scores consumer loan applications for probability of default and "
        "explains the drivers behind each decision. Answer from tool results "
        "only: never estimate a default probability yourself, and never invent "
        "a reason the tool did not return."
    ),
)

_explainer: RiskExplainer | None = None


def get_explainer() -> RiskExplainer:
    """Load the promoted bundle once, on first use.

    Loading at import time would run inside every test collection and inside
    any client that merely inspects the module, so it is deferred -- the same
    lazy pattern the FastAPI app uses.
    """
    global _explainer
    if _explainer is None:
        predictor = CreditRiskPredictor(
            MODEL_PATH,
            THRESHOLD_PATH,
            FEATURE_NAMES_PATH,
            PREPROCESSOR_PATH,
            # Exact TreeSHAP needs the native booster, which an MLflow pyfunc
            # wrapper does not expose.
            use_mlflow=False,
        )
        _explainer = RiskExplainer(predictor)
        logger.info("Credit risk bundle loaded")
    return _explainer


@server.tool()
def assess_loan_application(application: LoanApplication) -> dict:
    """Score a loan application and explain the decision.

    Call this whenever the user gives applicant details and asks whether the
    loan would be approved, how risky it is, or why a decision came out the
    way it did. It answers all three in one call.

    Returns the probability of default, the decision at the model's tuned
    threshold, and the reason codes that drove it, strongest first.

    Reading `reason_codes` correctly:
      * `contribution` is in log-odds, not probability. Contributions add up,
        but they are NOT shares of the probability -- never present one as a
        percentage of the risk, and never sum them into a percentage.
      * `direction` and the ordering are always safe to report: a positive
        contribution pushes the application towards default, a negative one
        away from it, and the first entry is the strongest driver.
      * A reason code is a group of related applicant attributes, so its net
        can be small even when the attributes inside it matter individually.
    """
    result = get_explainer().explain(application.model_dump(mode="json"))
    return {
        "probability_default": round(result["probability_default"], 4),
        "threshold": result["threshold_used"],
        "decision": (
            "reject"
            if result["probability_default"] >= result["threshold_used"]
            else "approve"
        ),
        "average_applicant_probability": round(result["base_probability"], 4),
        "reason_codes": [
            {
                "reason": group["reason"],
                "contribution": round(group["contribution"], 4),
                "direction": group["direction"],
            }
            for group in result["reason_codes"]
        ],
    }


@server.tool()
def get_model_info() -> dict:
    """Describe the scoring model itself.

    Call this for questions about the model rather than about an applicant --
    what it was trained on, which attributes it uses, or where its decision
    threshold sits. Useful for grounding an answer before scoring anything.
    """
    info = get_explainer().predictor.get_model_info()
    return {
        "model_type": info["model_type"],
        "decision_threshold": info["threshold"],
        # Tuned on validation data, not left at 0.5: below it the application
        # is approved, at or above it rejected.
        "threshold_note": "Applications scoring at or above the threshold are rejected.",
        "num_features": info["num_features"],
        "features": info["features"],
    }


if __name__ == "__main__":
    server.run(transport="stdio")
