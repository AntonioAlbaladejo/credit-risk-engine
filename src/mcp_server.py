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
from src.retriever import CorpusRetriever

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
        "a reason the tool did not return. It also searches the GDPR and the "
        "EU AI Act for the provisions bearing on a question; the same rule "
        "applies there, and an empty regulatory result means the corpus does "
        "not cover the question, not that no law exists."
    ),
)

_explainer: RiskExplainer | None = None
_retriever: CorpusRetriever | None = None


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


def get_retriever() -> CorpusRetriever:
    """Load the regulatory corpus and its index once, on first use.

    Kept separate from the scoring bundle so that each is paid for only when
    used: a client that never asks about regulation does not load the
    embedding model, and a checkout with no index built still serves the two
    scoring tools.

    Returns:
        A retriever over the corpus, abstaining below `MIN_SCORE`.

    Raises:
        FileNotFoundError: The corpus or its index has not been built. Run
            `uv run python -m scripts.ingest_corpus`.
    """
    global _retriever
    if _retriever is None:
        _retriever = CorpusRetriever.from_files()
        logger.info("Regulatory corpus loaded")
    return _retriever


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


@server.tool()
def search_regulation(question: str, hypothetical_passage: str = "") -> dict:
    """Find the passages of EU law that bear on a question about this system.

    Covers the GDPR and the EU AI Act in full. Call it whenever the user asks
    what the law requires, permits or forbids -- automated decisions, the right
    to an explanation, high-risk classification, record-keeping, human
    oversight -- so that the answer cites the provision instead of recalling
    it.

    Always fill in `hypothetical_passage`. Before calling, write the provision
    you would expect to find if the answer existed: two or three sentences in
    the register of EU legislation ("shall", "the controller", "providers of
    high-risk AI systems"), with no invented article numbers. It is matched
    against the corpus in place of the question, and questions are written in
    business language the legislation never uses, so this roughly halves the
    passages the search misses. Write it even when unsure what the law says --
    a wrong guess in the right register still retrieves better than the
    question alone, and `question` alone decides whether anything is returned,
    so a bad guess cannot make the tool answer something it should not.

    Quote and cite only what comes back. Every passage carries the `citation`
    naming it and the `source_url` and `retrieved_on` that let a reader check
    it. A claim about the law that no returned passage supports must not be
    presented as grounded, however confident you are that it is true.

    An empty `passages` list is an answer, not a failure. It means either that
    nothing in the corpus is close enough, or that the question asks what this
    organisation actually did rather than what the law requires -- legislation
    states requirements and holds no record of anyone's compliance with them.
    Say so rather than answering from your own knowledge of the law.

    The corpus is EU legislation and nothing else. It holds no internal policy,
    no record of what this organisation has actually done, and no US
    regulation, so it cannot say what *we* do -- only what the law requires.
    """
    return get_retriever().search_payload(
        question, hypothetical_passage=hypothetical_passage
    )


if __name__ == "__main__":
    server.run(transport="stdio")
