"""Explanations checked against the real artifacts in models/.

The autouse mock in conftest.py returns a Mock with no ``get_booster``, so an
explainer test running under it would prove nothing at all. These are marked
``real_artifacts`` for the same reason test_inference_real.py is.
"""

import pytest

from src.config import (
    FEATURE_NAMES_PATH,
    MODEL_PATH,
    PREPROCESSOR_PATH,
    THRESHOLD_PATH,
)
from src.explainer import REASON_GROUPS, RiskExplainer
from src.predictor import CreditRiskPredictor

ARTIFACTS = [MODEL_PATH, PREPROCESSOR_PATH, FEATURE_NAMES_PATH, THRESHOLD_PATH]

pytestmark = [
    pytest.mark.real_artifacts,
    pytest.mark.skipif(
        not all(p.exists() for p in ARTIFACTS),
        reason="models/ artifacts not present in this checkout",
    ),
]

# Same applicant, two affordability profiles: a 24% loan-to-income on a low
# salary against a 5% one on a high salary. Everything else is held fixed.
STRESSED = {
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
COMFORTABLE = {**STRESSED, "person_income": 160000, "loan_percent_income": 0.05}


@pytest.fixture(scope="module")
def predictor():
    return CreditRiskPredictor(
        model_path=MODEL_PATH,
        threshold_path=THRESHOLD_PATH,
        feature_names_path=FEATURE_NAMES_PATH,
        preprocessor_path=PREPROCESSOR_PATH,
        use_mlflow=False,
    )


@pytest.fixture(scope="module")
def explainer(predictor):
    return RiskExplainer(predictor)


def test_taxonomy_covers_the_promoted_bundle(explainer):
    """Construction is the guard: a retrain that changes features must fail."""
    grouped = [name for names in REASON_GROUPS.values() for name in names]
    assert sorted(grouped) == sorted(explainer.feature_names)
    assert len(grouped) == len(set(grouped)), "a feature is in two reason codes"


def test_contributions_add_up_to_the_margin(explainer):
    """SHAP additivity. Without it the numbers are not Shapley values at all."""
    result = explainer.explain(STRESSED)
    total = result["base_value"] + sum(result["feature_contributions"].values())
    assert total == pytest.approx(result["margin"], abs=1e-5)


def test_reason_codes_partition_the_feature_contributions(explainer):
    result = explainer.explain(STRESSED)
    grouped_total = sum(g["contribution"] for g in result["reason_codes"])
    assert grouped_total == pytest.approx(
        sum(result["feature_contributions"].values()), abs=1e-5
    )
    assert len(result["reason_codes"]) == len(REASON_GROUPS)


def test_explained_probability_matches_the_served_prediction(explainer, predictor):
    """The explanation must describe the decision the API actually returns.

    Explaining a probability that differs from the served one is the failure
    mode that would make every generated narrative subtly wrong.
    """
    for application in (STRESSED, COMFORTABLE):
        served = predictor.predict(application)["probability_default"]
        assert explainer.explain(application)["probability_default"] == pytest.approx(
            served, abs=1e-6
        )


def test_reason_codes_are_ranked_by_magnitude(explainer):
    magnitudes = [
        abs(g["contribution"]) for g in explainer.explain(STRESSED)["reason_codes"]
    ]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_affordability_flips_direction_with_the_income_profile(explainer):
    """Directional sanity: the same loan against 33.6k and against 160k."""
    stressed = _reason(explainer.explain(STRESSED), "affordability")
    comfortable = _reason(explainer.explain(COMFORTABLE), "affordability")

    assert stressed["direction"] == "increases_risk"
    assert comfortable["direction"] == "decreases_risk"
    assert comfortable["contribution"] < stressed["contribution"]


def test_direction_agrees_with_the_sign_of_every_contribution(explainer):
    """The sigmoid is monotonic, so sign is the one thing always safe to report."""
    for group in explainer.explain(STRESSED)["reason_codes"]:
        expected = "increases_risk" if group["contribution"] > 0 else "decreases_risk"
        assert group["direction"] == expected


def test_rejects_a_model_without_a_booster(predictor):
    """An MLflow pyfunc bundle has no exact TreeSHAP; fail loudly, not silently."""

    class Wrapped:
        def predict(self, X):
            raise NotImplementedError

    stub = type("Stub", (), {"model": Wrapped(), "feature_names": ["a"]})()
    with pytest.raises(TypeError, match="native XGBoost model"):
        RiskExplainer(stub)


def _reason(result: dict, code: str) -> dict:
    return next(g for g in result["reason_codes"] if g["reason"] == code)
