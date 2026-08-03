"""End-to-end inference against the real artifacts in models/.

Every other test in this suite runs against the autouse mock in conftest.py,
which returns random numbers from a fake preprocessor. That mock keeps the suite
fast, but it means a green run says nothing about whether the shipped bundle
still turns a raw application into the right probability: a reordered feature
list, a preprocessor from a different training run, or a stale threshold all
pass unnoticed.

The cases below are real rows from the held-out test split, with the probability
the promoted model assigned them, recorded by running the serving path itself.
They pin the whole chain -- derived features, ColumnTransformer, feature
selection and order, model, threshold -- to a fixed answer.

Regenerate after promoting a new model:
    uv run python scripts/train.py --save clean-unweighted
and re-record the expected probabilities; they are training-run specific and are
supposed to change when the bundle does.
"""

import sys
from pathlib import Path

import pytest

from src.config import (
    DATA_DIR,
    FEATURE_NAMES_PATH,
    MODEL_PATH,
    PREPROCESSOR_PATH,
    THRESHOLD_PATH,
)
from src.predictor import CreditRiskPredictor

# scripts/ is not a package, but train.py is where the hyperparameters of the
# promoted bundle are written down, so the guard below imports them from there.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from train import XGB_PARAMS  # noqa: E402

ARTIFACTS = [MODEL_PATH, PREPROCESSOR_PATH, FEATURE_NAMES_PATH, THRESHOLD_PATH]
FE_TRAIN_PATH = DATA_DIR / "credit_risk_fe_train.csv"

pytestmark = [
    pytest.mark.real_artifacts,
    pytest.mark.skipif(
        not all(p.exists() for p in ARTIFACTS),
        reason="models/ artifacts not present in this checkout",
    ),
]

# Rows drawn from the 6,336-row test split, spread across the score range so a
# shift confined to one region is still visible. `expected` is P(default) from
# models/best_tuned_model_xgboost.joblib; `actual` is the observed loan_status,
# kept for context -- the model is not expected to get every one of them right.
CASES = [
    {
        "row": 25266,
        "application": {
            "person_age": 32,
            "person_income": 33600,
            "person_home_ownership": "OWN",
            "person_emp_length": 0.0,
            "loan_intent": "EDUCATION",
            "loan_grade": "D",
            "loan_amnt": 8000,
            "loan_int_rate": 16.02,
            "loan_percent_income": 0.24,
            "cb_person_default_on_file": 0,
        },
        "expected": 0.005599,
        "actual": 0,
    },
    {
        "row": 28063,
        "application": {
            "person_age": 31,
            "person_income": 77000,
            "person_home_ownership": "MORTGAGE",
            "person_emp_length": 4.0,
            "loan_intent": "HOMEIMPROVEMENT",
            "loan_grade": "C",
            "loan_amnt": 10000,
            "loan_int_rate": 14.65,
            "loan_percent_income": 0.13,
            "cb_person_default_on_file": 0,
        },
        "expected": 0.176354,
        "actual": 0,
    },
    {
        "row": 20304,
        "application": {
            "person_age": 30,
            "person_income": 44449,
            "person_home_ownership": "MORTGAGE",
            "person_emp_length": 2.0,
            "loan_intent": "VENTURE",
            "loan_grade": "C",
            "loan_amnt": 21250,
            "loan_int_rate": 13.48,
            "loan_percent_income": 0.48,
            "cb_person_default_on_file": 1,
        },
        "expected": 0.365252,
        "actual": 0,
    },
    {
        "row": 20593,
        "application": {
            "person_age": 35,
            "person_income": 95028,
            "person_home_ownership": "RENT",
            "person_emp_length": 2.0,
            "loan_intent": "HOMEIMPROVEMENT",
            "loan_grade": "D",
            "loan_amnt": 5000,
            "loan_int_rate": 14.42,
            "loan_percent_income": 0.05,
            "cb_person_default_on_file": 1,
        },
        "expected": 0.573968,
        "actual": 1,
    },
    {
        "row": 8112,
        "application": {
            "person_age": 25,
            "person_income": 58650,
            "person_home_ownership": "RENT",
            "person_emp_length": 5.0,
            "loan_intent": "MEDICAL",
            "loan_grade": "D",
            "loan_amnt": 7500,
            "loan_int_rate": 15.37,
            "loan_percent_income": 0.11,
            "cb_person_default_on_file": 0,
        },
        "expected": 0.950414,
        "actual": 1,
    },
    {
        # Grade G, the rarest and riskiest grade: 64 rows in the whole dataset,
        # 13 in this split. It also carries a VENTURE intent and RENT ownership,
        # so it exercises the three one-hot blocks that a per-column filter used
        # to leave incomplete.
        "row": 6191,
        "application": {
            "person_age": 23,
            "person_income": 34255,
            "person_home_ownership": "RENT",
            "person_emp_length": 7.0,
            "loan_intent": "VENTURE",
            "loan_grade": "G",
            "loan_amnt": 6000,
            "loan_int_rate": 22.11,
            "loan_percent_income": 0.18,
            "cb_person_default_on_file": 1,
        },
        "expected": 0.956337,
        "actual": 1,
    },
]

EXPECTED_THRESHOLD = 0.39
EXPECTED_FEATURES = 24

# Loose enough for float noise across platforms, tight enough that any real
# change in the chain -- a swapped feature, a different scaler -- blows past it.
TOLERANCE = 1e-4


@pytest.fixture(scope="module")
def predictor():
    """The real bundle, loaded once. MLflow is off so this stays a joblib test."""
    return CreditRiskPredictor(
        model_path=MODEL_PATH,
        threshold_path=THRESHOLD_PATH,
        feature_names_path=FEATURE_NAMES_PATH,
        preprocessor_path=PREPROCESSOR_PATH,
        use_mlflow=False,
    )


def test_bundle_is_the_promoted_one(predictor):
    """Threshold and feature count identify the training run that produced it.

    A bundle mixed across runs is the failure the CASES tolerances cannot see,
    because the probabilities would move together and still look self-consistent.
    """
    assert predictor.threshold == pytest.approx(EXPECTED_THRESHOLD)
    assert len(predictor.feature_names) == EXPECTED_FEATURES


def test_every_loan_grade_scores_differently(predictor):
    """One application, seven grades, seven distinct probabilities.

    B, F and G used to return the same number to ten decimals. The importance
    filter dropped their three columns out of the seven-column one-hot block, so
    all three became the same all-zeros pattern and fused into one category that
    is 97% grade B -- meaning F (70.3% observed default rate) and G (98.4%) were
    scored as if they were B (15.9%).

    Nothing in the aggregate metrics moved: F and G together are 0.94% of the
    dataset. Only a per-grade check like this one sees it.
    """
    application = dict(CASES[0]["application"])
    scores = {
        grade: predictor.predict(application | {"loan_grade": grade})[
            "probability_default"
        ]
        for grade in "ABCDEFG"
    }

    assert len(set(scores.values())) == 7, f"grades share a score: {scores}"
    # The riskiest grade must outrank the safest; the middle is the model's call.
    assert scores["G"] > scores["A"]


def test_model_carries_the_hyperparameters_train_py_documents(predictor):
    """The served model must be the one scripts/train.py describes.

    Promoting a bundle trained with other parameters leaves the training script,
    the README and model_selection.ipynb documenting a model nobody is serving.
    """
    params = predictor.model.get_params()
    assert {k: params[k] for k in XGB_PARAMS} == pytest.approx(XGB_PARAMS)


@pytest.mark.skipif(
    not FE_TRAIN_PATH.exists(),
    reason="run notebooks/feature_engineering.ipynb to generate the split",
)
def test_feature_list_matches_the_engineered_split(predictor):
    """The notebook's feature selection and the shipped list are one selection."""
    with FE_TRAIN_PATH.open() as f:
        columns = f.readline().strip().split(",")

    assert columns == [*predictor.feature_names, "loan_status"]


@pytest.mark.parametrize("case", CASES, ids=lambda c: f"row{c['row']}")
def test_known_application_gets_known_probability(predictor, case):
    result = predictor.predict(case["application"])

    assert result["probability_default"] == pytest.approx(
        case["expected"], abs=TOLERANCE
    )
    # The decision, not just the score: catches a threshold artifact that no
    # longer belongs to this model.
    assert result["prediction"] == int(case["expected"] >= EXPECTED_THRESHOLD)
    assert result["probability_non_default"] == pytest.approx(
        1 - result["probability_default"]
    )


def test_batch_matches_single(predictor):
    """batch_preprocess is a separate implementation; it must not diverge."""
    applications = [c["application"] for c in CASES]
    batch = predictor.batch_predict(applications)

    assert len(batch) == len(CASES)
    for result, case in zip(batch, CASES, strict=True):
        assert result["probability_default"] == pytest.approx(
            case["expected"], abs=TOLERANCE
        )
