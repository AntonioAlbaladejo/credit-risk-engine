"""Shared test fixtures and configuration"""

from unittest.mock import Mock

import numpy as np
import pytest

# The columns the fitted ColumnTransformer emits, in its order, and the subset
# the promoted bundle selects. Kept in step with models/*.joblib by
# tests/test_inference_real.py, which loads the real artifacts and compares.
ENCODED_FEATURES = [
    "person_age",
    "person_income",
    "person_emp_length",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "loan_to_income",
    "employ_to_age",
    "default_flag",
    "person_home_ownership_MORTGAGE",
    "person_home_ownership_OTHER",
    "person_home_ownership_OWN",
    "person_home_ownership_RENT",
    "loan_intent_DEBTCONSOLIDATION",
    "loan_intent_EDUCATION",
    "loan_intent_HOMEIMPROVEMENT",
    "loan_intent_MEDICAL",
    "loan_intent_PERSONAL",
    "loan_intent_VENTURE",
    "loan_grade_A",
    "loan_grade_B",
    "loan_grade_C",
    "loan_grade_D",
    "loan_grade_E",
    "loan_grade_F",
    "loan_grade_G",
    "cb_person_default_on_file_N",
    "cb_person_default_on_file_Y",
    "age_bucket_18-24",
    "age_bucket_25-34",
    "age_bucket_35-44",
    "age_bucket_45-54",
    "age_bucket_55-64",
    "age_bucket_65+",
    "emp_length_bin_0",
    "emp_length_bin_1",
    "emp_length_bin_10+",
    "emp_length_bin_2-3",
    "emp_length_bin_4-5",
    "emp_length_bin_6-10",
]

SELECTED_FEATURES = [
    "person_age",
    "person_income",
    "person_emp_length",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "default_flag",
    "person_home_ownership_MORTGAGE",
    "person_home_ownership_OTHER",
    "person_home_ownership_OWN",
    "person_home_ownership_RENT",
    "loan_intent_DEBTCONSOLIDATION",
    "loan_intent_EDUCATION",
    "loan_intent_HOMEIMPROVEMENT",
    "loan_intent_MEDICAL",
    "loan_intent_PERSONAL",
    "loan_intent_VENTURE",
    "loan_grade_A",
    "loan_grade_B",
    "loan_grade_C",
    "loan_grade_D",
    "loan_grade_E",
    "loan_grade_F",
    "loan_grade_G",
]


# Mock the model loading to avoid numpy._core import issues
@pytest.fixture(autouse=True)
def mock_model_loading(request, monkeypatch):
    """Mock model loading to avoid numpy compatibility issues"""

    # Tests marked real_artifacts load models/*.joblib on purpose; mocking
    # joblib.load for them would remove the only thing they verify.
    if request.node.get_closest_marker("real_artifacts"):
        return

    # Create a mock model that behaves like XGBoost
    mock_model = Mock()

    # Make predict_proba return reasonable probabilities
    def mock_predict_proba(X):
        """Mock predict_proba method"""
        n_samples = X.shape[0] if len(X.shape) > 1 else 1
        # Return probabilities [prob_non_default, prob_default]
        probs_default = np.random.uniform(0.2, 0.8, n_samples)
        probs_non_default = 1 - probs_default
        return np.column_stack([probs_non_default, probs_default])

    mock_model.predict_proba = mock_predict_proba

    # Patch joblib.load to return our mocks
    import joblib

    original_load = joblib.load

    def mocked_load(filename):
        """Return appropriate mock based on filename"""
        filename_str = str(filename)
        if "model" in filename_str and "best_tuned_model" in filename_str:
            return mock_model
        elif "threshold" in filename_str:
            return 0.5  # Default threshold
        elif "feature_names" in filename_str:
            # The features the promoted bundle selects, in its order.
            return SELECTED_FEATURES
        elif "preprocessor" in filename_str:
            # Return a mock preprocessor
            mock_preprocessor = Mock()

            def mock_transform(X):
                """Mock transform: right shape, meaningless values"""
                rows = len(X) if isinstance(X, list) else X.shape[0]
                return np.random.randn(rows, len(ENCODED_FEATURES))

            def mock_get_feature_names_out():
                """The columns transform() returns, in order.

                Unprefixed: the num__/cat__ stripping is pinned against the real
                encoder in test_preprocessing.py, not faked here.
                """
                return ENCODED_FEATURES

            mock_preprocessor.fit_transform = mock_transform
            mock_preprocessor.transform = mock_transform
            mock_preprocessor.get_feature_names_out = mock_get_feature_names_out
            return mock_preprocessor
        else:
            return original_load(filename)

    monkeypatch.setattr(joblib, "load", mocked_load)


@pytest.fixture
def valid_application():
    """Sample valid loan application"""
    return {
        "person_age": 35,
        "person_income": 50000,
        "person_home_ownership": "OWN",
        "person_emp_length": 10,
        "loan_intent": "PERSONAL",
        "loan_grade": "A",
        "loan_amnt": 5000,
        "loan_int_rate": 11.5,
        "loan_percent_income": 0.1,
        "cb_person_default_on_file": 0,
    }


@pytest.fixture
def valid_data():
    """Sample valid data for preprocessing"""
    return {
        "person_age": 35,
        "person_income": 50000,
        "person_emp_length": 10,
        "person_home_ownership": "OWN",
        "loan_intent": "PERSONAL",
        "loan_grade": "A",
        "loan_amnt": 5000,
        "loan_int_rate": 11.5,
        "loan_percent_income": 0.1,
        "cb_person_default_on_file": 0,
    }
