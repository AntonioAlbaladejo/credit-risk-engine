"""Shared test fixtures and configuration"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import sys


# Mock the model loading to avoid numpy._core import issues
@pytest.fixture(autouse=True)
def mock_model_loading(monkeypatch):
    """Mock model loading to avoid numpy compatibility issues"""

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
            # These are the 17 features that the model was trained on
            return [
                "loan_percent_income",
                "person_income",
                "loan_int_rate",
                "loan_amnt",
                "loan_grade_D",
                "person_home_ownership_MORTGAGE",
                "person_age",
                "person_emp_length",
                "person_home_ownership_OWN",
                "loan_intent_DEBTCONSOLIDATION",
                "loan_intent_MEDICAL",
                "loan_grade_E",
                "loan_grade_C",
                "default_flag",
                "loan_intent_HOMEIMPROVEMENT",
                "loan_grade_A",
                "loan_intent_EDUCATION",
            ]
        elif "preprocessor" in filename_str:
            # Return a mock preprocessor
            mock_preprocessor = Mock()

            def mock_fit_transform(X):
                """Mock fit_transform"""
                return np.random.randn(
                    len(X) if isinstance(X, list) else X.shape[0], 41
                )

            def mock_transform(X):
                """Mock transform"""
                return np.random.randn(
                    len(X) if isinstance(X, list) else X.shape[0], 41
                )

            # All 41 features that come out of the ColumnTransformer
            all_features_41 = [
                # Numeric features (10 total)
                "person_age",
                "person_income",
                "person_emp_length",
                "loan_amnt",
                "loan_int_rate",
                "loan_percent_income",
                "cb_person_cred_hist_length",
                "loan_to_income",
                "employ_to_age",
                "default_flag",
                # Categorical features (31 total)
                "person_home_ownership_RENT",
                "person_home_ownership_OWN",
                "person_home_ownership_MORTGAGE",
                "person_home_ownership_OTHER",
                "loan_intent_PERSONAL",
                "loan_intent_EDUCATION",
                "loan_intent_MEDICAL",
                "loan_intent_VENTURE",
                "loan_intent_HOMEIMPROVEMENT",
                "loan_intent_DEBTCONSOLIDATION",
                "loan_grade_A",
                "loan_grade_B",
                "loan_grade_C",
                "loan_grade_D",
                "loan_grade_E",
                "loan_grade_F",
                "loan_grade_G",
                "cb_person_default_on_file_0",
                "cb_person_default_on_file_1",
                "age_bucket_18-24",
                "age_bucket_25-34",
                "age_bucket_35-44",
                "age_bucket_45-54",
                "age_bucket_55-64",
                "age_bucket_65+",
                "emp_length_bin_0",
                "emp_length_bin_1",
                "emp_length_bin_2-3",
                "emp_length_bin_4-5",
                "emp_length_bin_6-10",
                "emp_length_bin_10+",
            ]

            def mock_get_feature_names_out():
                """Return the 41 feature names"""
                return all_features_41

            mock_preprocessor.fit_transform = mock_fit_transform
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
        "loan_int_rate": 0.08,
        "loan_percent_income": 0.1,
        "cb_person_default_on_file": 0,
        "cb_person_cred_hist_length": 8,
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
        "loan_int_rate": 0.08,
        "loan_percent_income": 0.1,
        "cb_person_default_on_file": 0,
        "cb_person_cred_hist_length": 8,
    }
