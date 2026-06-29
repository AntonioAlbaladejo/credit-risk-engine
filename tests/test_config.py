"""Tests for configuration module"""

from pathlib import Path

from src.config import (
    API_TITLE,
    API_VERSION,
    BASE_DIR,
    DATA_DIR,
    FEATURE_NAMES_PATH,
    MAX_AGE,
    MAX_EMP_LENGTH,
    MAX_LOAN_AMOUNT,
    MIN_AGE,
    MIN_EMP_LENGTH,
    MIN_LOAN_AMOUNT,
    MODEL_PATH,
    MODELS_DIR,
    PREPROCESSOR_PATH,
    THRESHOLD_PATH,
)


class TestConfigPaths:
    """Test configuration paths"""

    def test_base_dir_exists(self):
        """Test that base directory exists"""
        assert BASE_DIR.exists()
        assert BASE_DIR.is_dir()

    def test_models_dir_path(self):
        """Test that models directory path is valid"""
        assert isinstance(MODELS_DIR, Path)
        assert str(MODELS_DIR).endswith("models")

    def test_data_dir_path(self):
        """Test that data directory path is valid"""
        assert isinstance(DATA_DIR, Path)
        assert str(DATA_DIR).endswith("data")

    def test_model_paths_are_paths(self):
        """Test that model paths are Path objects"""
        assert isinstance(MODEL_PATH, Path)
        assert isinstance(THRESHOLD_PATH, Path)
        assert isinstance(FEATURE_NAMES_PATH, Path)
        assert isinstance(PREPROCESSOR_PATH, Path)

    def test_model_paths_have_correct_names(self):
        """Test that model file names are correct"""
        assert MODEL_PATH.name == "best_tuned_model_xgboost.joblib"
        assert THRESHOLD_PATH.name == "optimal_threshold.joblib"
        assert FEATURE_NAMES_PATH.name == "feature_names.joblib"
        assert PREPROCESSOR_PATH.name == "preprocessor.joblib"


class TestAPIConfig:
    """Test API configuration"""

    def test_api_title(self):
        """Test API title is set"""
        assert API_TITLE == "Credit Risk Engine API"

    def test_api_version(self):
        """Test API version is set"""
        assert API_VERSION == "1.0.0"


class TestValidationLimits:
    """Test input validation limits"""

    def test_age_limits(self):
        """Test age validation limits"""
        assert MIN_AGE == 18
        assert MAX_AGE == 100
        assert MIN_AGE < MAX_AGE

    def test_employment_limits(self):
        """Test employment length validation limits"""
        assert MIN_EMP_LENGTH == 0
        assert MAX_EMP_LENGTH == 80
        assert MIN_EMP_LENGTH <= MAX_EMP_LENGTH

    def test_loan_amount_limits(self):
        """Test loan amount validation limits"""
        assert MIN_LOAN_AMOUNT == 500
        assert MAX_LOAN_AMOUNT == 100000
        assert MIN_LOAN_AMOUNT < MAX_LOAN_AMOUNT
