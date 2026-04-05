"""Tests for configuration module"""

import pytest
from pathlib import Path
from src.config import (
    BASE_DIR,
    MODELS_DIR,
    DATA_DIR,
    MODEL_PATH,
    THRESHOLD_PATH,
    FEATURE_NAMES_PATH,
    PREPROCESSOR_PATH,
    API_TITLE,
    API_VERSION,
    MIN_AGE,
    MAX_AGE,
    MIN_EMP_LENGTH,
    MAX_EMP_LENGTH,
    MIN_LOAN_AMOUNT,
    MAX_LOAN_AMOUNT,
)


class TestConfigPaths:
    """Test configuration paths"""

    def test_base_dir_exists(self):
        """Test that base directory exists"""
        assert BASE_DIR.exists()
        assert BASE_DIR.is_dir()

    def test_models_dir_exists(self):
        """Test that models directory exists"""
        assert MODELS_DIR.exists()
        assert MODELS_DIR.is_dir()

    def test_data_dir_exists(self):
        """Test that data directory exists"""
        assert DATA_DIR.exists()
        assert DATA_DIR.is_dir()

    def test_model_paths_are_paths(self):
        """Test that model paths are Path objects"""
        assert isinstance(MODEL_PATH, Path)
        assert isinstance(THRESHOLD_PATH, Path)
        assert isinstance(FEATURE_NAMES_PATH, Path)
        assert isinstance(PREPROCESSOR_PATH, Path)

    def test_model_files_exist(self):
        """Test that required model files exist"""
        assert MODEL_PATH.exists(), f"Model file not found at {MODEL_PATH}"
        assert THRESHOLD_PATH.exists(), f"Threshold file not found at {THRESHOLD_PATH}"
        assert FEATURE_NAMES_PATH.exists(), (
            f"Feature names file not found at {FEATURE_NAMES_PATH}"
        )
        assert PREPROCESSOR_PATH.exists(), (
            f"Preprocessor file not found at {PREPROCESSOR_PATH}"
        )


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
