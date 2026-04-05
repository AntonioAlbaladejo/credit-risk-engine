"""Tests for data preprocessing module"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from src.preprocessing import DataPreprocessor
from src.config import PREPROCESSOR_PATH, FEATURE_NAMES_PATH


class TestDataPreprocessorInitialization:
    """Test DataPreprocessor initialization"""

    def test_preprocessor_initialization(self):
        """Test successful preprocessor initialization"""
        preprocessor = DataPreprocessor(PREPROCESSOR_PATH, FEATURE_NAMES_PATH)
        assert preprocessor.preprocessor is not None
        assert preprocessor.feature_names is not None
        assert len(preprocessor.feature_names) > 0

    def test_feature_names_are_list(self):
        """Test that feature names are a list"""
        preprocessor = DataPreprocessor(PREPROCESSOR_PATH, FEATURE_NAMES_PATH)
        assert isinstance(preprocessor.feature_names, (list, np.ndarray))


class TestDataValidation:
    """Test data validation"""

    @pytest.fixture
    def preprocessor(self):
        """Initialize preprocessor"""
        return DataPreprocessor(PREPROCESSOR_PATH, FEATURE_NAMES_PATH)

    @pytest.fixture
    def valid_data(self):
        """Sample valid data"""
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

    def test_valid_data_passes_validation(self, preprocessor, valid_data):
        """Test that valid data passes validation"""
        # Should not raise any exception
        preprocessor._validate_input(valid_data)

    def test_zero_income_valid(self, preprocessor, valid_data):
        """Test that zero income is valid in preprocessing"""
        valid_data["person_income"] = 0
        # Should not raise any exception
        preprocessor._validate_input(valid_data)

    def test_zero_income_in_preprocess(self, preprocessor, valid_data):
        """Test preprocessing handles zero income (loan_to_income ratio)"""
        valid_data["person_income"] = 0
        result = preprocessor.preprocess(valid_data)
        # Should return processed data with no NaN values
        assert not result.isnull().any().any()
        assert result.shape[0] == 1

    def test_invalid_home_ownership(self, preprocessor, valid_data):
        """Test validation fails with invalid home ownership"""
        valid_data["person_home_ownership"] = "INVALID"
        with pytest.raises(ValueError, match="person_home_ownership"):
            preprocessor._validate_input(valid_data)

    def test_invalid_loan_intent(self, preprocessor, valid_data):
        """Test validation fails with invalid loan intent"""
        valid_data["loan_intent"] = "INVALID"
        with pytest.raises(ValueError, match="loan_intent"):
            preprocessor._validate_input(valid_data)

    def test_invalid_loan_grade(self, preprocessor, valid_data):
        """Test validation fails with invalid loan grade"""
        valid_data["loan_grade"] = "H"
        with pytest.raises(ValueError, match="loan_grade"):
            preprocessor._validate_input(valid_data)

    def test_age_below_minimum(self, preprocessor, valid_data):
        """Test validation fails with age below minimum"""
        valid_data["person_age"] = 15
        with pytest.raises(ValueError, match="person_age"):
            preprocessor._validate_input(valid_data)

    def test_age_above_maximum(self, preprocessor, valid_data):
        """Test validation fails with age above maximum"""
        valid_data["person_age"] = 110
        with pytest.raises(ValueError, match="person_age"):
            preprocessor._validate_input(valid_data)

    def test_negative_income(self, preprocessor, valid_data):
        """Test validation fails with negative income"""
        valid_data["person_income"] = -1000
        with pytest.raises(ValueError, match="person_income"):
            preprocessor._validate_input(valid_data)

    def test_zero_income_valid(self, preprocessor, valid_data):
        """Test that zero income is valid in validation"""
        valid_data["person_income"] = 0
        # Should not raise any exception
        preprocessor._validate_input(valid_data)

    def test_loan_amount_below_minimum(self, preprocessor, valid_data):
        """Test validation fails with loan amount below minimum"""
        valid_data["loan_amnt"] = 100
        with pytest.raises(ValueError, match="loan_amnt"):
            preprocessor._validate_input(valid_data)

    def test_invalid_default_flag(self, preprocessor, valid_data):
        """Test validation fails with invalid default flag"""
        valid_data["cb_person_default_on_file"] = 2
        with pytest.raises(ValueError, match="cb_person_default_on_file"):
            preprocessor._validate_input(valid_data)


class TestPreprocessing:
    """Test preprocessing functionality"""

    @pytest.fixture
    def preprocessor(self):
        """Initialize preprocessor"""
        return DataPreprocessor(PREPROCESSOR_PATH, FEATURE_NAMES_PATH)

    @pytest.fixture
    def valid_data(self):
        """Sample valid data"""
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

    def test_preprocess_valid_data(self, preprocessor, valid_data):
        """Test preprocessing valid data"""
        result = preprocessor.preprocess(valid_data)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        # Preprocessed data should have multiple columns (scaled + one-hot encoded)
        assert result.shape[1] > 0

    def test_preprocess_returns_correct_shape(self, preprocessor, valid_data):
        """Test that preprocessed data has correct shape"""
        result = preprocessor.preprocess(valid_data)
        assert result.shape[0] == 1
        # Preprocessed data should have multiple columns after one-hot encoding
        assert result.shape[1] > len(valid_data)

    def test_preprocess_no_missing_values(self, preprocessor, valid_data):
        """Test that preprocessed data has no missing values"""
        result = preprocessor.preprocess(valid_data)
        assert not result.isnull().any().any()

    def test_batch_preprocess_valid_data(self, preprocessor, valid_data):
        """Test batch preprocessing with valid data"""
        data_list = [valid_data, valid_data]
        result = preprocessor.batch_preprocess(data_list)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        # Preprocessed data should have multiple columns after one-hot encoding
        assert result.shape[1] > 0

    def test_batch_preprocess_single_record(self, preprocessor, valid_data):
        """Test batch preprocessing with single record"""
        result = preprocessor.batch_preprocess([valid_data])
        assert len(result) == 1
        # Preprocessed data should have multiple columns after one-hot encoding
        assert result.shape[1] > 0

    def test_batch_preprocess_invalid_record(self, preprocessor, valid_data):
        """Test batch preprocessing fails with invalid record"""
        invalid_data = valid_data.copy()
        invalid_data["person_age"] = 10
        with pytest.raises(ValueError):
            preprocessor.batch_preprocess([valid_data, invalid_data])

    def test_get_feature_info(self, preprocessor):
        """Test getting feature information"""
        info = preprocessor.get_feature_info()
        assert "valid_home_ownership" in info
        assert "valid_loan_intent" in info
        assert "valid_loan_grade" in info
        assert "num_features_after_preprocessing" in info
        assert "final_features" in info

    def test_zero_income_in_preprocess(self, preprocessor, valid_data):
        """Test preprocessing handles zero income (loan_to_income ratio)"""
        valid_data["person_income"] = 0
        result = preprocessor.preprocess(valid_data)
        # Should return processed data with no NaN values
        assert not result.isnull().any().any()
        assert result.shape[0] == 1

    def test_zero_income_batch_preprocess(self, preprocessor, valid_data):
        """Test batch preprocessing handles zero income correctly"""
        valid_data["person_income"] = 0
        result = preprocessor.batch_preprocess([valid_data])
        # Should return processed data with no NaN values
        assert not result.isnull().any().any()
        assert result.shape[0] == 1
