import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
import logging
from src.config import (
    MIN_AGE,
    MAX_AGE,
    MIN_EMP_LENGTH,
    MAX_EMP_LENGTH,
    MIN_LOAN_AMOUNT,
    MAX_LOAN_AMOUNT,
)

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Handle the transformation of raw input data to the format expected by the model"""

    # Valid values for categorical features
    VALID_HOME_OWNERSHIP = ["RENT", "OWN", "MORTGAGE", "OTHER"]
    VALID_LOAN_INTENT = [
        "PERSONAL",
        "EDUCATION",
        "MEDICAL",
        "VENTURE",
        "HOMEIMPROVEMENT",
        "DEBTCONSOLIDATION",
    ]
    VALID_LOAN_GRADE = ["A", "B", "C", "D", "E", "F", "G"]
    VALID_DEFAULT_FILE_FLAG = [0, 1]

    def __init__(self, preprocessor_path: Path, feature_names_path: Path):
        """
        Initialize the preprocessor

        Args:
            preprocessor_path: Path to the saved preprocessor (sklearn ColumnTransformer)
            feature_names_path: Path to the names of transformed features
        """
        try:
            self.preprocessor = joblib.load(preprocessor_path)
            self.feature_names = joblib.load(feature_names_path)
            logger.info("Preprocessor loaded successfully")
        except Exception as e:
            logger.error(f"Error loading preprocessor: {e}")
            raise

    def _validate_input(self, data: Dict) -> None:
        """
        Validates that the input data is correct

        Args:
            data: Dictionary with the input data

        Raises:
            ValueError: If there are invalid values
        """
        # Validate categorical features
        if data.get("person_home_ownership") not in self.VALID_HOME_OWNERSHIP:
            raise ValueError(
                f"person_home_ownership must be one of {self.VALID_HOME_OWNERSHIP}, "
                f"received: {data.get('person_home_ownership')}"
            )

        if data.get("loan_intent") not in self.VALID_LOAN_INTENT:
            raise ValueError(
                f"loan_intent must be one of {self.VALID_LOAN_INTENT}, "
                f"received: {data.get('loan_intent')}"
            )

        if data.get("loan_grade") not in self.VALID_LOAN_GRADE:
            raise ValueError(
                f"loan_grade must be one of {self.VALID_LOAN_GRADE}, "
                f"received: {data.get('loan_grade')}"
            )

        if data.get("cb_person_default_on_file") not in self.VALID_DEFAULT_FILE_FLAG:
            raise ValueError(
                f"cb_person_default_on_file must be one of {self.VALID_DEFAULT_FILE_FLAG}, "
                f"received: {data.get('cb_person_default_on_file')}"
            )

        # Validate numerical ranges
        if data.get("person_age") < MIN_AGE or data.get("person_age") > MAX_AGE:
            raise ValueError(f"person_age must be between {MIN_AGE} and {MAX_AGE}")

        if (
            data.get("person_emp_length") < MIN_EMP_LENGTH
            or data.get("person_emp_length") > MAX_EMP_LENGTH
        ):
            raise ValueError(
                f"person_emp_length must be between {MIN_EMP_LENGTH} and {MAX_EMP_LENGTH}"
            )

        if (
            data.get("loan_amnt") < MIN_LOAN_AMOUNT
            or data.get("loan_amnt") > MAX_LOAN_AMOUNT
        ):
            raise ValueError(
                f"loan_amnt must be between {MIN_LOAN_AMOUNT} and {MAX_LOAN_AMOUNT}"
            )

        if data.get("person_income") <= 0:
            raise ValueError(f"person_income must be greater than 0")

        if data.get("loan_amnt") <= 0:
            raise ValueError(f"loan_amnt must be greater than 0")

        if data.get("loan_int_rate") < 0:
            raise ValueError(f"loan_int_rate must be a positive value")

        if data.get("loan_percent_income") < 0 or data.get("loan_percent_income") > 1:
            raise ValueError(f"loan_percent_income must be between 0 and 1")

    def _create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates derived features from the original features (as done in feature engineering notebook)

        Args:
            df: DataFrame with original features

        Returns:
            DataFrame with derived features added
        """
        df_fe = df.copy()

        # Loan to income ratio
        df_fe["loan_to_income"] = df_fe["loan_amnt"] / (
            df_fe["person_income"].replace(0, np.nan)
        )
        # Employment length to age ratio
        df_fe["employ_to_age"] = df_fe["person_emp_length"] / (
            df_fe["person_age"].replace(0, np.nan)
        )

        # Age buckets
        df_fe["age_bucket"] = pd.cut(
            df_fe["person_age"],
            bins=[18, 25, 35, 45, 55, 65, 120],
            labels=["18-24", "25-34", "35-44", "45-54", "55-64", "65+"],
        )

        # Employment length bins
        df_fe["emp_length_bin"] = pd.cut(
            df_fe["person_emp_length"].fillna(0),
            bins=[-1, 0, 1, 3, 5, 10, 100],
            labels=["0", "1", "2-3", "4-5", "6-10", "10+"],
        )

        # Default flag
        df_fe["default_flag"] = df_fe["cb_person_default_on_file"].astype(int)

        return df_fe

    def preprocess(self, data: Dict) -> pd.DataFrame:
        """
        Transforms raw input data into the format expected by the model.

        Args:
            data: Dictionary with raw features

        Returns:
            DataFrame transformed with features in the expected order
        """
        try:
            # Validate input data
            self._validate_input(data)
            logger.debug(f"Input data validated successfully")

            # Convert dictionary to DataFrame
            df = pd.DataFrame([data])

            # Create derived features
            df_fe = self._create_derived_features(df)
            logger.debug(f"Derived features created")

            # Apply preprocessor (imputation, scaling, one-hot encoding)
            X_transformed = self.preprocessor.transform(df_fe)

            # Convert to DataFrame selecting only the features expected by the model
            X_df = pd.DataFrame(X_transformed, columns=self.feature_names)
            logger.debug(f"Data transformed successfully")

            return X_df

        except Exception as e:
            logger.error(f"Error preprocessing: {e}")
            raise

    def batch_preprocess(self, data_list: List[Dict]) -> pd.DataFrame:
        """
        Transforms multiple records into dataframe format expected by the model.

        Args:
            data_list: List of dictionaries with raw features

        Returns:
            DataFrame transformed with all records
        """
        try:
            # Validate each record in the batch
            for i, data in enumerate(data_list):
                try:
                    self._validate_input(data)
                except ValueError as e:
                    raise ValueError(f"Error in record {i}: {str(e)}")

            # Convert dictionary list to DataFrame
            df = pd.DataFrame(data_list)

            # Create derived features
            df_fe = self._create_derived_features(df)

            # Apply preprocessor
            X_transformed = self.preprocessor.transform(df_fe)

            # Convert to DataFrame
            X_df = pd.DataFrame(X_transformed, columns=self.feature_names)
            logger.debug(f"Batch of {len(data_list)} records transformed successfully")

            return X_df

        except Exception as e:
            logger.error(f"Error in batch preprocessing: {e}")
            raise

    def get_feature_info(self) -> Dict:
        """Returns information about the expected features"""
        return {
            "valid_home_ownership": self.VALID_HOME_OWNERSHIP,
            "valid_loan_intent": self.VALID_LOAN_INTENT,
            "valid_loan_grade": self.VALID_LOAN_GRADE,
            "num_features_after_preprocessing": len(self.feature_names),
            "final_features": self.feature_names,
        }
