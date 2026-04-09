import joblib
import pandas as pd
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

        if data.get("person_income") < 0:
            raise ValueError("person_income must be non-negative (0 is allowed)")

        if data.get("loan_int_rate") < 0:
            raise ValueError("loan_int_rate must be a positive value")

        if data.get("loan_percent_income") < 0 or data.get("loan_percent_income") > 1:
            raise ValueError("loan_percent_income must be between 0 and 1")

    def _get_feature_names_from_preprocessor(self) -> List[str]:
        """
        Reconstructs feature names from the ColumnTransformer since get_feature_names_out()
        may not be available due to sklearn version incompatibility.

        The order matches the ColumnTransformer structure:
        1. Numeric features (scaled, not renamed): 11 total
        2. Categorical features (one-hot encoded): 29 total (4+6+7+6+6)

        Returns:
            List of 41 feature names in the order produced by preprocessor.transform()
        """
        try:
            # Try to get feature names directly (works with newer sklearn)
            return list(self.preprocessor.get_feature_names_out())
        except (AttributeError, TypeError):
            pass

        # Fallback: reconstruct feature names based on ColumnTransformer structure
        # This is necessary due to sklearn 1.2.2 -> 1.7.2 version mismatch

        feature_names = []

        # Numeric features (10 total) - passed through StandardScaler, names unchanged
        numeric_features = [
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
        ]
        feature_names.extend(numeric_features)

        # Categorical features (31 total after one-hot encoding)
        # Order: person_home_ownership, loan_intent, loan_grade, cb_person_default_on_file, age_bucket, emp_length_bin
        categorical_encoded = [
            # person_home_ownership (4 categories)
            "person_home_ownership_RENT",
            "person_home_ownership_OWN",
            "person_home_ownership_MORTGAGE",
            "person_home_ownership_OTHER",
            # loan_intent (6 categories)
            "loan_intent_PERSONAL",
            "loan_intent_EDUCATION",
            "loan_intent_MEDICAL",
            "loan_intent_VENTURE",
            "loan_intent_HOMEIMPROVEMENT",
            "loan_intent_DEBTCONSOLIDATION",
            # loan_grade (7 categories)
            "loan_grade_A",
            "loan_grade_B",
            "loan_grade_C",
            "loan_grade_D",
            "loan_grade_E",
            "loan_grade_F",
            "loan_grade_G",
            # cb_person_default_on_file (2 categories - binary one-hot)
            "cb_person_default_on_file_0",
            "cb_person_default_on_file_1",
            # age_bucket (6 categories from pd.cut)
            "age_bucket_18-24",
            "age_bucket_25-34",
            "age_bucket_35-44",
            "age_bucket_45-54",
            "age_bucket_55-64",
            "age_bucket_65+",
            # emp_length_bin (6 categories from pd.cut)
            "emp_length_bin_0",
            "emp_length_bin_1",
            "emp_length_bin_2-3",
            "emp_length_bin_4-5",
            "emp_length_bin_6-10",
            "emp_length_bin_10+",
        ]
        feature_names.extend(categorical_encoded)

        # Verify we have the expected number
        if len(feature_names) != 41:
            logger.warning(
                f"Expected 41 features but generated {len(feature_names)}. "
                f"This may indicate a mismatch in preprocessor configuration."
            )

        return feature_names

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
            df_fe["person_income"].replace(0, 0.01)  # Avoid division by zero
        )

        # Employment length to age ratio
        df_fe["employ_to_age"] = df_fe["person_emp_length"] / (
            df_fe["person_age"].replace(0, 0.01)  # Avoid division by zero
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
            DataFrame with only the 17 selected features in correct order
        """
        try:
            # Validate input data
            self._validate_input(data)
            logger.debug("Input data validated successfully")

            # Convert dictionary to DataFrame
            df = pd.DataFrame([data])

            # Create derived features
            df_fe = self._create_derived_features(df)
            logger.debug("Derived features created")

            # Apply preprocessor (imputation, scaling, one-hot encoding)
            X_transformed = self.preprocessor.transform(df_fe)

            # Get feature names - use the reconstructed method
            feature_names_all = self._get_feature_names_from_preprocessor()

            # Create DataFrame with all transformed features
            X_df_all = pd.DataFrame(X_transformed, columns=feature_names_all)
            logger.debug(f"Data transformed. Initial shape: {X_df_all.shape}")

            # Select only the features that the model was trained on
            # self.feature_names contains the 17 selected features
            try:
                X_df_selected = X_df_all[self.feature_names]
                logger.debug(
                    f"Selected {len(self.feature_names)} features. Final shape: {X_df_selected.shape}"
                )
            except KeyError as e:
                logger.error(
                    f"Could not select features. Available: {list(X_df_all.columns)}, Needed: {self.feature_names}. Error: {e}"
                )
                raise

            return X_df_selected

        except Exception as e:
            logger.error(f"Error preprocessing: {e}")
            raise

    def batch_preprocess(self, data_list: List[Dict]) -> pd.DataFrame:
        """
        Transforms multiple records into dataframe format expected by the model.

        Args:
            data_list: List of dictionaries with raw features

        Returns:
            DataFrame with only the 17 selected features for all records
        """
        try:
            # Handle empty list case
            if not data_list:
                logger.debug(
                    "Empty batch received, returning empty DataFrame with correct columns"
                )
                return pd.DataFrame(columns=self.feature_names)

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

            # Get feature names - use the reconstructed method
            feature_names_all = self._get_feature_names_from_preprocessor()

            # Create DataFrame with all transformed features
            X_df_all = pd.DataFrame(X_transformed, columns=feature_names_all)
            logger.debug(
                f"Batch of {len(data_list)} records transformed. Initial shape: {X_df_all.shape}"
            )

            # Select only the features that the model was trained on
            try:
                X_df_selected = X_df_all[self.feature_names]
                logger.debug(
                    f"Selected {len(self.feature_names)} features. Final shape: {X_df_selected.shape}"
                )
            except KeyError as e:
                logger.error(
                    f"Could not select features. Available: {list(X_df_all.columns)}, Needed: {self.feature_names}. Error: {e}"
                )
                raise

            return X_df_selected

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

    def get_transformed_feature_names(self) -> List[str]:
        """
        Returns the names of features after transformation.
        Tries to get them from the preprocessor, falls back to generic names.
        """
        try:
            # Try to get feature names from preprocessor (sklearn 1.3+)
            return list(self.preprocessor.get_feature_names_out())
        except (AttributeError, TypeError):
            # Fallback: generate generic names based on output shape
            # This is a workaround for sklearn version compatibility
            return [
                str(i) for i in range(len(self.feature_names) * 2)
            ]  # Estimate based on one-hot encoding
