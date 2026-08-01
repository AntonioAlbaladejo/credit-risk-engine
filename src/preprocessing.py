import logging
from pathlib import Path

import joblib
import pandas as pd

from src.config import (
    MAX_AGE,
    MAX_EMP_LENGTH,
    MAX_LOAN_AMOUNT,
    MAX_LOAN_INT_RATE,
    MIN_AGE,
    MIN_EMP_LENGTH,
    MIN_LOAN_AMOUNT,
    MIN_LOAN_INT_RATE,
)

logger = logging.getLogger(__name__)

# Bin edges shared by training and serving. The lower age edge is 17, not 18, so
# that an applicant aged exactly MIN_AGE lands in the first bucket instead of
# falling outside every interval and being imputed to the mode.
AGE_BINS = [17, 25, 35, 45, 55, 65, 120]
AGE_LABELS = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
EMP_LENGTH_BINS = [-1, 0, 1, 3, 5, 10, 100]
EMP_LENGTH_LABELS = ["0", "1", "2-3", "4-5", "6-10", "10+"]


def create_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create the engineered features the model expects.

    Single source of truth for training and inference. Any divergence between
    the two produces train/serve skew that no test would catch, so both paths
    import this function rather than reimplementing it.

    Args:
        df: DataFrame with the raw applicant and loan columns.

    Returns:
        A copy of ``df`` with the derived columns added.
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
    df_fe["age_bucket"] = pd.cut(df_fe["person_age"], bins=AGE_BINS, labels=AGE_LABELS)

    # Employment length bins
    df_fe["emp_length_bin"] = pd.cut(
        df_fe["person_emp_length"].fillna(0),
        bins=EMP_LENGTH_BINS,
        labels=EMP_LENGTH_LABELS,
    )

    # Default flag. The raw dataset encodes this as Y/N while the API sends 0/1,
    # so both spellings have to land on the same numeric flag.
    default_on_file = df_fe["cb_person_default_on_file"]
    if default_on_file.dtype == object:
        df_fe["default_flag"] = (
            default_on_file.astype(str).str.upper().map({"Y": 1, "N": 0}).astype(int)
        )
    else:
        df_fe["default_flag"] = default_on_file.astype(int)

    return df_fe


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
            # Resolved once, at load time, so a preprocessor that cannot name
            # its own output fails here -- where /health reports it -- instead
            # of on the first prediction.
            self._transformed_names = self._get_feature_names_from_preprocessor()
            logger.info("Preprocessor loaded successfully")
        except Exception as e:
            logger.error(f"Error loading preprocessor: {e}")
            raise

    def _validate_input(self, data: dict) -> None:
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

        if (
            data.get("loan_int_rate") < MIN_LOAN_INT_RATE
            or data.get("loan_int_rate") > MAX_LOAN_INT_RATE
        ):
            raise ValueError(
                f"loan_int_rate is a percentage and must be between "
                f"{MIN_LOAN_INT_RATE} and {MAX_LOAN_INT_RATE}"
            )

        if data.get("loan_percent_income") < 0 or data.get("loan_percent_income") > 1:
            raise ValueError("loan_percent_income must be between 0 and 1")

    def _get_feature_names_from_preprocessor(self) -> list[str]:
        """
        Asks the fitted ColumnTransformer for the names of the columns it emits.

        The names are only meaningful in the order transform() produces them:
        they are attached positionally to an unlabelled array, so an order that
        does not come from the fitted transformer itself relabels the columns
        without changing the values. That mislabelling is silent -- every
        downstream consumer keeps working on the wrong feature.

        Returns:
            The transformed column names, with the ``num__`` / ``cat__`` step
            prefixes stripped.

        Raises:
            AttributeError: If the preprocessor cannot report its output names.
        """
        if not hasattr(self.preprocessor, "get_feature_names_out"):
            raise AttributeError(
                f"{type(self.preprocessor).__name__} does not expose "
                "get_feature_names_out(), so the transformed columns cannot be "
                "named. Rebuild the artifact bundle with the scikit-learn "
                "version pinned in pyproject.toml."
            )

        return [
            name.split("__", 1)[1] if "__" in name else name
            for name in self.preprocessor.get_feature_names_out()
        ]

    def _create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Delegates to the shared implementation used by the training pipeline."""
        return create_derived_features(df)

    def preprocess(self, data: dict) -> pd.DataFrame:
        """
        Transforms raw input data into the format expected by the model.

        Args:
            data: Dictionary with raw features

        Returns:
            DataFrame with only the selected features, in the order the model
            was trained on.
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

            # Create DataFrame with all transformed features
            X_df_all = pd.DataFrame(X_transformed, columns=self._transformed_names)
            logger.debug(f"Data transformed. Initial shape: {X_df_all.shape}")

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
            logger.error(f"Error preprocessing: {e}")
            raise

    def batch_preprocess(self, data_list: list[dict]) -> pd.DataFrame:
        """
        Transforms multiple records into dataframe format expected by the model.

        Args:
            data_list: List of dictionaries with raw features

        Returns:
            DataFrame with only the selected features for all records.
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
                    raise ValueError(f"Error in record {i}: {str(e)}") from e

            # Convert dictionary list to DataFrame
            df = pd.DataFrame(data_list)

            # Create derived features
            df_fe = self._create_derived_features(df)

            # Apply preprocessor
            X_transformed = self.preprocessor.transform(df_fe)

            # Create DataFrame with all transformed features
            X_df_all = pd.DataFrame(X_transformed, columns=self._transformed_names)
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

    def get_feature_info(self) -> dict:
        """Returns information about the expected features"""
        return {
            "valid_home_ownership": self.VALID_HOME_OWNERSHIP,
            "valid_loan_intent": self.VALID_LOAN_INTENT,
            "valid_loan_grade": self.VALID_LOAN_GRADE,
            "num_features_after_preprocessing": len(self.feature_names),
            "final_features": self.feature_names,
        }
