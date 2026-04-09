from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

# Model paths
MODEL_PATH = MODELS_DIR / "best_tuned_model_xgboost.joblib"
THRESHOLD_PATH = MODELS_DIR / "optimal_threshold.joblib"
FEATURE_NAMES_PATH = MODELS_DIR / "feature_names.joblib"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"

# API Config
API_TITLE = "Credit Risk Engine API"
API_VERSION = "1.0.0"
API_DESCRIPTION = "Credit Risk Engine API to predict loan default risk based on applicant and loan features."

# Logging
LOG_LEVEL = "INFO"

# Input validation limits
MIN_AGE = 18
MAX_AGE = 100
MIN_EMP_LENGTH = 0
MAX_EMP_LENGTH = 80
MIN_LOAN_AMOUNT = 500
MAX_LOAN_AMOUNT = 100000
