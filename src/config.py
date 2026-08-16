import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

# Retrieval corpus (regulatory text). Deliberately outside data/, which holds
# training data and is never committed: this corpus is an input to retrieval,
# not to training, and nothing is fitted on it.
CORPUS_DIR = BASE_DIR / "corpus"
CORPUS_RAW_DIR = CORPUS_DIR / "raw"
CORPUS_PATH = CORPUS_DIR / "eu_regulation.jsonl"
# Vectors are regenerated rather than versioned: searching needs the embedding
# model loaded anyway, to turn the query into a vector, so shipping the matrix
# would save a clone nothing.
CORPUS_INDEX_PATH = CORPUS_DIR / "embeddings.npz"
# Chunk size belongs to the embedding model, not to the text: the model reads a
# fixed window and drops the rest, so changing model means re-chunking.
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# Model paths
MODEL_PATH = MODELS_DIR / "best_tuned_model_xgboost.joblib"
THRESHOLD_PATH = MODELS_DIR / "optimal_threshold.joblib"
FEATURE_NAMES_PATH = MODELS_DIR / "feature_names.joblib"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"

# MLFlow
# Opt-in on purpose: when MLFLOW_TRACKING_URI is unset the predictor loads
# straight from joblib. Attempting an unreachable tracking server costs ~247s of
# urllib3 retry backoff before failing, which stalls container startup.
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MLFLOW_MODEL_URI = os.getenv("MLFLOW_MODEL_URI", "models:/CreditScorer/Staging")

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
# Annual interest rate as a PERCENTAGE, matching the training data
# (observed range 5.42 - 23.22). Bounds are wider than observed to leave
# operational headroom, but the floor stays at 1.0 on purpose: a caller passing
# the rate as a fraction (0.08 for 8%) must be rejected rather than silently
# scaled to ~3.5 standard deviations below anything the model has seen.
MIN_LOAN_INT_RATE = 1.0
MAX_LOAN_INT_RATE = 50.0
