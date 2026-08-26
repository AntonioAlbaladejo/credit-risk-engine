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
# Hand-written evaluation set: questions with the units that should answer them.
# Committed, unlike the vectors -- it is the labelled data any retrieval change
# is measured against, not a build artifact.
GOLDEN_SET_PATH = CORPUS_DIR / "golden_questions.jsonl"
# Chunk size belongs to the embedding model, not to the text: the model reads a
# fixed window and drops the rest, so changing model means re-chunking.
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
# Prepended to the question, never to the passage. This model is trained for
# asymmetric retrieval -- a short question against a long passage -- and its
# card asks for this exact wording on the query side. It moves with the model:
# a different encoder wants different wording, or none at all, and the wrong
# one degrades ranking without erroring.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
# Similarity is multiplied by the weight of the unit the passage belongs to.
# Recitals are the preamble: explanatory prose that reads like a question and
# therefore embeds closer to one than the article it explains, which is what
# actually binds.
#
# 0.90 is the weakest penalty that gets the full effect -- the sweep over
# corpus/golden_questions.jsonl is flat from here down to 0.0, so a harder
# penalty buys nothing and only discards more signal. Be aware that at this
# weight recitals stop reaching the top 5 almost entirely: the mechanism is a
# multiplier, but on this corpus it behaves close to a filter. Re-run the sweep
# if the corpus or the embedding model changes; the right value is a property
# of both, not a constant.
UNIT_WEIGHTS = {"article": 1.0, "annex": 1.0, "recital": 0.90}
# Below this ranking score the retriever returns nothing rather than its best
# guess. Most questions put to this corpus have no answer in it -- they are
# about the product, the model or the business -- and a passage cited for one
# of those is worse than silence: it reads like grounding and is not.
#
# Fitted on the `fit` split of corpus/golden_questions.jsonl only (59 of 101
# questions), from the middle of a plateau -- 0.64 to 0.67 all handle 43-44 of
# those 59 correctly -- rather than from the peak, so it does not sit on a
# cliff edge. Re-fit it whenever the corpus, the unit weights or the embedding
# model change: it is a property of that similarity scale, and the scale moves
# with all three. `uv run python -m scripts.evaluate_retrieval` prints the
# sweep and reports the held-out split separately.
#
# What it does NOT buy: on the held-out split this threshold still serves 6
# wrong citations out of 42 questions, against 10 right ones. An earlier
# reading of "zero wrong citations" came from 31 questions that had also
# chosen the value, and did not survive contact with unseen ones.
MIN_SCORE = 0.66

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
