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
# Fitted on the `fit` split of corpus/golden_questions.jsonl only (94 of 161
# questions), from the middle of the 0.635-0.670 plateau rather than its peak,
# so it does not sit on a cliff edge.
#
# Re-fit it whenever the corpus, the unit weights, the embedding model or **the
# shape of the questions asked** change. That last one is not padding: widening
# the set from 101 to 161 questions with terse fragments and rambling ones
# moved the optimum on its own, with everything else held fixed.
#
# What it does NOT buy: on the held-out split it still serves 13 wrong
# citations out of 67 questions, against 21 right ones.
# `uv run python -m scripts.evaluate_retrieval` prints the sweep.
MIN_SCORE = 0.65
# Same veto, re-fitted for the passage-led ranking, which finds an answer for
# more questions and so pays off further down. Middle of a narrow 0.590-0.600
# plateau on `fit`; held out, 47 of 67 right against 39 for the question alone.
# Re-fit alongside MIN_SCORE: the right value depends on the corpus, the unit
# weights, the embedding model AND the shape of the questions asked.
MIN_SCORE_WITH_PASSAGE = 0.595

# A second veto arm, on the modality of the question rather than on similarity.
# The corpus states what the law requires, so it can answer "must we do X" and
# structurally cannot answer "did we do X" -- and for the second it returns the
# provision governing X, which every relevance model agrees is a good match.
DEONTIC_ANCHORS = [
    "What does the law require in this situation?",
    "Are we obliged to do this, and under what conditions?",
    "Is this permitted, and what conditions apply to it?",
]
EVIDENTIAL_ANCHORS = [
    "What did our organisation actually do in this case?",
    "Show me our internal record of what happened.",
    "What is our current measured figure for this?",
]
# Middle of the -0.094..-0.048 plateau on `fit`. Cross-validated there it wins
# 7 seeds of 8 against the question score alone: +4 correct answers and 7 fewer
# wrong citations per fold set. Re-fit alongside MIN_SCORE_WITH_PASSAGE.
MIN_ANCHOR_SCORE = -0.071

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
