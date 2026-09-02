import asyncio
import logging
import threading
import time
from contextlib import asynccontextmanager
from enum import Enum

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthCheck,
    LoanApplication,
    ModelInfo,
    PredictionResponse,
    RegulationSearchRequest,
    RegulationSearchResponse,
)
from src.config import (
    API_DESCRIPTION,
    API_TITLE,
    API_VERSION,
    FEATURE_NAMES_PATH,
    LOG_LEVEL,
    MODEL_PATH,
    PREPROCESSOR_PATH,
    RATE_LIMIT_REQUESTS,
    RATE_LIMIT_WINDOW_SECONDS,
    THRESHOLD_PATH,
)
from src.predictor import CreditRiskPredictor
from src.retriever import CorpusRetriever

# Configure logging
logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)


def _dump_model(model: BaseModel) -> dict:
    """Flatten a request model to the raw dict the preprocessor expects.

    Args:
        model: A validated Pydantic model.

    Returns:
        Its fields, with enum members replaced by their values.
    """
    return {
        k: v.value if isinstance(v, Enum) else v for k, v in model.model_dump().items()
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage app startup and shutdown events"""
    # Startup
    try:
        get_predictor()
        logger.info("Model loaded at startup")
    except Exception as e:
        logger.error(f"Error loading model at startup: {e}")

    # Warm the corpus off the startup path. Loading it costs 12.6 s on the
    # 0.5 vCPU the task is sized at, which the first caller after every rollout
    # would otherwise pay. Blocking here instead would push startup from 8 s to
    # ~20 s, and the health check ECS actually reads lives in the task
    # definition in AWS, not in this repo -- its startPeriod is not ours to
    # widen, so startup time stays where it was measured.
    asyncio.create_task(asyncio.to_thread(_warm_corpus))

    yield  # App runs here

    # Shutdown (optional cleanup)
    logger.info("App shutting down")


# Create FastAPI app
app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=API_DESCRIPTION,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS - TODO Configure according to security requirements (currently allows all origins, methods, and headers for simplicity)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# One window shared by every caller: when it rolls the whole table is dropped,
# which bounds memory to a window's worth of distinct addresses. The known cost
# is a burst across the boundary, twice the limit in a moment; a sliding window
# smooths that in exchange for per-caller bookkeeping nobody here needs.
_rate_window_start = 0.0
_rate_counts: dict[str, int] = {}


@app.middleware("http")
async def rate_limit(request: Request, call_next):
    """Cap requests per client IP so one caller cannot take the task's CPU.

    No lock: the middleware is async and the table is only touched from the
    event loop, between awaits. Two ceilings worth knowing -- the count is per
    process, so `--workers N` multiplies the limit by N, and `request.client`
    is the real caller only while nothing proxies in front. Behind a load
    balancer every request arrives with its address, and the limit becomes one
    global budget unless it starts reading `X-Forwarded-For`.

    Args:
        request: The incoming request.
        call_next: The rest of the stack.

    Returns:
        The downstream response, or 429 with `Retry-After` once over budget.
    """
    global _rate_window_start
    # Health checks are never limited: ECS reads /health to decide whether the
    # task lives, and starving it would turn a busy minute into a restart.
    if request.url.path != "/health":
        now = time.monotonic()
        if now - _rate_window_start >= RATE_LIMIT_WINDOW_SECONDS:
            _rate_window_start = now
            _rate_counts.clear()
        caller = request.client.host if request.client else "unknown"
        _rate_counts[caller] = _rate_counts.get(caller, 0) + 1
        if _rate_counts[caller] > RATE_LIMIT_REQUESTS:
            logger.warning(f"Rate limit hit by {caller}")
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={"detail": "Too many requests."},
                headers={
                    "Retry-After": str(
                        int(RATE_LIMIT_WINDOW_SECONDS - (now - _rate_window_start)) + 1
                    )
                },
            )
    return await call_next(request)


# Load model and preprocessor at startup
_predictor: CreditRiskPredictor | None = None
_retriever: CorpusRetriever | None = None
# Sync endpoints run in a threadpool, so two callers can reach the lazy build
# at once and each load their own 240 MB copy of the model.
_retriever_lock = threading.Lock()


def get_predictor() -> CreditRiskPredictor:
    """Get predictor instance, loading it lazily if needed"""
    global _predictor
    if _predictor is None:
        try:
            _predictor = CreditRiskPredictor(
                MODEL_PATH,
                THRESHOLD_PATH,
                FEATURE_NAMES_PATH,
                PREPROCESSOR_PATH,
                use_mlflow=True,  # Enable MLFlow loading with fallback to joblib
            )
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    return _predictor


def get_retriever() -> CorpusRetriever:
    """Get the corpus retriever, loading it on first use.

    Lazy rather than constructed at import so that a checkout with no index
    built still serves the scoring endpoints, and so tests can substitute one.
    In the deployed service `lifespan` warms it in the background, so first use
    is normally the warm-up rather than a caller.

    Returns:
        A retriever over the regulatory corpus.

    Raises:
        FileNotFoundError: The corpus or its index is missing from the image.
    """
    global _retriever
    with _retriever_lock:
        if _retriever is None:
            _retriever = CorpusRetriever.from_files()
            logger.info("Regulatory corpus loaded")
    return _retriever


def _warm_corpus() -> None:
    """Build the retriever ahead of the first request, tolerating its absence.

    A failure here must not take the service down: the scoring endpoints do
    not need the corpus, and /regulation/search answers 503 on its own.
    """
    try:
        get_retriever()
    except Exception as e:
        logger.warning(f"Regulatory corpus not warmed: {e}")


# ==================== HEALTH CHECK ====================


@app.get("/health", response_model=HealthCheck)
async def health_check(response: Response):
    """Verifies the status of the API and the model.

    Returns 503 when the model is unavailable so that container and load
    balancer health checks fail on an instance that cannot serve predictions.
    """
    try:
        # Try to ensure model is loaded
        predictor = get_predictor()
        model_available = predictor is not None
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        model_available = False

    if not model_available:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return HealthCheck(
        status="healthy" if model_available else "unhealthy",
        model_loaded=model_available,
        version=API_VERSION,
    )


# ==================== MODEL INFO ====================


@app.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """Returns information about the loaded model"""
    try:
        predictor = get_predictor()
        return predictor.get_model_info()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model not loaded: {str(e)}",
        ) from e


# ==================== PREDICTIONS ====================


@app.post("/predict", response_model=PredictionResponse)
async def predict(application: LoanApplication):
    """
    Prediction of loan default risk based on the input features of the loan application.

    Args:
        application: LoanApplication object with the input features for the prediction
    Returns:
        PredictionResponse object with the prediction result, probabilities, risk level, and recommendation
    """
    try:
        predictor = get_predictor()
        features = _dump_model(application)
        prediction = predictor.predict(features)
        return PredictionResponse(**prediction)

    except Exception as e:
        logger.error(f"Error processing prediction: {e}")
        if "Model not loaded" in str(e):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded",
            ) from e
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing prediction: {str(e)}",
        ) from e


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """
    Makes batch predictions for multiple loan applications.
    """
    try:
        predictor = get_predictor()
        features_list = [_dump_model(app) for app in request.applications]
        predictions = predictor.batch_predict(features_list)

        return BatchPredictionResponse(
            success=True,
            num_predictions=len(predictions),
            predictions=[PredictionResponse(**p) for p in predictions],
        )

    except Exception as e:
        logger.error(f"Error processing batch prediction: {e}")
        if "Model not loaded" in str(e):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded",
            ) from e
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing batch prediction: {str(e)}",
        ) from e


# ==================== REGULATION ====================


@app.post("/regulation/search", response_model=RegulationSearchResponse)
async def search_regulation(request: RegulationSearchRequest):
    """Find the passages of EU law that bear on a question about this system.

    Covers the GDPR and the EU AI Act in full. Use it whenever the answer
    should cite the provision instead of recalling it -- automated decisions,
    the right to an explanation, high-risk classification, record-keeping,
    human oversight.

    Quote and cite only what comes back. Every passage carries the `citation`
    naming it and the `source_url` and `retrieved_on` that let a reader check
    it. A claim about the law that no returned passage supports must not be
    presented as grounded, however confident you are that it is true. That
    includes provisions a returned passage merely names: the corpus is
    searched, not followed, so "without prejudice to Article 78" does not
    bring Article 78 with it.

    An empty `passages` list is an answer, not a failure. It means either that
    nothing in the corpus is close enough, or that the question asks what this
    organisation actually did rather than what the law requires -- legislation
    states requirements and holds no record of anyone's compliance with them.

    The corpus is EU legislation and nothing else: no internal policy, no
    record of what this organisation has done, no US regulation.

    Args:
        request: The question, and optionally a hypothetical passage to rank
            by. Filling in the second roughly halves the passages missed.

    Returns:
        RegulationSearchResponse with the matching passages, best first.

    Raises:
        HTTPException: 503 when the corpus is not present in the deployment.
    """
    try:
        retriever = get_retriever()
    except FileNotFoundError as e:
        logger.error(f"Regulatory corpus unavailable: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The regulatory corpus is not available in this deployment.",
        ) from e

    return RegulationSearchResponse(
        **retriever.search_payload(
            request.question, hypothetical_passage=request.hypothetical_passage
        )
    )


# ==================== ROOT ====================


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": API_TITLE,
        "version": API_VERSION,
        "description": API_DESCRIPTION,
        "docs_url": "/docs",
        "redoc_url": "/redoc",
        "endpoints": {
            "health": "/health",
            "model_info": "/model/info",
            "predict": "/predict (POST)",
            "batch_predict": "/predict/batch (POST)",
            "search_regulation": "/regulation/search (POST)",
        },
    }
