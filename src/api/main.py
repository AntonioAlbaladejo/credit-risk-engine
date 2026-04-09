from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
import logging
from contextlib import asynccontextmanager


from src.config import (
    API_TITLE,
    API_VERSION,
    API_DESCRIPTION,
    MODEL_PATH,
    THRESHOLD_PATH,
    FEATURE_NAMES_PATH,
    PREPROCESSOR_PATH,
    LOG_LEVEL,
)
from src.predictor import CreditRiskPredictor
from src.api.schemas import (
    LoanApplication,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthCheck,
    ModelInfo,
)

# Configure logging
logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage app startup and shutdown events"""
    # Startup
    try:
        get_predictor()
        logger.info("Model loaded at startup")
    except Exception as e:
        logger.error(f"Error loading model at startup: {e}")

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

# Load model and preprocessor at startup
_predictor: CreditRiskPredictor = None


def get_predictor() -> CreditRiskPredictor:
    """Get predictor instance, loading it lazily if needed"""
    global _predictor
    if _predictor is None:
        try:
            _predictor = CreditRiskPredictor(
                MODEL_PATH, THRESHOLD_PATH, FEATURE_NAMES_PATH, PREPROCESSOR_PATH
            )
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    return _predictor


# ==================== HEALTH CHECK ====================


@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Verifies the status of the API and the model"""
    try:
        # Try to ensure model is loaded
        predictor = get_predictor()
        model_available = predictor is not None
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        model_available = False

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
        )


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
        # Handle both Pydantic v1 and v2 - use model_dump() for v2, dict() for v1
        features = (
            application.model_dump()
            if hasattr(application, "model_dump")
            else application.dict()
        )
        prediction = predictor.predict(features)
        return PredictionResponse(**prediction)

    except Exception as e:
        logger.error(f"Error processing prediction: {e}")
        if "Model not loaded" in str(e):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded",
            )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing prediction: {str(e)}",
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """
    Makes batch predictions for multiple loan applications.
    """
    try:
        predictor = get_predictor()
        # Handle both Pydantic v1 and v2 - use model_dump() for v2, dict() for v1
        features_list = [
            app.model_dump() if hasattr(app, "model_dump") else app.dict()
            for app in request.applications
        ]
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
            )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing batch prediction: {str(e)}",
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
        },
    }
