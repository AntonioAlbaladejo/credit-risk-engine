from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
import logging
from pathlib import Path

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

# Create FastAPI app
app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=API_DESCRIPTION,
    docs_url="/docs",
    redoc_url="/redoc",
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
predictor: CreditRiskPredictor = None


@app.on_event("startup")
async def load_model():
    """Load the model when the application starts"""
    global predictor
    try:
        predictor = CreditRiskPredictor(
            MODEL_PATH, THRESHOLD_PATH, FEATURE_NAMES_PATH, PREPROCESSOR_PATH
        )
        logger.info("Model loaded successfully at startup")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


# ==================== HEALTH CHECK ====================


@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Verifies the status of the API and the model"""
    return HealthCheck(
        status="healthy", model_loaded=predictor is not None, version=API_VERSION
    )


# ==================== MODEL INFO ====================


@app.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """Returns information about the loaded model"""
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model not loaded"
        )
    return predictor.get_model_info()


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
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model not loaded"
        )

    try:
        features = application.dict()
        prediction = predictor.predict(features)
        return PredictionResponse(**prediction)

    except Exception as e:
        logger.error(f"Error processing prediction: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing prediction: {str(e)}",
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """
    Makes batch predictions for multiple loan applications.
    """
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model not loaded"
        )

    try:
        features_list = [app.dict() for app in request.applications]
        predictions = predictor.batch_predict(features_list)

        return BatchPredictionResponse(
            success=True,
            num_predictions=len(predictions),
            predictions=[PredictionResponse(**p) for p in predictions],
        )

    except Exception as e:
        logger.error(f"Error processing batch prediction: {e}")
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
