import joblib
from pathlib import Path
from typing import Dict, List
import logging

# MLFlow imports
try:
    import mlflow
    import mlflow.pyfunc
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# Data preprocessor to handle raw input transformation
from src.preprocessing import DataPreprocessor

logger = logging.getLogger(__name__)


class CreditRiskPredictor:
    """Wrapper to handle model loading, prediction, and input validation for credit risk prediction"""

    def __init__(
        self,
        model_path: Path,
        threshold_path: Path,
        feature_names_path: Path,
        preprocessor_path: Path,
        use_mlflow: bool = True,
    ):
        """
        Initialize the predictor by loading the model, threshold, feature names, and preprocessor

        Args:
            model_path: Path to the saved model (joblib file) or MLFlow model URI
            threshold_path: Path to the saved optimal threshold (joblib file)
            feature_names_path: Path to the saved feature names (joblib file)
            preprocessor_path: Path to the saved preprocessor (joblib file)
            use_mlflow: Try to load from MLFlow first, with fallback to joblib
        """
        self.model = None
        self.threshold = None
        self.feature_names = None
        self.preprocessor = None
        self.model_source = None
        
        # Try MLFlow first if available and enabled
        if use_mlflow and MLFLOW_AVAILABLE:
            try:
                mlflow.set_tracking_uri("http://localhost:5000")
                # Try to load the registered model from MLFlow
                self.model = mlflow.pyfunc.load_model("models:/CreditScorer/Staging")
                self.model_source = "MLFlow (CreditScorer/Staging)"
                logger.info("Model loaded from MLFlow successfully")
            except Exception as e:
                logger.warning(f"Could not load model from MLFlow: {e}. Falling back to joblib...")
                self.model = None
        
        # Fallback to joblib if MLFlow loading failed or not enabled
        if self.model is None:
            try:
                self.model = joblib.load(model_path)
                self.model_source = "Joblib (local file)"
                logger.info("Model loaded from joblib successfully")
            except Exception as e:
                logger.error(f"Error loading model from joblib: {e}")
                raise
        
        # Load threshold, feature names, and preprocessor from joblib
        try:
            self.threshold = joblib.load(threshold_path)
            self.feature_names = joblib.load(feature_names_path)
            self.preprocessor = DataPreprocessor(preprocessor_path, feature_names_path)
            logger.info(f"Model source: {self.model_source}")
            logger.info("Predictor initialized successfully")
        except Exception as e:
            logger.error(f"Error loading predictor components: {e}")
            raise

    def predict(self, features: Dict[str, float]) -> Dict:
        """
        Performs a single prediction with raw data

        Args:
            features: Dictionary with the raw features of the client

        Returns:
            Dict with prediction, probability, and risk level
        """
        try:
            # Preprocess the input data to get the format expected by the model
            X_processed = self.preprocessor.preprocess(features)

            # Get default probability from the model
            # Handle both sklearn models (with predict_proba) and MLFlow pyfunc models
            if hasattr(self.model, 'predict_proba'):
                prob_default = self.model.predict_proba(X_processed)[0][1]
            else:
                # MLFlow pyfunc model - use predict method
                predictions = self.model.predict(X_processed)
                prob_default = float(predictions[0][1]) if len(predictions[0]) > 1 else float(predictions[0])

            # Apply threshold
            prediction = 1 if prob_default >= self.threshold else 0

            # Map to readable text
            risk_label = "high_risk" if prediction == 1 else "low_risk"

            return {
                "prediction": int(prediction),
                "probability_default": float(prob_default),
                "probability_non_default": float(1 - prob_default),
                "risk_level": risk_label,
                "threshold_used": float(self.threshold),
                "recommendation": "Reject application"
                if prediction == 1
                else "Approve application",
            }
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            raise

    def batch_predict(self, features_list: List[Dict]) -> List[Dict]:
        """
        Performs batch predictions with raw data

        Args:
            features_list: List of dictionaries with raw features

        Returns:
            List of predictions
        """
        try:
            # Preprocess the input data to get the format expected by the model
            X_processed = self.preprocessor.batch_preprocess(features_list)

            # Handle empty input - return empty predictions list
            if len(X_processed) == 0:
                logger.debug("Empty batch received, returning empty predictions")
                return []

            # Get default probabilities from the model
            # Handle both sklearn models (with predict_proba) and MLFlow pyfunc models
            if hasattr(self.model, 'predict_proba'):
                probs_default = self.model.predict_proba(X_processed)[:, 1]
            else:
                # MLFlow pyfunc model - use predict method
                predictions_raw = self.model.predict(X_processed)
                probs_default = predictions_raw[:, 1] if predictions_raw.shape[1] > 1 else predictions_raw.flatten()

            # Loop through probabilities and apply threshold to get predictions and risk levels
            predictions = []
            for i, prob_default in enumerate(probs_default):
                prediction = 1 if prob_default >= self.threshold else 0
                risk_label = "high_risk" if prediction == 1 else "low_risk"

                predictions.append(
                    {
                        "prediction": int(prediction),
                        "probability_default": float(prob_default),
                        "probability_non_default": float(1 - prob_default),
                        "risk_level": risk_label,
                        "threshold_used": float(self.threshold),
                        "recommendation": "Reject application"
                        if prediction == 1
                        else "Approve application",
                    }
                )

            return predictions
        except Exception as e:
            logger.error(f"Error in batch predictions: {e}")
            raise

    def get_feature_names(self) -> List[str]:
        """Returns the list of expected feature names"""
        return list(self.feature_names)

    def get_model_info(self) -> Dict:
        """Returns information about the model"""
        return {
            "model_type": str(type(self.model).__name__),
            "model_source": self.model_source,
            "threshold": float(self.threshold),
            "num_features": len(self.feature_names),
            "features": self.get_feature_names(),
        }
