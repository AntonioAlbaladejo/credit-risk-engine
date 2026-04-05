import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List
import logging

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
    ):
        """
        Initialize the predictor by loading the model, threshold, feature names, and preprocessor

        Args:
            model_path: Path to the saved model (joblib file)
            threshold_path: Path to the saved optimal threshold (joblib file)
            feature_names_path: Path to the saved feature names (joblib file)
            preprocessor_path: Path to the saved preprocessor (joblib file)
        """
        try:
            self.model = joblib.load(model_path)
            self.threshold = joblib.load(threshold_path)
            self.feature_names = joblib.load(feature_names_path)
            self.preprocessor = DataPreprocessor(preprocessor_path, feature_names_path)
            logger.info("Model and preprocessor loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
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
            prob_default = self.model.predict_proba(X_processed)[0][1]

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
            probs_default = self.model.predict_proba(X_processed)[:, 1]

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
            "threshold": float(self.threshold),
            "num_features": len(self.feature_names),
            "features": self.get_feature_names(),
        }
