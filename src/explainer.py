"""Per-application explanations for the credit risk model.

``CreditRiskPredictor`` answers *what* the decision is. This module answers
*why*: how much each input pushed the score towards default or away from it.

Contributions come from XGBoost's own TreeSHAP implementation.

Two properties of the output matter to every caller:

* Contributions are in **log-odds**, not probability. They add up exactly
  (``base_value + sum(contributions) == raw margin``), which is what makes the
  grouping below legitimate. They cannot be read as shares of the probability:
  the sigmoid is non-linear, so the same log-odds step moves P(default) a lot in
  the middle of the range and almost nothing at the extremes. To quote a
  probability, subtract in log-odds and convert at the end.
* The sigmoid is monotonic, so **signs and ranking are always safe to report**
  even though magnitudes are not directly translatable.
"""

import logging

import numpy as np
import xgboost as xgb

from src.predictor import CreditRiskPredictor

logger = logging.getLogger(__name__)


REASON_GROUPS: dict[str, tuple[str, ...]] = {
    "affordability": ("person_income", "loan_amnt", "loan_percent_income"),
    "loan_grade": ("loan_grade_A", "loan_grade_C", "loan_grade_D", "loan_grade_E"),
    "interest_rate": ("loan_int_rate",),
    "stability": (
        "person_emp_length",
        "person_age",
        "person_home_ownership_MORTGAGE",
        "person_home_ownership_OWN",
    ),
    "loan_purpose": (
        "loan_intent_DEBTCONSOLIDATION",
        "loan_intent_EDUCATION",
        "loan_intent_HOMEIMPROVEMENT",
        "loan_intent_MEDICAL",
        "loan_intent_PERSONAL",
    ),
    "credit_history": ("default_flag",),
}


class RiskExplainer:
    """Explains a single scored application in terms of reason codes.

    Takes an already-built predictor so the bundle -- model, preprocessor,
    feature names, threshold -- stays loaded once and cannot drift between the
    score and its explanation.
    """

    def __init__(self, predictor: CreditRiskPredictor):
        """
        Args:
            predictor: An initialised predictor holding the promoted bundle.

        Raises:
            TypeError: The model is not a native XGBoost estimator, so exact
                TreeSHAP is unavailable (an MLflow ``pyfunc`` wrapper, say).
            ValueError: ``REASON_GROUPS`` does not cover exactly the features
                the loaded bundle expects.
        """
        if not hasattr(predictor.model, "get_booster"):
            raise TypeError(
                f"RiskExplainer needs a native XGBoost model, got "
                f"{type(predictor.model).__name__}. Load the bundle with "
                f"use_mlflow=False, or unwrap the pyfunc model first."
            )

        self.predictor = predictor
        self.booster = predictor.model.get_booster()
        self.feature_names = list(predictor.feature_names)

        # A retrain that changes the feature list must fail here rather than
        # silently explaining 17 of 18 features.
        grouped = [name for names in REASON_GROUPS.values() for name in names]
        missing = set(self.feature_names) - set(grouped)
        unknown = set(grouped) - set(self.feature_names)
        if missing or unknown:
            raise ValueError(
                f"REASON_GROUPS does not match the loaded bundle. "
                f"Ungrouped features: {sorted(missing) or 'none'}. "
                f"Grouped but not in the model: {sorted(unknown) or 'none'}."
            )

    def explain(self, features: dict) -> dict:
        """Break a single application's score down into reason codes.

        Args:
            features: Raw application, same shape ``predictor.predict`` takes.

        Returns:
            Dict with the raw application echoed back, the model's baseline,
            the resulting probability, and contributions per reason code
            (sorted by magnitude, strongest first) and per feature.
        """
        X = self.predictor.preprocessor.preprocess(features)
        contribs = self.booster.predict(
            xgb.DMatrix(X, feature_names=self.feature_names), pred_contribs=True
        )[0]

        # Last column is the base value: the model's expected log-odds over the
        # training data. Every contribution is relative to it, not absolute.
        base_value = float(contribs[-1])
        per_feature = {
            name: float(value)
            for name, value in zip(self.feature_names, contribs[:-1], strict=True)
        }
        margin = base_value + sum(per_feature.values())

        groups = [
            {
                "reason": code,
                "contribution": sum(per_feature[name] for name in names),
            }
            for code, names in REASON_GROUPS.items()
        ]
        for group in groups:
            group["direction"] = (
                "increases_risk" if group["contribution"] > 0 else "decreases_risk"
            )
        groups.sort(key=lambda g: abs(g["contribution"]), reverse=True)

        return {
            # Echoed whole: contributions describe the effect of a value, never
            # the value itself, and the scaled features the model sees are not
            # something to show a human.
            "application": dict(features),
            "base_value": base_value,
            "base_probability": _sigmoid(base_value),
            "margin": margin,
            # Derived from the same margin as the contributions, so the
            # explanation can never disagree with the number it explains.
            "probability_default": _sigmoid(margin),
            "threshold_used": float(self.predictor.threshold),
            "reason_codes": groups,
            "feature_contributions": per_feature,
        }


def _sigmoid(margin: float) -> float:
    """Convert log-odds to probability."""
    return float(1.0 / (1.0 + np.exp(-margin)))
