"""Evidently drift and classification-quality report for the promoted bundle.

Reference is the training split and current the held-out test split, as written
by notebooks/feature_engineering.ipynb. Both are already transformed by the
bundle's preprocessor, so the model scores them directly; regenerate them
whenever models/ is retrained, or the report scores one run's data with another.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from evidently import ColumnMapping
from evidently.metric_preset import (
    ClassificationPreset,
    DataDriftPreset,
    TargetDriftPreset,
)
from evidently.report import Report

from src.config import (
    BASE_DIR,
    DATA_DIR,
    FEATURE_NAMES_PATH,
    MODEL_PATH,
    THRESHOLD_PATH,
)

logger = logging.getLogger(__name__)

RESULTS_DIR = BASE_DIR / "results"
TARGET = "loan_status"
PREDICTION = "prediction"


def _load_artifacts() -> tuple[Any, list[str], float]:
    """Load the model, its feature order and its decision threshold.

    Returns:
        ``(model, feature_names, threshold)`` from ``models/``.
    """
    return (
        joblib.load(MODEL_PATH),
        joblib.load(FEATURE_NAMES_PATH),
        joblib.load(THRESHOLD_PATH),
    )


def build_monitoring_report(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    model: Any,
    feature_names: list[str],
    threshold: float,
    output_path: str | os.PathLike[str],
    target: str = TARGET,
) -> Report:
    """Score both datasets with the model and save an Evidently HTML report.

    Args:
        reference_data: Transformed features plus the label, usually train.
        current_data: Same columns, the data being checked.
        model: Fitted classifier exposing ``predict_proba``.
        feature_names: Columns the model expects, in its order.
        threshold: Decision threshold for the classification metrics.
        output_path: Where the HTML report is written.
        target: The 0/1 label column.

    Returns:
        The run report; ``as_dict()`` holds the computed metrics.

    Raises:
        ValueError: If a dataset lacks a feature or the target, or the target
            is not 0/1.
    """
    scored = []
    for name, data in (("reference", reference_data), ("current", current_data)):
        missing = [c for c in [*feature_names, target] if c not in data.columns]
        if missing:
            raise ValueError(
                f"{name} data lacks {missing}; regenerate the splits with "
                "notebooks/feature_engineering.ipynb"
            )
        if not data[target].isin([0, 1]).all():
            raise ValueError(
                f"{name} column '{target}' is not binary 0/1; pass the label "
                "column, not a feature"
            )
        frame = data[[*feature_names, target]].copy()
        frame[PREDICTION] = model.predict_proba(frame[feature_names])[:, 1]
        scored.append(frame)

    report = Report(
        metrics=[
            DataDriftPreset(),
            TargetDriftPreset(),
            ClassificationPreset(probas_threshold=threshold),
        ]
    )
    report.run(
        reference_data=scored[0],
        current_data=scored[1],
        column_mapping=ColumnMapping(target=target, prediction=PREDICTION, pos_label=1),
    )
    # A metric that fails is kept as an error widget in the HTML; this re-raises.
    report.as_dict()
    report.save_html(str(output_path))
    return report


def generate_monitoring_report(
    reference_path: str | os.PathLike[str] | None = None,
    current_path: str | os.PathLike[str] | None = None,
    output_path: str | os.PathLike[str] | None = None,
    target: str = TARGET,
) -> Path:
    """Build the report for the promoted bundle from CSV splits.

    Args:
        reference_path: Defaults to the training split.
        current_path: Defaults to the held-out test split.
        output_path: Defaults to ``results/model_monitoring_report.html``.
        target: The 0/1 label column.

    Returns:
        Path of the written HTML report.
    """
    output_path = Path(output_path or RESULTS_DIR / "model_monitoring_report.html")
    model, feature_names, threshold = _load_artifacts()
    build_monitoring_report(
        pd.read_csv(reference_path or DATA_DIR / "credit_risk_fe_train.csv"),
        pd.read_csv(current_path or DATA_DIR / "credit_risk_fe_test.csv"),
        model,
        feature_names,
        threshold,
        output_path,
        target,
    )
    return output_path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger.info("Monitoring report saved to %s", generate_monitoring_report())


if __name__ == "__main__":
    main()
