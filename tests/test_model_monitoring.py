from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, roc_auc_score

from src.model_monitoring import build_monitoring_report

FEATURES = ["shifted", "stable"]
THRESHOLD = 0.39


def _split(rng: np.random.Generator, shift: float, n: int = 1000) -> pd.DataFrame:
    data = pd.DataFrame(
        {"shifted": rng.normal(shift, 1.0, n), "stable": rng.integers(0, 2, n)}
    )
    p_default = 1 / (1 + np.exp(-2 * data["shifted"]))
    data["loan_status"] = (rng.random(n) < p_default).astype(int)
    return data


@pytest.fixture
def splits() -> tuple[pd.DataFrame, pd.DataFrame, LogisticRegression]:
    rng = np.random.default_rng(42)
    reference, current = _split(rng, 0.0), _split(rng, 1.5)
    model = LogisticRegression(random_state=42)
    model.fit(reference[FEATURES], reference["loan_status"])
    return reference, current, model


def test_report_computes_real_metrics(splits, tmp_path: Path) -> None:
    reference, current, model = splits
    output = tmp_path / "report.html"

    report = build_monitoring_report(
        reference, current, model, FEATURES, THRESHOLD, output
    )

    metrics = {m["metric"]: m["result"] for m in report.as_dict()["metrics"]}
    quality = metrics["ClassificationQualityMetric"]["current"]
    proba = model.predict_proba(current[FEATURES])[:, 1]
    y = current["loan_status"]
    assert quality["roc_auc"] == pytest.approx(roc_auc_score(y, proba))
    assert quality["recall"] == pytest.approx(recall_score(y, proba >= THRESHOLD))

    drift = metrics["DataDriftTable"]["drift_by_columns"]
    assert drift["shifted"]["drift_detected"]
    assert not drift["stable"]["drift_detected"]
    assert output.stat().st_size > 0


def test_rejects_a_feature_as_target(splits, tmp_path: Path) -> None:
    # The old report guessed its target and landed on a scaled feature.
    reference, current, model = splits
    with pytest.raises(ValueError, match="not binary"):
        build_monitoring_report(
            reference,
            current,
            model,
            FEATURES,
            THRESHOLD,
            tmp_path / "r.html",
            target="shifted",
        )


def test_rejects_data_missing_a_model_feature(splits, tmp_path: Path) -> None:
    reference, current, model = splits
    with pytest.raises(ValueError, match="lacks"):
        build_monitoring_report(
            reference,
            current.drop(columns="stable"),
            model,
            FEATURES,
            THRESHOLD,
            tmp_path / "r.html",
        )
