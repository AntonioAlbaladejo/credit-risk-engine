from __future__ import annotations

import html
import os
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

try:
    from evidently.metric_preset import (
        ClassificationPreset,
        DataDriftPreset,
        TargetDriftPreset,
    )
    from evidently.report import Report

    EVIDENTLY_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on environment
    ClassificationPreset = None
    DataDriftPreset = None
    TargetDriftPreset = None
    Report = None
    EVIDENTLY_AVAILABLE = False

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data"
MODELS_PATH = ROOT_DIR / "models"
RESULTS_PATH = ROOT_DIR / "results"


def _load_artifacts() -> tuple[Any, Any, list[str], Any]:
    model = joblib.load(MODELS_PATH / "best_tuned_model_xgboost.joblib")
    preprocessor = joblib.load(MODELS_PATH / "preprocessor.joblib")
    feature_names = joblib.load(MODELS_PATH / "feature_names.joblib")
    threshold = joblib.load(MODELS_PATH / "optimal_threshold.joblib")
    return model, preprocessor, feature_names, threshold


def _resolve_target_column(
    reference_data: pd.DataFrame, current_data: pd.DataFrame, target: str
) -> str:
    candidate_names = [target]
    if target != "default":
        candidate_names.append("default")

    candidate_names.extend(["default_flag", "loan_status", "target", "y"])

    for candidate in candidate_names:
        if candidate in reference_data.columns and candidate in current_data.columns:
            return candidate

    for column in reference_data.columns:
        if column in current_data.columns and pd.api.types.is_integer_dtype(
            reference_data[column]
        ):
            return column

    raise ValueError(
        f"Could not identify a compatible target column from {list(reference_data.columns)}"
    )


def _build_simple_html_report(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    output_path: Path,
    target: str,
) -> Path:
    resolved_target = _resolve_target_column(reference_data, current_data, target)

    reference_copy = reference_data.copy()
    current_copy = current_data.copy()

    if resolved_target not in reference_copy.columns:
        raise ValueError(
            f"Target column '{resolved_target}' not found in reference data"
        )
    if resolved_target not in current_copy.columns:
        raise ValueError(f"Target column '{resolved_target}' not found in current data")

    reference_target = reference_copy[resolved_target].astype(int)
    current_target = current_copy[resolved_target].astype(int)

    feature_columns = [col for col in reference_copy.columns if col != resolved_target]
    summary_rows: list[str] = []

    for column in feature_columns:
        if column not in current_copy.columns:
            continue

        reference_values = reference_copy[column]
        current_values = current_copy[column]

        if pd.api.types.is_numeric_dtype(
            reference_values
        ) and pd.api.types.is_numeric_dtype(current_values):
            reference_mean = float(reference_values.mean())
            current_mean = float(current_values.mean())
            drift_value = abs(current_mean - reference_mean)
            metric_text = f"mean shift: {drift_value:.3f}"
        else:
            reference_modes = (
                reference_values.astype(str).value_counts().head(3).to_dict()
            )
            current_modes = current_values.astype(str).value_counts().head(3).to_dict()
            metric_text = f"reference categories: {reference_modes}; current categories: {current_modes}"

        summary_rows.append(
            f"<tr><td>{html.escape(str(column))}</td><td>{html.escape(metric_text)}</td></tr>"
        )

    html_content = f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <title>Monitoring summary</title>
  <style>body{{font-family:Arial,sans-serif; margin:2rem;}} table{{border-collapse:collapse; width:100%;}} th, td{{border:1px solid #ccc; padding:0.5rem; text-align:left;}} th{{background:#f5f5f5;}}</style>
</head>
<body>
  <h1>Monitoring summary</h1>
  <p>Reference rows: {len(reference_copy)} | Current rows: {len(current_copy)}</p>
  <p>Reference target rate: {reference_target.mean():.2%} | Current target rate: {current_target.mean():.2%}</p>
  <h2>Target drift</h2>
  <p>Target prevalence changed by {abs(float(current_target.mean()) - float(reference_target.mean())):.3%}.</p>
  <h2>Feature-level summary</h2>
  <table>
    <thead><tr><th>Feature</th><th>Observation</th></tr></thead>
    <tbody>
      {"".join(summary_rows) if summary_rows else '<tr><td colspan="2">No comparable feature columns were found.</td></tr>'}
    </tbody>
  </table>
</body>
</html>
"""
    output_path.write_text(html_content, encoding="utf-8")
    return output_path


def build_simple_monitoring_report(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    output_path: str | os.PathLike[str],
    target: str = "default",
) -> Path:
    """Create a lightweight monitoring report that does not depend on Evidently."""
    output_path = Path(output_path)
    return _build_simple_html_report(reference_data, current_data, output_path, target)


def build_monitoring_report(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    output_path: str | os.PathLike[str],
    target: str = "default",
) -> Path:
    """Create a monitoring report using Evidently when available, otherwise fall back to a simple report."""
    output_path = Path(output_path)

    if EVIDENTLY_AVAILABLE and Report is not None:
        try:
            reference_data_with_target = reference_data.copy()
            current_data_with_target = current_data.copy()
            resolved_target = _resolve_target_column(
                reference_data_with_target, current_data_with_target, target
            )

            if resolved_target not in reference_data_with_target.columns:
                raise ValueError(
                    f"Target column '{resolved_target}' not found in reference data"
                )
            if resolved_target not in current_data_with_target.columns:
                raise ValueError(
                    f"Target column '{resolved_target}' not found in current data"
                )

            report = Report(
                metrics=[
                    DataDriftPreset(),
                    TargetDriftPreset(),
                    ClassificationPreset(
                        target_name=resolved_target,
                        prediction_feature_names=["prediction"],
                        classification_task="binary",
                        probas_features=["prediction"],
                    ),
                ]
            )

            if "prediction" not in reference_data_with_target.columns:
                reference_data_with_target["prediction"] = 0.0
            if "prediction" not in current_data_with_target.columns:
                current_data_with_target["prediction"] = 0.0

            report.run(
                reference_data=reference_data_with_target,
                current_data=current_data_with_target,
            )
            report.save_html(str(output_path))
            return output_path
        except Exception as exc:  # pragma: no cover - depends on environment
            print(
                f"Falling back to simple monitoring report because Evidently failed: {exc}"
            )

    return _build_simple_html_report(reference_data, current_data, output_path, target)


def generate_monitoring_report(
    reference_path: str | os.PathLike[str] | None = None,
    current_path: str | os.PathLike[str] | None = None,
    output_path: str | os.PathLike[str] | None = None,
    target: str = "default",
) -> Path:
    """Generate a monitoring report from CSV files using the project data directory by default."""
    reference_path = Path(reference_path or DATA_PATH / "credit_risk_fe.csv")
    current_path = Path(current_path or DATA_PATH / "test_samples.csv")
    output_path = Path(output_path or RESULTS_PATH / "model_monitoring_report.html")

    reference_data = pd.read_csv(reference_path)
    current_data = pd.read_csv(current_path)
    return build_monitoring_report(reference_data, current_data, output_path, target)


def main() -> None:
    report_path = generate_monitoring_report()
    print(f"Monitoring report saved to {report_path}")


if __name__ == "__main__":
    main()
