from pathlib import Path

import pandas as pd

from src.model_monitoring import build_simple_monitoring_report


def test_build_simple_monitoring_report_creates_html_summary(tmp_path: Path) -> None:
    reference_data = pd.DataFrame(
        {
            "default": [0, 1, 0, 1],
            "score": [0.2, 0.6, 0.3, 0.8],
        }
    )
    current_data = pd.DataFrame(
        {
            "default": [0, 1, 1, 1],
            "score": [0.25, 0.55, 0.7, 0.9],
        }
    )

    output_path = tmp_path / "monitoring_report.html"
    result_path = build_simple_monitoring_report(
        reference_data=reference_data,
        current_data=current_data,
        output_path=output_path,
        target="default",
    )

    assert result_path == output_path
    assert output_path.exists()
    html = output_path.read_text(encoding="utf-8")
    assert "Monitoring summary" in html
    assert "Target drift" in html


def test_build_simple_monitoring_report_auto_detects_target_column(
    tmp_path: Path,
) -> None:
    reference_data = pd.DataFrame(
        {
            "default_flag": [0, 1, 0, 1],
            "score": [0.2, 0.6, 0.3, 0.8],
        }
    )
    current_data = pd.DataFrame(
        {
            "default_flag": [0, 1, 1, 1],
            "score": [0.25, 0.55, 0.7, 0.9],
        }
    )

    output_path = tmp_path / "monitoring_report.html"
    result_path = build_simple_monitoring_report(
        reference_data=reference_data,
        current_data=current_data,
        output_path=output_path,
        target="default",
    )

    assert result_path == output_path
    assert output_path.exists()
    html = output_path.read_text(encoding="utf-8")
    assert "Target drift" in html
