"""Render the figures the README relies on.

All come from measured data, never from hand-drawn numbers: the threshold
sweep is read straight from results/threshold_optimization.csv, and the
calibration curve, confusion matrix and reason-code impact are recomputed on
the held-out test split with the shipped model.

The split is rebuilt through train.prepare() rather than reimplemented, so the
figures cannot silently drift from the pipeline that produced the artifacts.

Usage:
    uv run python scripts/plot_results.py
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import xgboost as xgb  # noqa: E402
from sklearn.metrics import ConfusionMatrixDisplay  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import (  # noqa: E402
    DATA_DIR,
    FEATURE_NAMES_PATH,
    MODEL_PATH,
    THRESHOLD_PATH,
)
from src.explainer import REASON_GROUPS  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = ROOT / "results"
ASSETS_DIR = ROOT / "assets"
N_BINS = 10

# Matches the muted palette of the badges, and stays legible in both GitHub themes.
COLORS = {"precision": "#4C78A8", "recall": "#E45756", "f1": "#54A24B"}


def load_train_module():
    """Import scripts/train.py, which is a script rather than a package."""
    spec = importlib.util.spec_from_file_location(
        "train", ROOT / "scripts" / "train.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def plot_threshold_sweep(path: Path) -> None:
    """Precision, recall and F1 across the decision threshold, on validation."""
    table = pd.read_csv(RESULTS_DIR / "threshold_optimization.csv")
    chosen = float(table.loc[table["F1-Score"].idxmax(), "Threshold"])

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for column, color, label in (
        ("Precision", COLORS["precision"], "Precision"),
        ("Recall", COLORS["recall"], "Recall"),
        ("F1-Score", COLORS["f1"], "F1"),
    ):
        ax.plot(table["Threshold"], table[column], color=color, label=label, lw=2)

    ax.axvline(chosen, color="#333333", ls="--", lw=1.2)
    ax.annotate(
        f"selected: {chosen:.2f}",
        xy=(chosen, 0.05),
        xytext=(chosen + 0.03, 0.05),
        fontsize=9,
        color="#333333",
    )

    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Score")
    ax.set_title("Threshold sweep on the validation split", fontsize=11)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("wrote %s (threshold %.2f)", path, chosen)


def calibration_by_decile(proba, actual) -> tuple[pd.DataFrame, float]:
    """Bin predictions into deciles and compare mean predicted to observed rate."""
    binned = pd.DataFrame({"proba": proba, "actual": actual})
    binned["bin"] = pd.qcut(binned["proba"], N_BINS, labels=False, duplicates="drop")
    grouped = binned.groupby("bin").agg(
        predicted=("proba", "mean"), observed=("actual", "mean")
    )
    return grouped, (grouped["predicted"] - grouped["observed"]).abs().mean()


def load_split(train) -> dict:
    """Rebuild the splits and check they match the shipped bundle."""
    data = train.prepare(pd.read_csv(DATA_DIR / "credit_risk_cleaned.csv"), leaky=False)
    expected = list(joblib.load(FEATURE_NAMES_PATH))
    if data["features"] != expected:
        raise RuntimeError(
            "Feature selection no longer matches models/feature_names.joblib. "
            "Rerun scripts/train.py --save clean-unweighted before plotting."
        )
    return data


def plot_calibration(path: Path, train, data: dict, model) -> None:
    """Predicted probability against observed default rate, by decile.

    Plots the weighted arm alongside the shipped one: the gap between the two
    curves is the whole argument for dropping scale_pos_weight.
    """
    actual = data["y_test"].to_numpy()
    shipped, ece = calibration_by_decile(model.predict_proba(data["Xte"])[:, 1], actual)

    # Retrained here rather than stored: it is a counter-example for the figure,
    # not a deployment candidate, so it has no business in models/.
    y_train = data["y_train"]
    ratio = float((y_train == 0).sum() / (y_train == 1).sum())
    weighted_model = train.XGBClassifier(
        **train.XGB_PARAMS,
        random_state=train.RANDOM_STATE,
        eval_metric="logloss",
        scale_pos_weight=ratio,
    ).fit(data["Xtr"], y_train)
    weighted, weighted_ece = calibration_by_decile(
        weighted_model.predict_proba(data["Xte"])[:, 1], actual
    )

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([0, 1], [0, 1], color="#999999", ls="--", lw=1, label="Perfect calibration")
    ax.plot(
        weighted["predicted"],
        weighted["observed"],
        marker="s",
        ms=4,
        color=COLORS["recall"],
        lw=1.6,
        alpha=0.85,
        label=f"with scale_pos_weight (ECE {weighted_ece:.4f})",
    )
    ax.plot(
        shipped["predicted"],
        shipped["observed"],
        marker="o",
        ms=5,
        color=COLORS["precision"],
        lw=2,
        label=f"shipped, unweighted (ECE {ece:.4f})",
    )

    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed default rate")
    ax.set_title("Calibration by decile on the test split", fontsize=11)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("wrote %s (ECE %.4f)", path, ece)


def plot_confusion_matrix(path: Path, data: dict, model) -> None:
    """Test-split outcomes at the tuned threshold, as counts."""
    threshold = float(joblib.load(THRESHOLD_PATH))
    predicted = (model.predict_proba(data["Xte"])[:, 1] >= threshold).astype(int)

    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay.from_predictions(
        data["y_test"],
        predicted,
        display_labels=["Repaid", "Defaulted"],
        cmap="Blues",
        colorbar=False,
        values_format=",d",
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Test split, threshold {threshold:.2f}", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("wrote %s", path)


def mean_abs_by_reason(contribs: pd.DataFrame) -> pd.Series:
    """Mean |SHAP| per reason code, summing within a group before taking abs.

    Summing first matches how RiskExplainer reports a reason: dummies of one
    block offset each other, and abs-then-sum would overstate the block.
    """
    return pd.Series(
        {
            code: contribs[list(names)].sum(axis=1).abs().mean()
            for code, names in REASON_GROUPS.items()
        }
    ).sort_values()


def plot_reason_importance(path: Path, data: dict, model) -> None:
    """Average impact of each reason code on the test split, in log-odds."""
    features = data["features"]
    contribs = model.get_booster().predict(
        xgb.DMatrix(data["Xte"], feature_names=features), pred_contribs=True
    )
    impact = mean_abs_by_reason(pd.DataFrame(contribs[:, :-1], columns=features))

    fig, ax = plt.subplots(figsize=(7, 4))
    labels = [code.replace("_", " ").capitalize() for code in impact.index]
    bars = ax.barh(labels, impact.to_numpy(), color=COLORS["precision"], height=0.6)
    ax.bar_label(bars, fmt="%.2f", padding=4, fontsize=9, color="#333333")
    ax.set_xlabel("Mean |SHAP contribution| (log-odds)")
    ax.set_title("What drives the score, on the test split", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("wrote %s", path)


def main() -> None:
    ASSETS_DIR.mkdir(exist_ok=True)
    train = load_train_module()
    data = load_split(train)
    model = joblib.load(MODEL_PATH)
    plot_threshold_sweep(ASSETS_DIR / "threshold_sweep.png")
    plot_calibration(ASSETS_DIR / "calibration.png", train, data, model)
    plot_confusion_matrix(ASSETS_DIR / "confusion_matrix.png", data, model)
    plot_reason_importance(ASSETS_DIR / "reason_importance.png", data, model)


if __name__ == "__main__":
    main()
