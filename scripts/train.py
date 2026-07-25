"""Leak-free training pipeline for the credit risk model.

Replaces the notebook path, where the preprocessor was fitted and features were
selected on the full dataset *before* the train/test split (see docs/AUDIT.md C5).
Here every fit happens on training data only, the decision threshold is chosen on
a validation split, and the test split is touched exactly once, at the end.

Runs three arms so the two open questions can be answered with numbers:

  leaky-weighted     replica of the old pipeline, to quantify the leakage bias
  clean-weighted     correct split, scale_pos_weight (current production setup)
  clean-unweighted   correct split, no imbalance handling

Usage:
    uv run python scripts/train.py                 # compare arms, write no artifacts
    uv run python scripts/train.py --save <arm>    # promote one arm to models/
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import (  # noqa: E402
    DATA_DIR,
    FEATURE_NAMES_PATH,
    MODEL_PATH,
    PREPROCESSOR_PATH,
    THRESHOLD_PATH,
)
from src.preprocessing import create_derived_features  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

RANDOM_STATE = 42
TARGET = "loan_status"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

# Hyperparameters from the GridSearchCV run in model_selection.ipynb. Held fixed
# across arms so each comparison isolates a single variable.
XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.1,
    "subsample": 0.8,
}

# Feature selection thresholds, matching the original recipe.
CORRELATION_THRESHOLD = 0.85
IMPORTANCE_THRESHOLD = 0.01
VARIANCE_THRESHOLD = 0.01


def build_preprocessor(
    numeric_cols: list[str], cat_cols: list[str]
) -> ColumnTransformer:
    """Median-impute + scale numerics, mode-impute + one-hot the categoricals."""
    return ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_cols,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "ohe",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                cat_cols,
            ),
        ],
        remainder="drop",
    )


def select_features(X: pd.DataFrame, y: pd.Series) -> list[str]:
    """Three-stage filter: correlation, tree importance, then variance.

    Must only ever see training data. Running this on the full dataset is what
    made the original metrics optimistic.
    """
    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    dropped = [c for c in upper.columns if any(upper[c] > CORRELATION_THRESHOLD)]
    X = X.drop(columns=dropped)
    logger.info("  correlation  : -%d features -> %d", len(dropped), X.shape[1])

    rf = RandomForestClassifier(
        n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1
    ).fit(X, y)
    keep = X.columns[rf.feature_importances_ >= IMPORTANCE_THRESHOLD].tolist()
    X = X[keep]
    logger.info("  importance   : -> %d features", X.shape[1])

    selector = VarianceThreshold(threshold=VARIANCE_THRESHOLD).fit(X)
    keep = X.columns[selector.get_support()].tolist()
    logger.info("  variance     : -> %d features", len(keep))
    return keep


def strip_prefix(names) -> list[str]:
    return [n.split("__", 1)[1] if "__" in n else n for n in names]


def pick_threshold(y_true, proba) -> tuple[float, pd.DataFrame]:
    """Choose the F1-maximising threshold on the validation split.

    Selecting it on test would leak the test set into the reported operating
    point, which is the same class of mistake as C5 itself.
    """
    grid = np.round(np.arange(0.05, 0.96, 0.01), 2)
    rows = [
        {
            "Threshold": t,
            "Precision": precision_score(y_true, proba >= t, zero_division=0),
            "Recall": recall_score(y_true, proba >= t, zero_division=0),
            "F1-Score": f1_score(y_true, proba >= t, zero_division=0),
            "Accuracy": accuracy_score(y_true, proba >= t),
        }
        for t in grid
    ]
    table = pd.DataFrame(rows)
    return float(table.loc[table["F1-Score"].idxmax(), "Threshold"]), table


def evaluate(y_true, proba, threshold: float) -> dict:
    pred = (proba >= threshold).astype(int)
    return {
        "ROC-AUC": roc_auc_score(y_true, proba),
        "PR-AUC": average_precision_score(y_true, proba),
        "Brier": brier_score_loss(y_true, proba),
        "mean_predicted": float(proba.mean()),
        "Accuracy": accuracy_score(y_true, pred),
        "Precision": precision_score(y_true, pred, zero_division=0),
        "Recall": recall_score(y_true, pred, zero_division=0),
        "F1-Score": f1_score(y_true, pred, zero_division=0),
    }


def run_arm(name: str, df: pd.DataFrame, *, leaky: bool, weighted: bool) -> dict:
    """Train and evaluate one configuration end to end."""
    logger.info("\n%s\n=== %s ===", "-" * 70, name)

    y_all = df[TARGET]
    X_all = create_derived_features(df.drop(columns=[TARGET]))
    numeric_cols = X_all.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X_all.select_dtypes(include=["object", "category"]).columns.tolist()

    # 64/16/20. The test split is held out from every fit and from the threshold
    # choice, so it stays a genuine estimate of unseen performance.
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X_all, y_all, test_size=0.2, random_state=RANDOM_STATE, stratify=y_all
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=0.2, random_state=RANDOM_STATE, stratify=y_tmp
    )

    if leaky:
        # Reproduces the notebook: fit and select on everything, split afterwards.
        pre = build_preprocessor(numeric_cols, cat_cols).fit(X_all)
        names = strip_prefix(pre.get_feature_names_out())
        selected = select_features(
            pd.DataFrame(pre.transform(X_all), columns=names), y_all
        )
    else:
        pre = build_preprocessor(numeric_cols, cat_cols).fit(X_train)
        names = strip_prefix(pre.get_feature_names_out())
        selected = select_features(
            pd.DataFrame(pre.transform(X_train), columns=names), y_train
        )

    def prep(X):
        return pd.DataFrame(pre.transform(X), columns=names)[selected]

    Xtr, Xval, Xte = prep(X_train), prep(X_val), prep(X_test)

    ratio = float((y_train == 0).sum() / (y_train == 1).sum())
    model = XGBClassifier(
        **XGB_PARAMS,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        scale_pos_weight=ratio if weighted else 1.0,
    ).fit(Xtr, y_train)

    threshold, threshold_table = pick_threshold(y_val, model.predict_proba(Xval)[:, 1])
    metrics = evaluate(y_test, model.predict_proba(Xte)[:, 1], threshold)
    metrics |= {"threshold": threshold, "n_features": len(selected)}

    logger.info(
        "  threshold=%.2f  ROC-AUC=%.4f  PR-AUC=%.4f  Brier=%.4f  F1=%.4f",
        threshold,
        metrics["ROC-AUC"],
        metrics["PR-AUC"],
        metrics["Brier"],
        metrics["F1-Score"],
    )

    return {
        "arm": name,
        "metrics": metrics,
        "model": model,
        "preprocessor": pre,
        "features": selected,
        "threshold": threshold,
        "threshold_table": threshold_table,
    }


def log_to_mlflow(results: list[dict]) -> None:
    """Log every arm as an MLflow run. Skipped silently if MLflow is unavailable."""
    try:
        import mlflow
    except ImportError:
        logger.warning("\nmlflow not installed, skipping experiment tracking")
        return

    import os

    uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("credit-risk-leakage-and-weighting")

    for r in results:
        with mlflow.start_run(run_name=r["arm"]):
            mlflow.log_params(
                XGB_PARAMS
                | {
                    "arm": r["arm"],
                    "n_features": r["metrics"]["n_features"],
                    "threshold": r["threshold"],
                }
            )
            mlflow.log_metrics(
                {k: v for k, v in r["metrics"].items() if isinstance(v, int | float)}
            )
    logger.info("\nLogged %d runs to MLflow at %s", len(results), uri)
    logger.info("View with: uv run mlflow ui --backend-store-uri %s", uri)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--save",
        choices=["clean-weighted", "clean-unweighted"],
        help="promote this arm's artifacts to models/",
    )
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()

    df = pd.read_csv(DATA_DIR / "credit_risk_cleaned.csv")
    logger.info("dataset: %d rows, %.2f%% positives", len(df), 100 * df[TARGET].mean())

    results = [
        run_arm("leaky-weighted", df, leaky=True, weighted=True),
        run_arm("clean-weighted", df, leaky=False, weighted=True),
        run_arm("clean-unweighted", df, leaky=False, weighted=False),
    ]

    table = pd.DataFrame(
        [{"arm": r["arm"], **r["metrics"]} for r in results]
    ).set_index("arm")
    logger.info("\n%s\nTEST SET COMPARISON\n%s", "=" * 100, "=" * 100)
    logger.info(table.round(4).to_string())

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / "leakage_and_weighting_comparison.csv"
    table.to_csv(out)
    logger.info("\nSaved comparison to %s", out)

    if not args.no_mlflow:
        log_to_mlflow(results)

    if args.save:
        chosen = next(r for r in results if r["arm"] == args.save)
        joblib.dump(chosen["model"], MODEL_PATH)
        joblib.dump(chosen["preprocessor"], PREPROCESSOR_PATH)
        joblib.dump(chosen["features"], FEATURE_NAMES_PATH)
        joblib.dump(chosen["threshold"], THRESHOLD_PATH)

        # Threshold sweep on validation, so the operating point can be moved on
        # business grounds without rerunning training.
        chosen["threshold_table"].to_csv(
            RESULTS_DIR / "threshold_optimization.csv", index=False
        )
        logger.info(
            "\nPromoted '%s' to models/ (threshold %.2f)",
            args.save,
            chosen["threshold"],
        )


if __name__ == "__main__":
    main()
