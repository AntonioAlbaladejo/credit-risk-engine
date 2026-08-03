"""Leak-free training pipeline for the credit risk model.

Replaces the notebook path, where the preprocessor was fitted and features were
selected on the full dataset *before* the train/test split (see docs/AUDIT.md C5).
Here every fit happens on training data only, the decision threshold is chosen on
a validation split, and the test split is touched exactly once, at the end.

Runs three arms so the two open questions can be answered with numbers:

  leaky-weighted     replica of the old pipeline, to quantify the leakage bias
  clean-weighted     correct split, scale_pos_weight (current production setup)
  clean-unweighted   correct split, no imbalance handling

A fourth mode compares the candidate algorithms themselves under that same
clean pipeline, which is what backs the model-selection table in the README.

Usage:
    uv run python scripts/train.py                 # compare arms, write no artifacts
    uv run python scripts/train.py --save <arm>    # promote one arm to models/
    uv run python scripts/train.py --baselines     # compare LR / RF / XGBoost / SVM
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
from sklearn.linear_model import LogisticRegression
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
from sklearn.svm import SVC
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

# Present in the raw data but deliberately not offered to the pipeline. Credit
# history length correlates 0.878 with person_age, so the correlation filter has
# dropped it on every run; keeping it as an input meant the API demanded a value
# that could never reach the model. Excluded here so the fitted preprocessor
# stops expecting it and the request schema can drop the field.
UNUSED_COLUMNS = ["cb_person_cred_hist_length"]

# Hyperparameters from the GridSearchCV run in model_selection.ipynb. The grid is
# scored on PR-AUC and selected with the one-standard-error rule: 15 of the 81
# configurations land within one standard error of the best, and this is the
# simplest of them. Held fixed across arms so each comparison isolates a single
# variable.
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


def one_hot_blocks(
    preprocessor: ColumnTransformer, cat_cols: list[str]
) -> list[list[str]]:
    """Name the encoded columns that belong to each categorical variable.

    Read off the fitted encoder rather than guessed from column prefixes, for
    the same reason the serving path does it: the categories and their spelling
    are a property of the fit, not of the source column names.

    Args:
        preprocessor: A fitted ColumnTransformer built by ``build_preprocessor``.
        cat_cols: The categorical columns, in the order they were passed to it.

    Returns:
        One list of encoded column names per categorical variable.
    """
    ohe = preprocessor.named_transformers_["cat"]["ohe"]
    return [
        [f"{col}_{category}" for category in categories]
        for col, categories in zip(cat_cols, ohe.categories_, strict=True)
    ]


def keep_blocks_whole(
    selected: list[str], available: list[str], blocks: list[list[str]]
) -> list[str]:
    """Re-add the siblings of any one-hot column that survived selection.

    Dropping a single column out of a one-hot block does not remove
    information: the block is exhaustive, so the missing category is still
    identified by the all-zeros pattern. Dropping *two or more* does, because
    the survivors become indistinguishable from each other. That is silent --
    the model simply learns the merged group's average risk.

    It bit loan_grade: the importance filter kept A, C, D and E, leaving B, F
    and G fused into one all-zeros bucket that is 97% grade B. F (70.3% default)
    and G (98.4%) inherited B's 15.9%, and with F and G at 0.94% of the rows no
    aggregate metric moved.

    So a block is kept whole or dropped whole. The cost is a handful of columns
    a per-column filter would have called weak; the alternative is a model that
    is confidently wrong about its riskiest applicants.

    Args:
        selected: Columns the filters kept.
        available: Everything the preprocessor emits, in its order -- measured
            *before* the correlation filter runs, since that filter is no better
            suited to one-hot blocks than the others. It dropped
            person_home_ownership_RENT for correlating 0.853 with
            person_home_ownership_MORTGAGE, but that figure is mechanical: two
            mutually exclusive dummies at 50% and 41% prevalence are
            anti-correlated by construction. Compare loan_to_income against
            loan_percent_income at 0.999, which is real duplication.
        blocks: One-hot blocks, as returned by ``one_hot_blocks``.

    Returns:
        ``selected`` plus the reinstated siblings, in ``available`` order.
    """
    keep = set(selected)
    for block in blocks:
        present = [c for c in block if c in available]
        if keep.intersection(present):
            keep.update(present)
    return [c for c in available if c in keep]


def select_features(
    X: pd.DataFrame, y: pd.Series, blocks: list[list[str]]
) -> list[str]:
    """Three-stage filter: correlation, tree importance, then variance.

    Must only ever see training data. Running this on the full dataset is what
    made the original metrics optimistic.

    The three filters score columns one at a time, which is wrong for one-hot
    blocks, so ``keep_blocks_whole`` repairs their verdict at the end rather
    than each filter having to know about encoding.

    Args:
        X: Transformed training matrix.
        y: Training labels.
        blocks: One-hot blocks, as returned by ``one_hot_blocks``.

    Returns:
        The selected column names, in the order ``X`` presents them.
    """
    available = X.columns.tolist()

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

    keep = keep_blocks_whole(keep, available, blocks)
    logger.info("  whole blocks : -> %d features", len(keep))
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


def prepare(df: pd.DataFrame, *, leaky: bool) -> dict:
    """Split, fit the preprocessor and select features.

    Shared by every caller on purpose: a second copy of the split is exactly
    how the fit-before-split bug would come back.
    """
    y_all = df[TARGET]
    X_all = create_derived_features(df.drop(columns=[TARGET, *UNUSED_COLUMNS]))
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
        fit_X, fit_y = X_all, y_all
    else:
        fit_X, fit_y = X_train, y_train

    pre = build_preprocessor(numeric_cols, cat_cols).fit(fit_X)
    names = strip_prefix(pre.get_feature_names_out())
    selected = select_features(
        pd.DataFrame(pre.transform(fit_X), columns=names),
        fit_y,
        one_hot_blocks(pre, cat_cols),
    )

    def prep(X):
        return pd.DataFrame(pre.transform(X), columns=names)[selected]

    return {
        "Xtr": prep(X_train),
        "Xval": prep(X_val),
        "Xte": prep(X_test),
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "preprocessor": pre,
        "features": selected,
    }


def score_model(model, data: dict) -> tuple[dict, pd.DataFrame]:
    """Fit on train, choose the threshold on validation, score once on test."""
    model.fit(data["Xtr"], data["y_train"])
    threshold, threshold_table = pick_threshold(
        data["y_val"], model.predict_proba(data["Xval"])[:, 1]
    )
    metrics = evaluate(
        data["y_test"], model.predict_proba(data["Xte"])[:, 1], threshold
    )
    metrics |= {"threshold": threshold, "n_features": len(data["features"])}

    logger.info(
        "  threshold=%.2f  ROC-AUC=%.4f  PR-AUC=%.4f  Brier=%.4f  F1=%.4f",
        threshold,
        metrics["ROC-AUC"],
        metrics["PR-AUC"],
        metrics["Brier"],
        metrics["F1-Score"],
    )
    return metrics, threshold_table


def run_arm(name: str, df: pd.DataFrame, *, leaky: bool, weighted: bool) -> dict:
    """Train and evaluate one configuration end to end."""
    logger.info("\n%s\n=== %s ===", "-" * 70, name)

    data = prepare(df, leaky=leaky)
    y_train = data["y_train"]

    ratio = float((y_train == 0).sum() / (y_train == 1).sum())
    model = XGBClassifier(
        **XGB_PARAMS,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        scale_pos_weight=ratio if weighted else 1.0,
    )
    metrics, threshold_table = score_model(model, data)

    return {
        "arm": name,
        # The two factors go in as separate params, not just fused into the arm
        # name, so the comparison can be grouped and filtered by either one.
        "params": XGB_PARAMS
        | {
            "leaky": leaky,
            "weighted": weighted,
            "scale_pos_weight": model.scale_pos_weight,
        },
        "metrics": metrics,
        "model": model,
        "preprocessor": data["preprocessor"],
        "features": data["features"],
        "threshold": metrics["threshold"],
        "threshold_table": threshold_table,
    }


def run_baselines(df: pd.DataFrame) -> list[dict]:
    """Compare the four candidate algorithms under the clean pipeline.

    Same split, same preprocessor, same feature set and no imbalance handling
    anywhere, so the only variable left is the algorithm. The XGBoost row is by
    construction the model that ships.
    """
    data = prepare(df, leaky=False)

    candidates = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            **XGB_PARAMS, random_state=RANDOM_STATE, eval_metric="logloss"
        ),
        # Features are already scaled by the preprocessor, so the RBF kernel is
        # on a sane footing. probability=True adds an internal Platt calibration
        # pass, which is what makes the Brier column comparable.
        "SVM (RBF)": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
    }

    results = []
    for name, model in candidates.items():
        logger.info("\n%s\n=== %s ===", "-" * 70, name)
        metrics, _ = score_model(model, data)
        results.append(
            {
                "arm": name,
                "params": model.get_params(),
                "metrics": metrics,
                "threshold": metrics["threshold"],
            }
        )
    return results


def log_to_mlflow(results: list[dict], experiment: str) -> None:
    """Log every arm as an MLflow run. Skipped silently if MLflow is unavailable."""
    try:
        import mlflow
    except ImportError:
        logger.warning("\nmlflow not installed, skipping experiment tracking")
        return

    import os

    uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment)

    for r in results:
        with mlflow.start_run(run_name=r["arm"]):
            mlflow.log_params(
                r["params"]
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
    parser.add_argument(
        "--baselines",
        action="store_true",
        help="compare the four candidate algorithms instead of the three arms",
    )
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()

    if args.baselines and args.save:
        parser.error("--save promotes an arm; it does not apply to --baselines")

    df = pd.read_csv(DATA_DIR / "credit_risk_cleaned.csv")
    logger.info("dataset: %d rows, %.2f%% positives", len(df), 100 * df[TARGET].mean())

    if args.baselines:
        results = run_baselines(df)
        experiment = "credit-risk-baselines"
        out = RESULTS_DIR / "baseline_comparison.csv"
        index_name = "model"
    else:
        results = [
            run_arm("leaky-weighted", df, leaky=True, weighted=True),
            run_arm("clean-weighted", df, leaky=False, weighted=True),
            run_arm("clean-unweighted", df, leaky=False, weighted=False),
        ]
        experiment = "credit-risk-leakage-and-weighting"
        out = RESULTS_DIR / "leakage_and_weighting_comparison.csv"
        index_name = "arm"

    table = pd.DataFrame(
        [{index_name: r["arm"], **r["metrics"]} for r in results]
    ).set_index(index_name)
    logger.info("\n%s\nTEST SET COMPARISON\n%s", "=" * 100, "=" * 100)
    logger.info(table.round(4).to_string())

    RESULTS_DIR.mkdir(exist_ok=True)
    table.to_csv(out)
    logger.info("\nSaved comparison to %s", out)

    if not args.no_mlflow:
        log_to_mlflow(results, experiment)

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
