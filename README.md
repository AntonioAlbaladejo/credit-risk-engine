# Credit Risk Engine

Probability-of-default scoring for consumer loans: a calibrated XGBoost model served as a FastAPI
service, containerised and deployed to AWS ECS Fargate.

[![CI](https://github.com/AntonioAlbaladejo/credit-risk-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/AntonioAlbaladejo/credit-risk-engine/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-black.svg)](https://github.com/astral-sh/ruff)

---

## Overview

Given a loan application — applicant income, employment, home ownership, requested amount, interest
rate, credit history — the service returns the probability that the loan will default, a binary
decision at a tuned threshold, and the threshold it used.

The model is trained on 31,679 historical applications with a 21.5% default rate. On a held-out test
split it reaches **0.9460 ROC-AUC** and **0.9019 PR-AUC**, and its output is calibrated closely
enough to be read as a real probability of default rather than an arbitrary score
(expected calibration error 0.0076).

Two properties drive most of the design:

- **The probability is the product, not the label.** A calibrated score can be repriced, turned into
  expected loss, or moved to a different operating point without retraining. A binary yes/no cannot.
- **Training and serving share one implementation.** Feature derivation lives in a single function
  imported by both the training script and the API, because divergence between the two is the
  failure mode that no test suite catches on its own.

The repository covers the full path: exploration notebooks, a reproducible training pipeline, the
serving code, the container, and the CI/CD workflows that deploy it.

---

## Architecture

```mermaid
flowchart TB
    subgraph training["Training  ·  scripts/train.py"]
        direction LR
        raw[("credit_risk_cleaned.csv<br/>31,679 rows")]
        derive["create_derived_features()"]
        split["Stratified split<br/>64 / 16 / 20"]
        fit["Fit preprocessor<br/>+ select features<br/>(train split only)"]
        train["XGBoost"]
        thr["Pick threshold<br/>(on validation)"]
        raw --> derive --> split --> fit --> train --> thr
    end

    subgraph bundle["Versioned artifact bundle  ·  models/"]
        art["preprocessor · model<br/>feature_names · threshold"]
    end

    subgraph serving["Serving  ·  src/api"]
        direction LR
        api["FastAPI<br/>/predict · /health"]
        pre["DataPreprocessor<br/>(same derivation)"]
        api --> pre --> art
    end

    subgraph deploy["Delivery  ·  GitHub Actions"]
        direction LR
        ci["CI: ruff + pytest"]
        img["Docker build<br/>+ live /health and /predict check"]
        ecr[("Amazon ECR")]
        ecs["ECS Fargate<br/>eu-west-1"]
        ci --> img --> ecr --> ecs
    end

    thr --> art
    derive -.->|shared code| pre
    serving --> img

    mlflow[("MLflow<br/>params · metrics · runs")]
    training -.-> mlflow
```

The four artifacts are produced together by one training run and loaded together at startup. They
are never mixed across runs: the preprocessor that scaled the training data, the feature list it
produced, the model fitted on it and the threshold tuned for it are one unit.

---

## Results

All figures below are measured on the 6,336-row test split, which is held out from every `fit` call
and from the threshold search. Seed `42` throughout.

### Production model

XGBoost, `n_estimators=300`, `max_depth=4`, `learning_rate=0.1`, `subsample=0.8`, no class weighting,
decision threshold 0.39, 18 features.

| Metric | Value | |
|---|---|---|
| ROC-AUC | **0.9460** | ranking quality |
| PR-AUC | **0.9019** | ranking quality on the minority class |
| Brier score | **0.0520** | probability accuracy (lower is better) |
| Expected calibration error | **0.0076** | mean gap between predicted and observed, by decile |
| Recall | 0.7553 | share of real defaults caught |
| Precision | 0.9313 | share of rejections that were real defaults |
| F1 | 0.8341 | at threshold 0.39 |
| Accuracy | 0.9353 | |
| Mean predicted P(default) | 0.2159 | against an observed rate of 0.2154 |

Reproduce with `uv run python scripts/train.py`; the table is written to
[`results/leakage_and_weighting_comparison.csv`](results/leakage_and_weighting_comparison.csv).

### Model selection

Four algorithms under an identical pipeline — same split, same preprocessor, same 18 features, no
imbalance handling anywhere — so the only variable between rows is the algorithm itself.

| Model | ROC-AUC | PR-AUC | Brier ↓ | Precision | Recall | F1 | Threshold |
|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.8722 | 0.7325 | 0.1005 | 0.6605 | 0.7026 | 0.6809 | 0.33 |
| Random Forest | 0.9309 | 0.8823 | 0.0568 | 0.9256 | 0.7473 | 0.8269 | 0.42 |
| **XGBoost** | **0.9460** | **0.9019** | **0.0520** | **0.9313** | **0.7553** | **0.8341** | 0.39 |
| SVM (RBF) | 0.9012 | 0.8415 | 0.0711 | 0.8524 | 0.7026 | 0.7703 | 0.40 |

XGBoost wins on every column simultaneously, so the choice hides no trade-off. Logistic regression
is the informative loser: it trails by 7.4 ROC-AUC points and 17 PR-AUC points, which says the
decision boundary here is genuinely non-linear — grade, intent and home ownership interact with
loan-to-income rather than adding up.

`uv run python scripts/train.py --baselines` →
[`results/baseline_comparison.csv`](results/baseline_comparison.csv).

### Threshold selection

![Precision, recall and F1 across the decision threshold](assets/threshold_sweep.png)

The threshold is a tuned artifact, not a default of 0.5. It is chosen on the **validation** split by
maximising F1, then applied unchanged to test — selecting it on test would leak the test set into
the reported operating point.

The curve is deliberately flat between roughly 0.3 and 0.7, which is the useful part: moving the
operating point trades precision for recall along a shallow slope, so the cut-off can be set on
business grounds without collapsing the model. At 0.39 the model catches 75.5% of defaults while
93.1% of its rejections are real ones. The full sweep is in
[`results/threshold_optimization.csv`](results/threshold_optimization.csv).

### Calibration

![Calibration by decile, with and without class weighting](assets/calibration.png)

This is the reason the model ships without `scale_pos_weight`. Both curves come from the same
algorithm, the same features and the same split; the only difference is the class weighting.

The weighted model systematically over-predicts risk — its expected calibration error is 0.0903
against 0.0076, twelve times worse — and it buys nothing for it: +0.0005 ROC-AUC, which is noise.
Weighting inflates mean predicted probability to 0.3046 against a true default rate of 0.2154, and
the optimal threshold drifts up to 0.69 to compensate.

Without it, mean predicted probability lands at 0.2159 against that same 0.2154, no decile deviates
more than 1.6 percentage points, and the Brier score improves by 24%.

### Feature importance

| Feature | Gain | Feature | Gain |
|---|---|---|---|
| `loan_percent_income` | 0.142 | `loan_intent_DEBTCONSOLIDATION` | 0.057 |
| `person_home_ownership_MORTGAGE` | 0.112 | `loan_intent_HOMEIMPROVEMENT` | 0.047 |
| `person_home_ownership_OWN` | 0.111 | `loan_intent_MEDICAL` | 0.041 |
| `loan_grade_A` | 0.100 | `person_income` | 0.035 |
| `loan_grade_C` | 0.088 | `loan_grade_E` | 0.034 |
| `loan_grade_D` | 0.075 | `person_emp_length` | 0.025 |
| `loan_int_rate` | 0.067 | `person_age` | 0.018 |

Debt burden relative to income dominates, followed by housing status and the lender's own grade.
Absolute loan amount ranks near the bottom — what matters is the amount *relative to income*, which
is exactly the derived feature the pipeline adds.

---

## Design decisions

**No SMOTE, and no class weighting either.** At 21.5% positives (3.64:1) the imbalance is mild.
Three SMOTE variants were measured and all three degraded both ROC-AUC and PR-AUC; 15% of the
synthetic rows carried a `loan_grade` block that was not a valid one-hot, because interpolating over
already-encoded columns produces categories that do not exist. More decisively, once the threshold
is tuned every arm converges to an F1 between 0.8370 and 0.8447 — the benefit oversampling promises
is the problem threshold tuning already solves. Class weighting was then dropped for the calibration
reason above.

**Every `fit` happens inside the training split.** The preprocessor, the feature selector and the
threshold search see training or validation data only; test is touched once, at the end. An earlier
notebook pipeline fitted and selected over the full dataset before splitting. Measured side by side,
that leak turned out **not** to inflate the reported metrics — the leaky arm scored marginally worse
(0.9447 vs 0.9455 ROC-AUC) — but the methodology was wrong regardless, and a pipeline that is only
accidentally correct will not stay correct.

| Arm | ROC-AUC | PR-AUC | Brier ↓ | Mean predicted | Threshold |
|---|---|---|---|---|---|
| Old pipeline: leaky + weighted | 0.9447 | 0.9001 | 0.0696 | 0.3063 | 0.71 |
| Clean split + weighted | 0.9455 | 0.9013 | 0.0683 | 0.3046 | 0.69 |
| **Clean split, unweighted** (shipped) | **0.9460** | **0.9019** | **0.0520** | **0.2159** | 0.39 |

![Brier score across the three arms, compared in MLflow](assets/mlflow_brier_comparison.png)

Each arm is logged as an MLflow run with `leaky` and `weighted` as separate parameters, so either
factor can be isolated. Brier is the metric that actually separates them: the two weighted arms sit
together near 0.069 while the unweighted one drops to 0.052, and fixing the leak alone moves it
almost nothing.

**Recall is the primary metric, with PR-AUC as the guardrail.** A missed default costs the principal;
a false rejection costs a customer. The asymmetry justifies favouring recall, but optimising it alone
degenerates to approving nothing, so ROC-AUC and PR-AUC constrain the search and the threshold sets
the final balance.

**One feature-derivation implementation.** `create_derived_features()` in
[`src/preprocessing.py`](src/preprocessing.py) is imported by both the training script and the
serving path. It previously existed twice and the copies had already drifted — different
zero-division handling, different bucket dtypes. That is train/serve skew that produces no error and
no failing test, just quietly wrong predictions.

**ECS Fargate rather than a managed inference endpoint.** The artifact bundle is under 500 KB and
inference is a single tree ensemble; the workload needs a container, not a specialised serving stack.
Fargate keeps it as a plain HTTP service with no vendor-specific container contract, deployable and
testable identically on a laptop and in the cloud.

**Model artifacts are versioned in git.** They total roughly 500 KB. At that size, object storage or
LFS adds infrastructure and a failure mode without buying anything, and keeping the artifacts beside
the code that loads them means any checkout builds a working image.

**MLflow is opt-in.** The registry lookup runs only when `MLFLOW_TRACKING_URI` is set; otherwise the
predictor loads from joblib directly. Probing an unreachable tracking server cost 247 seconds of
retry backoff on startup — the cost is the retries, not a hanging socket, so no timeout setting fixes
it. A dependency that is unreachable must degrade, not block startup.

---

## Quickstart

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
git clone https://github.com/AntonioAlbaladejo/credit-risk-engine.git
cd credit-risk-engine
uv sync --all-groups
uv run uvicorn src.api.main:app --reload --port 8000
```

Interactive docs at <http://localhost:8000/docs>.

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"person_age": 23, "person_income": 24000, "person_home_ownership": "RENT",
       "person_emp_length": 1, "loan_intent": "DEBTCONSOLIDATION", "loan_grade": "E",
       "loan_amnt": 12000, "loan_int_rate": 16.0, "loan_percent_income": 0.5,
       "cb_person_default_on_file": 1, "cb_person_cred_hist_length": 3}'
```

```json
{
  "prediction": 1,
  "probability_default": 0.9998610019683838,
  "probability_non_default": 0.00013899803161621094,
  "risk_level": "high_risk",
  "threshold_used": 0.39,
  "recommendation": "Reject application"
}
```

### Docker

```bash
docker build -t credit-risk-engine:local .
docker run --rm -p 8000:8000 credit-risk-engine:local
curl http://localhost:8000/health
```

The image carries the model artifacts, so a fresh clone builds a container that serves real
predictions. It becomes healthy in about 8 seconds.

### Retraining

```bash
uv run python scripts/train.py                          # compare the three arms
uv run python scripts/train.py --baselines              # compare the four algorithms
uv run python scripts/train.py --save clean-unweighted  # promote a run to models/
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

---

## API reference

![The /predict endpoint in the generated OpenAPI documentation](assets/swagger_predict.png)

![The response returned by the service for that request](assets/swagger_response.png)

FastAPI generates this documentation from the Pydantic schemas, so it cannot drift from what the
service actually accepts, and requests can be sent straight from the page. Note `threshold_used` in
the response: the service reports the operating point it applied rather than leaving the caller to
assume 0.5.

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Service metadata and endpoint index |
| `GET` | `/health` | `200` when the model is loaded, `503` when it is not |
| `GET` | `/model/info` | Model type, threshold, and the 18 feature names in order |
| `POST` | `/predict` | Score one application |
| `POST` | `/predict/batch` | Score up to 100 applications in one call |
| `GET` | `/docs` · `/redoc` | OpenAPI documentation |

`/health` returning `503` rather than `200` with an `unhealthy` body is deliberate: a load balancer
reads the status code, so an instance that cannot serve predictions has to fail the check rather than
report its problem in a payload nobody parses.

### Request contract

Pydantic v2 schemas are the contract, and their bounds live in [`src/config.py`](src/config.py) so
they cannot drift from the ranges seen in training.

| Field | Type | Constraint |
|---|---|---|
| `person_age` | int | 18–100 |
| `person_income` | float | ≥ 0 |
| `person_emp_length` | float | 0–80 years |
| `person_home_ownership` | enum | `RENT` · `OWN` · `MORTGAGE` · `OTHER` |
| `loan_intent` | enum | `PERSONAL` · `EDUCATION` · `MEDICAL` · `VENTURE` · `HOMEIMPROVEMENT` · `DEBTCONSOLIDATION` |
| `loan_grade` | enum | `A`–`G` |
| `loan_amnt` | float | 500–100,000 |
| `loan_int_rate` | float | 1.0–50.0, **as a percentage** |
| `loan_percent_income` | float | 0–1 |
| `cb_person_default_on_file` | int | 0 or 1 |
| `cb_person_cred_hist_length` | int | ≥ 0 |

`loan_int_rate` has a floor of 1.0 rather than 0 on purpose. Training data ranges from 5.42 to 23.22
in percent units, so a caller sending a fraction (`0.08` for 8%) has to be rejected with a `422`
rather than silently scaled to 3.5 standard deviations below anything the model has seen.

---

## Project structure

```
credit-risk-engine/
├── src/
│   ├── config.py              # paths, API metadata, validation bounds
│   ├── preprocessing.py       # raw dict -> model matrix; shared feature derivation
│   ├── predictor.py           # artifact loading (MLflow -> joblib fallback), predict
│   ├── model_monitoring.py    # Evidently drift and quality reports
│   └── api/
│       ├── main.py            # FastAPI app, lifespan, endpoints
│       └── schemas.py         # Pydantic request/response contracts
├── scripts/
│   ├── train.py               # leak-free training pipeline, arms and baselines
│   └── plot_results.py        # regenerates the figures in assets/
├── models/                    # the four versioned artifacts, loaded as one bundle
├── results/                   # measured comparison tables
├── notebooks/                 # ingestion -> EDA -> feature engineering -> model selection
├── tests/                     # 88 tests
└── .github/workflows/         # ci.yml (ruff, pytest) · cd.yml (build, ECR, Fargate)
```

---

## Development

```bash
uv run pytest                                        # 88 tests, ~4s
uv run pytest --cov=src --cov-report=term-missing    # coverage
uv run ruff check . && uv run ruff format --check .  # exactly what CI runs
```

**CI** (`ci.yml`) runs ruff, then the test suite with coverage, on every push and pull request to
`main` and `develop`.

**CD** (`cd.yml`) triggers only on a successful CI run, so nothing reaches AWS without passing the
suite. It builds the image, **starts the container and calls `/health` and `/predict` against it**,
then pushes to ECR and deploys to Fargate. That container check matters more than it looks: the step
it replaced ran `python -c "import src"`, which passes even when the model artifacts are missing from
the image entirely.

**Monitoring.** [`src/model_monitoring.py`](src/model_monitoring.py) generates Evidently drift and
quality reports into `results/`.

![The service running on ECS Fargate](assets/fargate_service.png)

The deployed service on Fargate in `eu-west-1`, serving from task definition revision 10.

---

## Tech stack

| Layer | Tools |
|---|---|
| Modelling | XGBoost, scikit-learn, pandas, NumPy |
| Tracking | MLflow (params, metrics, artifacts; SQLite backend locally) |
| Monitoring | Evidently |
| Serving | FastAPI, Pydantic v2, uvicorn |
| Packaging | uv, Docker multi-stage build |
| Quality | pytest, pytest-cov, ruff |
| Delivery | GitHub Actions, Amazon ECR, ECS Fargate (eu-west-1) |

---

## Dataset

The [Credit Risk Dataset](https://www.kaggle.com/datasets/laotse/credit-risk-dataset) from Kaggle:
32,581 loan applications with 11 features and a binary `loan_status` target. Cleaning drops
implausible ages (≥ 100) and employment lengths (> 60 years) together with the 895 records that have
no employment length, and median-imputes the 3,116 missing interest rates. That leaves **31,679 rows
at a 21.5% default rate** (6,825 defaults, a 3.64:1 ratio).

Feature engineering adds five derived columns — `loan_to_income`, `employ_to_age`, `age_bucket`,
`emp_length_bin`, `default_flag` — which expand to 41 columns after one-hot encoding. A three-stage
filter (correlation > 0.85, tree importance, variance) fitted on the training split alone reduces
that to the **18 features** the model uses.

Raw data is not committed; the notebooks and `scripts/train.py` read it from `data/`.

---

## Limitations and open work

Stated plainly, because they are the honest state of the repository rather than a roadmap pitch.

- **The notebooks still document the old pipeline.** `scripts/train.py` is the source of truth for
  the shipped artifacts, but `feature_engineering.ipynb` and `model_selection.ipynb` still show the
  fit-before-split ordering and class weighting. The repository currently tells two stories; aligning
  them is the next task.
- **The test suite mocks `joblib.load` with an autouse fixture**, so a green run says little about
  the real inference path. Recent tests exercise the real feature-derivation logic, but the test that
  would matter most — a known application scored against the real artifacts to an expected
  probability — does not exist yet. The CD pipeline's live container check is what covers that gap
  today.
- **The container is 2.94 GB**, down from 4.08 GB. What remains is dominated by `mlflow`, `evidently`,
  `seaborn` and an unused CUDA library sitting in the runtime dependency set; moving them to optional
  groups should bring it to a few hundred megabytes.
- **CORS is wide open** (`allow_origins=["*"]`), which suits a demonstration service and nothing
  exposed to real traffic.
- **The Evidently report compares against a three-row hand-written reference file**, so its drift
  numbers are not meaningful yet.

---

## License

MIT — see [LICENSE](LICENSE).

## Author

**Antonio Albaladejo Soriano** ·
[LinkedIn](https://www.linkedin.com/in/antonio-albaladejo-soriano-3133211b7/)
