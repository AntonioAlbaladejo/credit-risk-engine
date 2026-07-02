# Credit Risk Engine

A production-ready machine learning system for credit scoring and loan default probability prediction. The repository includes a FastAPI inference service, serialized model artifacts, a reproducible dependency workflow, Docker deployment, and CI/CD automation.

---

## ⚡ Executive Summary

### 🏆 Final outcome
- **Production model**: XGBoost classifier
- **Primary evaluation goal**: recall and ROC-AUC for reliable default detection
- **Model threshold**: optimized to 0.55 for balanced decision-making
- **API service**: FastAPI with validated inference and batch scoring

### 💡 What makes this project strong
- Reproducible dependency management with `uv`, `pyproject.toml`, and `uv.lock`
- FastAPI inference service with typed request/response schemas
- Docker multi-stage build with runtime-only artifacts
- GitHub Actions workflows for linting, testing, and image validation
- Model inference path designed for production readiness

### 🧰 Tech stack
- Python 3.10+
- FastAPI, Uvicorn
- XGBoost, scikit-learn, pandas, numpy
- joblib
- GitHub Actions
- Docker

---

## 📦 Repository layout

```
credit-risk-engine/
├── README.md
├── LICENSE
├── Dockerfile
├── pyproject.toml
├── uv.lock
├── .dockerignore
├── .github/workflows/
│   ├── ci.yml
│   └── cd.yml
├── data/
├── models/
├── results/
├── notebooks/
├── src/
│   ├── api/
│   │   ├── main.py
│   │   └── schemas.py
│   ├── config.py
│   ├── predictor.py
│   ├── preprocessing.py
│   ├── model_monitoring.py
│   └── __init__.py
└── tests/
    ├── test_api.py
    ├── test_config.py
    ├── test_preprocessing.py
    └── test_schemas.py
```

> The runtime Docker image includes only `src/` and `models/`; notebooks, raw data, and intermediate results are excluded from production packaging.

---

## 📈 Results summary

### Baseline test metrics

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 80.57% | 53.34% | 78.39% | 0.635 | 0.872 |
| Random Forest | 93.61% | 96.88% | 72.67% | 0.830 | 0.932 |
| **XGBoost** | **92.35%** | **83.79%** | **79.93%** | **0.818** | **0.945** |
| SVM | 88.27% | 71.21% | 76.48% | 0.738 | 0.909 |

### 5-fold cross-validation

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|----|---------|
| Logistic Regression | 0.8062 (±0.0035) | 0.5348 (±0.0061) | 0.7733 (±0.0077) | 0.6323 (±0.0050) | 0.8659 (±0.0034) |
| Random Forest | 0.9340 (±0.0020) | 0.9698 (±0.0038) | 0.7160 (±0.0083) | 0.8238 (±0.0060) | 0.9284 (±0.0042) |
| **XGBoost** | **0.9191 (±0.0035)** | **0.8241 (±0.0095)** | **0.7938 (±0.0085)** | **0.8087 (±0.0083)** | **0.9432 (±0.0035)** |
| SVM | 0.8785 (±0.0044) | 0.7015 (±0.0123) | 0.7599 (±0.0054) | 0.7294 (±0.0078) | 0.9044 (±0.0040) |

---

## 📡 API reference

The production API is defined in `src/api/main.py` and provides the following endpoints:

- `GET /health` — returns service health and whether the model is loaded successfully
- `GET /model/info` — returns model metadata and supported feature names
- `POST /predict` — returns a single loan application prediction
- `POST /predict/batch` — returns batch predictions for up to 100 applications
- `GET /` — root endpoint with API summary

### Input schema
The API accepts loan application data validated by `src/api/schemas.py`.

Required fields:
- `person_age`
- `person_income`
- `person_emp_length`
- `person_home_ownership`
- `loan_intent`
- `loan_grade`
- `loan_amnt`
- `loan_int_rate`
- `loan_percent_income`
- `cb_person_default_on_file`
- `cb_person_cred_hist_length`

### Response fields
- `prediction`: binary default risk label (`1` = default, `0` = no default)
- `probability_default`
- `probability_non_default`
- `risk_level`
- `threshold_used`
- `recommendation`

---

## 🧠 Architecture

### `src/predictor.py`
- Loads model artifacts from `models/`
- Attempts MLflow model loading first, then falls back to joblib
- Applies an optimized threshold to probability outputs
- Produces structured prediction metadata for API responses

### `src/preprocessing.py`
- Validates raw API inputs
- Transforms numeric and categorical fields into model-ready features
- Supports single and batch preprocessing flows

### `src/config.py`
- Central configuration for artifact paths
- API metadata and validation limits

### `src/model_monitoring.py`
- Contains drift monitoring and evaluation support for production monitoring workflows

---

## 🧪 Testing

### Run tests
```bash
uv run pytest tests/ -v
```

### Coverage goals
- API endpoints
- schema validation
- preprocessing pipeline
- model artifact loading and inference

---

## 🐳 Docker

The repository includes a production Docker image built with a multi-stage `Dockerfile`.

### Build and run
```bash
docker build -t credit-risk-engine:latest .
docker run --rm -p 8000:8000 credit-risk-engine:latest
```

### Deployment details
- Base image: `python:3.11.10-slim`
- Uses `uv` in the builder stage to install pinned dependencies
- Copies only runtime artifacts into the final image
- Runs the app as a non-root user
- Defines a healthcheck against `/health`

---

## ⚙️ CI/CD

### GitHub Actions
This repository includes two workflows:

- `.github/workflows/ci.yml`
  - runs on `push` and `pull_request` to `main` and `develop`
  - installs `uv` and Python 3.11
  - installs dependencies with `uv sync --frozen`
  - runs `ruff check .` and `ruff format --check .`
  - runs `pytest tests/ -v --cov=src --cov-report=xml`
  - uploads coverage reports to Codecov

- `.github/workflows/cd.yml`
  - triggers on successful `main` builds
  - builds the Docker image
  - validates the image by loading it and running a smoke import
  - includes a commented registry push step for future deployment

---

## ⚙️ Local setup

### Preferred workflow
```bash
git clone https://github.com/yourusername/credit-risk-engine.git
cd credit-risk-engine
uv sync
source .venv/bin/activate
```

### Legacy fallback
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt.bak
```

### Run the API locally
```bash
uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

---

## 📌 Reviewer notes

- `uv` is the primary package manager and dependency source for reproducibility
- `requirements.txt.bak` remains available for compatibility with legacy pip workflows
- The Docker image excludes notebooks, raw data, and results for production efficiency
- The FastAPI app currently allows broad CORS for ease of integration; production should narrow origins
- The CD workflow is ready for registry push once credentials are configured

---

## 👤 Author
**Antonio Albaladejo Soriano**

[LinkedIn](https://www.linkedin.com/in/antonio-albaladejo-soriano-3133211b7/)

---

**Last Updated**: February 2026
