# Credit Risk Engine

A production-ready machine learning system to predict credit risk and loan default probability. Built with advanced techniques for handling imbalanced data and optimized for business decision-making.

---

## ⚡ Quick Summary (For Recruiters)

### 🏆 Final Results
- **Selected Model**: XGBoost (Gradient Boosting)
- **ROC-AUC**: 0.9432 (5-fold CV) - Excellent discrimination ability
- **Test Set Performance**:
  - Accuracy: 92.35%
  - Recall: 79.93% (misses only ~20% of defaults)
  - Precision: 83.79% (avoids discarding credit-worthy clients)
  - F1-Score: 0.818 (excellent balance between Precission and Recall)

### 💻 Tech Stack
| Category | Technologies |
|----------|---------------|
| **Languages** | Python 3.8+ |
| **Data Processing** | Pandas, NumPy |
| **ML & Modeling** | Scikit-learn, XGBoost |
| **Evaluation** | Cross-validation, Grid Search, ROC-AUC |
| **Visualization** | Matplotlib, Seaborn |
| **Model Serialization** | Joblib |
| **Development** | Jupyter Notebooks |

### 🎯 Key Decisions
1. **Model Selection**: XGBoost chosen over Logistic Regression, Random Forest, and SVM due to:
   - Highest ROC-AUC (0.945)
   - Best stability across validation folds
   - Superior handling of imbalanced classes with `scale_pos_weight`
   - Well-calibrated probability estimates

2. **Optimization Metric**: 
   - **Recall** (minimize missed defaults - costly in credit risk)

3. **Threshold Selection**: 
   - **Optimal threshold: 0.55** (from grid search 0.1-0.9)
   - Balances precision (86%) and recall (79%) per business requirements

4. **Imbalance Handling**:
   - `scale_pos_weight` in XGBoost
   - Stratified train-test split (80-20)
   - 5-fold stratified cross-validation

### 📊 Quick Usage
```python
import joblib

model = joblib.load('models/best_tuned_model_xgboost.joblib')
threshold = joblib.load('models/optimal_threshold.joblib')

# Get prediction for new applicant
probability = model.predict_proba(X_new)[:, 1]
is_default_risk = probability >= threshold  # 0.55
```

---

## 📋 Detailed Project Information

## 🎯 Project Overview

The **Credit Risk Engine** is a credit scoring system that predicts the probability of loan default using financial and operational indicators. This system employs multiple machine learning algorithms, rigorous cross-validation, and threshold optimization to balance precision and recall for optimal decision-making in credit risk management.

### Key Characteristics
- **Objective**: Predict probability of default (PD) on loan applications
- **Approach**: Supervised classification with imbalanced binary classification handling
- **Models Evaluated**: 4 baseline models + hyperparameter tuning
- **Validation Strategy**: Stratified train-test split + 5-fold cross-validation
- **Threshold Optimization**: Data-driven threshold selection for business alignment

---

## 📊 Detailed Model Performance

### Baseline Models Comparison (Test Set)

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 80.57% | 53.34% | 78.39% | 0.635 | 0.872 |
| Random Forest | 93.61% | 96.88% | 72.67% | 0.830 | 0.932 |
| **XGBoost** | **92.35%** | **83.79%** | **79.93%** | **0.818** | **0.945** |
| SVM | 88.27% | 71.21% | 76.48% | 0.738 | 0.909 |

### Cross-Validation Results (5-Fold Stratified)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|----| ---------|
| Logistic Regression | 0.8062 (±0.0035) | 0.5348 (±0.0061) | 0.7733 (±0.0077) | 0.6323 (±0.0050) | 0.8659 (±0.0034) |
| Random Forest | 0.9340 (±0.0020) | 0.9698 (±0.0038) | 0.7160 (±0.0083) | 0.8238 (±0.0060) | 0.9284 (±0.0042) |
| **XGBoost** | **0.9191 (±0.0035)** | **0.8241 (±0.0095)** | **0.7938 (±0.0085)** | **0.8087 (±0.0083)** | **0.9432 (±0.0035)** |
| SVM | 0.8785 (±0.0044) | 0.7015 (±0.0123) | 0.7599 (±0.0054) | 0.7294 (±0.0078) | 0.9044 (±0.0040) |

---

## 📁 Repository Structure

```
credit-risk-engine/
├── README.md                          # Project documentation
├── LICENSE                            # Project license
├── requirements.txt                   # Python dependencies
│
├── notebooks/                         # Jupyter notebooks for analysis
│   ├── data_ingestion.ipynb          # Data loading and initial exploration
│   ├── exploratory_data_analysis.ipynb # Statistical analysis and visualization
│   ├── feature_engineering.ipynb      # Feature creation and preprocessing
│   └── model_selection.ipynb          # Model training, evaluation, and selection
│
├── data/                              # Datasets (raw and processed)
│   ├── credit_risk_dataset.csv        # Original raw dataset
│   ├── credit_risk_cleaned.csv        # Cleaned dataset (missing values, outliers)
│   ├── credit_risk_fe.csv             # Feature-engineered dataset
│   └── credit_risk_unskewed.csv       # Balanced dataset variant
│
├── models/                            # Trained models and artifacts
│   ├── best_model_xgboost.joblib     # Production XGBoost model
│   ├── best_tuned_model_xgboost.joblib # Tuned XGBoost (after GridSearchCV)
│   ├── preprocessor.joblib            # Scikit-learn pipeline for preprocessing
│   ├── feature_names.joblib           # Feature list for inference
│   └── optimal_threshold.joblib       # Optimal decision threshold
│
├── results/                           # Model evaluation results
│   ├── model_comparison.csv           # Performance metrics comparison
│   ├── cross_validation_results.csv   # 5-fold CV metrics by model
│   ├── threshold_optimization.csv     # Threshold sensitivity analysis
│   └── tuned_models_comparison.csv    # Hyperparameter tuned models results
│
├── src/                               # Source code (production modules)
│   └── (To be populated with API and deployment code)
│
└── tests/                             # Unit tests
    └── (To be populated with test suite)
```

---

## 🔄 Machine Learning Pipeline

### 1️⃣ **Data Ingestion** (`data_ingestion.ipynb`)
- Loaded raw credit risk dataset from Kaggle
- Initial data exploration (shape, types, missing values)
- Basic statistical summaries

### 2️⃣ **Exploratory Data Analysis** (`exploratory_data_analysis.ipynb`)
- Distribution analysis of features and target variable
- Correlation analysis and multicollinearity detection
- Class imbalance assessment
- Visualization of key relationships
- Outlier detection and analysis

### 3️⃣ **Feature Engineering** (`feature_engineering.ipynb`)
- Missing value handling (imputation strategies)
- Categorical encoding (one-hot, label encoding)
- Numerical feature scaling (StandardScaler)
- Feature creation from domain knowledge
- Dataset balancing techniques
- Dimensionality reduction excluding irrelevant features
- Output: `credit_risk_fe.csv` with 50+ engineered features

### 4️⃣ **Model Selection & Evaluation** (`model_selection.ipynb`)

#### Data Preparation
- Train-test split: 80-20 with stratification
- Class balance preservation in splits

#### Models Evaluated
1. **Logistic Regression** - Baseline linear model with balanced class weights
2. **Random Forest** - Ensemble method with balanced class weights
3. **XGBoost** - Gradient boosting with scale_pos_weight for imbalance
4. **SVM (RBF)** - Support Vector Machines with probability estimates

#### Evaluation Methodology
- **Primary Metric**: Recall (reduce number of false negatives)
- **Secondary Metrics**: ROC-AUC (handles imbalance well), Precision an F1-Score
- **Validation**: 5-fold Stratified Cross-Validation
- **Threshold Optimization**: Grid search from 0.1 to 0.9

#### Hyperparameter Tuning
- GridSearchCV applied to XGBoost
- Parameters tuned: max_depth, learning_rate, n_estimators, subsample
- Results saved in `tuned_models_comparison.csv`

---

## � Model Insights & Analysis

### Strengths
 - **High ROC-AUC (0.945)** - Excellent discrimination between default/non-default  
 - **Strong Recall (79.93%)** - Captures ~80% of actual defaults (critical for credit risk)  
 - **Good Precision (83.79%)** - Minimizes false positives, reduces unnecessary rejections  
 - **Stable Performance** - Consistent across 5-fold cross-validation folds  
 - **Well-calibrated Probabilities** - Suitable for business-driven threshold optimization  

### Technical Considerations
 - **Imbalanced Classes** - Handled with `scale_pos_weight` and stratified sampling  
 - **Feature Importance** - XGBoost provides native feature importance for interpretability  
 - **Threshold Dependency** - Final performance depends on selected decision threshold (0.55 optimal)  

### Model Trade-offs
 - **vs Logistic Regression**: XGBoost captures non-linear patterns better (+19.5% ROC-AUC)
 - **vs Random Forest**: XGBoost has better recall (+7.3%) while maintaining similar precision
 - **vs SVM**: XGBoost more stable and faster to train while maintaining better discriminative power

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- pip or conda

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/credit-risk-engine.git
cd credit-risk-engine

# Install required dependencies
pip install -r requirements.txt

# Launch Jupyter for notebook exploration
jupyter notebook
```
---

## 📚 Detailed Implementation

---

## 📄 File Reference Guide

| File | Purpose |
|------|---------|
| `best_tuned_model_xgboost.joblib` | Production-ready XGBoost model |
| `optimal_threshold.joblib` | Decision threshold (0.55) for binary classification |
| `preprocessor.joblib` | Scikit-learn pipeline with feature scaling and encoding |
| `feature_names.joblib` | Feature list for ensuring consistency |
| `model_comparison.csv` | Performance comparison of 4 baseline models |
| `cross_validation_results.csv` | 5-fold CV metrics with standard deviations |
| `threshold_optimization.csv` | Sensitivity analysis across decision thresholds |

---

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Antonio Albaladejo Soriano**  
[LinkedIn Profile](https://www.linkedin.com/in/antonio-albaladejo-soriano-3133211b7/)

---

**Last Updated**: February 2026
