# MedGuardAI - Health Insurance Fraud Detection using Big Data

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end machine learning pipeline for detecting healthcare fraud in Medicare Part B claims data. This project implements a **distributed ensemble model** (XGBoost + LightGBM + Logistic Regression stacker) trained on ~8 million provider records with extreme class imbalance (0.04% fraud rate).

## 🎯 Overview

MedGuardAI addresses the critical challenge of Medicare fraud detection by:

- **Processing CMS Medicare Part B data** (2018-2023) with PySpark on GCP Dataproc
- **Engineering peer-relative features** (Z-scores, risk ratios) to detect anomalous billing patterns
- **Training distributed ensemble models** using Dask, Optuna HPO, and MLflow tracking
- **Handling extreme imbalance** (0.04% positive class) via strategic undersampling and F2-Score optimization

### Key Features

| Feature | Description |
|---------|-------------|
| **Distributed Training** | Vertex AI + Dask cluster for 8M+ records |
| **Ensemble Architecture** | XGBoost + LightGBM with Logistic Regression stacker |
| **Fraud Labeling** | LEIE (List of Excluded Individuals/Entities) integration |
| **Provider-Stratified CV** | Prevents data leakage across validation folds |
| **MLflow Tracking** | Full experiment tracking with GCS artifact storage |

---

## 📁 Project Structure

```
medGuardAI-fraudDetection/
├── config/
│   └── pipeline_config.yaml      # Centralized pipeline configuration
├── src/
│   ├── data/
│   │   ├── ingest_cms_data.py    # CMS API data ingestion
│   │   ├── data_preprocessing.py # PySpark feature engineering
│   │   └── preprocess.sh         # Dataproc job submission
│   ├── models/
│   │   └── train.py              # Main training orchestrator
│   └── utils/
│       ├── data_loader.py        # Data loading utilities
│       ├── cv_splitter.py        # Provider-stratified CV
│       └── evaluation.py         # Metrics (AUCPR, F2, TPR, TNR)
├── training/
│   ├── Dockerfile                # Training container image
│   └── train.sh                  # Vertex AI job submission
├── mlflow-server/
│   ├── Dockerfile                # MLflow server container
│   └── deploy_mlflow_server.sh   # Cloud Run deployment script
└── requirements.txt
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Google Cloud SDK (`gcloud`)
- Docker (for Vertex AI training)
- GCP Project with enabled APIs (Vertex AI, Cloud Storage, BigQuery)

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/naman2398/medGuardAI-fraudDetection.git
cd medGuardAI-fraudDetection

# Install dependencies
pip install -r requirements.txt
```

### 2. GCP Authentication

```bash
# Option A: User credentials
gcloud auth application-default login

# Option B: Service account
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account-key.json"
```

### 3. Configuration

Edit `config/pipeline_config.yaml` to customize:

```yaml
data:
  gcs_bucket: "your-bucket-name"
  training_data_path: "gs://your-bucket/data_processed"

models:
  xgboost:
    n_trials: 20    # Optuna HPO trials
    cv_folds: 5     # Cross-validation folds
  lightgbm:
    n_trials: 20
    cv_folds: 5

sampling:
  undersample_enabled: true
  undersample_ratio: 100  # 1:100 fraud-to-normal ratio
```

---

## 📊 Data Pipeline

### Data Sources

| Source | Description | Link |
|--------|-------------|------|
| **CMS Part B by Provider & Service** | Claims-level billing data (~9M rows/year) | [CMS Data](https://data.cms.gov/provider-summary-by-type-of-service/medicare-physician-other-practitioners/medicare-physician-other-practitioners-by-provider-and-service) |
| **CMS Part B Summary by Provider** | Provider-level aggregates (~1M rows/year) | [CMS Data](https://data.cms.gov/provider-summary-by-type-of-service/medicare-physician-other-practitioners/medicare-physician-other-practitioners-by-provider/data) |
| **LEIE Exclusion List** | OIG fraud labels (~75K records) | [OIG LEIE](https://oig.hhs.gov/exclusions/exclusions_list.asp) |

### Preprocessing Workflow

```
Raw CMS Data → PySpark Cleaning → Aggregation → Enrichment → Feature Engineering → Model-Ready Data
```

**Key transformations:**
1. **PII Removal** - Drop identifying columns
2. **Type Casting** - Convert currency/count fields
3. **Provider Aggregation** - 6 statistics (min, max, mean, median, sum, std) per metric
4. **Z-Score Features** - Peer-relative comparisons by specialty
5. **Risk Ratios** - Billing inflation, service density
6. **Fraud Labeling** - Join with LEIE exclusion data

### Run Data Ingestion

```bash
# Ingest CMS data from API to GCS
python src/data/ingest_cms_data.py

# Run PySpark preprocessing on Dataproc
./src/data/preprocess.sh
```

---

## 🤖 Model Training

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     TRAINING PIPELINE                        │
├─────────────────────────────────────────────────────────────┤
│  1. Data Load (Dask)         → 8M rows distributed          │
│  2. XGBoost HPO (Optuna)     → 20 trials, 5-fold CV         │
│  3. LightGBM HPO (Optuna)    → 20 trials, 5-fold CV         │
│  4. OOF Predictions          → Generate stacker training    │
│  5. Stacker Training         → Logistic Regression          │
│  6. Final Models             → Train on 100% data           │
└─────────────────────────────────────────────────────────────┘
```

### Run Training Locally

```bash
# Quick test (sampled data)
python src/models/train.py \
    --n_rows_sample 100000 \
    --xgb_trials 5 \
    --lgbm_trials 5 \
    --cv_folds 3
```

### Run Training on Vertex AI

```bash
# Full distributed training
./training/train.sh

# Custom configuration
PRIMARY_MACHINE=n2-highmem-16 \
WORKER_COUNT=2 \
XGB_TRIALS=20 \
LGBM_TRIALS=20 \
CV_FOLDS=5 \
./training/train.sh
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `N_ROWS_SAMPLE` | `-1` (all) | Number of rows to sample |
| `XGB_TRIALS` | `20` | Optuna trials for XGBoost |
| `LGBM_TRIALS` | `20` | Optuna trials for LightGBM |
| `CV_FOLDS` | `5` | Cross-validation folds |
| `PRIMARY_MACHINE` | `e2-standard-4` | Primary node machine type |
| `WORKER_COUNT` | `2` | Number of Dask workers |

---

## 📈 Evaluation Metrics

### Primary Metrics

| Metric | Description | Why Used |
|--------|-------------|----------|
| **AUCPR** | Area Under Precision-Recall Curve | Best for extreme imbalance |
| **F2-Score** | Weighted F-score (β=2) | Prioritizes Recall |

### Threshold-Based Metrics (at 0.04% threshold)

- **TPR (Recall)** - True Positive Rate
- **TNR (Specificity)** - True Negative Rate  
- **Precision** - Positive Predictive Value

### Class Imbalance Strategy

```
Original Data:     0.04% Fraud (1:2500 ratio)
Undersampled:      1.00% Fraud (1:100 ratio)
scale_pos_weight:  [25-100] search range
```

---

## 📊 MLflow Experiment Tracking

MLflow server runs on **Google Cloud Run** with PostgreSQL backend and GCS artifact storage.

### Deploy MLflow Server

```bash
# Deploy MLflow server to Cloud Run
cd mlflow-server
./deploy_mlflow_server.sh
```

### Access MLflow UI

Once deployed, access the MLflow UI directly in your browser:
```
https://mlflow-server-<PROJECT_NUMBER>.<REGION>.run.app
```

The tracking URI is configured in `config/pipeline_config.yaml`.

### Tracked Artifacts

- HPO trial parameters and metrics
- Best hyperparameters per model
- Trained model files (`.bst`, `.joblib`)
- Feature importance rankings
- Cross-validation results

---

## 🔧 Feature Engineering Details

### Peer-Relative Z-Scores

Compares each provider to their specialty peer group:

```python
Z_score = (X - μ_specialty) / σ_specialty
Ratio = X / μ_specialty
```

**Features created:**
- `tot_srvcs_zscore`, `tot_srvcs_ratio`
- `avg_mdcr_pymt_amt_zscore`, `avg_mdcr_pymt_amt_ratio`
- `tot_benes_zscore`, `tot_benes_ratio`
- `avg_sbmtd_chrg_zscore`, `avg_sbmtd_chrg_ratio`

### Risk Ratios

```python
billing_inflation = avg_submitted_charge / avg_medicare_payment
service_density = total_services / total_beneficiaries
```

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|--------------|
| **Data Processing** | PySpark, Pandas, PyArrow |
| **ML Frameworks** | XGBoost, LightGBM, Scikit-learn |
| **Distributed Compute** | Dask, GCP Vertex AI, Dataproc |
| **HPO** | Optuna |
| **Experiment Tracking** | MLflow |
| **Cloud Storage** | Google Cloud Storage, BigQuery |
| **Containerization** | Docker, Artifact Registry |
