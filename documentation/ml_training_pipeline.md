# Distributed Ensemble ML Pipeline

**Source File:** `src/models/train.py`  
**Execution Environment:** GCP Vertex AI Custom Jobs with Dask

---

## Overview

This document describes the distributed training pipeline for a stacked ensemble model (XGBoost + LightGBM + Logistic Regression) designed to detect Medicare fraud. The pipeline handles ~8 million records with extreme class imbalance (0.04% fraud rate) using provider-stratified cross-validation, Optuna hyperparameter optimization, and MLflow experiment tracking.

---

## System Architecture

### Components

| Component | Purpose | Technology |
|-----------|---------|------------|
| **Compute** | Distributed training cluster | GCP Vertex AI Custom Jobs |
| **Parallelization** | Data loading & training | Dask (LocalCluster or Distributed) |
| **HPO** | Hyperparameter tuning | Optuna |
| **Experiment Tracking** | Metrics, parameters, artifacts | MLflow (Cloud Run hosted) |
| **Storage** | Data & model artifacts | Google Cloud Storage |
| **Container Registry** | Training image | GCP Artifact Registry |

### Cluster Configuration

From `config/pipeline_config.yaml`:

| Node | Machine Type | vCPUs | Memory |
|------|--------------|-------|--------|
| Primary | n2-highmem-16 | 16 | 128 GB |
| Workers | n2-highmem-16 | 16 | 128 GB each |

**Default:** Single-node execution (worker_count=0) for cost efficiency.

---

## Pipeline Execution Flow

### Entry Point

```bash
# Submit training job
PRIMARY_MACHINE=n2-highmem-16 WORKER_COUNT=0 \
XGB_TRIALS=5 LGBM_TRIALS=5 CV_FOLDS=3 \
./training/submit_job.sh
```

### Main Class: `DistributedTrainer`

**Initialization:**
```python
trainer = DistributedTrainer(config)
trainer.run_training()
```

### Execution Steps

| Step | Method | Description |
|------|--------|-------------|
| 1 | `setup_dask()` | Initialize Dask cluster (local or distributed) |
| 2 | `setup_mlflow()` | Connect to MLflow tracking server |
| 3 | `load_data()` | Load data via Dask, create provider-stratified folds |
| 4 | `_precompute_folds()` | Undersample and cache fold data for HPO efficiency |
| 5 | `optimize_xgboost()` | Optuna HPO for XGBoost (5 trials, 3-fold CV) |
| 6 | `optimize_lightgbm()` | Optuna HPO for LightGBM (5 trials, 3-fold CV) |
| 7 | `generate_oof_predictions()` | Create out-of-fold predictions for stacking |
| 8 | `train_stacker()` | Train Logistic Regression on OOF predictions |
| 9 | `train_final_models()` | Retrain XGB/LGBM on full (undersampled) data |
| 10 | `save_models()` | Save artifacts to GCS, log to MLflow |

---

## Data Handling

### Loading

**Function:** `load_training_data()` (from `src/utils/data_loader.py`)

- Reads Parquet files from `gs://medguard_rawdata/data_processed/`
- Returns Dask DataFrame for distributed processing
- Optional row sampling via `n_rows_sample` config

### Provider-Stratified Cross-Validation

**File:** `src/utils/cv_splitter.py`

**Purpose:** Prevent data leakage by splitting at provider level, not row level.

```python
# Create stratified folds at provider level
folds = create_provider_stratified_folds(
    provider_labels,  # DataFrame with (provider_npi, fraud_label)
    n_folds=3,
    random_state=42
)
```

**Implementation:**
1. Group all records by `rndrng_npi` to get one fraud label per provider
2. Apply `StratifiedKFold` on provider list (ensures ~0.04% fraud in each fold)
3. Map provider folds back to row-level train/test splits

### Undersampling Strategy

**Purpose:** Address extreme class imbalance (0.04% fraud) by training on a controlled ratio.

**Configuration:**
```yaml
sampling:
  undersample_enabled: true
  undersample_ratio: 100  # 1:100 fraud-to-normal ratio
```

**Implementation:** `undersample_dask_dataframe()`
1. Keep ALL fraud cases (N_fraud ≈ 3,200)
2. Sample N_fraud × 100 normal cases (~320,000)
3. Result: ~350K training rows per fold instead of 8M

**Key Points:**
- Validation folds remain untouched (real-world imbalance)
- Undersampling applied only to training folds
- Fold data precomputed once before HPO to avoid repeated computation

---

## Model Training

### XGBoost HPO

**Function:** `optimize_xgboost()`

**Optuna Search Space:**

| Parameter | Range | Scale |
|-----------|-------|-------|
| `max_depth` | 3-10 | int |
| `learning_rate` | 0.01-0.3 | log |
| `subsample` | 0.6-1.0 | float |
| `colsample_bytree` | 0.6-1.0 | float |
| `min_child_weight` | 1-10 | int |
| `gamma` | 0-5 | float |
| `scale_pos_weight` | 25-100 | float |

**Fixed Parameters:**
- `objective`: `binary:logistic`
- `eval_metric`: `aucpr`
- `tree_method`: `hist`

### LightGBM HPO

**Function:** `optimize_lightgbm()`

**Optuna Search Space:**

| Parameter | Range | Scale |
|-----------|-------|-------|
| `max_depth` | 3-10 | int |
| `learning_rate` | 0.01-0.3 | log |
| `num_leaves` | 20-150 | int |
| `subsample` | 0.6-1.0 | float |
| `colsample_bytree` | 0.6-1.0 | float |
| `min_child_weight` | 0.001-10 | log |
| `reg_alpha` | 0-10 | float |
| `reg_lambda` | 0-10 | float |
| `scale_pos_weight` | 25-100 | float |

### Stacking Ensemble

**Function:** `train_stacker()`

**Architecture:**
```
Input Features → XGBoost → pred_xgb ─┐
                                      ├─→ LogisticRegression → Final Prediction
Input Features → LightGBM → pred_lgbm┘
```

**Stacker Model:** `sklearn.linear_model.LogisticRegression`
- Input: 2 columns (XGB OOF predictions, LGBM OOF predictions)
- Output: Ensemble fraud probability
- Logged coefficients show relative weight of each base model

---

## Evaluation Metrics

**File:** `src/utils/evaluation.py`

### Primary Metric

| Metric | Function | Purpose |
|--------|----------|---------|
| **AUCPR** | `calculate_aucpr()` | Primary optimization target for imbalanced data |

### Secondary Metrics (at threshold=0.0004)

| Metric | Formula | Purpose |
|--------|---------|---------|
| **AUROC** | `calculate_auroc()` | Overall discrimination ability |
| **F2-Score** | `fbeta_score(beta=2)` | Recall-weighted precision-recall balance |
| **TPR (Recall)** | TP / (TP + FN) | Fraud capture rate |
| **TNR (Specificity)** | TN / (TN + FP) | False positive control |
| **Precision** | TP / (TP + FP) | Alert accuracy |

### Threshold Selection

**Default:** 0.0004 (matches prior fraud probability)

**Rationale:** Setting threshold at class prior ensures balanced sensitivity to both classes.

---

## MLflow Tracking

### Tracking URI

```
https://mlflow-server-855767627985.us-central1.run.app
```

### Experiment Structure

```
healthcare-fraud-detection/
├── xgb-hpo-study_{timestamp}/       # XGBoost HPO parent run
│   ├── xgb-trial-0                  # Trial runs with params/metrics
│   ├── xgb-trial-1
│   └── ...
├── lgbm-hpo-study_{timestamp}/      # LightGBM HPO parent run
│   ├── lgbm-trial-0
│   └── ...
└── final-models_{timestamp}/        # Final artifacts
    ├── model_xgb.json
    ├── model_lgbm.txt
    ├── stacker_model.joblib
    ├── best_params.json
    └── feature_importance.csv
```

### Logged Metrics

| Metric | Scope |
|--------|-------|
| `cv_aucpr`, `cv_auroc`, `cv_f2` | Per trial |
| `cv_tpr`, `cv_tnr`, `cv_precision` | Per trial |
| `best_cv_aucpr` | Best trial per model |
| `final_aucpr`, `final_auroc`, `final_f2` | Final ensemble |
| `oof_xgb_aucpr`, `oof_lgbm_aucpr` | OOF performance |
| `stacker_weight_xgb`, `stacker_weight_lgbm` | Ensemble weights |

---

## Model Artifacts

**Output Location:** `gs://medguard_rawdata/models/artifacts/{job_name}/`

| Artifact | Format | Description |
|----------|--------|-------------|
| `model_xgb.json` | XGBoost JSON | Final XGBoost model |
| `model_lgbm.txt` | LightGBM text | Final LightGBM model |
| `stacker_model.joblib` | Joblib pickle | Logistic Regression stacker |
| `best_params.json` | JSON | Best hyperparameters for both models |
| `feature_importance.csv` | CSV | Top 20 features by average importance |

---

## Feature Importance

**Function:** `_save_feature_importance()`

**Output:** Top 20 features ranked by average importance across XGBoost and LightGBM.

| Column | Description |
|--------|-------------|
| `feature` | Feature name |
| `importance_xgb` | XGBoost feature importance (gain) |
| `importance_lgbm` | LightGBM feature importance |
| `importance_avg` | Average of both |

---

## Configuration Reference

**File:** `config/pipeline_config.yaml`

```yaml
models:
  xgboost:
    n_trials: 5        # Optuna trials
    cv_folds: 3        # Cross-validation folds
    random_state: 42
  lightgbm:
    n_trials: 5
    cv_folds: 3
    random_state: 42
  stacker:
    max_iter: 1000
    random_state: 42

sampling:
  undersample_enabled: true
  undersample_ratio: 100

evaluation:
  decision_threshold: 0.0004
  primary_metric: "f2"

validation:
  stratify_by: "rndrng_npi"
  target_column: "fraud_label"
  positive_class_ratio: 0.0004
```

---

## Execution Commands

### Local Training (Development)

```bash
cd /path/to/medGuardAI-fraudDetection
python src/models/train.py \
    --xgb_trials 5 \
    --lgbm_trials 5 \
    --cv_folds 3 \
    --worker_count 0
```

### Vertex AI Training (Production)

```bash
PRIMARY_MACHINE=n2-highmem-16 \
WORKER_COUNT=0 \
XGB_TRIALS=5 \
LGBM_TRIALS=5 \
CV_FOLDS=3 \
./training/submit_job.sh
```

---

## Cost Optimization

| Strategy | Implementation |
|----------|----------------|
| **Preemptible VMs** | `use_preemptible: true` in config |
| **Single-Node Default** | `worker_count=0` reduces cluster overhead |
| **Fold Caching** | Undersampled folds computed once before HPO |
| **Efficient HPO** | 5 trials × 3 folds = 15 model trains per algorithm |
