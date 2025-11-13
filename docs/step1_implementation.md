# Step 1 Implementation: Utility Modules

## Overview
This implements the foundational utility modules for the distributed ML pipeline:
1. **Data Loader** - Loads training data from GCS
2. **CV Splitter** - Creates provider-level stratified folds
3. **Evaluation** - Calculates AUCPR and other metrics

## Files Created

### 1. `src/utils/data_loader.py`
**Purpose:** Load and validate training data from GCS parquet files

**Key Functions:**
- `load_config()` - Load pipeline configuration from YAML
- `load_training_data()` - Load parquet files from GCS into Dask DataFrame
- `validate_data()` - Validate schema and check class distribution
- `get_provider_labels()` - Extract provider-level fraud labels for stratification

**Features:**
- Loads multiple parquet files from GCS path
- Optional sampling for testing
- Data validation with class balance checks
- Provider-level label aggregation (critical for preventing data leakage)

### 2. `src/utils/cv_splitter.py`
**Purpose:** Create stratified K-fold splits at provider level

**Key Functions:**
- `create_provider_stratified_folds()` - Create provider-level stratified folds
- `split_data_by_providers()` - Split Dask DataFrame by provider lists
- `get_fold_indices()` - Convert provider folds to row-level indices

**Features:**
- No provider appears in both train and test sets
- Maintains correct fraud/non-fraud ratio in each fold
- Detailed logging of fold statistics

### 3. `src/utils/evaluation.py`
**Purpose:** Evaluation metrics for extreme class imbalance

**Key Functions:**
- `calculate_aucpr()` - Primary metric (Area Under Precision-Recall Curve)
- `calculate_auroc()` - Supplementary metric
- `evaluate_fold()` - Evaluate single fold with multiple metrics
- `evaluate_cv_folds()` - Aggregate metrics across folds
- `log_metrics_to_mlflow()` - Optional MLflow integration

**Features:**
- AUCPR as primary metric (optimal for 0.04% positive class)
- AUROC as supplementary metric
- Class distribution tracking
- MLflow integration ready

## Configuration Updates

### `config/pipeline_config.yaml`
- Updated `stratify_by` to use `rndrng_npi` (matches preprocessing output)
- All GCS paths configurable
- Sample size option for testing

### `requirements.txt`
Added missing dependencies:
- `google-cloud-bigquery>=3.11.0`
- `fsspec>=2023.10.0`
- `gcsfs>=2023.10.0`

## Design Principles

✅ **Minimalistic & Clean**
- No unnecessary complexity
- Each function has single responsibility
- Clear docstrings

✅ **Logger Everywhere**
- No print statements
- Consistent logging format
- Info, warning, and error levels

✅ **Config-Driven**
- All paths from config file
- Easy to switch between environments
- No hardcoded values

✅ **Provider-Level Stratification**
- Critical for preventing data leakage
- Maintains class balance in each fold
- Follows design document specifications

## Testing

Test script created: `scripts/test_utilities.py`

**To test (once dependencies installed):**
```bash
python scripts/test_utilities.py
```

This will:
1. Load configuration
2. Load sample training data from GCS
3. Validate data schema
4. Compute provider labels
5. Create stratified CV folds

## Next Steps

**Step 2: Model Training Components**
- HPO objective functions for XGBoost and LightGBM
- OOF prediction generation
- Stacker training
- Main training orchestrator

**Step 3: Containerization & Deployment**
- Dockerfile
- Vertex AI submission script
- Dask cluster setup logic

Would you like to proceed with Step 2?
