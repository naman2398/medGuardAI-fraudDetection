# Design Specification: MedGuardAI High-Imbalance Optimization

**Status:** Draft  
**Version:** 1.5  
**Target Metric:** 10x Lift in AUCPR (Goal: >0.05), F2-Score Stability

## 1. Executive Summary

The current model performs well against random baselines but suffers from low precision due to extreme class imbalance (0.04% positive). The current scale_pos_weight strategy (~2000) maximizes Recall at the expense of Precision, rendering the Alert List unusable.

This enhancement plan implements three critical changes derived from state-of-the-art fraud detection techniques and the reference paper (Johnson & Khoshgoftaar, 2023):

1. **Contextual Feature Engineering:** Moving from absolute counts to peer-relative Z-scores.
2. **Training Stabilization:** Replacing massive class weights with aggressive undersampling (1:100 ratio).
3. **Metric Expansion:** Adopting F2-Score to balance the business need for Recall with the operational need for Precision.

## 2. Feature Engineering Enhancements (PySpark)

**File:** `src/data/data_preprocessing.py`

### 2.1 Peer Comparison Features (Z-Scores)

**Problem:** Absolute numbers (e.g., "100 services") lack context. 100 services is low for a Dermatologist but impossible for a Neurosurgeon. 

**Solution:** Calculate Z-scores relative to the provider's specialty (`rndrng_prvdr_type`).

**Implementation Logic:** For each numerical column (e.g., `tot_srvcs`, `avg_mdcr_pymt_amt`):

1. Partition data by `rndrng_prvdr_type`.
2. Calculate Group Mean (μ_spec) and Group StdDev (σ_spec).
3. Compute Z-Score: Z = (X - μ_spec) / σ_spec
4. Compute Ratio: R = X / μ_spec

**New Feature List:**
- `tot_srvcs_zscore`, `tot_srvcs_ratio`
- `avg_mdcr_pymt_amt_zscore`, `avg_mdcr_pymt_amt_ratio`
- `tot_benes_zscore`, `tot_benes_ratio`
- `avg_sbmtd_chrg_zscore`, `avg_sbmtd_chrg_ratio`

### 2.2 Risk Ratios

**Implementation Logic:** Create explicit interaction features known to correlate with fraud:

- **Billing Inflation:** `avg_sbmtd_chrg / avg_mdcr_pymt_amt` (High values indicate excessive upcoding).
- **Service Density:** `tot_srvcs / tot_benes` (High values indicate potential churning/unnecessary services).

## 3. Training Pipeline Modifications (Dask/XGBoost)

**File:** `src/models/train_opensource.py`

### 3.1 Aggressive Undersampling

**Problem:** 8M rows of noise distract the model.

**Solution:** Train on a controlled ratio of 1:100 (Fraud:Normal).

**Implementation Logic (Inside CV Loop):**

1. **Validation Fold:** Keep untouched (Real-world imbalance).
2. **Training Fold:**
   - Isolate all Fraud cases (N_fraud).
   - Sample Normal cases (N_normal = N_fraud × 100).
   - Concat and shuffle.

**Configuration Strategy:**
- **Target Ratio:** 1:100.
- **Goal:** Balanced approach. Reduces noise by ~96% (keeping ~350k normal rows) while retaining enough variance to prevent overfitting.
- **Result:** Training set size ~350k - 400k rows.
- **Pros:** Faster training than full data, better specificity than 1:20.

### 3.2 Weight Calibration (Critical)

**Problem:** `scale_pos_weight` must change based on the Undersampling Ratio. The original weight of 2000 is now incorrect because the data imbalance is only 1:100.

**Optuna Search Space Update:**

For Ratio 1:100:
- `scale_pos_weight`: [25, 100]

**Reasoning:** A weight of 100 is mathematically balanced for this ratio. Searching in the 25-100 range allows the model to prioritize Precision (by underweighting) if needed.

## 4. Metric & Observability Updates (Implemented)

**File:** `src/utils/evaluation.py`  

### 4.1 Implemented Metrics & Logic

The evaluation module has been updated to support a threshold-aware F2-Score, prioritizing Recall while penalizing False Positives.

- **F2-Score:** Implemented using `sklearn.metrics.fbeta_score(beta=2)`. This weights Recall twice as heavily as Precision.
- **Thresholding:** A fixed decision threshold of 0.0004 (0.04%) is applied to probability outputs to match the prior class probability (replicating the reference paper's methodology).

**Secondary Metrics:**
- **TPR (Recall/Sensitivity):** TP / (TP + FN)
- **TNR (Specificity):** TN / (TN + FP)
- **Precision:** TP / (TP + FP)
- **AUROC:** Calculated for paper comparison only (not an optimization target).

### 4.2 Code Interface

The module now exposes the following primary functions:

1. `calculate_threshold_metrics(y_true, y_pred_proba, threshold=0.0004)`: Returns dictionary with `f2`, `tpr`, `tnr`, and `precision`.
2. `evaluate_fold(...)`: Logs all metrics for individual folds.
3. `evaluate_cv_folds(...)`: Aggregates mean/std for all metrics across folds.

### 4.3 Feature Importance Logging (New)

**Goal:** Understand why the model makes predictions, ensuring it's learning fraud patterns (e.g., Z-scores) rather than noise.

**Implementation Logic:** After training the final model (Best XGBoost/LightGBM):

1. Extract feature importance scores (Gain/Weight).
2. Create a DataFrame of `[Feature Name, Importance Score]`.
3. Sort descending and take top 20.
4. Save to `feature_importance.csv`.
5. Log artifact to MLflow: `mlflow.log_artifact("feature_importance.csv")`.

## 5. Execution Plan

1. **Phase 1 (Code):** Update `evaluation.py`.
2. **Phase 2 (Data):** Run PySpark job with new Z-score logic. Save to `gs://.../enriched_v2/`.
3. **Phase 3 (Training):** Run `train_opensource.py` with Ratio 1:100 and Weight [25, 100].
