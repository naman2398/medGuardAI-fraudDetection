# Design Document: Distributed Ensemble ML Pipeline

## 1. Objective

The goal of this pipeline is to automate the training, evaluation, and versioning of a stacked ensemble (XGBoost + LightGBM + Logistic Regression) model for detecting healthcare fraud.  
This pipeline is designed to be fully open-source (using Optuna, MLflow, and Dask) while leveraging Google Cloud Platform (Vertex AI) purely for distributed compute.  
It will run on an 8-million-record dataset with a 0.04% positive class (fraud), and must implement robust validation to handle this extreme imbalance and prevent data leakage.

## 2. System Architecture

The pipeline uses GCP for compute and storage, and open-source libraries for all ML operations.

- **Google Cloud Storage (GCS):** Acts as the central "hard drive." It will store:
  1. The model-ready training data (8M records as multiple Parquet files).
  2. All MLflow experiment logs, metrics, and parameters.
  3. The final, trained model artifacts (XGBoost, LightGBM, and the Logistic Regression stacker).
- **Artifact Registry:** A GCP service that hosts our custom Docker container. This is required to ensure all cluster nodes run an identical environment.
- **Vertex AI (Compute Only):** Used only to provision and run our compute cluster.
  - Custom Job: We will launch a CustomJob (not an HPO job).
  - Cluster: This job will request a single 1+2 cluster (1 Primary, 2 Workers) which is torn down when the job is complete.
- **Open-Source MLOps (The "Brain"):** All logic is inside our container.
  - Dask: The distributed computing framework used to parallelize data loading and training across the 2 worker nodes.
  - Optuna: The hyperparameter tuning library. It will orchestrate the HPO loops on the Dask cluster.
  - MLflow: The experiment tracking library. It will log all HPO trials and model artifacts to our GCS bucket.

## 3. Cluster & Data Flow

This outlines the precise execution flow from start to finish.

1. **Submission:** The `training/submit_job_opensource.sh` script is executed.
2. **Provisioning:** Vertex AI provisions the 1+2 cluster:
   - Worker Pool 0 (Primary): 1x e2-standard-4 (4 vCPU, 16 GB RAM).
   - Worker Pool 1 (Workers): 2x c3-standard-8 (8 vCPU, 32 GB RAM each).
3. **Container Pull:** All 3 nodes pull the Docker image from Artifact Registry.
4. **Cluster Formation:**
   - `src/models/train_opensource.py` starts on all 3 nodes.
   - The script reads the CLUSTER_SPEC environment variable.
   - The Primary node starts a Dask Scheduler.
   - The 2 Worker nodes start Dask Workers and connect to the Scheduler.
5. **Main Orchestration (on Primary):** The Primary node, now acting as the "boss," begins the `run_training` function.
6. **Distributed Data Load:** The Primary node tells the 2 Workers to load the Parquet files from GCS. The 8M-row dataset is now partitioned and held in the cluster's distributed memory (64 GB total on the workers).
7. **HPO & Training:** The Primary node orchestrates the entire ML pipeline (see below) on the Dask cluster.
8. **Shutdown:** Once the script finishes, Vertex AI automatically destroys all 3 VMs.

## 4. Pipeline Execution Logic (in train_opensource.py)

The Primary node executes these steps sequentially on the same Dask cluster:

**Step 1: MLOps Setup**
- MLflow: Initialize mlflow, setting the tracking URI to a GCS path (e.g., `gs://[BUCKET]/models/mlflow_logs/`).
- Data Prep: Load the 8M rows into a Dask DataFrame. Crucially, compute and cache the `provider_labels` (the master fraud label for each unique provider) for stratification.

**Step 2: HPO for XGBoost**
- An MLflow "Parent Run" is created for this study.
- `optuna.create_study()` is called.
- `study.optimize(objective_xgb, n_trials=20)` is run.
  - For each of the 20 trials, Optuna suggests new params.
  - `objective_xgb` runs a full 5-fold Stratified-by-Provider CV on the Dask cluster.
  - The cv_average_aucpr is logged to a nested MLflow run and returned to Optuna.
- The best_params_xgb are saved.

**Step 3: HPO for LightGBM**
- A new MLflow "Parent Run" is created.
- `optuna.create_study()` is called.
- `study.optimize(objective_lgbm, n_trials=20)` is run.
  - The same 20-trial, 5-fold CV process is repeated for LightGBM.
- The best_params_lgbm are saved.

**Step 4: Stacking (Ensemble) Data Generation**
- Generate OOF Predictions: The 5-fold CV is run one more time.
  - Fold 1: Train XGB (best params) and LGBM (best params) on Folds 2-5. Predict on Fold 1.
  - Fold 2: Train on Folds 1, 3-5. Predict on Fold 2.
  - ...and so on.
- Result: This process produces two new datasets (OOF predictions for XGB and LGBM) that are the same length as the 8M-record dataset. This is the training data for our stacker.

**Step 5: Stacker Model Training**
- A LogisticRegression model is trained on the OOF predictions.
- Note: This step is very fast and runs on the Primary node, as the data is just 8M rows x 2 columns.
- The trained `stacker_model.joblib` is saved.

**Step 6: Final Base Model Training**
- Final XGB Model: Train one XGBoost model on 100% of the data using best_params_xgb. Save `model_xgb.bst`.
- Final LGBM Model: Train one LightGBM model on 100% of the data using best_params_lgbm. Save `model_lgbm.bst`.

**Step 7: Artifact Saving**
- All three models (`model_xgb.bst`, `model_lgbm.bst`, `stacker_model.joblib`) are uploaded to the GCS bucket (`gs://[BUCKET]/models/[JOB_NAME]/`) and logged as artifacts in MLflow.

## 5. Validation & Evaluation Strategy

- **Validation:** The core of this pipeline is Stratified K-Fold-by-Entity.
  1. We group by provider_npi to get a "master" fraud label for each provider.
  2. We use StratifiedKFold on this list of providers.
  3. This ensures no provider's data leaks between train and test sets, and guarantees each fold has the correct 0.04% proportion of fraudulent providers.
- **Primary Metric:** aucpr (Area Under the Precision-Recall Curve). This is the only reliable metric for this level of imbalance and is used to optimize the HPO.
- **Feature Importance:** SHAP will be used for model explainability during post-analysis.

## 6. Artifacts & Tracking (via MLflow)

The MLflow UI will show:
- 2 "Parent" Runs (Studies): xgb-hpo-study and lgbm-hpo-study.
- 40 "Child" Runs (Trials): 20 nested under each parent, logging the params and cv_average_aucpr for each trial.
- 1 "Parent" Run (Final): A final run logging the 3 saved model artifacts (.bst, .bst, .joblib) and their GCS paths.

## 7. Cost Management (Budget: $250)

- **Compute:**
  1. Preemptible VMs: All 3 nodes in the cluster will be set as preemptible, providing a 60-80% cost reduction.
  2. Efficient Cluster: The c3-standard-8 workers are used for their high-speed (100 Gbps) networking, which is critical for Dask's shuffle performance.
  3. Efficient HPO: This design is far cheaper than the managed-service (Vizier) approach. We pay the cluster spin-up and data-loading cost only once, not 40 times (once per trial).
- **Storage:** GCS and Artifact Registry costs are minimal (pennies per month).
