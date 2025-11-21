"""
Main training orchestrator for distributed ensemble ML pipeline.
Implements HPO for XGBoost and LightGBM, then trains a stacked ensemble.
"""

import logging
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
import joblib

import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression

import optuna
import mlflow
import mlflow.xgboost
import mlflow.lightgbm

from dask.distributed import Client, LocalCluster
import dask.dataframe as dd

import sys
sys.path.append('src')

from utils.data_loader import load_config, load_training_data, validate_data, get_provider_labels
from utils.cv_splitter import create_provider_stratified_folds, split_data_by_providers
from utils.evaluation import calculate_aucpr, evaluate_cv_folds

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_runtime_config():
    """
    Override config with environment variables for flexible deployment.
    Allows runtime parameterization without changing config files.
    """
    # Read base config
    base_config = load_config("config/pipeline_config.yaml")
    
    # Get environment variable overrides
    n_rows_sample = os.getenv('N_ROWS_SAMPLE')
    xgb_trials = os.getenv('XGB_TRIALS')
    lgbm_trials = os.getenv('LGBM_TRIALS')
    cv_folds = os.getenv('CV_FOLDS')
    worker_count = os.getenv('WORKER_COUNT')
    
    # Apply overrides if provided
    if n_rows_sample is not None:
        n_rows = int(n_rows_sample)
        base_config['data']['n_rows_sample'] = None if n_rows == -1 else n_rows
    
    if xgb_trials is not None:
        base_config['models']['xgboost']['n_trials'] = int(xgb_trials)
    
    if lgbm_trials is not None:
        base_config['models']['lightgbm']['n_trials'] = int(lgbm_trials)
    
    if cv_folds is not None:
        base_config['models']['xgboost']['cv_folds'] = int(cv_folds)
        base_config['models']['lightgbm']['cv_folds'] = int(cv_folds)
    
    if worker_count is not None:
        base_config['compute']['workers']['count'] = int(worker_count)
    
    # Note: Keep tracking_uri from config (supports both local ./mlruns and GCS)
    # Artifact location will be set to GCS if specified in config
    
    # Log runtime configuration
    logger.info("="*80)
    logger.info("RUNTIME CONFIGURATION")
    logger.info("="*80)
    data_rows_info = 'ALL DATA' if base_config['data']['n_rows_sample'] is None else f"{base_config['data']['n_rows_sample']:,}"
    logger.info(f"Data rows: {data_rows_info}")
    logger.info(f"XGBoost trials: {base_config['models']['xgboost']['n_trials']}")
    logger.info(f"LightGBM trials: {base_config['models']['lightgbm']['n_trials']}")
    logger.info(f"CV folds: {base_config['models']['xgboost']['cv_folds']}")
    logger.info(f"Worker count: {base_config['compute']['workers']['count']}")
    logger.info(f"Distributed mode: {base_config['dask']['use_distributed']}")
    logger.info(f"MLflow URI: {base_config['mlflow']['tracking_uri']}")
    logger.info("="*80)
    
    return base_config


class DistributedTrainer:
    """Main trainer class for distributed ensemble training."""
    
    def __init__(self, config=None):
        """Initialize trainer with configuration."""
        self.config = config if config is not None else get_runtime_config()
        self.client = None
        self.df = None
        self.provider_labels = None
        self.folds = None
        self.best_params_xgb = None
        self.best_params_lgbm = None
        
    def setup_dask(self):
        """Setup Dask cluster (local or distributed)."""
        if self.config['dask']['use_distributed']:
            # Try to get scheduler address from environment
            scheduler_address = os.environ.get('DASK_SCHEDULER_ADDRESS')
            
            if scheduler_address:
                logger.info(f"Connecting to distributed Dask cluster at {scheduler_address}")
                self.client = Client(scheduler_address)
            else:
                # Vertex AI cluster formation logic
                cluster_spec = os.environ.get('CLUSTER_SPEC')
                
                if cluster_spec:
                    logger.info("Vertex AI cluster detected, setting up Dask coordination...")
                    import json
                    spec = json.loads(cluster_spec)
                    
                    # Determine if this is primary or worker node
                    # Primary is worker pool 0, Workers are worker pool 1
                    task_type = os.environ.get('CLOUD_ML_TASK_TYPE', 'master')
                    
                    if task_type == 'master' or 'workerpool0' in task_type.lower():
                        # This is the primary node - start scheduler
                        logger.info("Starting Dask Scheduler on Primary node...")
                        from dask.distributed import Scheduler
                        scheduler = Scheduler()
                        scheduler.start(port=8786)
                        scheduler_address = f"tcp://0.0.0.0:8786"
                        logger.info(f"Dask Scheduler started at {scheduler_address}")
                        self.client = Client(scheduler_address)
                    else:
                        # This is a worker node - connect to primary
                        logger.info("Worker node detected, connecting to Primary scheduler...")
                        # In Vertex AI, primary node is accessible via cluster networking
                        primary_address = "tcp://workerpool0-0:8786"
                        self.client = Client(primary_address)
                        logger.info(f"Connected to scheduler at {primary_address}")
                else:
                    logger.warning("Distributed mode enabled but no cluster info found, falling back to local")
                    self._setup_local_cluster()
        else:
            self._setup_local_cluster()
            
        logger.info(f"Dask dashboard: {self.client.dashboard_link}")
        logger.info(f"Dask cluster workers: {len(self.client.scheduler_info()['workers'])}")
        
    def _setup_local_cluster(self):
        """Setup local Dask cluster for testing."""
        logger.info("Setting up local Dask cluster")
        cluster = LocalCluster(
            n_workers=self.config['dask']['n_workers'],
            threads_per_worker=self.config['dask']['threads_per_worker'],
            memory_limit=self.config['dask']['memory_limit']
        )
        self.client = Client(cluster)
        
    def setup_mlflow(self):
        """Initialize MLflow tracking."""
        tracking_uri = self.config['mlflow']['tracking_uri']
        experiment_name = self.config['mlflow']['experiment_name']
        artifact_location = self.config['mlflow'].get('artifact_location')
        
        mlflow.set_tracking_uri(tracking_uri)
        
        # Set experiment with artifact location if specified
        if artifact_location:
            mlflow.set_experiment(experiment_name, artifact_location=artifact_location)
        else:
            mlflow.set_experiment(experiment_name)
        
        logger.info(f"MLflow tracking URI: {tracking_uri}")
        logger.info(f"MLflow experiment: {experiment_name}")
        if artifact_location:
            logger.info(f"MLflow artifact location: {artifact_location}")
        
    def load_data(self):
        """Load and prepare training data."""
        logger.info("="*60)
        logger.info("STEP 1: Data Loading and Preparation")
        logger.info("="*60)
        
        # Load data
        self.df = load_training_data(self.config, use_sample=True)
        validate_data(self.df, self.config)
        
        # Get provider labels for stratification
        self.provider_labels = get_provider_labels(self.df, self.config)
        
        # Create stratified folds
        n_folds = self.config['models']['xgboost']['cv_folds']
        random_state = self.config['models']['xgboost']['random_state']
        
        self.folds = create_provider_stratified_folds(
            self.provider_labels,
            n_folds=n_folds,
            random_state=random_state
        )
        
        logger.info(f"Data preparation complete. Ready for training with {n_folds} folds")
        
    def optimize_xgboost(self):
        """Run HPO for XGBoost using Optuna."""
        logger.info("="*60)
        logger.info("STEP 2: XGBoost Hyperparameter Optimization")
        logger.info("="*60)
        
        with mlflow.start_run(run_name="xgb-hpo-study") as parent_run:
            
            def objective(trial):
                """Optuna objective for XGBoost."""
                # Suggest hyperparameters
                params = {
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                    'gamma': trial.suggest_float('gamma', 0, 5),
                    'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1, 100),
                    'objective': 'binary:logistic',
                    'eval_metric': 'aucpr',
                    'tree_method': 'hist',
                    'random_state': self.config['models']['xgboost']['random_state']
                }
                
                # Run CV
                cv_score = self._run_cv_xgboost(params, trial.number)
                
                # Log to MLflow
                with mlflow.start_run(run_name=f"xgb-trial-{trial.number}", nested=True):
                    mlflow.log_params(params)
                    mlflow.log_metric("cv_aucpr", cv_score)
                
                return cv_score
            
            # Create study and optimize
            study = optuna.create_study(direction='maximize')
            n_trials = self.config['models']['xgboost']['n_trials']
            
            logger.info(f"Starting Optuna optimization with {n_trials} trials")
            study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
            
            # Save best params
            self.best_params_xgb = study.best_params
            self.best_params_xgb.update({
                'objective': 'binary:logistic',
                'eval_metric': 'aucpr',
                'tree_method': 'hist',
                'random_state': self.config['models']['xgboost']['random_state']
            })
            
            logger.info(f"Best XGBoost AUCPR: {study.best_value:.4f}")
            logger.info(f"Best params: {self.best_params_xgb}")
            
            # Log best params to parent run
            mlflow.log_params(self.best_params_xgb)
            mlflow.log_metric("best_cv_aucpr", study.best_value)
            
    def _run_cv_xgboost(self, params, trial_num):
        """Run cross-validation for XGBoost."""
        logger.info(f"XGBoost trial {trial_num}: Running {len(self.folds)}-fold CV")
        
        target_col = self.config['validation']['target_column']
        stratify_col = self.config['validation']['stratify_by']
        
        aucpr_scores = []
        
        for fold_idx, (train_providers, test_providers) in enumerate(self.folds):
            # Split data
            train_df, test_df = split_data_by_providers(
                self.df, train_providers, test_providers, stratify_col
            )
            
            # Prepare data
            X_train = train_df.drop(columns=[target_col, stratify_col]).compute()
            y_train = train_df[target_col].compute()
            X_test = test_df.drop(columns=[target_col, stratify_col]).compute()
            y_test = test_df[target_col].compute()
            
            # Train model
            model = xgb.XGBClassifier(**params)
            model.fit(X_train, y_train, verbose=False)
            
            # Predict and evaluate
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            aucpr = calculate_aucpr(y_test, y_pred_proba)
            aucpr_scores.append(aucpr)
            
            logger.info(f"  Fold {fold_idx + 1}: AUCPR = {aucpr:.4f}")
        
        cv_mean = np.mean(aucpr_scores)
        cv_std = np.std(aucpr_scores)
        logger.info(f"Trial {trial_num} CV AUCPR: {cv_mean:.4f} ± {cv_std:.4f}")
        
        return cv_mean
        
    def optimize_lightgbm(self):
        """Run HPO for LightGBM using Optuna."""
        logger.info("="*60)
        logger.info("STEP 3: LightGBM Hyperparameter Optimization")
        logger.info("="*60)
        
        with mlflow.start_run(run_name="lgbm-hpo-study") as parent_run:
            
            def objective(trial):
                """Optuna objective for LightGBM."""
                # Suggest hyperparameters
                params = {
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'num_leaves': trial.suggest_int('num_leaves', 20, 150),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'min_child_weight': trial.suggest_float('min_child_weight', 0.001, 10, log=True),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                    'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1, 100),
                    'objective': 'binary',
                    'metric': 'auc',
                    'random_state': self.config['models']['lightgbm']['random_state'],
                    'verbose': -1
                }
                
                # Run CV
                cv_score = self._run_cv_lightgbm(params, trial.number)
                
                # Log to MLflow
                with mlflow.start_run(run_name=f"lgbm-trial-{trial.number}", nested=True):
                    mlflow.log_params(params)
                    mlflow.log_metric("cv_aucpr", cv_score)
                
                return cv_score
            
            # Create study and optimize
            study = optuna.create_study(direction='maximize')
            n_trials = self.config['models']['lightgbm']['n_trials']
            
            logger.info(f"Starting Optuna optimization with {n_trials} trials")
            study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
            
            # Save best params
            self.best_params_lgbm = study.best_params
            self.best_params_lgbm.update({
                'objective': 'binary',
                'metric': 'auc',
                'random_state': self.config['models']['lightgbm']['random_state'],
                'verbose': -1
            })
            
            logger.info(f"Best LightGBM AUCPR: {study.best_value:.4f}")
            logger.info(f"Best params: {self.best_params_lgbm}")
            
            # Log best params to parent run
            mlflow.log_params(self.best_params_lgbm)
            mlflow.log_metric("best_cv_aucpr", study.best_value)
            
    def _run_cv_lightgbm(self, params, trial_num):
        """Run cross-validation for LightGBM."""
        logger.info(f"LightGBM trial {trial_num}: Running {len(self.folds)}-fold CV")
        
        target_col = self.config['validation']['target_column']
        stratify_col = self.config['validation']['stratify_by']
        
        aucpr_scores = []
        
        for fold_idx, (train_providers, test_providers) in enumerate(self.folds):
            # Split data
            train_df, test_df = split_data_by_providers(
                self.df, train_providers, test_providers, stratify_col
            )
            
            # Prepare data
            X_train = train_df.drop(columns=[target_col, stratify_col]).compute()
            y_train = train_df[target_col].compute()
            X_test = test_df.drop(columns=[target_col, stratify_col]).compute()
            y_test = test_df[target_col].compute()
            
            # Train model
            model = lgb.LGBMClassifier(**params)
            model.fit(X_train, y_train)
            
            # Predict and evaluate
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            aucpr = calculate_aucpr(y_test, y_pred_proba)
            aucpr_scores.append(aucpr)
            
            logger.info(f"  Fold {fold_idx + 1}: AUCPR = {aucpr:.4f}")
        
        cv_mean = np.mean(aucpr_scores)
        cv_std = np.std(aucpr_scores)
        logger.info(f"Trial {trial_num} CV AUCPR: {cv_mean:.4f} ± {cv_std:.4f}")
        
        return cv_mean
        
    def generate_oof_predictions(self):
        """Generate out-of-fold predictions for stacking."""
        logger.info("="*60)
        logger.info("STEP 4: Generating OOF Predictions for Stacking")
        logger.info("="*60)
        
        target_col = self.config['validation']['target_column']
        stratify_col = self.config['validation']['stratify_by']
        
        # Get full data length
        total_rows = len(self.df)
        
        # Initialize OOF arrays
        oof_xgb = np.zeros(total_rows)
        oof_lgbm = np.zeros(total_rows)
        oof_targets = np.zeros(total_rows)
        
        logger.info(f"Generating OOF predictions for {total_rows:,} rows across {len(self.folds)} folds")
        
        # Compute provider column once for indexing
        provider_col = self.df[stratify_col].compute()
        
        for fold_idx, (train_providers, test_providers) in enumerate(self.folds):
            logger.info(f"Processing fold {fold_idx + 1}/{len(self.folds)}")
            
            # Split data
            train_df, test_df = split_data_by_providers(
                self.df, train_providers, test_providers, stratify_col
            )
            
            # Prepare data
            X_train = train_df.drop(columns=[target_col, stratify_col]).compute()
            y_train = train_df[target_col].compute()
            X_test = test_df.drop(columns=[target_col, stratify_col]).compute()
            y_test = test_df[target_col].compute()
            
            # Get test indices
            test_mask = provider_col.isin(test_providers)
            test_indices = np.where(test_mask)[0]
            
            # Train XGBoost
            model_xgb = xgb.XGBClassifier(**self.best_params_xgb)
            model_xgb.fit(X_train, y_train, verbose=False)
            oof_xgb[test_indices] = model_xgb.predict_proba(X_test)[:, 1]
            
            # Train LightGBM
            model_lgbm = lgb.LGBMClassifier(**self.best_params_lgbm)
            model_lgbm.fit(X_train, y_train)
            oof_lgbm[test_indices] = model_lgbm.predict_proba(X_test)[:, 1]
            
            # Store targets
            oof_targets[test_indices] = y_test.values
            
            logger.info(f"  Fold {fold_idx + 1} complete")
        
        # Store OOF predictions
        self.oof_predictions = pd.DataFrame({
            'xgb_pred': oof_xgb,
            'lgbm_pred': oof_lgbm,
            'target': oof_targets
        })
        
        # Evaluate OOF performance
        aucpr_xgb = calculate_aucpr(oof_targets, oof_xgb)
        aucpr_lgbm = calculate_aucpr(oof_targets, oof_lgbm)
        
        logger.info(f"OOF AUCPR - XGBoost: {aucpr_xgb:.4f}, LightGBM: {aucpr_lgbm:.4f}")
        
    def train_stacker(self):
        """Train stacker model on OOF predictions."""
        logger.info("="*60)
        logger.info("STEP 5: Training Stacker Model")
        logger.info("="*60)
        
        # Prepare stacking data
        X_stack = self.oof_predictions[['xgb_pred', 'lgbm_pred']].values
        y_stack = self.oof_predictions['target'].values
        
        # Train logistic regression stacker
        self.stacker = LogisticRegression(
            max_iter=self.config['models']['stacker']['max_iter'],
            random_state=self.config['models']['stacker']['random_state']
        )
        
        logger.info("Training stacker on OOF predictions...")
        self.stacker.fit(X_stack, y_stack)
        
        # Evaluate stacker
        y_stack_pred = self.stacker.predict_proba(X_stack)[:, 1]
        aucpr_stack = calculate_aucpr(y_stack, y_stack_pred)
        
        logger.info(f"Stacker AUCPR: {aucpr_stack:.4f}")
        logger.info(f"Stacker coefficients - XGB: {self.stacker.coef_[0][0]:.4f}, LGBM: {self.stacker.coef_[0][1]:.4f}")
        
    def train_final_models(self):
        """Train final models on full dataset."""
        logger.info("="*60)
        logger.info("STEP 6: Training Final Models on Full Data")
        logger.info("="*60)
        
        target_col = self.config['validation']['target_column']
        stratify_col = self.config['validation']['stratify_by']
        
        # Prepare full dataset
        X_full = self.df.drop(columns=[target_col, stratify_col]).compute()
        y_full = self.df[target_col].compute()
        
        logger.info(f"Training on full dataset: {len(X_full):,} rows")
        
        # Train final XGBoost
        logger.info("Training final XGBoost model...")
        self.final_xgb = xgb.XGBClassifier(**self.best_params_xgb)
        self.final_xgb.fit(X_full, y_full, verbose=False)
        logger.info("XGBoost training complete")
        
        # Train final LightGBM
        logger.info("Training final LightGBM model...")
        self.final_lgbm = lgb.LGBMClassifier(**self.best_params_lgbm)
        self.final_lgbm.fit(X_full, y_full)
        logger.info("LightGBM training complete")
        
        logger.info("All final models trained successfully")
        
    def save_models(self, output_dir=None):
        """Save all model artifacts."""
        logger.info("="*60)
        logger.info("STEP 7: Saving Model Artifacts")
        logger.info("="*60)
        
        # Determine output directory
        if output_dir is None:
            job_name = os.environ.get('JOB_NAME', 'local-training')
            if self.config['dask']['use_distributed']:
                # Save to GCS with job-specific path
                bucket = self.config['data']['gcs_bucket']
                output_dir = f"gs://{bucket}/models/artifacts/{job_name}"
            else:
                # Local path
                output_dir = f"models/artifacts/{job_name}"
        
        logger.info(f"Saving models to: {output_dir}")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        with mlflow.start_run(run_name="final-models") as run:
            
            # Save XGBoost
            xgb_path = output_path / "model_xgb.json"
            self.final_xgb.save_model(xgb_path)
            mlflow.log_artifact(xgb_path)
            logger.info(f"Saved XGBoost model to {xgb_path}")
            
            # Save LightGBM
            lgbm_path = output_path / "model_lgbm.txt"
            self.final_lgbm.booster_.save_model(str(lgbm_path))
            mlflow.log_artifact(lgbm_path)
            logger.info(f"Saved LightGBM model to {lgbm_path}")
            
            # Save Stacker
            stacker_path = output_path / "stacker_model.joblib"
            joblib.dump(self.stacker, stacker_path)
            mlflow.log_artifact(stacker_path)
            logger.info(f"Saved Stacker model to {stacker_path}")
            
            # Save best params
            params_path = output_path / "best_params.json"
            with open(params_path, 'w') as f:
                json.dump({
                    'xgboost': self.best_params_xgb,
                    'lightgbm': self.best_params_lgbm
                }, f, indent=2)
            mlflow.log_artifact(params_path)
            logger.info(f"Saved best params to {params_path}")
            
            logger.info(f"All artifacts saved to {output_path}")
            logger.info(f"MLflow run ID: {run.info.run_id}")
    
    def run_training(self):
        """Main training pipeline execution."""
        logger.info("="*60)
        logger.info("DISTRIBUTED ENSEMBLE TRAINING PIPELINE")
        logger.info("="*60)
        
        try:
            # Setup
            self.setup_dask()
            self.setup_mlflow()
            
            # Load data
            self.load_data()
            
            # HPO
            self.optimize_xgboost()
            self.optimize_lightgbm()
            
            # Stacking
            self.generate_oof_predictions()
            self.train_stacker()
            
            # Final training
            self.train_final_models()
            
            # Save
            self.save_models()
            
            logger.info("="*60)
            logger.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("="*60)
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {e}", exc_info=True)
            raise
        
        finally:
            if self.client:
                self.client.close()
                logger.info("Dask client closed")


def main():
    """Entry point for training script."""
    trainer = DistributedTrainer()
    trainer.run_training()


if __name__ == "__main__":
    main()
