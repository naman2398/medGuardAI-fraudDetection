"""
Main training orchestrator for distributed ensemble ML pipeline.
Implements HPO for XGBoost and LightGBM, then trains a stacked ensemble.
"""

import argparse
import logging
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
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
import subprocess
import time

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


def undersample_dask_dataframe(train_df, target_col, ratio=100, random_state=42):
    """
    Undersample within Dask BEFORE .compute() to avoid memory issues.
    
    Per design doc Section 3.1:
    - Training fold: 1:100 ratio (Fraud:Normal)
    - Isolate all Fraud cases (N_fraud)
    - Sample Normal cases (N_normal = N_fraud × 100)
    - Reduces ~5M rows to ~100K rows before memory hit
    
    Args:
        train_df: Dask DataFrame with training data
        target_col: Name of target column (fraud_label)
        ratio: Target ratio of normal to fraud cases (default: 100)
        random_state: Random seed for reproducibility
        
    Returns:
        Undersampled Dask DataFrame (lazy - not yet computed)
    """
    logger.info(f"    [DASK UNDERSAMPLE] Starting Dask-level undersampling (ratio 1:{ratio})...")
    
    # CRITICAL FIX: Reset index to avoid Dask partition alignment issues
    # When filtering Dask DataFrames with boolean masks, the mask must align with
    # the DataFrame's partition boundaries. Resetting index ensures clean alignment.
    logger.info("    [DASK UNDERSAMPLE] Resetting index for partition alignment...")
    train_df = train_df.reset_index(drop=True)
    
    # Persist the dataframe to materialize it before filtering
    # This ensures the boolean mask aligns with actual data partitions
    train_df = train_df.persist()
    
    # Separate fraud and normal using boolean indexing (now safe after reset_index + persist)
    fraud_df = train_df[train_df[target_col] == 1]
    normal_df = train_df[train_df[target_col] == 0]
    
    # Get counts - compute() needed for Dask delayed scalars
    logger.info("    [DASK UNDERSAMPLE] Computing class counts...")
    n_fraud = fraud_df.shape[0].compute()
    n_normal = normal_df.shape[0].compute()
    
    logger.info(f"    [DASK UNDERSAMPLE] Original: Fraud={n_fraud:,}, Normal={n_normal:,}")
    
    # Calculate sampling fraction per design doc: N_normal = N_fraud × ratio
    n_normal_target = min(n_fraud * ratio, n_normal)
    sample_frac = n_normal_target / n_normal if n_normal > 0 else 0
    
    logger.info(f"    [DASK UNDERSAMPLE] Sampling {sample_frac:.4%} of normal cases ({n_normal_target:,} rows)")
    
    # Sample normal rows in Dask (lazy operation)
    sampled_normal = normal_df.sample(frac=sample_frac, random_state=random_state)
    
    # Concatenate fraud + sampled normal (lazy operation)
    undersampled_df = dd.concat([fraud_df, sampled_normal])
    
    # Repartition to optimize downstream compute
    n_partitions = max(1, (n_fraud + n_normal_target) // 50000)  # ~50K rows per partition
    undersampled_df = undersampled_df.repartition(npartitions=n_partitions)
    
    logger.info(f"    [DASK UNDERSAMPLE] Result: ~{n_fraud + n_normal_target:,} rows in {n_partitions} partitions")
    logger.info(f"    >>> UNDERSAMPLING COMPLETE: Ratio 1:{ratio} (Fraud:{n_fraud:,}, Normal:{n_normal_target:,}) <<<")
    
    return undersampled_df


def parse_args():
    """Parse command-line arguments for training configuration."""
    parser = argparse.ArgumentParser(
        description='Distributed Fraud Detection Training Pipeline'
    )
    
    parser.add_argument('--n_rows_sample', type=int, default=-1,
                        help='Number of rows to sample (-1 for all)')
    parser.add_argument('--xgb_trials', type=int, default=20,
                        help='Optuna trials for XGBoost')
    parser.add_argument('--lgbm_trials', type=int, default=20,
                        help='Optuna trials for LightGBM')
    parser.add_argument('--cv_folds', type=int, default=5,
                        help='Cross-validation folds')
    parser.add_argument('--worker_count', type=int, default=2,
                        help='Dask worker nodes')
    
    return parser.parse_args()


def get_runtime_config(args):
    """Apply command-line arguments to base config."""
    base_config = load_config("config/pipeline_config.yaml")
    
    # Apply arguments
    base_config['data']['n_rows_sample'] = None if args.n_rows_sample == -1 else args.n_rows_sample
    base_config['models']['xgboost']['n_trials'] = args.xgb_trials
    base_config['models']['xgboost']['cv_folds'] = args.cv_folds
    base_config['models']['lightgbm']['n_trials'] = args.lgbm_trials
    base_config['models']['lightgbm']['cv_folds'] = args.cv_folds
    base_config['compute']['workers']['count'] = args.worker_count
    
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
    
    # Log sampling configuration prominently (per design doc Section 3.1)
    sampling_config = base_config.get('sampling', {})
    undersample_enabled = sampling_config.get('undersample_enabled', False)
    undersample_ratio = sampling_config.get('undersample_ratio', 100)
    logger.info("SAMPLING CONFIGURATION")
    logger.info("="*80)
    if undersample_enabled:
        logger.info(f">>> UNDERSAMPLING ENABLED <<<")
        logger.info(f"    Ratio: 1:{undersample_ratio} (Fraud:Normal)")
        logger.info(f"    Method: Dask-level (before .compute() to avoid OOM)")
        logger.info(f"    Per design doc Section 3.1: Aggressive Undersampling")
    else:
        logger.info(">>> UNDERSAMPLING DISABLED <<<")
        logger.info("    Using full training data (may cause memory issues)")
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
        
        # Cache sampling config (accessed frequently)
        sampling_config = self.config.get('sampling', {})
        self._undersample_enabled = sampling_config.get('undersample_enabled', False)
        self._undersample_ratio = sampling_config.get('undersample_ratio', 100)
    
    def _prepare_fold_data(self, train_df, test_df, fold_idx, random_state, phase="CV"):
        """
        Prepare train/test data for a fold with optional undersampling.
        
        Centralizes the repeated pattern of:
        1. Apply Dask-level undersampling on training fold
        2. Compute training data
        3. Compute test data (full, no undersampling)
        
        Args:
            train_df: Dask DataFrame for training fold
            test_df: Dask DataFrame for test fold  
            fold_idx: Current fold index (0-based)
            random_state: Base random state for reproducibility
            phase: Logging context ("CV", "OOF", etc.)
            
        Returns:
            X_train, y_train, X_test, y_test as pandas objects
        """
        target_col = self.config['validation']['target_column']
        stratify_col = self.config['validation']['stratify_by']
        
        # Apply Dask-level undersampling BEFORE .compute() to avoid OOM
        if self._undersample_enabled:
            logger.info(f"  Fold {fold_idx + 1}: Applying Dask undersampling...")
            train_df = undersample_dask_dataframe(
                train_df, target_col,
                ratio=self._undersample_ratio,
                random_state=random_state + fold_idx
            )
        
        # Compute training data (undersampled if enabled)
        logger.info(f"  Fold {fold_idx + 1}: Computing training data...")
        X_train = train_df.drop(columns=[target_col, stratify_col]).compute()
        y_train = train_df[target_col].compute()
        logger.info(f"  Fold {fold_idx + 1}: Train ready - {len(X_train):,} rows {'[UNDERSAMPLED]' if self._undersample_enabled else ''}")
        
        # Test set: Keep full (per design doc - real-world imbalance for validation)
        logger.info(f"  Fold {fold_idx + 1}: Computing test data...")
        X_test = test_df.drop(columns=[target_col, stratify_col]).compute()
        y_test = test_df[target_col].compute()
        logger.info(f"  Fold {fold_idx + 1}: Test ready - {len(X_test):,} rows")
        
        return X_train, y_train, X_test, y_test
        
    def setup_dask(self):
        """Setup Dask cluster for distributed training."""
        if not self.config['dask']['use_distributed']:
            logger.info("Distributed mode disabled, using LocalCluster")
            self._setup_local_cluster()
            return
        
        # Check for pre-configured external scheduler
        scheduler_address = os.environ.get('DASK_SCHEDULER_ADDRESS')
        if scheduler_address:
            logger.info(f"Connecting to external scheduler at {scheduler_address}")
            self.client = Client(scheduler_address)
            logger.info(f"Connected - Dashboard: {self.client.dashboard_link}")
            logger.info(f"Workers: {len(self.client.scheduler_info()['workers'])}")
            return
        
        # Vertex AI: Check which worker pool we're in
        worker_pool = int(os.environ.get('CLOUD_ML_WORKER_POOL_INDEX', -1))
        
        if worker_pool == -1:
            # Local environment
            logger.info("Local environment, using LocalCluster")
            self._setup_local_cluster()
        elif worker_pool == 0:
            # Primary: Start scheduler and connect
            self._start_scheduler_and_connect()
        else:
            # Worker: Start worker process and block
            self._start_worker_and_block()
        
    def _start_scheduler_and_connect(self):
        """Start Dask scheduler subprocess on primary node."""
        logger.info("[PRIMARY] Starting Dask Scheduler...")
        
        # Start dask-scheduler in background
        scheduler_proc = subprocess.Popen(
            ['dask-scheduler', '--port', '8786', '--dashboard-address', ':8787'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Give scheduler time to start
        time.sleep(5)
        
        # Connect client
        scheduler_address = 'tcp://localhost:8786'
        logger.info(f"Connecting to scheduler at {scheduler_address}")
        self.client = Client(scheduler_address, timeout='60s')
        logger.info(f"Dashboard: {self.client.dashboard_link}")
        logger.info("Waiting for worker nodes to connect...")
    
    def _start_worker_and_block(self):
        """Start Dask worker subprocess and block (worker nodes don't run training)."""
        logger.info("[WORKER] Starting Dask Worker...")
        
        # Construct primary node hostname (Vertex AI naming convention)
        # Format: {job-name}-workerpool0-0
        job_name = os.environ.get('CLOUD_ML_JOB_ID', 'training')
        primary_host = f"{job_name}-workerpool0-0"
        scheduler_address = f"tcp://{primary_host}:8786"
        
        logger.info(f"Connecting to scheduler at {scheduler_address}")
        
        # Start dask-worker and block (this process becomes the worker)
        subprocess.run([
            'dask-worker',
            scheduler_address,
            '--nthreads', str(self.config['dask']['threads_per_worker']),
            '--memory-limit', self.config['dask']['memory_limit']
        ])
        
        # If worker exits, log and exit process
        logger.info("Worker process completed")
        exit(0)
    
    def _setup_local_cluster(self):
        """Setup local Dask cluster for testing."""
        logger.info("Setting up local Dask cluster")
        cluster = LocalCluster(
            n_workers=self.config['dask']['n_workers'],
            threads_per_worker=self.config['dask']['threads_per_worker'],
            memory_limit=self.config['dask']['memory_limit']
        )
        self.client = Client(cluster)
        logger.info(f"Dask dashboard: {self.client.dashboard_link}")
        logger.info(f"Dask cluster workers: {len(self.client.scheduler_info()['workers'])}")
        
    def setup_mlflow(self):
        """Initialize MLflow tracking."""
        tracking_uri = self.config['mlflow']['tracking_uri']
        experiment_name = self.config['mlflow']['experiment_name']
        artifact_location = self.config['mlflow'].get('artifact_location')
        
        mlflow.set_tracking_uri(tracking_uri)
        
        # Set experiment (artifact_location is configured server-side or via env var)
        mlflow.set_experiment(experiment_name)
        
        logger.info(f"MLflow tracking URI: {tracking_uri}")
        logger.info(f"MLflow experiment: {experiment_name}")
        if artifact_location:
            logger.info(f"MLflow artifact location (configured): {artifact_location}")
        
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
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with mlflow.start_run(run_name=f"xgb-hpo-study_{timestamp}") as parent_run:
            
            def objective(trial):
                """Optuna objective for XGBoost."""
                # Suggest hyperparameters
                # scale_pos_weight: [25, 100] calibrated for 1:100 undersampling ratio
                params = {
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                    'gamma': trial.suggest_float('gamma', 0, 5),
                    'scale_pos_weight': trial.suggest_float('scale_pos_weight', 25, 100),
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
            
    def _run_cv(self, params, trial_num, model_type='xgboost'):
        """
        Unified cross-validation runner for both XGBoost and LightGBM.
        
        Args:
            params: Model hyperparameters
            trial_num: Optuna trial number
            model_type: 'xgboost' or 'lightgbm'
        """
        model_name = 'XGBoost' if model_type == 'xgboost' else 'LightGBM'
        logger.info(f"{model_name} trial {trial_num}: Running {len(self.folds)}-fold CV")
        
        stratify_col = self.config['validation']['stratify_by']
        random_state = self.config['models'][model_type]['random_state']
        
        # Log undersampling status once per trial
        if self._undersample_enabled:
            logger.info(f"  >>> UNDERSAMPLING ACTIVE: Ratio 1:{self._undersample_ratio} (Dask-level) <<<")
        else:
            logger.info("  [UNDERSAMPLE] Disabled - using full training data")
        
        aucpr_scores = []
        
        for fold_idx, (train_providers, test_providers) in enumerate(self.folds):
            logger.info(f"  Fold {fold_idx + 1}/{len(self.folds)}: Starting...")
            
            # Split data (lazy Dask operation)
            train_df, test_df = split_data_by_providers(
                self.df, train_providers, test_providers, stratify_col
            )
            
            # Prepare fold data (handles undersampling + compute)
            X_train, y_train, X_test, y_test = self._prepare_fold_data(
                train_df, test_df, fold_idx, random_state
            )
            
            # Train model
            logger.info(f"  Fold {fold_idx + 1}: Training {model_name} on {len(X_train):,} rows...")
            if model_type == 'xgboost':
                model = xgb.XGBClassifier(**params)
                model.fit(X_train, y_train, verbose=False)
            else:
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
    
    def _run_cv_xgboost(self, params, trial_num):
        """Run cross-validation for XGBoost."""
        return self._run_cv(params, trial_num, model_type='xgboost')
        
    def optimize_lightgbm(self):
        """Run HPO for LightGBM using Optuna."""
        logger.info("="*60)
        logger.info("STEP 3: LightGBM Hyperparameter Optimization")
        logger.info("="*60)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with mlflow.start_run(run_name=f"lgbm-hpo-study_{timestamp}") as parent_run:
            
            def objective(trial):
                """Optuna objective for LightGBM."""
                # Suggest hyperparameters
                # scale_pos_weight: [25, 100] calibrated for 1:100 undersampling ratio
                params = {
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'num_leaves': trial.suggest_int('num_leaves', 20, 150),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'min_child_weight': trial.suggest_float('min_child_weight', 0.001, 10, log=True),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                    'scale_pos_weight': trial.suggest_float('scale_pos_weight', 25, 100),
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
        return self._run_cv(params, trial_num, model_type='lightgbm')
        
    def generate_oof_predictions(self):
        """Generate out-of-fold predictions for stacking."""
        logger.info("="*60)
        logger.info("STEP 4: Generating OOF Predictions for Stacking")
        logger.info("="*60)
        
        stratify_col = self.config['validation']['stratify_by']
        
        if self._undersample_enabled:
            logger.info(f">>> OOF: UNDERSAMPLING ENABLED - Ratio 1:{self._undersample_ratio} <<<")
        else:
            logger.info(">>> OOF: UNDERSAMPLING DISABLED - using full training data <<<")
        
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
            
            # Split data (lazy Dask operation)
            train_df, test_df = split_data_by_providers(
                self.df, train_providers, test_providers, stratify_col
            )
            
            # Prepare fold data (handles undersampling + compute)
            X_train, y_train, X_test, y_test = self._prepare_fold_data(
                train_df, test_df, fold_idx, random_state=42, phase="OOF"
            )
            
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
        """Train final models on full dataset with undersampling."""
        logger.info("="*60)
        logger.info("STEP 6: Training Final Models on Full Data")
        logger.info("="*60)
        
        target_col = self.config['validation']['target_column']
        stratify_col = self.config['validation']['stratify_by']
        
        if self._undersample_enabled:
            logger.info(f">>> FINAL MODELS: UNDERSAMPLING ENABLED - Ratio 1:{self._undersample_ratio} <<<")
        else:
            logger.info(">>> FINAL MODELS: UNDERSAMPLING DISABLED - using full data <<<")
        
        # Apply Dask-level undersampling BEFORE .compute() (per design doc 3.1)
        df_to_train = self.df
        if self._undersample_enabled:
            logger.info("Applying Dask-level undersampling on full dataset...")
            df_to_train = undersample_dask_dataframe(
                self.df, target_col,
                ratio=self._undersample_ratio,
                random_state=42
            )
        
        # NOW .compute() on (undersampled) data
        logger.info("Computing training data...")
        X_full = df_to_train.drop(columns=[target_col, stratify_col]).compute()
        y_full = df_to_train[target_col].compute()
        
        logger.info(f"Training on dataset: {len(X_full):,} rows")
        
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
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with mlflow.start_run(run_name=f"final-models_{timestamp}") as run:
            
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
            
            # Save feature importance (per design doc Section 4.3)
            self._save_feature_importance(output_path)
            
            logger.info(f"All artifacts saved to {output_path}")
            logger.info(f"MLflow run ID: {run.info.run_id}")
    
    def _save_feature_importance(self, output_path):
        """
        Extract and save top 20 feature importances to CSV and MLflow.
        
        Per design doc: Log feature importance to understand why the model
        makes predictions, ensuring it learns fraud patterns (Z-scores) not noise.
        """
        logger.info("Extracting feature importance...")
        
        try:
            # Get feature names from training data
            target_col = self.config['validation']['target_column']
            stratify_col = self.config['validation']['stratify_by']
            feature_names = [c for c in self.df.columns 
                           if c not in [target_col, stratify_col]]
            
            # XGBoost feature importance
            xgb_importance = self.final_xgb.feature_importances_
            xgb_df = pd.DataFrame({
                'feature': feature_names,
                'importance_xgb': xgb_importance
            }).sort_values('importance_xgb', ascending=False)
            
            # LightGBM feature importance  
            lgbm_importance = self.final_lgbm.feature_importances_
            lgbm_df = pd.DataFrame({
                'feature': feature_names,
                'importance_lgbm': lgbm_importance
            }).sort_values('importance_lgbm', ascending=False)
            
            # Merge and compute average importance
            importance_df = xgb_df.merge(lgbm_df, on='feature')
            importance_df['importance_avg'] = (
                importance_df['importance_xgb'] + importance_df['importance_lgbm']
            ) / 2
            importance_df = importance_df.sort_values('importance_avg', ascending=False)
            
            # Save top 20 features
            top_20 = importance_df.head(20)
            importance_path = output_path / "feature_importance.csv"
            top_20.to_csv(importance_path, index=False)
            mlflow.log_artifact(importance_path)
            
            logger.info(f"Saved top 20 feature importance to {importance_path}")
            logger.info("Top 5 features:")
            for _, row in top_20.head(5).iterrows():
                logger.info(f"  {row['feature']}: {row['importance_avg']:.4f}")
                
        except Exception as e:
            logger.warning(f"Failed to save feature importance: {e}")
    
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
    args = parse_args()
    config = get_runtime_config(args)
    
    trainer = DistributedTrainer(config=config)
    trainer.run_training()


if __name__ == "__main__":
    main()
