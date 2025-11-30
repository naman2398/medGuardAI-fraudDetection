"""Training orchestrator for distributed ensemble ML pipeline."""

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
from utils.evaluation import calculate_aucpr, evaluate_fold

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def undersample_dask_dataframe(train_df, target_col, ratio=100, random_state=42):
    """Undersample within Dask BEFORE .compute() to avoid memory issues."""
    logger.info(f"Undersampling with ratio 1:{ratio}...")
    
    train_df = train_df.reset_index(drop=True)
    train_df = train_df.persist()
    
    fraud_df = train_df[train_df[target_col] == 1]
    normal_df = train_df[train_df[target_col] == 0]
    
    n_fraud = fraud_df.shape[0].compute()
    n_normal = normal_df.shape[0].compute()
    
    logger.info(f"Original: Fraud={n_fraud:,}, Normal={n_normal:,}")
    
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
    """Parse command-line arguments."""
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
    
    base_config['data']['n_rows_sample'] = None if args.n_rows_sample == -1 else args.n_rows_sample
    base_config['models']['xgboost']['n_trials'] = args.xgb_trials
    base_config['models']['xgboost']['cv_folds'] = args.cv_folds
    base_config['models']['lightgbm']['n_trials'] = args.lgbm_trials
    base_config['models']['lightgbm']['cv_folds'] = args.cv_folds
    base_config['compute']['workers']['count'] = args.worker_count
    
    data_rows_info = 'ALL DATA' if base_config['data']['n_rows_sample'] is None else f"{base_config['data']['n_rows_sample']:,}"
    logger.info(f"Config: {data_rows_info} rows, XGB={base_config['models']['xgboost']['n_trials']} trials, LGBM={base_config['models']['lightgbm']['n_trials']} trials, {base_config['models']['xgboost']['cv_folds']} folds")
    
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
        
        sampling_config = self.config.get('sampling', {})
        self._undersample_enabled = sampling_config.get('undersample_enabled', False)
        self._undersample_ratio = sampling_config.get('undersample_ratio', 100)
        self._fold_cache = None
    
    def _fix_dtypes(self, df):
        """Fix data types for XGBoost/LightGBM compatibility."""
        for col in df.columns:
            if df[col].dtype == 'object' or str(df[col].dtype) == 'string':
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
                except Exception:
                    df[col] = df[col].astype('category')
        return df
    
    def _prepare_fold_data(self, train_df, test_df, fold_idx, random_state, phase="CV"):
        """Prepare train/test data for a fold with optional undersampling."""
        target_col = self.config['validation']['target_column']
        stratify_col = self.config['validation']['stratify_by']
        
        if self._undersample_enabled:
            train_df = undersample_dask_dataframe(
                train_df, target_col,
                ratio=self._undersample_ratio,
                random_state=random_state + fold_idx
            )
        
        X_train = train_df.drop(columns=[target_col, stratify_col]).compute()
        y_train = train_df[target_col].compute()
        
        X_test = test_df.drop(columns=[target_col, stratify_col]).compute()
        y_test = test_df[target_col].compute()
        
        X_train = self._fix_dtypes(X_train)
        X_test = self._fix_dtypes(X_test)
        
        return X_train, y_train, X_test, y_test
    
    def _precompute_folds(self):
        """Precompute undersampled fold data ONCE before HPO."""
        logger.info("Precomputing fold data...")
        
        stratify_col = self.config['validation']['stratify_by']
        random_state = self.config['models']['xgboost']['random_state']
        
        self._fold_cache = {}
        
        for fold_idx, (train_providers, test_providers) in enumerate(self.folds):
            train_df, test_df = split_data_by_providers(
                self.df, train_providers, test_providers, stratify_col
            )
            
            X_train, y_train, X_test, y_test = self._prepare_fold_data(
                train_df, test_df, fold_idx, random_state
            )
            
            self._fold_cache[fold_idx] = (X_train, y_train, X_test, y_test)
        
        logger.info(f"Fold cache complete: {len(self._fold_cache)} folds ready")
        
    def setup_dask(self):
        """Setup Dask cluster for distributed training."""
        if not self.config['dask']['use_distributed']:
            self._setup_local_cluster()
            return
        
        scheduler_address = os.environ.get('DASK_SCHEDULER_ADDRESS')
        if scheduler_address:
            logger.info(f"Connecting to scheduler at {scheduler_address}")
            self.client = Client(scheduler_address)
            return
        
        worker_pool = int(os.environ.get('CLOUD_ML_WORKER_POOL_INDEX', -1))
        
        if worker_pool == -1:
            self._setup_local_cluster()
        elif worker_pool == 0:
            self._start_scheduler_and_connect()
        else:
            self._start_worker_and_block()
        
    def _start_scheduler_and_connect(self):
        """Start Dask scheduler subprocess on primary node."""
        logger.info("Starting Dask Scheduler...")
        
        scheduler_proc = subprocess.Popen(
            ['dask-scheduler', '--port', '8786', '--dashboard-address', ':8787'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        time.sleep(5)
        
        scheduler_address = 'tcp://localhost:8786'
        self.client = Client(scheduler_address, timeout='60s')
        logger.info(f"Dashboard: {self.client.dashboard_link}")
    
    def _start_worker_and_block(self):
        """Start Dask worker subprocess and block."""
        logger.info("Starting Dask Worker...")
        
        job_name = os.environ.get('CLOUD_ML_JOB_ID', 'training')
        primary_host = f"{job_name}-workerpool0-0"
        scheduler_address = f"tcp://{primary_host}:8786"
        
        subprocess.run([
            'dask-worker',
            scheduler_address,
            '--nthreads', str(self.config['dask']['threads_per_worker']),
            '--memory-limit', self.config['dask']['memory_limit']
        ])
        
        exit(0)
    
    def _setup_local_cluster(self):
        """Setup local Dask cluster."""
        cluster = LocalCluster(
            n_workers=self.config['dask']['n_workers'],
            threads_per_worker=self.config['dask']['threads_per_worker'],
            memory_limit=self.config['dask']['memory_limit']
        )
        self.client = Client(cluster)
        logger.info(f"Dask dashboard: {self.client.dashboard_link}")
        
    def setup_mlflow(self):
        """Initialize MLflow tracking."""
        tracking_uri = self.config['mlflow']['tracking_uri']
        experiment_name = self.config['mlflow']['experiment_name']
        
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        logger.info(f"MLflow: {tracking_uri}, experiment: {experiment_name}")
        
    def load_data(self):
        """Load and prepare training data."""
        logger.info("Loading and preparing data...")
        
        self.df = load_training_data(self.config, use_sample=True)
        validate_data(self.df, self.config)
        
        self.provider_labels = get_provider_labels(self.df, self.config)
        
        n_folds = self.config['models']['xgboost']['cv_folds']
        random_state = self.config['models']['xgboost']['random_state']
        
        self.folds = create_provider_stratified_folds(
            self.provider_labels,
            n_folds=n_folds,
            random_state=random_state
        )
        
        logger.info(f"Data ready with {n_folds} folds")
        
    def optimize_xgboost(self):
        """Run HPO for XGBoost using Optuna."""
        logger.info("XGBoost HPO starting...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with mlflow.start_run(run_name=f"xgb-hpo-study_{timestamp}") as parent_run:
            
            def objective(trial):
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
                cv_score, cv_metrics = self._run_cv_xgboost(params, trial.number)
                
                # Store metrics in trial for best trial retrieval
                trial.set_user_attr('cv_metrics', cv_metrics)
                
                # Log to MLflow
                with mlflow.start_run(run_name=f"xgb-trial-{trial.number}", nested=True):
                    mlflow.log_params(params)
                    mlflow.log_metric("cv_aucpr", cv_score)
                    mlflow.log_metric("cv_auroc", cv_metrics['auroc'])
                    mlflow.log_metric("cv_f2", cv_metrics['f2'])
                    mlflow.log_metric("cv_tpr", cv_metrics['tpr'])
                    mlflow.log_metric("cv_tnr", cv_metrics['tnr'])
                    mlflow.log_metric("cv_precision", cv_metrics['precision'])
                
                return cv_score
            
            study = optuna.create_study(direction='maximize')
            n_trials = self.config['models']['xgboost']['n_trials']
            
            study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
            
            self.best_params_xgb = study.best_params
            self.best_params_xgb.update({
                'objective': 'binary:logistic',
                'eval_metric': 'aucpr',
                'tree_method': 'hist',
                'random_state': self.config['models']['xgboost']['random_state']
            })
            
            logger.info(f"Best XGBoost AUCPR: {study.best_value:.4f}")
            
            mlflow.log_params(self.best_params_xgb)
            mlflow.log_metric("best_cv_aucpr", study.best_value)
            self.best_xgb_metrics = study.best_trial.user_attrs.get('cv_metrics', {})
            
    def _run_cv(self, params, trial_num, model_type='xgboost'):
        """Cross-validation runner for both XGBoost and LightGBM."""
        model_name = 'XGBoost' if model_type == 'xgboost' else 'LightGBM'
        
        stratify_col = self.config['validation']['stratify_by']
        random_state = self.config['models'][model_type]['random_state']
        
        aucpr_scores = []
        auroc_scores = []
        f2_scores = []
        tpr_scores = []
        tnr_scores = []
        precision_scores = []
        
        for fold_idx, (train_providers, test_providers) in enumerate(self.folds):
            if self._fold_cache is not None and fold_idx in self._fold_cache:
                X_train, y_train, X_test, y_test = self._fold_cache[fold_idx]
            else:
                train_df, test_df = split_data_by_providers(
                    self.df, train_providers, test_providers, stratify_col
                )
                X_train, y_train, X_test, y_test = self._prepare_fold_data(
                    train_df, test_df, fold_idx, random_state
                )
            
            if model_type == 'xgboost':
                model = xgb.XGBClassifier(**params)
                model.fit(X_train, y_train, verbose=False)
            else:
                model = lgb.LGBMClassifier(**params)
                model.fit(X_train, y_train)
            
            # Predict and evaluate using full metrics
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            fold_metrics = evaluate_fold(y_test, y_pred_proba, fold_idx=fold_idx)
            aucpr_scores.append(fold_metrics['aucpr'])
            auroc_scores.append(fold_metrics['auroc'])
            f2_scores.append(fold_metrics['f2'])
            tpr_scores.append(fold_metrics['tpr'])
            tnr_scores.append(fold_metrics['tnr'])
            precision_scores.append(fold_metrics['precision'])
        
        cv_mean = np.mean(aucpr_scores)
        
        cv_metrics = {
            'aucpr': cv_mean,
            'auroc': np.mean(auroc_scores),
            'f2': np.mean(f2_scores),
            'tpr': np.mean(tpr_scores),
            'tnr': np.mean(tnr_scores),
            'precision': np.mean(precision_scores)
        }
        return cv_mean, cv_metrics
    
    def _run_cv_xgboost(self, params, trial_num):
        """Run cross-validation for XGBoost."""
        return self._run_cv(params, trial_num, model_type='xgboost')
        
    def optimize_lightgbm(self):
        """Run HPO for LightGBM using Optuna."""
        logger.info("LightGBM HPO starting...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with mlflow.start_run(run_name=f"lgbm-hpo-study_{timestamp}") as parent_run:
            
            def objective(trial):
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
                cv_score, cv_metrics = self._run_cv_lightgbm(params, trial.number)
                
                # Store metrics in trial for best trial retrieval
                trial.set_user_attr('cv_metrics', cv_metrics)
                
                # Log to MLflow
                with mlflow.start_run(run_name=f"lgbm-trial-{trial.number}", nested=True):
                    mlflow.log_params(params)
                    mlflow.log_metric("cv_aucpr", cv_score)
                    mlflow.log_metric("cv_auroc", cv_metrics['auroc'])
                    mlflow.log_metric("cv_f2", cv_metrics['f2'])
                    mlflow.log_metric("cv_tpr", cv_metrics['tpr'])
                    mlflow.log_metric("cv_tnr", cv_metrics['tnr'])
                    mlflow.log_metric("cv_precision", cv_metrics['precision'])
                
                return cv_score
            
            study = optuna.create_study(direction='maximize')
            n_trials = self.config['models']['lightgbm']['n_trials']
            
            study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
            
            self.best_params_lgbm = study.best_params
            self.best_params_lgbm.update({
                'objective': 'binary',
                'metric': 'auc',
                'random_state': self.config['models']['lightgbm']['random_state'],
                'verbose': -1
            })
            
            logger.info(f"Best LightGBM AUCPR: {study.best_value:.4f}")
            
            mlflow.log_params(self.best_params_lgbm)
            mlflow.log_metric("best_cv_aucpr", study.best_value)
            self.best_lgbm_metrics = study.best_trial.user_attrs.get('cv_metrics', {})
            
    def _run_cv_lightgbm(self, params, trial_num):
        """Run cross-validation for LightGBM."""
        return self._run_cv(params, trial_num, model_type='lightgbm')
        
    def generate_oof_predictions(self):
        """Generate out-of-fold predictions for stacking."""
        logger.info("Generating OOF predictions...")
        
        stratify_col = self.config['validation']['stratify_by']
        total_rows = len(self.df)
        
        oof_xgb = np.zeros(total_rows)
        oof_lgbm = np.zeros(total_rows)
        oof_targets = np.zeros(total_rows)
        
        provider_col = self.df[stratify_col].compute()
        
        for fold_idx, (train_providers, test_providers) in enumerate(self.folds):
            if self._fold_cache is not None and fold_idx in self._fold_cache:
                X_train, y_train, X_test, y_test = self._fold_cache[fold_idx]
            else:
                train_df, test_df = split_data_by_providers(
                    self.df, train_providers, test_providers, stratify_col
                )
                X_train, y_train, X_test, y_test = self._prepare_fold_data(
                    train_df, test_df, fold_idx, random_state=42, phase="OOF"
                )
            
            test_mask = provider_col.isin(test_providers)
            test_indices = np.where(test_mask)[0]
            
            model_xgb = xgb.XGBClassifier(**self.best_params_xgb)
            model_xgb.fit(X_train, y_train, verbose=False)
            oof_xgb[test_indices] = model_xgb.predict_proba(X_test)[:, 1]
            
            model_lgbm = lgb.LGBMClassifier(**self.best_params_lgbm)
            model_lgbm.fit(X_train, y_train)
            oof_lgbm[test_indices] = model_lgbm.predict_proba(X_test)[:, 1]
            
            oof_targets[test_indices] = y_test.values
        
        self.oof_predictions = pd.DataFrame({
            'xgb_pred': oof_xgb,
            'lgbm_pred': oof_lgbm,
            'target': oof_targets
        })
        
        self.oof_metrics_xgb = evaluate_fold(oof_targets, oof_xgb)
        self.oof_metrics_lgbm = evaluate_fold(oof_targets, oof_lgbm)
        
        logger.info(f"OOF AUCPR - XGB: {self.oof_metrics_xgb['aucpr']:.4f}, LGBM: {self.oof_metrics_lgbm['aucpr']:.4f}")
        
    def train_stacker(self):
        """Train stacker model on OOF predictions."""
        logger.info("Training stacker model...")
        
        X_stack = self.oof_predictions[['xgb_pred', 'lgbm_pred']].values
        y_stack = self.oof_predictions['target'].values
        
        self.stacker = LogisticRegression(
            max_iter=self.config['models']['stacker']['max_iter'],
            random_state=self.config['models']['stacker']['random_state']
        )
        self.stacker.fit(X_stack, y_stack)
        
        y_stack_pred = self.stacker.predict_proba(X_stack)[:, 1]
        self.stacker_metrics = evaluate_fold(y_stack, y_stack_pred)
        
        self.stacker_coef_xgb = self.stacker.coef_[0][0]
        self.stacker_coef_lgbm = self.stacker.coef_[0][1]
        
        logger.info(f"Stacker AUCPR: {self.stacker_metrics['aucpr']:.4f}")
        
    def train_final_models(self):
        """Train final models on full dataset with undersampling."""
        logger.info("Training final models...")
        
        target_col = self.config['validation']['target_column']
        stratify_col = self.config['validation']['stratify_by']
        
        df_to_train = self.df
        if self._undersample_enabled:
            df_to_train = undersample_dask_dataframe(
                self.df, target_col,
                ratio=self._undersample_ratio,
                random_state=42
            )
        
        X_full = df_to_train.drop(columns=[target_col, stratify_col]).compute()
        y_full = df_to_train[target_col].compute()
        X_full = self._fix_dtypes(X_full)
        
        logger.info(f"Training on {len(X_full):,} rows")
        
        self.final_xgb = xgb.XGBClassifier(**self.best_params_xgb)
        self.final_xgb.fit(X_full, y_full, verbose=False)
        
        self.final_lgbm = lgb.LGBMClassifier(**self.best_params_lgbm)
        self.final_lgbm.fit(X_full, y_full)
        
        logger.info("Final models trained")
        
    def save_models(self, output_dir=None):
        """Save all model artifacts."""
        logger.info("Saving model artifacts...")
        
        if output_dir is None:
            job_name = os.environ.get('JOB_NAME', 'local-training')
            if self.config['dask']['use_distributed']:
                bucket = self.config['data']['gcs_bucket']
                output_dir = f"gs://{bucket}/models/artifacts/{job_name}"
            else:
                output_dir = f"models/artifacts/{job_name}"
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with mlflow.start_run(run_name=f"final-models_{timestamp}") as run:
            
            xgb_path = output_path / "model_xgb.json"
            self.final_xgb.save_model(xgb_path)
            mlflow.log_artifact(xgb_path)
            
            lgbm_path = output_path / "model_lgbm.txt"
            self.final_lgbm.booster_.save_model(str(lgbm_path))
            mlflow.log_artifact(lgbm_path)
            
            stacker_path = output_path / "stacker_model.joblib"
            joblib.dump(self.stacker, stacker_path)
            mlflow.log_artifact(stacker_path)
            
            params_path = output_path / "best_params.json"
            with open(params_path, 'w') as f:
                json.dump({
                    'xgboost': self.best_params_xgb,
                    'lightgbm': self.best_params_lgbm
                }, f, indent=2)
            mlflow.log_artifact(params_path)
            
            self._save_feature_importance(output_path)
            
            mlflow.log_metric("final_aucpr", self.stacker_metrics['aucpr'])
            mlflow.log_metric("final_auroc", self.stacker_metrics['auroc'])
            mlflow.log_metric("final_f2", self.stacker_metrics['f2'])
            mlflow.log_metric("oof_xgb_aucpr", self.oof_metrics_xgb['aucpr'])
            mlflow.log_metric("oof_lgbm_aucpr", self.oof_metrics_lgbm['aucpr'])
            mlflow.log_metric("stacker_weight_xgb", self.stacker_coef_xgb)
            mlflow.log_metric("stacker_weight_lgbm", self.stacker_coef_lgbm)
            
            logger.info(f"Artifacts saved to {output_path}")
            logger.info(f"Final ensemble AUCPR: {self.stacker_metrics['aucpr']:.4f}")
    
    def _save_feature_importance(self, output_path):
        """Extract and save top 20 feature importances."""
        try:
            target_col = self.config['validation']['target_column']
            stratify_col = self.config['validation']['stratify_by']
            feature_names = [c for c in self.df.columns 
                           if c not in [target_col, stratify_col]]
            
            xgb_importance = self.final_xgb.feature_importances_
            xgb_df = pd.DataFrame({
                'feature': feature_names,
                'importance_xgb': xgb_importance
            }).sort_values('importance_xgb', ascending=False)
            
            lgbm_importance = self.final_lgbm.feature_importances_
            lgbm_df = pd.DataFrame({
                'feature': feature_names,
                'importance_lgbm': lgbm_importance
            }).sort_values('importance_lgbm', ascending=False)
            
            importance_df = xgb_df.merge(lgbm_df, on='feature')
            importance_df['importance_avg'] = (
                importance_df['importance_xgb'] + importance_df['importance_lgbm']
            ) / 2
            importance_df = importance_df.sort_values('importance_avg', ascending=False)
            
            top_20 = importance_df.head(20)
            importance_path = output_path / "feature_importance.csv"
            top_20.to_csv(importance_path, index=False)
            mlflow.log_artifact(importance_path)
                
        except Exception as e:
            logger.warning(f"Failed to save feature importance: {e}")
    
    def run_training(self):
        """Main training pipeline execution."""
        logger.info("Starting distributed ensemble training pipeline...")
        
        try:
            self.setup_dask()
            self.setup_mlflow()
            self.load_data()
            self._precompute_folds()
            
            self.optimize_xgboost()
            self.optimize_lightgbm()
            
            self.generate_oof_predictions()
            self.train_stacker()
            
            self.train_final_models()
            self.save_models()
            
            logger.info("Training pipeline completed successfully")
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {e}", exc_info=True)
            raise
        
        finally:
            if self.client:
                self.client.close()


def main():
    """Entry point for training script."""
    args = parse_args()
    config = get_runtime_config(args)
    trainer = DistributedTrainer(config=config)
    trainer.run_training()


if __name__ == "__main__":
    main()
