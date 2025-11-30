"""Data loading utilities for distributed ML pipeline."""

import logging
import dask.dataframe as dd
from google.cloud import storage
import yaml

logger = logging.getLogger(__name__)


def load_config(config_path="config/pipeline_config.yaml"):
    """Load pipeline configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_training_data(config, use_sample=False):
    """Load training data from GCS parquet files into Dask DataFrame."""
    gcs_path = config['data']['training_data_path']
    n_rows_sample = config['data'].get('n_rows_sample')
    
    logger.info(f"Loading training data from: {gcs_path}")
    
    try:
        # Load all parquet files from the GCS path
        df = dd.read_parquet(
            f"{gcs_path}/*.parquet",
            engine='pyarrow'
        )
        
        # Apply sampling if configured
        if use_sample and n_rows_sample:
            logger.info(f"Sampling {n_rows_sample} rows for testing")
            df = df.head(n_rows_sample, npartitions=-1)
            df = dd.from_pandas(df, npartitions=4)
        
        total_rows = len(df)
        logger.info(f"Data loaded successfully. Total rows: {total_rows:,}")
        logger.info(f"Number of partitions: {df.npartitions}")
        
        return df
        
    except Exception as e:
        logger.error(f"Failed to load training data from {gcs_path}")
        logger.error(f"Error: {e}")
        raise


def validate_data(df, config):
    """Validate loaded data has required columns and correct types."""
    target_col = config['validation']['target_column']
    stratify_col = config['validation']['stratify_by']
    
    required_cols = [target_col, stratify_col]
    
    logger.info("Validating data schema...")
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    null_counts = df[required_cols].isnull().sum().compute()
    if null_counts.sum() > 0:
        logger.warning(f"Null values in required columns: {null_counts.to_dict()}")
    
    fraud_count = df[target_col].sum().compute()
    total_count = len(df)
    fraud_ratio = fraud_count / total_count if total_count > 0 else 0
    
    logger.info(f"Class distribution - Fraud: {fraud_count:,} ({fraud_ratio:.4%})")
    
    return True


def get_provider_labels(df, config):
    """Extract provider-level fraud labels for stratification."""
    stratify_col = config['validation']['stratify_by']
    target_col = config['validation']['target_column']
    
    provider_labels = (
        df.groupby(stratify_col)[target_col]
        .max()
        .compute()
        .reset_index()
    )
    provider_labels.columns = ['provider_npi', 'fraud_label']
    
    fraud_providers = provider_labels['fraud_label'].sum()
    total_providers = len(provider_labels)
    logger.info(f"Provider stats - Total: {total_providers:,}, Fraud: {fraud_providers:,}")
    
    return provider_labels
