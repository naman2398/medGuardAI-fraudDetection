"""
Data loading utilities for distributed ML pipeline.
Loads training data from GCS into Dask DataFrame.
"""

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
    """
    Load training data from GCS parquet files into Dask DataFrame.
    
    Args:
        config: Configuration dictionary
        use_sample: If True, loads only a sample of the data
        
    Returns:
        dask.dataframe.DataFrame: Loaded training data
    """
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
    """
    Validate loaded data has required columns and correct types.
    
    Args:
        df: Dask DataFrame to validate
        config: Configuration dictionary
        
    Returns:
        bool: True if validation passes
    """
    target_col = config['validation']['target_column']
    stratify_col = config['validation']['stratify_by']
    
    required_cols = [target_col, stratify_col]
    
    logger.info("Validating data schema...")
    
    # Check required columns exist
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    logger.info(f"Required columns present: {required_cols}")
    
    # Check for nulls in critical columns
    null_counts = df[required_cols].isnull().sum().compute()
    if null_counts.sum() > 0:
        logger.warning(f"Null values found in required columns: {null_counts.to_dict()}")
    
    # Check class balance
    fraud_count = df[target_col].sum().compute()
    total_count = len(df)
    fraud_ratio = fraud_count / total_count if total_count > 0 else 0
    
    logger.info(f"Class distribution - Fraud: {fraud_count:,} ({fraud_ratio:.4%}), Non-fraud: {total_count - fraud_count:,}")
    
    expected_ratio = config['validation']['positive_class_ratio']
    if abs(fraud_ratio - expected_ratio) > 0.001:
        logger.warning(f"Fraud ratio {fraud_ratio:.4%} differs from expected {expected_ratio:.4%}")
    
    logger.info("Data validation completed")
    return True


def get_provider_labels(df, config):
    """
    Extract provider-level fraud labels for stratification.
    This is critical for preventing data leakage in CV.
    
    Args:
        df: Dask DataFrame with training data
        config: Configuration dictionary
        
    Returns:
        pandas.DataFrame: Provider-level labels (rndrng_npi, fraud_label)
    """
    stratify_col = config['validation']['stratify_by']
    target_col = config['validation']['target_column']
    
    logger.info(f"Computing provider-level labels for stratification by {stratify_col}")
    
    # Get one fraud label per provider (max ensures if ANY record is fraud, provider is fraud)
    provider_labels = (
        df.groupby(stratify_col)[target_col]
        .max()
        .compute()
        .reset_index()
    )
    
    # Rename columns for consistency
    provider_labels.columns = ['provider_npi', 'fraud_label']
    
    fraud_providers = provider_labels['fraud_label'].sum()
    total_providers = len(provider_labels)
    
    logger.info(f"Provider-level stats - Total: {total_providers:,}, Fraud: {fraud_providers:,} ({fraud_providers/total_providers:.4%})")
    
    return provider_labels
