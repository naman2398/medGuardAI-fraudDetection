"""
Cross-validation splitting utilities for provider-level stratification.
Ensures no provider data leaks between train/test sets.
"""

import logging
import numpy as np
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)


def create_provider_stratified_folds(provider_labels, n_folds, random_state=42):
    """
    Create stratified K-fold splits at provider level.
    
    This ensures:
    1. No provider appears in both train and test sets
    2. Each fold maintains the correct fraud/non-fraud ratio
    
    Args:
        provider_labels: DataFrame with columns [provider_npi, fraud_label]
        n_folds: Number of CV folds
        random_state: Random seed for reproducibility
        
    Returns:
        list: List of (train_providers, test_providers) tuples for each fold
    """
    logger.info(f"Creating {n_folds}-fold stratified splits at provider level")
    
    provider_ids = provider_labels['provider_npi'].values
    fraud_labels = provider_labels['fraud_label'].values
    
    # Check class distribution
    fraud_count = fraud_labels.sum()
    total_count = len(fraud_labels)
    
    logger.info(f"Provider-level distribution - Total: {total_count:,}, Fraud: {fraud_count:,} ({fraud_count/total_count:.4%})")
    
    # Create stratified folds
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    folds = []
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(provider_ids, fraud_labels)):
        train_providers = provider_ids[train_idx]
        test_providers = provider_ids[test_idx]
        
        train_fraud = fraud_labels[train_idx].sum()
        test_fraud = fraud_labels[test_idx].sum()
        
        logger.info(
            f"Fold {fold_idx + 1}: "
            f"Train providers={len(train_providers):,} (fraud={train_fraud:,}), "
            f"Test providers={len(test_providers):,} (fraud={test_fraud:,})"
        )
        
        folds.append((train_providers, test_providers))
    
    logger.info(f"Created {n_folds} stratified folds successfully")
    return folds


def split_data_by_providers(df, train_providers, test_providers, stratify_col='provider_npi'):
    """
    Split Dask DataFrame into train/test based on provider lists.
    
    Args:
        df: Dask DataFrame with all data
        train_providers: Array of provider NPIs for training
        test_providers: Array of provider NPIs for testing
        stratify_col: Column name containing provider identifiers
        
    Returns:
        tuple: (train_df, test_df) as Dask DataFrames
    """
    logger.info(f"Splitting data by {len(train_providers):,} train and {len(test_providers):,} test providers")
    
    # Convert to sets for faster lookup
    train_set = set(train_providers)
    test_set = set(test_providers)
    
    # Filter data
    train_df = df[df[stratify_col].isin(train_set)]
    test_df = df[df[stratify_col].isin(test_set)]
    
    # Log split sizes (this triggers computation but needed for validation)
    train_size = len(train_df)
    test_size = len(test_df)
    
    logger.info(f"Train set: {train_size:,} rows, Test set: {test_size:,} rows")
    
    return train_df, test_df


def get_fold_indices(df, folds, stratify_col='provider_npi'):
    """
    Convert provider-level folds to row-level indices for the full dataset.
    
    Args:
        df: Dask DataFrame with all data
        folds: List of (train_providers, test_providers) tuples
        stratify_col: Column name containing provider identifiers
        
    Returns:
        list: List of (train_indices, test_indices) for each fold
    """
    logger.info("Converting provider-level folds to row-level indices")
    
    # Compute provider column once
    provider_col = df[stratify_col].compute()
    
    fold_indices = []
    for fold_idx, (train_providers, test_providers) in enumerate(folds):
        train_mask = provider_col.isin(train_providers)
        test_mask = provider_col.isin(test_providers)
        
        train_indices = np.where(train_mask)[0]
        test_indices = np.where(test_mask)[0]
        
        logger.info(
            f"Fold {fold_idx + 1}: "
            f"Train indices={len(train_indices):,}, Test indices={len(test_indices):,}"
        )
        
        fold_indices.append((train_indices, test_indices))
    
    return fold_indices
