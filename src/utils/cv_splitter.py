"""Cross-validation splitting utilities for provider-level stratification."""

import logging
import numpy as np
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)


def create_provider_stratified_folds(provider_labels, n_folds, random_state=42):
    """Create stratified K-fold splits at provider level."""
    logger.info(f"Creating {n_folds}-fold stratified splits")
    
    provider_ids = provider_labels['provider_npi'].values
    fraud_labels = provider_labels['fraud_label'].values
    
    fraud_count = fraud_labels.sum()
    total_count = len(fraud_labels)
    logger.info(f"Provider distribution - Total: {total_count:,}, Fraud: {fraud_count:,}")
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    folds = []
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(provider_ids, fraud_labels)):
        train_providers = provider_ids[train_idx]
        test_providers = provider_ids[test_idx]
        folds.append((train_providers, test_providers))
    
    logger.info(f"Created {n_folds} stratified folds")
    return folds


def split_data_by_providers(df, train_providers, test_providers, stratify_col='provider_npi'):
    """Split Dask DataFrame into train/test based on provider lists."""
    train_set = set(train_providers)
    test_set = set(test_providers)
    
    train_df = df[df[stratify_col].isin(train_set)]
    test_df = df[df[stratify_col].isin(test_set)]
    
    return train_df, test_df


def get_fold_indices(df, folds, stratify_col='provider_npi'):
    """Convert provider-level folds to row-level indices."""
    provider_col = df[stratify_col].compute()
    
    fold_indices = []
    for fold_idx, (train_providers, test_providers) in enumerate(folds):
        train_mask = provider_col.isin(train_providers)
        test_mask = provider_col.isin(test_providers)
        
        train_indices = np.where(train_mask)[0]
        test_indices = np.where(test_mask)[0]
        fold_indices.append((train_indices, test_indices))
    
    return fold_indices
