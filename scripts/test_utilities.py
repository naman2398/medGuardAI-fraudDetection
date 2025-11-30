"""Test script for utility modules."""

import logging
import sys
sys.path.append('src')

from utils.data_loader import load_config, load_training_data, validate_data, get_provider_labels
from utils.cv_splitter import create_provider_stratified_folds, split_data_by_providers
from utils.evaluation import evaluate_fold, evaluate_cv_folds

if not logging.getLogger().hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        force=True
    )

logger = logging.getLogger(__name__)


def test_utilities():
    """Test all utility modules."""
    logger.info("Testing utility modules...")
    
    config = load_config()
    df = load_training_data(config, use_sample=True)
    validate_data(df, config)
    provider_labels = get_provider_labels(df, config)
    
    n_folds = config['models']['xgboost']['cv_folds']
    folds = create_provider_stratified_folds(
        provider_labels, 
        n_folds=n_folds,
        random_state=config['models']['xgboost']['random_state']
    )
    
    target_col = config['validation']['target_column']
    stratify_col = config['validation']['stratify_by']
    
    for fold_idx, (train_providers, test_providers) in enumerate(folds):
        train_df, test_df = split_data_by_providers(
            df,
            train_providers,
            test_providers,
            stratify_col=stratify_col
        )
        
        train_records = len(train_df)
        test_records = len(test_df)
        train_frauds = train_df[target_col].sum().compute()
        test_frauds = test_df[target_col].sum().compute()
        
        logger.info(
            f"Fold {fold_idx + 1}: "
            f"Train={train_records:,} (fraud={train_frauds:,}), "
            f"Test={test_records:,} (fraud={test_frauds:,})"
        )
    
    logger.info("Utility module testing completed!")


if __name__ == "__main__":
    try:
        test_utilities()
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        sys.exit(1)
