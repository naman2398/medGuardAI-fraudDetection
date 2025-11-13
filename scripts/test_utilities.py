"""
Test script for utility modules.
Demonstrates data loading, CV splitting, and evaluation.
"""

import logging
import sys
sys.path.append('src')

from utils.data_loader import load_config, load_training_data, validate_data, get_provider_labels
from utils.cv_splitter import create_provider_stratified_folds, split_data_by_providers
from utils.evaluation import evaluate_fold, evaluate_cv_folds

# Setup logging - only if not already configured
if not logging.getLogger().hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        force=True
    )

logger = logging.getLogger(__name__)


def test_utilities():
    """Test all utility modules with sample data."""
    
    logger.info("="*60)
    logger.info("Testing Utility Modules")
    logger.info("="*60)
    
    # Step 1: Load configuration
    logger.info("\nStep 1: Loading configuration...")
    config = load_config()
    logger.info(f"Config loaded: {config['data']['training_data_path']}")
    
    # Step 2: Load training data
    logger.info("\nStep 2: Loading training data...")
    df = load_training_data(config, use_sample=True)
    
    # Step 3: Validate data
    logger.info("\nStep 3: Validating data...")
    validate_data(df, config)
    
    # Step 4: Get provider labels for stratification
    logger.info("\nStep 4: Computing provider-level labels...")
    provider_labels = get_provider_labels(df, config)
    
    # Step 5: Create stratified folds
    logger.info("\nStep 5: Creating stratified CV folds...")
    n_folds = config['models']['xgboost']['cv_folds']
    folds = create_provider_stratified_folds(
        provider_labels, 
        n_folds=n_folds,
        random_state=config['models']['xgboost']['random_state']
    )
    
    # Step 6: Verify actual data splits for each fold
    logger.info("\nStep 6: Verifying data record counts per fold...")
    target_col = config['validation']['target_column']
    stratify_col = config['validation']['stratify_by']
    
    for fold_idx, (train_providers, test_providers) in enumerate(folds):
        train_df, test_df = split_data_by_providers(
            df,
            train_providers,
            test_providers,
            stratify_col=stratify_col
        )
        
        # Count records and frauds
        train_records = len(train_df)
        test_records = len(test_df)
        train_frauds = train_df[target_col].sum().compute()
        test_frauds = test_df[target_col].sum().compute()
        
        logger.info(
            f"Fold {fold_idx + 1} Data Split: "
            f"Train records={train_records:,} (frauds={train_frauds:,}), "
            f"Test records={test_records:,} (frauds={test_frauds:,})"
        )
    
    logger.info("\n" + "="*60)
    logger.info("Utility module testing completed successfully!")
    logger.info("="*60)


if __name__ == "__main__":
    try:
        test_utilities()
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        sys.exit(1)
