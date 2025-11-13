"""
Test script for training pipeline.
Runs a quick training test with sample data.
"""

import logging
import sys
sys.path.append('src')

from models.train_opensource import DistributedTrainer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_training_pipeline():
    """Test the complete training pipeline with sample data."""
    
    logger.info("="*60)
    logger.info("Testing Training Pipeline")
    logger.info("="*60)
    
    try:
        # Create trainer
        trainer = DistributedTrainer(config_path="config/pipeline_config.yaml")
        
        # Run full pipeline
        trainer.run_training()
        
        logger.info("\n" + "="*60)
        logger.info("Training pipeline test completed successfully!")
        logger.info("="*60)
        logger.info("\nCheck the following:")
        logger.info("1. MLflow UI: mlflow ui --backend-store-uri ./mlruns")
        logger.info("2. Model artifacts: models/artifacts/")
        logger.info("3. Logs above for AUCPR scores")
        
    except Exception as e:
        logger.error(f"Training pipeline test failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    test_training_pipeline()
