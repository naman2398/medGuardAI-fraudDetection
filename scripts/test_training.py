"""Test script for training pipeline."""

import logging
import sys
sys.path.append('src')

from models.train_opensource import DistributedTrainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_training_pipeline():
    """Test the complete training pipeline."""
    try:
        trainer = DistributedTrainer(config_path="config/pipeline_config.yaml")
        trainer.run_training()
        logger.info("Training pipeline test completed successfully!")
    except Exception as e:
        logger.error(f"Training pipeline test failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    test_training_pipeline()
