"""Launch MLflow UI for viewing experiment results."""

import sys
import yaml
import logging
import subprocess
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def main():
    config_path = Path("config/pipeline_config.yaml")
    
    if not config_path.exists():
        logger.error("Config file not found")
        sys.exit(1)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    tracking_uri = config['mlflow']['tracking_uri']
    
    if tracking_uri.startswith('gs://'):
        import os
        if not os.getenv('GOOGLE_APPLICATION_CREDENTIALS'):
            logger.warning("GOOGLE_APPLICATION_CREDENTIALS not set")
    
    logger.info(f"Launching MLflow UI at http://localhost:5000")
    logger.info(f"Tracking URI: {tracking_uri}")
    
    try:
        subprocess.run([
            sys.executable, "-m", "mlflow", "ui",
            "--backend-store-uri", tracking_uri,
            "--port", "5000"
        ], check=True)
    except KeyboardInterrupt:
        logger.info("MLflow UI stopped")
    except Exception as e:
        logger.error(f"Failed to launch UI: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
