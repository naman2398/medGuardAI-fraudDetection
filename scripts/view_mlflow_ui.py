"""
Launch MLflow UI.
Supports both local (./mlruns) and GCS (gs://...) tracking URIs.

Usage: python scripts/view_mlflow_ui.py

Note: For GCS URIs, ensure GOOGLE_APPLICATION_CREDENTIALS is set:
  export GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json
"""

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
    
    # Check if GCS URI and warn about credentials
    if tracking_uri.startswith('gs://'):
        logger.info("Using GCS backend for MLflow")
        logger.info("Ensure GOOGLE_APPLICATION_CREDENTIALS is set")
        import os
        if not os.getenv('GOOGLE_APPLICATION_CREDENTIALS'):
            logger.warning("GOOGLE_APPLICATION_CREDENTIALS not set - authentication may fail")
    
    logger.info(f"Launching MLflow UI at http://localhost:5000")
    logger.info(f"Tracking URI: {tracking_uri}")
    logger.info("")
    logger.info("This will show experiments from:")
    logger.info("  - Local training runs (if any)")
    logger.info("  - Vertex AI training jobs")
    logger.info("")
    logger.info("Press Ctrl+C to stop the UI")
    
    try:
        subprocess.run([
            sys.executable, "-m", "mlflow", "ui",
            "--backend-store-uri", tracking_uri,
            "--port", "5000"
        ], check=True)
    except KeyboardInterrupt:
        logger.info("\nMLflow UI stopped")
    except Exception as e:
        logger.error(f"Failed to launch UI: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
