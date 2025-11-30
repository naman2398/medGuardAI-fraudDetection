"""Setup MLflow with GCS backend."""

import os
import sys
import yaml
import logging
from pathlib import Path
from google.cloud import storage

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def check_credentials():
    """Verify GCP credentials are set."""
    creds = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not creds or not Path(creds).exists():
        logger.error("GOOGLE_APPLICATION_CREDENTIALS not set or file not found")
        logger.info("Set with: export GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json")
        sys.exit(1)


def test_gcs_connection(bucket_name: str):
    """Test GCS bucket access."""
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob("mlflow_logs/.test")
        blob.upload_from_string("test")
        blob.delete()
        logger.info(f"GCS connection verified: gs://{bucket_name}")
    except Exception as e:
        logger.error(f"GCS connection failed: {e}")
        sys.exit(1)


def update_config(bucket_name: str):
    """Update config with GCS paths."""
    config_path = Path("config/pipeline_config.yaml")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    config['mlflow']['tracking_uri'] = f"gs://{bucket_name}/mlflow_logs"
    config['data']['training_data_path'] = f"gs://{bucket_name}/data/model_ready/"
    config['data']['gcs_bucket'] = bucket_name
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    logger.info("Config updated successfully")


def test_mlflow():
    """Test MLflow logging to GCS."""
    import mlflow
    
    with open("config/pipeline_config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    mlflow.set_experiment("test-setup")
    
    try:
        with mlflow.start_run(run_name="connection-test"):
            mlflow.log_param("test", "success")
            mlflow.log_metric("value", 1.0)
        logger.info("MLflow logging verified")
    except Exception as e:
        logger.error(f"MLflow logging failed: {e}")
        sys.exit(1)


def main():
    logger.info("Starting MLflow GCS setup")
    
    check_credentials()
    
    bucket_name = input("GCS bucket name: ").strip()
    if not bucket_name:
        logger.error("Bucket name required")
        sys.exit(1)
    
    test_gcs_connection(bucket_name)
    update_config(bucket_name)
    test_mlflow()
    
    logger.info("Setup complete. Run: python scripts/view_mlflow_ui.py")


if __name__ == "__main__":
    main()
