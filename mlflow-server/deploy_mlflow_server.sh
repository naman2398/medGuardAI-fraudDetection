#!/bin/bash
set -e

PROJECT_ID="involuted-fold-474521-h3"
REGION="us-central1"
IMAGE="us-central1-docker.pkg.dev/${PROJECT_ID}/medguardai-ml-training/mlflow-server:latest"

echo "Building and pushing MLflow server image..."
gcloud builds submit --tag ${IMAGE}

echo "Deploying to Cloud Run..."
gcloud run deploy mlflow-server \
    --image ${IMAGE} \
    --region ${REGION} \
    --memory 2Gi \
    --cpu 1 \
    --max-instances 2 \
    --set-env-vars "MLFLOW_BACKEND_URI=postgresql://mlflow:xg3LK2PkeKa4Brso@/mlflow?host=/cloudsql/involuted-fold-474521-h3:us-central1:mlflow-db" \
    --set-env-vars "MLFLOW_ARTIFACT_ROOT=gs://medguard_rawdata/models/mlflow_artifacts" \
    --add-cloudsql-instances "involuted-fold-474521-h3:us-central1:mlflow-db" \
    --allow-unauthenticated

echo "Done. MLflow server deployed."
