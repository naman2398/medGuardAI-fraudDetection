#!/bin/bash
# MedGuardAI - Vertex AI Training Job Submission
#
# USAGE:
#   Production (default):  ./training/submit_job.sh
#   Quick Test:            N_ROWS_SAMPLE=50000 XGB_TRIALS=5 LGBM_TRIALS=5 CV_FOLDS=3 WORKER_COUNT=1 WORKER_MACHINE=e2-standard-4 ./training/submit_job.sh
#   Medium Test:           N_ROWS_SAMPLE=200000 XGB_TRIALS=10 LGBM_TRIALS=10 WORKER_COUNT=1 ./training/submit_job.sh

set -e

# Training Parameters (defaults match pipeline_config.yaml production values)
export N_ROWS_SAMPLE=${N_ROWS_SAMPLE:--1}
export XGB_TRIALS=${XGB_TRIALS:-20}
export LGBM_TRIALS=${LGBM_TRIALS:-20}
export CV_FOLDS=${CV_FOLDS:-5}

# Compute Configuration (defaults match pipeline_config.yaml production values)
export PRIMARY_MACHINE=${PRIMARY_MACHINE:-e2-standard-4}
export WORKER_MACHINE=${WORKER_MACHINE:-n2-standard-8}
export WORKER_COUNT=${WORKER_COUNT:-2}
export USE_PREEMPTIBLE=${USE_PREEMPTIBLE:-true}

# GCP Configuration
export PROJECT_ID="involuted-fold-474521-h3"
export REGION="${REGION:-us-east1}"  # Default to us-east1, override with REGION env var
export BUCKET_NAME="medguard_rawdata"
export ARTIFACT_REGISTRY="us-central1-docker.pkg.dev/involuted-fold-474521-h3/medguardai-ml-training"

# Generate job identifiers
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
JOB_NAME="fraud-training-${TIMESTAMP}"

# Use timestamped tag for each submission + latest
IMAGE_TAG="${TIMESTAMP}"
IMAGE_URI="${ARTIFACT_REGISTRY}/fraud-detection:${IMAGE_TAG}"
IMAGE_URI_LATEST="${ARTIFACT_REGISTRY}/fraud-detection:latest"

echo "=========================================="
echo "Job:      ${JOB_NAME}"
echo "Data:     ${N_ROWS_SAMPLE} rows (-1 = all)"
echo "Trials:   XGB=${XGB_TRIALS}, LGBM=${LGBM_TRIALS}"
echo "Folds:    ${CV_FOLDS}"
echo "Cluster:  ${PRIMARY_MACHINE} + ${WORKER_COUNT}x ${WORKER_MACHINE}"
echo "=========================================="

echo "Building Docker image (Docker will use cache if nothing changed)..."
echo "Image tags: ${IMAGE_TAG}, latest"
docker build --platform linux/amd64 -t ${IMAGE_URI} -t ${IMAGE_URI_LATEST} -f training/Dockerfile .

echo "Pushing to Artifact Registry..."
docker push ${IMAGE_URI}
docker push ${IMAGE_URI_LATEST}

echo "Submitting Vertex AI job..."

# Always use latest tag so jobs get the most recent image
WORKER_POOL_0="machine-type=${PRIMARY_MACHINE},replica-count=1,container-image-uri=${IMAGE_URI_LATEST}"

# Only add worker pool if WORKER_COUNT > 0
if [ "${WORKER_COUNT}" -gt 0 ]; then
  WORKER_POOL_1="machine-type=${WORKER_MACHINE},replica-count=${WORKER_COUNT},container-image-uri=${IMAGE_URI_LATEST}"
  
  gcloud ai custom-jobs create \
    --region=${REGION} \
    --display-name=${JOB_NAME} \
    --worker-pool-spec="${WORKER_POOL_0}" \
    --worker-pool-spec="${WORKER_POOL_1}" \
    --args="--n_rows_sample=${N_ROWS_SAMPLE}" \
    --args="--xgb_trials=${XGB_TRIALS}" \
    --args="--lgbm_trials=${LGBM_TRIALS}" \
    --args="--cv_folds=${CV_FOLDS}" \
    --args="--worker_count=${WORKER_COUNT}" \
    --project=${PROJECT_ID}
else
  # Single machine mode - no worker pool
  gcloud ai custom-jobs create \
    --region=${REGION} \
    --display-name=${JOB_NAME} \
    --worker-pool-spec="${WORKER_POOL_0}" \
    --args="--n_rows_sample=${N_ROWS_SAMPLE}" \
    --args="--xgb_trials=${XGB_TRIALS}" \
    --args="--lgbm_trials=${LGBM_TRIALS}" \
    --args="--cv_folds=${CV_FOLDS}" \
    --args="--worker_count=${WORKER_COUNT}" \
    --project=${PROJECT_ID}
fi

echo ""
echo "✓ Job submitted: ${JOB_NAME}"
echo ""
echo "Monitor job (infrastructure/logs):"
echo "  https://console.cloud.google.com/vertex-ai/training/custom-jobs?project=${PROJECT_ID}"
echo ""
echo "View experiments (ML metrics):"
echo "  python scripts/view_mlflow_ui.py"
echo ""
echo "Model artifacts:"
echo "  gs://${BUCKET_NAME}/models/artifacts/${JOB_NAME}/"
echo "=========================================="
