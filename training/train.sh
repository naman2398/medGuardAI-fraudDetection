#!/bin/bash
# MedGuardAI - Vertex AI Training Job Submission
set -e

# Training Parameters
export N_ROWS_SAMPLE=${N_ROWS_SAMPLE:--1}
export XGB_TRIALS=${XGB_TRIALS:-20}
export LGBM_TRIALS=${LGBM_TRIALS:-20}
export CV_FOLDS=${CV_FOLDS:-5}

# Compute Configuration
export PRIMARY_MACHINE=${PRIMARY_MACHINE:-e2-standard-4}
export WORKER_MACHINE=${WORKER_MACHINE:-n2-standard-8}
export WORKER_COUNT=${WORKER_COUNT:-2}
export USE_PREEMPTIBLE=${USE_PREEMPTIBLE:-true}

# GCP Configuration
export PROJECT_ID="involuted-fold-474521-h3"
export REGION="${REGION:-us-east1}"
export BUCKET_NAME="medguard_rawdata"
export ARTIFACT_REGISTRY="us-central1-docker.pkg.dev/involuted-fold-474521-h3/medguardai-ml-training"

TIMESTAMP=$(date +%Y%m%d-%H%M%S)
JOB_NAME="fraud-training-${TIMESTAMP}"
IMAGE_TAG="${TIMESTAMP}"
IMAGE_URI="${ARTIFACT_REGISTRY}/fraud-detection:${IMAGE_TAG}"
IMAGE_URI_LATEST="${ARTIFACT_REGISTRY}/fraud-detection:latest"

echo "Job: ${JOB_NAME} | Data: ${N_ROWS_SAMPLE} rows | XGB=${XGB_TRIALS}, LGBM=${LGBM_TRIALS}, Folds=${CV_FOLDS}"

echo "Building Docker image..."
docker build --platform linux/amd64 -t ${IMAGE_URI} -t ${IMAGE_URI_LATEST} -f training/Dockerfile .

echo "Pushing to Artifact Registry..."
docker push ${IMAGE_URI}
docker push ${IMAGE_URI_LATEST}

echo "Submitting Vertex AI job..."

WORKER_POOL_0="machine-type=${PRIMARY_MACHINE},replica-count=1,container-image-uri=${IMAGE_URI_LATEST}"

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

echo "Job submitted: ${JOB_NAME}"
echo "Monitor: https://console.cloud.google.com/vertex-ai/training/custom-jobs?project=${PROJECT_ID}"
echo "Artifacts: gs://${BUCKET_NAME}/models/artifacts/${JOB_NAME}/"
