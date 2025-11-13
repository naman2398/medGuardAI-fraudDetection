#!/bin/bash
# Submits the MedGuardAI PySpark job to Dataproc Serverless.
set -e # Exit immediately if a command fails

# --- 1. Configuration Variables ---
export PROJECT_ID="involuted-fold-474521-h3"
export REGION="us-east1"
export STAGING_BUCKET="medguard_rawdata"
export SCRIPT_PATH="gs://medguard_rawdata/scripts/data_preprocessing.py"
export BATCH_ID="medguard-bq-create-$(date +%s)"
export SPARK_PROPERTIES="spark.executor.memory=8g,spark.executor.cores=4,spark.dynamicAllocation.enabled=true,spark.dynamicAllocation.minExecutors=2,spark.dynamicAllocation.maxExecutors=10,spark.dataproc.runtime.python.packages=google-cloud-bigquery"

# --- 2. Submit the Job ---
echo "Submitting Dataproc batch job: ${BATCH_ID}"

gcloud dataproc batches submit pyspark ${SCRIPT_PATH} \
    --project=${PROJECT_ID} \
    --region=${REGION} \
    --batch=${BATCH_ID} \
    --staging-bucket=${STAGING_BUCKET} \
    --properties="${SPARK_PROPERTIES}"

echo "Job ${BATCH_ID} submitted successfully."