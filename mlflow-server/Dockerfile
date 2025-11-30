FROM python:3.10-slim

RUN pip install --no-cache-dir mlflow psycopg2-binary google-cloud-storage

EXPOSE 8080

CMD ["sh", "-c", "mlflow server --host 0.0.0.0 --port 8080 --backend-store-uri $MLFLOW_BACKEND_URI --default-artifact-root $MLFLOW_ARTIFACT_ROOT"]
