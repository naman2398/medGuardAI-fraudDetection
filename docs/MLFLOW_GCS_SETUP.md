# MLflow Setup (GCS Backend)

## Setup

1. Set credentials:
```bash
# Windows (PowerShell)
$env:GOOGLE_APPLICATION_CREDENTIALS="C:\path\to\your\key.json"

# Linux/Mac
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json
```

2. Run setup:
```bash
python scripts/setup_mlflow_gcs.py
```

3. View UI:
```bash
python scripts/view_mlflow_ui.py
```

Open: http://localhost:5000

## How It Works

- Training jobs on Vertex AI log to `gs://bucket/mlflow_logs/`
- Local UI reads from same GCS path
- View experiments in real-time

## Troubleshooting

**Permission error:** Verify service account has `Storage Object Admin` role

**Bucket not found:** Check bucket exists with `gsutil ls`

**No experiments visible:** Refresh browser, check training job logs
