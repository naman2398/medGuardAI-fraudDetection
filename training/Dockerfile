# Dockerfile for MedGuardAI Fraud Detection Training Pipeline
# Optimized for Vertex AI distributed training

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for ML libraries
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY config/ ./config/

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Run training script (ENTRYPOINT allows args to be passed)
ENTRYPOINT ["python", "src/models/train_opensource.py"]
