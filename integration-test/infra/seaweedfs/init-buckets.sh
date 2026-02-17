#!/bin/bash
# Create S3 buckets in SeaweedFS after it's ready
set -e

ENDPOINT="http://localhost:8333"
MAX_RETRIES=30
RETRY_DELAY=2

echo "Waiting for SeaweedFS S3 API..."
for i in $(seq 1 $MAX_RETRIES); do
    if curl -s "$ENDPOINT" > /dev/null 2>&1; then
        echo "SeaweedFS S3 API is ready"
        break
    fi
    echo "  Retry $i/$MAX_RETRIES..."
    sleep $RETRY_DELAY
done

echo "Creating S3 buckets..."
aws --endpoint-url "$ENDPOINT" s3 mb s3://mbt-mlflow-artifacts 2>/dev/null || echo "Bucket mbt-mlflow-artifacts already exists"
aws --endpoint-url "$ENDPOINT" s3 mb s3://mbt-pipeline-artifacts 2>/dev/null || echo "Bucket mbt-pipeline-artifacts already exists"

echo "Verifying buckets..."
aws --endpoint-url "$ENDPOINT" s3 ls

echo "S3 buckets ready!"
