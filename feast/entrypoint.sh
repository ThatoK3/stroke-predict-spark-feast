#!/bin/bash

set -e

echo "Starting Feast Feature Store..."

while ! curl -f redis:6379 >/dev/null 2>&1; do
    echo "Redis not ready yet..."
    sleep 2
done

while ! curl -f http://minio:9000/minio/health/live >/dev/null 2>&1; do
    echo "MinIO not ready yet..."
    sleep 2
done

if [ ! -f /app/feast/feature_store.yaml ]; then
    cat > /app/feast/feature_store.yaml << EOF
project: stroke_prediction
registry: s3://feast/registry.db
provider: aws
online_store:
  type: redis
  connection_string: redis://redis:6379
offline_store:
  type: file
entity_key_serialization_version: 2
EOF
fi

cd /app/feast
feast apply

feast serve --host 0.0.0.0 --port 6565 &
FEAST_PID=$1

feast registry-server --host 0.0.0.0 --port 6566 &
REGISTRY_PID=$2

cleanup() {
    echo "Shutting down Feast services..."
    kill $FEAST_PID $REGISTRY_PID 2>/dev/null || true
    exit 0
}

trap cleanup SIGTERM SIGINT

wait
