#!/bin/bash

set -e  # Exit immediately if any command fails

# Check for optional flag to skip base image build
if [ "$1" != "--skip-base" ]; then
    echo "Building base image..."
    docker build -f Dockerfile.base -t rl-base:py312-cu118-v1 .
else
    echo "Skipping base image build"
fi

echo "Building hyperparameter image..."
docker build -f Dockerfile.hyperparameters -t hyperparameter-image .

echo "Creating shared volume..."
docker volume create mydata

echo "Running hyperparameter container..."
docker run \
    -v "$(pwd)/configs:/app/data/configs" \
    -v mydata:/app/data \
    -p 8080:8080 \
    --env-file .env \
    --gpus all \
    --rm -it hyperparameter-image

echo "Building reward tests image..."
docker build -f Dockerfile.rewardTests -t rewardtests-image .

echo "Hyperparameter container finished. Starting reward tests..."
docker run \
    -v mydata:/app/data \
    -p 8090:8090 \
    --env-file .env \
    --gpus all \
    --rm -it rewardtests-image

echo "All containers completed successfully."