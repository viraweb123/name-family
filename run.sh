#!/bin/bash

echo "Updating the Repository..."
git pull

echo "Setting permissions for 'train/' directory..."
chmod 777 train/

echo "Running the Docker container..."
docker run \
    --gpus all \
    --rm \
    -u $(id -u):$(id -g) \
    -v $(pwd)/train:/train \
    -e HF_HOME=/train/.cache/huggingface \
    -w "/train/code" \
    train:v1 \
    python3 main.py

if [ $? -ne 0 ]; then
  echo "Failed to run Docker container."
  exit 1
fi

echo "Script completed successfully."