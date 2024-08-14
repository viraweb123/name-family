#!/bin/bash

REPO_DIR="name-family"

if [ -d "$REPO_DIR" ]; then
  echo "Repository directory '$REPO_DIR' already exists. Pulling latest changes..."
  cd "$REPO_DIR" || { echo "Failed to change directory to '$REPO_DIR'."; exit 1; }
  git pull
else
  echo "Cloning the repository..."
  git clone https://github.com/viraweb123/name-family.git

  if [ $? -ne 0 ]; then
    echo "Failed to clone repository."
    exit 1
  fi

  cd "$REPO_DIR" || { echo "Failed to change directory to '$REPO_DIR'."; exit 1; }
fi

# echo "Building the Docker image..."
# docker build -t train:v1 .

# if [ $? -ne 0 ]; then
#   echo "Failed to build Docker image."
#   exit 1
# fi

echo "Setting permissions for 'train/' directory..."
chmod -fR 777 train/

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
