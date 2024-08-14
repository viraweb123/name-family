#!/bin/bash

REPO_DIR="name-family"

check_if_in_repo() {
  [ "$(basename "$(pwd)")" == "$REPO_DIR" ]
}

check_repo_exists() {
  [ -d "$REPO_DIR" ]
}

navigate_to_repo() {
  cd "$REPO_DIR" || { echo "Failed to change directory to '$REPO_DIR'."; exit 1; }
}

if check_if_in_repo; then
  echo "Already in the repository directory. Pulling latest changes..."
else
  if check_repo_exists; then
    echo "Repository directory '$REPO_DIR' exists. Changing to the directory..."
    navigate_to_repo
  else
    echo "Cloning the repository..."
    git clone https://github.com/viraweb123/name-family.git

    if [ $? -ne 0 ]; then
      echo "Failed to clone repository."
      exit 1
    fi

    navigate_to_repo
  fi

  git pull
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
