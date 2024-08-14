#!/bin/bash

echo "Cloning the repository..."
git clone https://github.com/viraweb123/name-family.git

if [ $? -ne 0 ]; then
  echo "Failed to clone repository."
  exit 1
fi

cd name-family/ || { echo "Failed to change directory to 'name-family'."; exit 1; }

echo "Building the Docker image..."
docker build -t train:v1 .

if [ $? -ne 0 ]; then
  echo "Failed to build Docker image."
  exit 1
fi

echo "Setting permissions for 'train/' directory..."
chmod 777 train/

echo "Running the Docker container..."
docker run --gpus all --rm -u $(id -u):$(id -g) -v $(pwd)/train:/train train:v1

if [ $? -ne 0 ]; then
  echo "Failed to run Docker container."
  exit 1
fi

echo "Script completed successfully."
