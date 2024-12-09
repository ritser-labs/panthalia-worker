#!/bin/bash

IMAGE_NAME="panthalia_plugin"
DOCKERFILE_PATH="Dockerfile"

# Remove existing image if it exists
docker rmi -f $IMAGE_NAME 2>/dev/null

echo "Building Docker image: $IMAGE_NAME"
docker build -t $IMAGE_NAME -f $DOCKERFILE_PATH .
if [[ $? -ne 0 ]]; then
    echo "Failed to build Docker image."
    exit 1
fi
