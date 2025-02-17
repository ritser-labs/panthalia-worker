#!/bin/bash

IMAGE_NAME="panthalia_plugin"
DOCKERFILE_PATH="Dockerfile"

# Remove existing image if it exists
docker rmi -f "$IMAGE_NAME" 2>/dev/null

# Enable BuildKit (ensure you're using Docker 18.09+)
export DOCKER_BUILDKIT=1

# Build the image with memory limits
docker build --memory=5g --memory-swap=5g -t "$IMAGE_NAME" -f "$DOCKERFILE_PATH" . --progress=plain

if [[ $? -ne 0 ]]; then
    echo "Failed to build Docker image."
    exit 1
fi
