#!/bin/bash

# List all containers (running and stopped)
all_containers=$(docker ps -aq)

# Check if there are any containers to destroy
if [[ -z "$all_containers" ]]; then
    echo "No Docker containers found. Nothing to destroy."
    exit 0
fi

# Stop all running containers
if docker ps -q | grep -q .; then
    echo "Stopping all running containers..."
    docker stop $(docker ps -q)
fi

# Remove all containers
echo "Removing all containers..."
docker rm $all_containers

# Confirmation of completion
echo "All Docker containers have been successfully destroyed."

