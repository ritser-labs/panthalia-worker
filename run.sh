#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Define the scripts to be executed
scripts=("./destroy-containers.sh" "./kill-processes.sh" "./desktop.sh")

# Iterate through each script and execute it
for script in "${scripts[@]}"; do
    if [[ -x "$script" ]]; then
        echo "Executing $script..."
        "$script"
        echo "$script completed successfully."
    else
        echo "Error: $script is not executable or not found."
        exit 1
    fi
done

echo "All scripts executed successfully."
