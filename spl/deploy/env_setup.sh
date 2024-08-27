#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Log in to GitHub using an environment variable for the GitHub token
if [ -z "$GITHUB_TOKEN" ]; then
  echo "Error: GITHUB_TOKEN is not set."
  exit 1
fi

# Set the desired branch
BRANCH_NAME="runpod"  # Replace with your branch name

# Clone the repository from GitHub
REPO_URL="https://github.com/ritser-labs/magnum.git"
CLONE_DIR="/app"

git clone https://$GITHUB_TOKEN@github.com/ritser-labs/magnum.git $CLONE_DIR

# Change to the repository directory
cd $CLONE_DIR

# Check out the desired branch
git checkout $BRANCH_NAME

# Set the working directory to /app/spl
cd spl

apt-get remove -y python3-blinker
# Install the dependencies
pip install --no-cache-dir -r requirements.txt

cd ..

# Command selection based on the SERVICE_TYPE argument
case $SERVICE_TYPE in
    worker)
        python -m spl.worker --task_types ${TASK_TYPES} --subnet_addresses ${SUBNET_ADDRESSES} --private_keys ${PRIVATE_KEYS} --rpc_url ${RPC_URL} --sot_url ${SOT_URL} --pool_address ${POOL_ADDRESS} --group ${GROUP} --backend ${BACKEND}
        ;;
    master)
        python -m spl.master --rpc_url ${RPC_URL} --wallets ${WALLETS} --sot_url ${SOT_URL} --subnet_addresses ${SUBNET_ADDRESSES} --max_concurrent_iterations ${MAX_CONCURRENT_ITERATIONS}
        ;;
    sot)
        gunicorn -w 4 -b "0.0.0.0:${SOT_PRIVATE_PORT}" "spl.sot:create_app('${PUBLIC_KEYS}')"
        ;;
    *)
        echo "Error: Unknown service type"
        exit 1
        ;;
esac
