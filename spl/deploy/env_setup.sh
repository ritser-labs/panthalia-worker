#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Log in to GitHub using an environment variable for the GitHub token
if [ -z "$GITHUB_TOKEN" ]; then
  echo "Error: GITHUB_TOKEN is not set."
  exit 1
fi

# Set the desired branch
BRANCH_NAME="main"  # Replace with your branch name

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
        # Build the command for the worker service
        CMD="python -m spl.worker --task_types ${TASK_TYPES} --subnet_addresses ${SUBNET_ADDRESSES} --private_keys ${PRIVATE_KEYS} --rpc_url ${RPC_URL} --sot_url ${SOT_URL} --pool_address ${POOL_ADDRESS} --group ${GROUP} --backend ${BACKEND} --db_url ${DB_URL}"

        # Append --torch_compile if TORCH_COMPILE is set to true
        if [ "$TORCH_COMPILE" = "true" ]; then
            CMD="$CMD --torch_compile"
        fi

        # Execute the command
        eval $CMD
        ;;
    master)
        python -m spl.master --rpc_url ${RPC_URL} --wallets ${WALLETS} --sot_url ${SOT_URL} --subnet_addresses ${SUBNET_ADDRESSES} --max_concurrent_iterations ${MAX_CONCURRENT_ITERATIONS} --job_id ${JOB_ID} --db_url ${DB_URL}
        ;;
    sot)
        #hypercorn "spl.sot:create_app('${PUBLIC_KEYS}')" --bind "0.0.0.0:${SOT_PRIVATE_PORT}"
        python -m spl.sot --sot_id ${SOT_ID} --db_url ${DB_URL} --private_key ${PRIVATE_KEY}
        ;;
    db)
        python -m spl.db.db_server --host ${DB_HOST} --port ${DB_PORT} --perm ${DB_PERM} --root_wallet ${ROOT_WALLET}
        ;;
    *)
        echo "Error: Unknown service type"
        exit 1
        ;;
esac
