#!/bin/bash

set -e

curl -L https://foundry.paradigm.xyz | bash


# Manually ensure the PATH includes the foundry directory
export PATH="$PATH:/root/.foundry/bin"

# Run foundryup to finalize installation
foundryup

echo "Done setting up Foundry, starting anvil"

# Start anvil (only if you want to start it immediately)
anvil --host 0.0.0.0