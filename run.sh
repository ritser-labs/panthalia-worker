#!/usr/bin/env bash

# Exit on any non-zero status
set -e

# Detect which shell we’re using and source the corresponding rc file
shell_name="$(basename -- "$SHELL")"
case "$shell_name" in
  "zsh")
    if [[ -f "${HOME}/.zshrc" ]]; then
      echo "Sourcing ~/.zshrc..."
      # shellcheck disable=SC1090
      source "${HOME}/.zshrc"
    else
      echo "Warning: ~/.zshrc not found."
    fi
    ;;
  "bash")
    if [[ -f "${HOME}/.bashrc" ]]; then
      echo "Sourcing ~/.bashrc..."
      # shellcheck disable=SC1090
      source "${HOME}/.bashrc"
    else
      echo "Warning: ~/.bashrc not found."
    fi
    ;;
  *)
    # Fallback to ~/.bashrc in case the shell is something else but still wants a typical config
    if [[ -f "${HOME}/.bashrc" ]]; then
      echo "Sourcing ~/.bashrc (fallback)..."
      # shellcheck disable=SC1090
      source "${HOME}/.bashrc"
    else
      echo "Warning: ~/.bashrc not found."
    fi
    ;;
esac

# Save the current working directory so we can return to it later
ORIGINAL_DIR="$(pwd)"

# Determine the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Change into the 'spl' directory relative to the script directory and execute build_image.sh
#cd "$SCRIPT_DIR/spl"
#echo "Changed directory to $(pwd). Running build_image.sh..."
#if [[ -x "./build_image.sh" ]]; then
#  ./build_image.sh
#  echo "build_image.sh completed successfully."
#else
#  echo "Error: build_image.sh is either not found or not executable."
#  exit 1
#fi

# Return to the original directory
#cd "$ORIGINAL_DIR"
#echo "Returned to original directory: $(pwd)"

# Define the scripts to be executed
scripts=(
  "./destroy-containers.sh"
  "./kill-processes.sh"
  "./desktop.sh"
)

# Run each script if executable
for script in "${scripts[@]}"; do
  if [[ -x "$script" ]]; then
    echo "Executing $script..."
    "$script"
    echo "$script completed successfully."
  else
    echo "Error: $script is either not found or not executable."
    exit 1
  fi
done

echo "All scripts executed successfully."

