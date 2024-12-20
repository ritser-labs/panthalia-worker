#!/bin/bash

# Description: This script kills all processes containing "python -m spl." in their name.

# Find and kill processes
processes=$(ps aux | grep "python -m spl." | grep -v grep | awk '{print $2}')

if [ -z "$processes" ]; then
  echo "No matching processes found."
  exit 0
fi

# Iterate and kill each process
for pid in $processes; do
  echo "Killing process with PID: $pid"
  kill -9 $pid
  if [ $? -eq 0 ]; then
    echo "Successfully killed process $pid."
  else
    echo "Failed to kill process $pid."
  fi
done

exit 0
