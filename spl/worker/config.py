# spl/worker/config.py
import argparse
import os
import json
import sys
from pathlib import Path

default_config = {
    "db_url": "http://localhost:5432",
    "docker_engine_url": "unix:///var/run/docker.sock",
    "subnet_id": 1,
}

def load_config():
    # Changed from .myworkerapp to .panthalia_worker
    config_dir = Path.home() / ".panthalia_worker"
    config_dir.mkdir(exist_ok=True)
    config_file = config_dir / "config.json"

    if config_file.exists():
        with config_file.open("r") as f:
            return json.load(f)
    else:
        # Save defaults
        with config_file.open("w") as f:
            json.dump(default_config, f)
        return default_config

def save_config(new_config):
    # Changed from .myworkerapp to .panthalia_worker
    config_dir = Path.home() / ".panthalia_worker"
    config_dir.mkdir(exist_ok=True)
    config_file = config_dir / "config.json"
    with config_file.open("w") as f:
        json.dump(new_config, f)

config_data = load_config()

def parse_args():
    parser = argparse.ArgumentParser(description="Panthalia Worker for processing tasks")
    # No longer required here. Let’s make it optional.
    parser.add_argument('--subnet_id', type=int, required=False, help="Subnet ID (CLI mode only)")
    parser.add_argument('--private_key', type=str, required=False, help="Private key of the worker (ignored in GUI mode)")
    parser.add_argument('--detailed_logs', action='store_true', help="Enable detailed logging for loss task")
    parser.add_argument('--max_stakes', type=int, default=2, help="Max number of stakes to maintain")
    parser.add_argument('--poll_interval', type=int, default=1, help="Interval (in seconds) for polling")
    parser.add_argument('--torch_compile', action='store_true', help="Enable torch.compile and model warmup")
    parser.add_argument('--max_tasks_handling', type=int, default=1, help="Max number of tasks allowed in queue")
    parser.add_argument('--db_url', type=str, default=config_data.get("db_url"), help="DB Server URL") 
    # Remove --gui. Instead add --cli to switch modes.
    parser.add_argument('--cli', action='store_true', help="Enable CLI mode instead of GUI")

    parser.add_argument('--docker_engine_url', type=str, default=config_data.get("docker_engine_url"), help="Docker engine URL")

    args = parser.parse_args()

    # Default to GUI mode if --cli not provided
    args.gui = not args.cli

    if args.gui:
        # In GUI mode, private_key and subnet_id are taken from the GUI config
        args.private_key = None
        args.subnet_id = None
    else:
        # In CLI mode, if subnet_id is not provided, we can fallback or require it.
        if args.subnet_id is None:
            parser.error("--subnet_id is required in CLI mode.")

    return args

args = parse_args()
