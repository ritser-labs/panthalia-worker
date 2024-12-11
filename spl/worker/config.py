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
    "private_key": None  # CHANGED: Ensure private_key is represented in config
}

def load_config():
    config_dir = Path.home() / ".panthalia_worker"
    config_dir.mkdir(exist_ok=True)
    config_file = config_dir / "config.json"

    if config_file.exists():
        with config_file.open("r") as f:
            return json.load(f)
    else:
        with config_file.open("w") as f:
            json.dump(default_config, f)
        return default_config

def save_config(new_config):
    config_dir = Path.home() / ".panthalia_worker"
    config_dir.mkdir(exist_ok=True)
    config_file = config_dir / "config.json"
    with config_file.open("w") as f:
        json.dump(new_config, f)

config_data = load_config()

def parse_args():
    parser = argparse.ArgumentParser(description="Panthalia Worker for processing tasks")
    # CHANGED: Add default for private_key from config_data
    parser.add_argument('--private_key', type=str, required=False, default=config_data.get("private_key"),
                        help="Private key of the worker")
    parser.add_argument('--subnet_id', type=int, required=False, help="Subnet ID (CLI mode only)", default=config_data.get("subnet_id"))
    parser.add_argument('--detailed_logs', action='store_true', help="Enable detailed logging for loss task")
    parser.add_argument('--max_stakes', type=int, default=2, help="Max number of stakes to maintain")
    parser.add_argument('--poll_interval', type=int, default=1, help="Interval (in seconds) for polling")
    parser.add_argument('--torch_compile', action='store_true', help="Enable torch.compile and model warmup")
    parser.add_argument('--max_tasks_handling', type=int, default=1, help="Max number of tasks allowed in queue")
    parser.add_argument('--db_url', type=str, default=config_data.get("db_url"), help="DB Server URL")
    parser.add_argument('--cli', action='store_true', help="Enable CLI mode instead of GUI")
    parser.add_argument('--docker_engine_url', type=str, default=config_data.get("docker_engine_url"), help="Docker engine URL")

    args = parser.parse_args()

    # If not CLI, GUI mode:
    args.gui = not args.cli

    # CHANGED: Do not overwrite private_key in GUI mode. Just rely on config if not provided.
    # Previously we did args.private_key = None in GUI mode, now we allow config_data.

    if args.cli:
        # If running in CLI mode and no private_key provided (or in config), error out.
        if not args.private_key:
            parser.error("--private_key is required in CLI mode if not set in config.")

    fields_to_check = {
        "private_key": args.private_key,
        "db_url": args.db_url,
        "docker_engine_url": args.docker_engine_url,
        "subnet_id": args.subnet_id
    }

    # Update config if changed
    new_config = dict(config_data)
    changed = False
    for field, new_value in fields_to_check.items():
        if new_config.get(field) != new_value:
            new_config[field] = new_value
            changed = True

    if changed:
        save_config(new_config)

    return args

args = parse_args()
