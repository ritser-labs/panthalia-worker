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
    "private_key": None,
    "limit_price": None
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
    
    # Other argument definitions...
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
    parser.add_argument('--worker_tag', type=str, default=None,
                    help="Override the persistent worker tag value")
    
    # --- FIX for limit_price ---
    # If a limit_price is saved in the config, it is stored as an integer (scaled by DOLLAR_AMOUNT).
    # For UI/CLI purposes we need the plain dollar amount (float), so divide by DOLLAR_AMOUNT.
    from ..models.schema import DOLLAR_AMOUNT
    default_limit_price = config_data.get("limit_price")
    if default_limit_price is not None:
        default_limit_price = float(default_limit_price) / DOLLAR_AMOUNT
    
    parser.add_argument('--limit_price', type=float, default=default_limit_price,
                        help="User-configured limit price for orders (in dollars)")
    # --------------------------------

    # New argument for cloud mode (default is off)
    parser.add_argument('--cloud', action='store_true', help="Enable cloud mode for plugin execution (default off)")

    args = parser.parse_args()

    # Determine GUI vs CLI mode
    args.gui = not args.cli

    if args.cli:
        if not args.private_key:
            parser.error("--private_key is required in CLI mode if not set in config.")
        if args.limit_price is None:
            parser.error("--limit_price is required in CLI mode if not set in config.")

    fields_to_check = {
        "private_key": args.private_key,
        "db_url": args.db_url,
        "docker_engine_url": args.docker_engine_url,
        "subnet_id": args.subnet_id,
        "limit_price": args.limit_price
    }

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
