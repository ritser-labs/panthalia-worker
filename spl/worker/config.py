# spl/worker/config.py
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Worker for processing tasks based on smart contract events")
    parser.add_argument('--subnet_id', type=int, required=True, help="Subnet ID")
    parser.add_argument('--private_key', type=str, required=True, help="Private key of the worker")
    parser.add_argument('--sot_url', type=str, required=True, help="Source of Truth URL for streaming gradient updates")
    parser.add_argument('--detailed_logs', action='store_true', help="Enable detailed logging for loss task")
    parser.add_argument('--max_stakes', type=int, default=2, help="Maximum number of stakes to maintain")
    parser.add_argument('--poll_interval', type=int, default=1, help="Interval (in seconds) for polling for new tasks")
    parser.add_argument('--torch_compile', action='store_true', help="Enable torch.compile and model warmup")
    parser.add_argument('--max_tasks_handling', type=int, default=1, help="Maximum number of tasks allowed in the queue")
    parser.add_argument('--db_url', type=str, required=True, help="URL of the database server")
    return parser.parse_args()

args = parse_args()
