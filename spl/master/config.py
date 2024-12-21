# spl/master/config.py
import argparse
import logging
import sys

# Default logger setup here (adjust as needed)
logging.basicConfig(
    level=logging.DEBUG, 
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Master process for task submission"
    )
    parser.add_argument(
        "--private_key",
        type=str,
        required=True,
        help="The wallet private key",
    )
    parser.add_argument(
        "--db_url",
        type=str,
        required=True,
        help="URL for the database",
    )
    parser.add_argument(
        "--detailed_logs",
        action="store_true",
        help="Enable detailed logs",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        help="Number of workers to start for each job",
    )
    parser.add_argument(
        "--num_master_wallets",
        type=int,
        help="Number of wallets to generate for the master",
        default=70,
    )
    parser.add_argument(
        "--deploy_type",
        type=str,
        required=True,
        help="Type of deployment (disabled, local, cloud)",
    )
    parser.add_argument(
        "--torch_compile",
        action="store_true",
        help="Enable torch.compile and model warmup",
    )
    return parser.parse_args()

args = parse_args()

# Adjust logging level if detailed logs are requested
if args.detailed_logs:
    logging.getLogger().setLevel(logging.DEBUG)
