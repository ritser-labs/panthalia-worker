# spl/worker/db_client.py
from ..db.db_adapter_client import DBAdapterClient
from .config import args

db_adapter = DBAdapterClient(args.db_url, args.private_key)
