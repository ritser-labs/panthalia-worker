# spl/db/server/db_server_instance.py
import logging
from .adapter import DBAdapterServer

db_adapter_server = DBAdapterServer()
logger = logging.getLogger(__name__)
