# spl/adapters/sot_adapter.py

import asyncio
import logging
from abc import ABC, abstractmethod

################################################################
# Base interface for any SOT Adapter
################################################################

class BaseSOTAdapter(ABC):
    """
    The abstract interface for any SOT adapter implementation.
    The plugin code should provide a class that implements these methods.
    """

    @abstractmethod
    def set_plugin_and_db_adapter(self, plugin, db_adapter):
        """
        Called once after we create the adapter, so it can store references
        to the plugin object and the DB adapter object if needed.
        """
        pass

    @abstractmethod
    async def initialize(self, sot_id, db_url, private_key, job_id, perm_db):
        """
        Called once by the SOT side, passing the relevant configuration (SOT ID,
        DB URL, private key, job ID, permission info, etc.). This is your chance
        to do filesystem or data structure initialization.
        """
        pass

    @abstractmethod
    async def handle_request(self, method, path, query, headers, body):
        """
        The main entry point for handling an HTTP request from the SOT proxy side.
        Return a dict: {
          'status': <int>,
          'headers': { ... },
          'body': <bytes or an async generator/iterator>
        }
        """
        pass

