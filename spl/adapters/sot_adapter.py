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
    async def initialize(self, sot_id, db_url, private_key, job_id, perm_db, port):
        """
        Called once by the SOT side, passing the relevant configuration (SOT ID,
        DB URL, private key, job ID, permission info, etc.). This is your chance
        to do filesystem or data structure initialization.
        """
        pass
