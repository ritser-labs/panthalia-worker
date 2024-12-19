# spl/adapters/sot_adapter.py
from abc import ABC, abstractmethod

class BaseSOTAdapter(ABC):
    @abstractmethod
    def get_state_dir(self):
        pass

    @abstractmethod
    async def initialize_directories(self):
        pass

    @abstractmethod
    async def initialize_all_tensors(self):
        pass

    @abstractmethod
    async def get_batch(self):
        pass

    @abstractmethod
    async def update_state(self, tensor_name, result_url, version_number, input_url, learning_params):
        pass

    @abstractmethod
    async def update_loss(self, loss_value, version_number):
        pass

    @abstractmethod
    async def get_loss(self):
        pass

    @abstractmethod
    async def upload_tensor(self, tensor_state, label):
        pass

    @abstractmethod
    async def get_data_file(self, filename):
        """
        Retrieve a file's entire contents at once.
        Return a dict like: {'data': bytes, 'mime_type': 'application/octet-stream'}
        """
        pass

    @abstractmethod
    async def stream_data_file(self, filename):
        """
        Return an async generator (or async iterable) that yields chunks of the file.
        This allows streaming large files without loading them entirely into memory.
        """
        pass

    @abstractmethod
    async def get_latest_state(self, tensor_name):
        pass
