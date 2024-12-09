import json
import base64
import torch
from safetensors.torch import save as safetensors_save
from safetensors.torch import load as safetensors_load
import logging

def serialize_data(obj):
    """
    Serialize data with type identifiers and return as a JSON-serializable dict.
    """
    try:
        if isinstance(obj, torch.Tensor):
            # Serialize tensor to bytes, encode as base64, and add a type identifier
            buffer = {'tensor': obj}
            serialized_bytes = safetensors_save(buffer)
            return {'type': 'tensor', 'data': base64.b64encode(serialized_bytes).decode('utf-8')}
        elif isinstance(obj, tuple):
            return {'type': 'tuple', 'data': [serialize_data(item) for item in obj]}
        elif isinstance(obj, list):
            return {'type': 'list', 'data': [serialize_data(item) for item in obj]}
        elif isinstance(obj, dict):
            return {'type': 'dict', 'data': {k: serialize_data(v) for k, v in obj.items()}}
        else:
            return {'type': 'value', 'data': obj}
    except (TypeError, OverflowError) as e:
        logging.error(f"Serialization error: {e}")
        raise

def deserialize_data(data):
    try:
        #logging.info(f"Deserializing: {data}")

        if 'type' not in data or 'data' not in data:
            raise ValueError("Serialized data must have 'type' and 'data' keys")
          
        obj_type = data.get('type')
        obj_data = data.get('data')

        if obj_type == 'tensor':
            # Decode base64 tensor data
            decoded = base64.b64decode(obj_data.encode('utf-8'))
            tensors = safetensors_load(decoded)
            return tensors['tensor']
        elif obj_type == 'tuple':
            return tuple(deserialize_data(item) for item in obj_data)
        elif obj_type == 'list':
            return [deserialize_data(item) for item in obj_data]
        elif obj_type == 'dict':
            return {k: deserialize_data(v) for k, v in obj_data.items()}
        else:
            return obj_data  # Treat as a basic value (e.g., int, str, etc.)

    except Exception as e:
        logging.error(f"Deserialization error: {e}")
        raise
