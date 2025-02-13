# spl/plugins/serialize.py
import json
import base64
import torch
from safetensors.torch import save as safetensors_save, load as safetensors_load
import logging

def serialize_data(obj):
    try:
        if isinstance(obj, torch.Tensor):
            # Serialize the tensor directly to bytes using safetensors.torch.save
            raw_bytes = safetensors_save({"tensor": obj})
            return {
                'type': 'tensor',
                'data': base64.b64encode(raw_bytes).decode('utf-8')
            }
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
        if 'type' not in data or 'data' not in data:
            raise ValueError("Serialized data must have 'type' and 'data' keys")
          
        obj_type = data.get('type')
        obj_data = data.get('data')

        if obj_type == 'tensor':
            encoded_str = obj_data
            raw_bytes = base64.b64decode(encoded_str.encode('utf-8'))
            # Load the tensor directly from the raw bytes
            loaded = safetensors_load(raw_bytes)  # returns dict[str, Tensor]
            return loaded['tensor']
        elif obj_type == 'tuple':
            return tuple(deserialize_data(item) for item in obj_data)
        elif obj_type == 'list':
            return [deserialize_data(item) for item in obj_data]
        elif obj_type == 'dict':
            return {k: deserialize_data(v) for k, v in obj_data.items()}
        else:
            return obj_data
    except Exception as e:
        logging.error(f"Deserialization error: {e}")
        raise
