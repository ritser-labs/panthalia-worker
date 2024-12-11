# spl/worker/gui_config.py
from .config import load_config, save_config

def get_config():
    return load_config()

def update_config(new_data):
    current = load_config()
    current.update(new_data)
    save_config(current)

# CHANGED: Replace api_key references with private_key
def get_private_key():
    config = get_config()
    return config.get("private_key")

def set_private_key(private_key):
    update_config({"private_key": private_key})

def get_db_url():
    return get_config().get("db_url")

def get_docker_engine_url():
    return get_config().get("docker_engine_url")

def get_subnet_id():
    c = get_config()
    return c.get("subnet_id")

def set_subnet_id(subnet_id):
    update_config({"subnet_id": subnet_id})
