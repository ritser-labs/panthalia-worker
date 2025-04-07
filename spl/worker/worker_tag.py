import os
import uuid
from .config import args  # import the parsed args

# Define a path to store the worker tag persistently.
HOME_DIR = os.path.expanduser("~")
WORKER_DIR = os.path.join(HOME_DIR, ".panthalia_worker")
os.makedirs(WORKER_DIR, exist_ok=True)
TAG_FILENAME = os.path.join(WORKER_DIR, "worker_tag.txt")

def get_worker_tag() -> str:
    """
    Returns the worker tag. If a --worker_tag argument is provided,
    that value is used (without persisting it); otherwise, it checks the
    persistent file. If the file is missing or empty, a new UUID is generated.
    """
    # If the user provided a worker_tag override via command line, use it.
    if args.worker_tag is not None:
        return args.worker_tag

    # Otherwise, use the persistent file system.
    if os.path.exists(TAG_FILENAME):
        with open(TAG_FILENAME, "r") as f:
            tag = f.read().strip()
            if tag:
                return tag
    # Generate a new unique tag and persist it.
    new_tag = str(uuid.uuid4())
    with open(TAG_FILENAME, "w") as f:
        f.write(new_tag)
    return new_tag

def set_worker_tag(new_tag: str) -> None:
    """
    Overwrites the persistent worker tag.
    """
    os.makedirs(WORKER_DIR, exist_ok=True)
    with open(TAG_FILENAME, "w") as f:
        f.write(new_tag)
