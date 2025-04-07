# spl/worker/logging_config.py
import logging
import sys

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Console handler
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

# In GUI mode, we will add a handler later that updates the GUI logs.
gui_log_handler = None

def set_gui_handler(handler):
    root = logging.getLogger()  # Get the root logger
    # Optionally, remove any existing GUI handlers from the root logger:
    for h in root.handlers:
        if isinstance(h, type(handler)):
            root.removeHandler(h)
    root.addHandler(handler)
