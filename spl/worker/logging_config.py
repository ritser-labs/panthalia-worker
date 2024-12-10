# spl/worker/logging_config.py
import logging
import sys

class SuppressTracebackFilter(logging.Filter):
    def filter(self, record):
        if 'ConnectionRefusedError' in record.getMessage() or 'MaxRetryError' in record.getMessage():
            return False
        return True

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

logger.addFilter(SuppressTracebackFilter())

# Console handler
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

# In GUI mode, we will add a handler later that updates the GUI logs.
gui_log_handler = None

def set_gui_handler(handler):
    global gui_log_handler
    if gui_log_handler:
        logger.removeHandler(gui_log_handler)
    gui_log_handler = handler
    if gui_log_handler:
        logger.addHandler(gui_log_handler)
