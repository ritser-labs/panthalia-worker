# spl/sot/utils.py
import logging
import psutil

def log_memory_usage(note='', enabled=False):
    if enabled:
        process = psutil.Process()
        mem_info = process.memory_info()
        logging.debug(
            f"Memory usage ({note}): RSS={mem_info.rss / 1024**2:.2f} MB, VMS={mem_info.vms / 1024**2:.2f} MB"
        )
