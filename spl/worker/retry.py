# spl/worker/retry.py

import asyncio
import functools
import logging

def async_retry(
    max_attempts=3,
    initial_delay=1,
    backoff=2,
    exceptions=(Exception,)
):
    """
    A decorator that retries an async function if it raises one of the specified exceptions.

    Parameters:
      max_attempts (int): Maximum number of attempts before giving up.
      initial_delay (float): Delay (in seconds) before the first retry.
      backoff (float): Multiplier to increase the delay between subsequent attempts.
      exceptions (tuple): A tuple of exception types that should trigger a retry.

    Usage example:
      @async_retry(max_attempts=5, initial_delay=2, backoff=2)
      async def my_operation(...):
          ...
    """

    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            attempt = 0
            delay = initial_delay
            while True:
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    attempt += 1
                    if attempt >= max_attempts:
                        logging.error(
                            f"{func.__name__} failed after {attempt} attempts: {e}",
                            exc_info=True
                        )
                        raise
                    logging.warning(
                        f"{func.__name__} failed on attempt {attempt}/{max_attempts} with error: {e}. "
                        f"Retrying in {delay} seconds..."
                    )
                    await asyncio.sleep(delay)
                    delay *= backoff

        return wrapper
    return decorator
