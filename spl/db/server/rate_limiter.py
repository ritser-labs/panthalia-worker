# rate_limiter.py
from datetime import timedelta
from quart_rate_limiter import RateLimiter, rate_limit, RateLimit

def init_rate_limiter(app, key_function=None):
    """
    Initialize the Quart Rate Limiter on your Quart app.

    Args:
        app (Quart): Your Quart application instance.
        key_function (coroutine, optional): A coroutine that returns a unique key
            per requester. If not provided, it defaults to using the client's IP.
    
    Returns:
        RateLimiter: The initialized rate limiter instance.
    """
    if key_function is None:
        # Default key function uses the remote IP address.
        async def default_key_function():
            from quart import request
            return request.remote_addr
        key_function = default_key_function

    rate_limiter = RateLimiter(app, key_function=key_function)
    return rate_limiter
