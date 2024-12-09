from quart import g

def get_user_id():
    """
    Retrieves the user's unique identifier (sub) from the JWT payload.

    Returns:
        str: The user ID from the JWT payload, or None if not authenticated.
    """
    if not hasattr(g, 'user') or not g.user:
        raise Exception('User not authenticated')
    return g.user.get('user_id')
