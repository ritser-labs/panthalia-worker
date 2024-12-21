def janky_url_replace(url: str) -> str:
    """
    Replace the janky URL with the correct one.

    Args:
    url (str): The janky URL.

    Returns:
    str: The correct URL.
    """
    return url.replace('localhost', 'host.docker.internal')