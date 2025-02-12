# spl/auth/auth0_management.py
import os
import requests

AUTH0_DOMAIN = os.environ.get("PANTHALIA_AUTH0_DOMAIN")
AUTH0_CLIENT_ID = os.environ.get("PANTHALIA_AUTH0_MGMT_CLIENT_ID")
AUTH0_CLIENT_SECRET = os.environ.get("PANTHALIA_AUTH0_MGMT_CLIENT_SECRET")
# The audience for management API (note the “api/v2”)
AUTH0_AUDIENCE = f"https://{AUTH0_DOMAIN}/api/v2/"

def get_management_token() -> str:
    url = f"https://{AUTH0_DOMAIN}/oauth/token"
    payload = {
        "client_id": AUTH0_CLIENT_ID,
        "client_secret": AUTH0_CLIENT_SECRET,
        "audience": AUTH0_AUDIENCE,
        "grant_type": "client_credentials",
        "scope": "delete:users"
    }
    response = requests.post(url, json=payload)
    if response.status_code != 200:
        # Log detailed error information
        error_info = response.json() if response.headers.get("Content-Type") == "application/json" else response.text
        raise Exception(f"Failed to get management token: {response.status_code} {error_info}")
    return response.json()["access_token"]


def delete_auth0_user(user_id: str) -> bool:
    token = get_management_token()
    url = f"https://{AUTH0_DOMAIN}/api/v2/users/{user_id}"
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.delete(url, headers=headers)
    response.raise_for_status()
    return True