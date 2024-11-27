import os
import base64
import hashlib
import httpx
import asyncio
import webbrowser
from aiohttp import web

# configurations
AUTH0_DOMAIN = 'your-domain.auth0.com'
CLIENT_ID = 'your-client-id'
AUDIENCE = 'your-api-identifier'
REDIRECT_URI = 'http://localhost:5000/callback'
AUTH_SCOPES = 'openid profile email offline_access'

# initialize global variables to store the access and refresh tokens
_access_token = None
_refresh_token = None

async def generate_pkce():
    """
    generate pkce code verifier and challenge.
    """
    code_verifier = base64.urlsafe_b64encode(os.urandom(40)).decode('utf-8').rstrip('=')
    code_challenge = base64.urlsafe_b64encode(
        hashlib.sha256(code_verifier.encode('utf-8')).digest()
    ).decode('utf-8').rstrip('=')
    return code_verifier, code_challenge

async def authenticate():
    """
    opens a browser for user login and retrieves an access token and refresh token.
    """
    global _access_token, _refresh_token
    code_verifier, code_challenge = await generate_pkce()

    # create authorization url
    auth_url = (
        f"https://{AUTH0_DOMAIN}/authorize?"
        f"response_type=code&client_id={CLIENT_ID}&redirect_uri={REDIRECT_URI}&scope={AUTH_SCOPES}"
        f"&audience={AUDIENCE}&code_challenge={code_challenge}&code_challenge_method=S256"
    )

    # create aiohttp web server to handle callback
    app = web.Application()
    loop = asyncio.get_event_loop()
    server = loop.create_server(app._make_handler(), '127.0.0.1', 5000)
    await asyncio.ensure_future(server)

    @app.route('/callback', methods=['GET'])
    async def callback(request):
        global _access_token, _refresh_token
        auth_code = request.query.get('code')
        if not auth_code:
            return web.Response(text='login failed!', status=400)

        # exchange authorization code for tokens
        token_url = f"https://{AUTH0_DOMAIN}/oauth/token"
        payload = {
            'grant_type': 'authorization_code',
            'client_id': CLIENT_ID,
            'code_verifier': code_verifier,
            'code': auth_code,
            'redirect_uri': REDIRECT_URI,
        }
        async with httpx.AsyncClient() as client:
            response = await client.post(token_url, json=payload)
            response.raise_for_status()
            tokens = response.json()
            _access_token = tokens['access_token']
            _refresh_token = tokens.get('refresh_token')
        return web.Response(text='authentication successful! you can close this window.')

    # open the browser and run the aiohttp server
    webbrowser.open(auth_url)
    await app.startup()
    await app.cleanup()

async def refresh_access_token():
    """
    uses the refresh token to obtain a new access token.
    """
    global _access_token
    if not _refresh_token:
        raise Exception("no refresh token available. authenticate first.")

    token_url = f"https://{AUTH0_DOMAIN}/oauth/token"
    payload = {
        'grant_type': 'refresh_token',
        'client_id': CLIENT_ID,
        'refresh_token': _refresh_token,
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(token_url, json=payload)
        response.raise_for_status()
        tokens = response.json()
        _access_token = tokens['access_token']

async def get_auth_header():
    """
    returns the authorization header with the bearer token.
    automatically refreshes the token if it has expired.
    """
    global _access_token

    if not _access_token:
        raise Exception('user not authenticated. call `authenticate()` first.')

    try:
        # test the access token by making a dummy request
        test_url = f"https://{AUTH0_DOMAIN}/userinfo"
        async with httpx.AsyncClient() as client:
            response = await client.get(test_url, headers={'Authorization': f'Bearer {_access_token}'})
            if response.status_code == 401:  # token is invalid/expired
                await refresh_access_token()
            elif response.status_code != 200:
                raise Exception('unexpected response during token validation')
    except httpx.RequestError as e:
        if e.response and e.response.status_code == 401:
            await refresh_access_token()
        else:
            raise e

    return {'Authorization': f'Bearer {_access_token}'}

# usage example:
# asyncio.run(authenticate())
# asyncio.run(get_auth_header())
