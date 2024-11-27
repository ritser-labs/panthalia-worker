import json
import requests
from jose import jwt
from functools import lru_cache

AUTH0_DOMAIN = 'your-domain.auth0.com'
API_AUDIENCE = 'your-api-identifier'
ALGORITHMS = ['RS256']

@lru_cache()
def get_jwks():
    jwks_url = f'https://{AUTH0_DOMAIN}/.well-known/jwks.json'
    response = requests.get(jwks_url)
    response.raise_for_status()
    return response.json()

def verify_jwt(token):
    jwks = get_jwks()
    unverified_header = jwt.get_unverified_header(token)
    rsa_key = {}
    for key in jwks['keys']:
        if key['kid'] == unverified_header['kid']:
            rsa_key = {
                'kty': key['kty'],
                'kid': key['kid'],
                'use': key['use'],
                'n': key['n'],
                'e': key['e']
            }
            break
    if not rsa_key:
        raise Exception('Unable to find appropriate key')

    payload = jwt.decode(
        token,
        rsa_key,
        algorithms=ALGORITHMS,
        audience=API_AUDIENCE,
        issuer=f'https://{AUTH0_DOMAIN}/'
    )
    return payload
