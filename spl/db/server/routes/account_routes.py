# spl/db/server/routes/account_routes.py
from .common import create_post_route, create_get_route, AuthMethod
from ..app import app
from ..db_server_instance import db_adapter_server

@app.route('/delete_account', methods=['POST'], endpoint='delete_account_endpoint')
async def delete_account_route():
    """
    DELETE ACCOUNT (GDPR):
    This endpoint deletes (anonymizes) the currently authenticated user’s account.
    It requires that the caller is authenticated via user JWT or signature.
    
    Request Body: (empty JSON is acceptable)
      {}
      
    Response:
      { "success": true }
    """
    route_func = create_post_route(
         db_adapter_server.delete_account,
         required_keys=[],  # no extra parameters required
         auth_method=AuthMethod.USER
    )
    return await route_func()

@app.route('/user/account', methods=['GET'], endpoint='get_account_endpoint')
async def get_account_route():
    # Use our adapter function via a get-route with USER auth
    route_func = create_get_route(
        method=db_adapter_server.get_account,
        params=[],
        auth_method=AuthMethod.USER
    )
    return await route_func()