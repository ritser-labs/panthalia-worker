# spl/db/server/routes/account_routes.py
from .common import create_post_route, AuthMethod
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
