# spl/db/server/routes/sot_uploads_routes.py

from ..app import app
from .common import create_get_route, create_post_route_return_id, create_post_route, AuthMethod
from ..db_server_instance import db_adapter_server

@app.route('/record_sot_upload', methods=['POST'])
async def record_sot_upload_route():
    route_func = create_post_route_return_id(
        db_adapter_server.record_sot_upload,
        ['job_id','user_id','s3_key','file_size_bytes'],
        'sot_upload_id',
        auth_method=AuthMethod.KEY  # or ADMIN, your call
    )
    return await route_func()

@app.route('/get_sot_upload_usage', methods=['GET'])
async def get_sot_upload_usage_route():
    route_func = create_get_route(
        method=db_adapter_server.get_sot_upload_usage,
        params=['user_id'],
        auth_method=AuthMethod.KEY  # or maybe user if you want user to see their usage
    )
    return await route_func()

@app.route('/prune_old_sot_uploads', methods=['POST'])
async def prune_old_sot_uploads_route():
    route_func = create_post_route(
        db_adapter_server.prune_old_sot_uploads,
        ['user_id'],
        auth_method=AuthMethod.KEY
    )
    return await route_func()

@app.route('/get_sot_download_url', methods=['GET'], endpoint='get_sot_download_url_endpoint')
async def get_sot_download_url_route():
    """
    GET /get_sot_download_url?job_id=XYZ

    Returns a JSON object like:
      { "download_url": "<presigned URL>" }
    
    or an error message if not found.
    """
    route_func = create_get_route(
        method=db_adapter_server.get_sot_download_url,
        params=['job_id'],
        auth_method=AuthMethod.USER
    )
    return await route_func()