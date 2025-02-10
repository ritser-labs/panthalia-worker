# spl/db/server/routes/instance_routes.py

from .common import create_get_route, create_post_route, create_post_route_return_id, AuthMethod
from ..app import app
from ..db_server_instance import db_adapter_server
from quart import request, jsonify

@app.route('/get_instance_by_service_type', methods=['GET'], endpoint='get_instance_by_service_type_endpoint')
async def get_instance_by_service_type_route():
    route_func = create_get_route(
        method=db_adapter_server.get_instance_by_service_type,
        params=['service_type','job_id'],
        auth_method=AuthMethod.KEY
    )
    return await route_func()

@app.route('/get_instances_by_job', methods=['GET'], endpoint='get_instances_by_job_endpoint')
async def get_instances_by_job_route():
    route_func = create_get_route(
        method=db_adapter_server.get_instances_by_job,
        params=['job_id'],
        auth_method=AuthMethod.KEY
    )
    return await route_func()

@app.route('/get_all_instances', methods=['GET'], endpoint='get_all_instances_endpoint')
async def get_all_instances_route():
    route_func = create_get_route(
        method=db_adapter_server.get_all_instances,
        params=[],
        auth_method=AuthMethod.KEY
    )
    return await route_func()

@app.route('/create_instance', methods=['POST'], endpoint='create_instance_endpoint')
async def create_instance_route():
    route_func = create_post_route_return_id(
        db_adapter_server.create_instance,
        ['name','service_type','job_id','private_key','pod_id','process_id'],
        'instance_id'
    )
    return await route_func()

@app.route('/delete_instance', methods=['POST'], endpoint='delete_instance_endpoint')
async def delete_instance_route():
    route_func = create_post_route(
        db_adapter_server.delete_instance,
        ['instance_id'],
        auth_method=AuthMethod.USER
    )
    return await route_func()

@app.route('/get_instance', methods=['GET'], endpoint='get_instance_endpoint')
async def get_instance_route():
    route_func = create_get_route(
        method=db_adapter_server.get_instance,
        params=['instance_id'],
        auth_method=AuthMethod.KEY
    )
    return await route_func()

@app.route('/get_free_instances_by_slot_type', methods=['GET'], endpoint='get_free_instances_by_slot_type_endpoint')
async def get_free_instances_by_slot_type_route():
    route_func = create_get_route(
        method=db_adapter_server.get_free_instances_by_slot_type,
        params=['slot_type'],
        auth_method=AuthMethod.KEY
    )
    return await route_func()

@app.route('/reserve_instance', methods=['POST'], endpoint='reserve_instance_endpoint')
async def reserve_instance_route():
    route_func = create_post_route(
        db_adapter_server.reserve_instance,
        ['instance_id','job_id'],
        auth_method=AuthMethod.KEY
    )
    return await route_func()

@app.route('/update_instance', methods=['POST'], endpoint='update_instance_endpoint')
async def update_instance():
    data = await request.get_json()
    if not data:
        return jsonify({'error':'Missing JSON body'}), 400
    if 'instance_id' not in data:
        return jsonify({'error':'Missing "instance_id"'}), 400

    success = await db_adapter_server.update_instance(data)
    return jsonify({'success': success}), 200
