# spl/db/server/routes/health_routes.py

from quart import jsonify
from .common import handle_errors
from ..app import app

@app.route('/health', methods=['GET'], endpoint='health_endpoint')
@handle_errors
async def health_check():
    return jsonify({'status': 'healthy'}), 200
