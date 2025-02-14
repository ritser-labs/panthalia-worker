# file: spl/db/server/app.py

import logging
from quart import Quart, jsonify
from quart_cors import cors
from .ephemeral_key import (
    generate_ephemeral_db_sot_key,
    get_db_sot_private_key,
    get_db_sot_address,
)
from .rate_limiter import init_rate_limiter  # import the initializer

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

original_app = Quart(__name__)
app = cors(original_app, allow_origin="http://localhost:3000")

# Global error handler for unhandled exceptions
@app.errorhandler(Exception)
async def global_error_handler(error):
    logger.exception("Unhandled error: %s", error)
    return jsonify({"error": str(error)}), 500

# Set up any globals you need (e.g. permission settings)
_perm_modify_db = None

def set_perm_modify_db(perm):
    global _perm_modify_db
    _perm_modify_db = perm

def get_perm_modify_db():
    return _perm_modify_db

# Initialize the rate limiter here:
init_rate_limiter(app)

###############################################################################
# Force-load route files
###############################################################################
from .routes import (
    auth_routes,
    debug_routes,
    health_routes,
    instance_routes,
    job_routes,
    master_state_routes,
    order_routes,
    perms_routes,
    plugin_routes,
    sot_routes,
    stripe_routes,
    subnet_routes,
    task_routes,
    withdrawal_routes,
    account_routes,
    global_routes,
)
