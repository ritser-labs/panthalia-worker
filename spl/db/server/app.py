# file: spl/db/server/app.py

import logging
from quart import Quart
from quart_cors import cors
from .ephemeral_key import (
    generate_ephemeral_db_sot_key,
    get_db_sot_private_key,
    get_db_sot_address,
)

# Possibly the same logger you used
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

original_app = Quart(__name__)
app = cors(original_app, allow_origin="http://localhost:3000")

# This sets or gets your "modify DB" permission int
_perm_modify_db = None

def set_perm_modify_db(perm):
    global _perm_modify_db
    _perm_modify_db = perm

def get_perm_modify_db():
    return _perm_modify_db


###############################################################################
#  Force-load route files (like you do) ...
###############################################################################
from .routes import auth_routes
from .routes import debug_routes
from .routes import health_routes
from .routes import instance_routes
from .routes import job_routes
from .routes import master_state_routes
from .routes import order_routes
from .routes import perms_routes
from .routes import plugin_routes
from .routes import sot_routes
from .routes import stripe_routes
from .routes import subnet_routes
from .routes import task_routes
from .routes import withdrawal_routes
from .routes import account_routes