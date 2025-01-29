# spl/db/server/app.py

import logging
import os
from quart import Quart
from quart_cors import cors

# Import your DBAdapterServer
from .adapter import db_adapter_server

logger = logging.getLogger(__name__)

# Create and configure the Quart app
original_app = Quart(__name__)
app = cors(original_app, allow_origin="http://localhost:3000")

# If you still need these two global variables and functions:
_perm_modify_db = None

def set_perm_modify_db(perm):
    global _perm_modify_db
    _perm_modify_db = perm

def get_perm_modify_db():
    return _perm_modify_db

# -------------------------------------------------------------------
# Force-load the route modules so they can attach routes to `app`.
# Make sure you import ALL the route files you created in spl/db/server/routes/.
# If you split them differently, list those files here.
# -------------------------------------------------------------------
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

# If you keep any special logic (like hooking extra frameworks),
# put it here at the bottom. Usually, you can just leave it.
#
# Done! Now all your routes are registered onto `app`.
# Usage: "python -m spl.db.server" or run from your main server code.
