import logging
import os
from quart import Quart
from quart_cors import cors

from .adapter import db_adapter_server

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Quart(__name__)
app = cors(app, allow_origin="http://localhost:3000")

_perm_modify_db = None

def set_perm_modify_db(perm):
    global _perm_modify_db
    _perm_modify_db = perm

def get_perm_modify_db():
    return _perm_modify_db

script_dir = os.path.dirname(os.path.abspath(__file__))
db_adapter = db_adapter_server

# Import routes after app is created so that route decorators have access to app
from . import routes
