# spl/db/server/adapter/__init__.py

from ....models import init_db
from .accounts import DBAdapterAccountsMixin
from .holds import DBAdapterHoldsMixin
from .orders_tasks import DBAdapterOrdersTasksMixin
from .jobs_plugins import DBAdapterJobsPluginsMixin
from .permissions import DBAdapterPermissionsMixin
from .instances import DBAdapterInstancesMixin
from .state_updates import DBAdapterStateUpdatesMixin
from .billing.stripe import DBAdapterStripeBillingMixin

# Import the original get_user_id here
from ....auth.view import get_user_id as default_get_user_id

class DBAdapterServer(
    DBAdapterAccountsMixin,
    DBAdapterHoldsMixin,
    DBAdapterOrdersTasksMixin,
    DBAdapterJobsPluginsMixin,
    DBAdapterPermissionsMixin,
    DBAdapterInstancesMixin,
    DBAdapterStateUpdatesMixin,
    DBAdapterStripeBillingMixin
):
    def __init__(self, user_id_getter=None):
        if user_id_getter is None:
            user_id_getter = default_get_user_id
        self._user_id_getter = user_id_getter

    def get_user_id(self):
        return self._user_id_getter()

db_adapter_server = DBAdapterServer()
