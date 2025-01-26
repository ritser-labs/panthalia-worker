# file: spl/db/server/adapter/__init__.py

from ....db.init import init_db, create_invariant_async_session

# Import all your mixins
from .accounts import DBAdapterAccountsMixin
from .holds import DBAdapterHoldsMixin
from .orders_tasks import DBAdapterOrdersTasksMixin
from .jobs_plugins import DBAdapterJobsPluginsMixin
from .permissions import DBAdapterPermissionsMixin
from .instances import DBAdapterInstancesMixin
from .state_updates import DBAdapterStateUpdatesMixin
from .billing.stripe import DBAdapterStripeBillingMixin
from .balance_details import DBAdapterBalanceDetailsMixin
from .withdrawals import DBAdapterWithdrawalsMixin

# If you need user_id_getter from auth:
from ....auth.view import (get_user_id as default_get_user_id,
                           is_key_auth as default_is_key_auth)

class DBAdapterServer(
    DBAdapterAccountsMixin,
    DBAdapterHoldsMixin,
    DBAdapterOrdersTasksMixin,
    DBAdapterJobsPluginsMixin,
    DBAdapterPermissionsMixin,
    DBAdapterInstancesMixin,
    DBAdapterStateUpdatesMixin,
    DBAdapterStripeBillingMixin,
    DBAdapterBalanceDetailsMixin,
    DBAdapterWithdrawalsMixin,
):
    def __init__(self, user_id_getter=None, is_key_auth=None):
        super().__init__()
        if user_id_getter is None:
            user_id_getter = default_get_user_id
        if is_key_auth is None:
            is_key_auth = default_is_key_auth
        self._user_id_getter = user_id_getter
        self._is_key_auth = is_key_auth

    def get_user_id(self):
        return self._user_id_getter()
    
    def is_key_auth(self):
        return self._is_key_auth()

    def get_async_session(self):
        """
        Returns a brand-new session that automatically checks invariants
        via create_invariant_async_session(db_adapter_server=...).
        """
        SessionFactory = create_invariant_async_session(db_adapter_server=self)
        return SessionFactory()

# Instantiate a single server instance
db_adapter_server = DBAdapterServer()
