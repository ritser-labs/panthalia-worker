# file: spl/db/server/adapter/__init__.py

# Instead of "from ....models import init_db, create_invariant_async_session", 
# we now import them from spl.db.init to avoid the circular import.
from ....db.init import init_db, create_invariant_async_session

from .accounts import DBAdapterAccountsMixin
from .holds import DBAdapterHoldsMixin
from .orders_tasks import DBAdapterOrdersTasksMixin
from .jobs_plugins import DBAdapterJobsPluginsMixin
from .permissions import DBAdapterPermissionsMixin
from .instances import DBAdapterInstancesMixin
from .state_updates import DBAdapterStateUpdatesMixin
from .billing.stripe import DBAdapterStripeBillingMixin
from .balance_details import DBAdapterBalanceDetailsMixin

from ....auth.view import get_user_id as default_get_user_id

class DBAdapterServer(
    DBAdapterAccountsMixin,
    DBAdapterHoldsMixin,
    DBAdapterOrdersTasksMixin,
    DBAdapterJobsPluginsMixin,
    DBAdapterPermissionsMixin,
    DBAdapterInstancesMixin,
    DBAdapterStateUpdatesMixin,
    DBAdapterStripeBillingMixin,
    DBAdapterBalanceDetailsMixin
):
    def __init__(self, user_id_getter=None):
        super().__init__()
        if user_id_getter is None:
            user_id_getter = default_get_user_id
        self._user_id_getter = user_id_getter

    def get_user_id(self):
        return self._user_id_getter()

    async def check_invariant(self) -> dict:
        """
        The method that verifies your ledger / account invariants.
        If it fails, return { 'invariant_holds': False, 'difference': ... } or similar.
        If it passes, return { 'invariant_holds': True, ... }.
        """
        # If you have a 'check_invariant()' in a mixin, call super(). 
        # Or implement it here. Example:
        return await super().check_invariant()

# Instantiate the server object once
db_adapter_server = DBAdapterServer()

def get_async_session():
    """
    Returns a brand-new session that automatically checks invariants
    before flush and after commit, via CheckingInvariantAsyncSession.
    """
    SessionFactory = create_invariant_async_session(db_adapter_server)
    return SessionFactory()
