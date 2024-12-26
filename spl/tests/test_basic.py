import pytest
from spl.db.server.app import original_app


@pytest.mark.asyncio
async def test_basic_setup(db_adapter_server_fixture):
    """
    Simple smoke test: ensures we can create a plugin, a subnet, and a job,
    then read it back from the DB.
    """
    async with original_app.test_request_context('/'):
        server = db_adapter_server_fixture

        plugin_id = await server.create_plugin(name="Test Plugin", code="print('hello')")
        subnet_id = await server.create_subnet(dispute_period=3600, solve_period=1800, stake_multiplier=1.5)
        job_id = await server.create_job(
            name="Test Job",
            plugin_id=plugin_id,
            subnet_id=subnet_id,
            sot_url="http://example.com",
            iteration=0
        )

        job = await server.get_job(job_id)
        assert job is not None
        assert job.plugin_id == plugin_id
        assert job.subnet_id == subnet_id
        assert job.done is False
        assert job.active is True
