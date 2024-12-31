# file: spl/tests/test_earnings_and_balances.py

import pytest
import json
from datetime import datetime, timedelta
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from spl.db.server.app import original_app
from spl.db.server.adapter import DBAdapterServer
from spl.models import (
    AsyncSessionLocal, Task, TaskStatus, OrderType
)


@pytest.mark.asyncio
async def test_earnings_and_balances_scenario():
    """
    Since the code enforces that only the job owner can place a Bid,
    we unify "buyer" and "job owner" to be the same user: "testbuyer".
    The solver is still "solveruser".
    
    Scenario tested:
      - "testbuyer" (the job owner) deposits indefinite credits => places a Bid => credits go down.
      - "solveruser" deposits indefinite credits => places an Ask => solver's credits go down for the stake.
      - If resolved correct:
          - solver's stake is freed
          - solver's earnings_balance increases by (bid_price - platform_fee)
          - buyer leftover is *not* refunded
      - If resolved incorrect:
          - solver's stake is lost => platform revenue
          - buyer's cost is refunded => leftover credits come back
          - solver's earnings remain unchanged

    We do two tasks in a row: first correct, second incorrect.
    """

    async with original_app.test_request_context('/'):

        #
        # 1) We have two DBAdapterServers:
        #    - buyer_server => user_id_getter = "testbuyer" (also the job owner).
        #    - solver_server => user_id_getter = "solveruser".
        #
        buyer_server = DBAdapterServer(user_id_getter=lambda: "testbuyer")
        solver_server = DBAdapterServer(user_id_getter=lambda: "solveruser")

        #
        # 2) Buyer (who is also job owner) creates plugin, subnet, job
        #
        plugin_id = await buyer_server.create_plugin(
            name="Test Plugin",
            code="print('earning')"
        )
        subnet_id = await buyer_server.create_subnet(
            dispute_period=3600,
            solve_period=1800,
            stake_multiplier=2.0
        )
        job_id = await buyer_server.create_job(
            name="BalanceTesting Job",
            plugin_id=plugin_id,
            subnet_id=subnet_id,
            sot_url="http://example.com",
            iteration=0
        )

        #
        # 3) Buyer => deposit 1000 => "testbuyer"
        #
        await buyer_server.admin_deposit_account(user_id="testbuyer", amount=1000.0)

        #
        # 4) Solver => deposit 500 => "solveruser"
        #
        await solver_server.admin_deposit_account(user_id="solveruser", amount=500.0)

        # Check initial balances
        async with AsyncSessionLocal() as session:
            buyer_acc = await buyer_server.get_or_create_account("testbuyer", session=session)
            solver_acc = await solver_server.get_or_create_account("solveruser", session=session)
            assert buyer_acc.credits_balance == 1000.0
            assert solver_acc.credits_balance == 500.0
            assert solver_acc.earnings_balance == 0.0

        #
        # 5) Create a new Task => owned by "testbuyer" => status=SelectingSolver
        #
        task_id = await buyer_server.create_task(
            job_id=job_id,
            job_iteration=1,
            status=TaskStatus.SelectingSolver,
            params="{}"
        )

        #
        # 6) Buyer (==job owner) => place a Bid=100 => buyer credits => 1000 => 900
        #
        bid_order_id = await buyer_server.create_order(
            task_id=task_id,
            subnet_id=subnet_id,
            order_type=OrderType.Bid,
            price=100.0,
            hold_id=None
        )
        # Confirm => buyer => 900
        async with AsyncSessionLocal() as session:
            buyer_acc = await buyer_server.get_or_create_account("testbuyer", session=session)
            assert buyer_acc.credits_balance == 900.0

        #
        # 7) Solver => place an Ask => stake=2 * 100=200 => solver => 500 => 300
        #
        ask_order_id = await solver_server.create_order(
            task_id=None,
            subnet_id=subnet_id,
            order_type=OrderType.Ask,
            price=100.0,
            hold_id=None
        )
        async with AsyncSessionLocal() as session:
            solver_acc = await solver_server.get_or_create_account("solveruser", session=session)
            assert solver_acc.credits_balance == 300.0

        #
        # 8) Solver => submit => scPending => finalize => is_valid=True => correct
        #
        await solver_server.submit_task_result(task_id, result=json.dumps({"output": "correct"}))
        await buyer_server.finalize_sanity_check(task_id, True)

        # After correct resolution:
        #  - solver's stake is freed => solver => 500
        #  - solver's earnings => +90 (platform_fee=10% => 100-10=90)
        #  - buyer leftover is not refunded => buyer => 900
        async with AsyncSessionLocal() as session:
            t = await session.get(Task, task_id)
            assert t.status == TaskStatus.ResolvedCorrect

            buyer_acc = await buyer_server.get_or_create_account("testbuyer", session=session)
            assert buyer_acc.credits_balance == 900.0, "Buyer leftover stays at 900"

            solver_acc = await solver_server.get_or_create_account("solveruser", session=session)
            assert solver_acc.credits_balance == 500.0, "Solver's stake freed"
            assert solver_acc.earnings_balance == 90.0,  "Earnings=90 after platform fee"

        #
        # 9) Create second Task => resolves incorrect
        #
        task_id_2 = await buyer_server.create_task(
            job_id=job_id,
            job_iteration=2,
            status=TaskStatus.SelectingSolver,
            params="{}"
        )

        # Buyer => place a Bid=150 => from 900 => 750
        bid_order_2 = await buyer_server.create_order(
            task_id=task_id_2,
            subnet_id=subnet_id,
            order_type=OrderType.Bid,
            price=150.0,
            hold_id=None
        )
        async with AsyncSessionLocal() as session:
            buyer_acc_2 = await buyer_server.get_or_create_account("testbuyer", session=session)
            assert buyer_acc_2.credits_balance == 750.0

        # Solver => place an Ask => stake=2*150=300 => from 500 => 200
        ask_order_2 = await solver_server.create_order(
            task_id=None,
            subnet_id=subnet_id,
            order_type=OrderType.Ask,
            price=150.0,
            hold_id=None
        )
        async with AsyncSessionLocal() as session:
            solver_acc_2 = await solver_server.get_or_create_account("solveruser", session=session)
            assert solver_acc_2.credits_balance == 200.0

        # solver => submit => finalize => is_valid=False => incorrect
        await solver_server.submit_task_result(task_id_2, result=json.dumps({"output": "incorrect"}))
        await buyer_server.finalize_sanity_check(task_id_2, False)

        # If incorrect => solver's stake is lost => buyer cost is refunded => leftover=900, solver's earnings stay=90
        async with AsyncSessionLocal() as session:
            t2 = await session.get(Task, task_id_2)
            assert t2.status == TaskStatus.ResolvedIncorrect

            buyer_after = await buyer_server.get_or_create_account("testbuyer", session=session)
            assert buyer_after.credits_balance == 900.0, "Buyer gets 150 refunded => back to 900"

            solver_after = await solver_server.get_or_create_account("solveruser", session=session)
            assert solver_after.credits_balance == 200.0, "Solver lost stake => no refund"
            assert solver_after.earnings_balance == 90.0, "Earnings remain the same"

        print("test_earnings_and_balances_scenario passed successfully!")
