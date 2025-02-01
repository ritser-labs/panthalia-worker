# file: spl/tests/test_earnings_and_balances.py

import pytest
import json
from datetime import datetime, timezone, timedelta
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from spl.db.server.app import original_app
from spl.db.server.adapter import DBAdapterServer
from spl.models import (
    TaskStatus, OrderType
)

@pytest.mark.asyncio
async def test_earnings_and_balances_scenario():
    """
    Scenario tested:
      - "testbuyer" deposits indefinite credits => places a Bid => leftover in credits holds changes.
      - "solveruser" places an Ask => leftover in solver's stake hold changes.
      - If resolved correct: solver's stake is freed => solver leftover => new earnings hold => derived solver.earnings_balance => (bid_price - fee).
      - If resolved incorrect: solver loses stake => buyer is refunded => leftover reverts.
    """

    async with original_app.test_request_context('/'):

        buyer_server = DBAdapterServer(user_id_getter=lambda: "testbuyer")
        solver_server = DBAdapterServer(user_id_getter=lambda: "solveruser")

        plugin_id = await buyer_server.create_plugin(name="Test Plugin", code="print('earning')")
        subnet_id = await buyer_server.create_subnet(dispute_period=3600, solve_period=1800, stake_multiplier=2.0)
        job_id = await buyer_server.create_job(
            name="BalanceTesting Job",
            plugin_id=plugin_id,
            subnet_id=subnet_id,
            sot_url="http://panthalia.com",
            iteration=0
        )

        # 1) Buyer => deposit 1000 => leftover => 1000
        await buyer_server.admin_deposit_account(user_id="testbuyer", amount=1000.0)
        balance_buyer_1 = await buyer_server.get_balance_details_for_user()
        assert balance_buyer_1["credits_balance"] == 1000.0

        # 2) Solver => deposit 500 => leftover => 500
        await solver_server.admin_deposit_account(user_id="solveruser", amount=500.0)
        balance_solver_1 = await solver_server.get_balance_details_for_user()
        assert balance_solver_1["credits_balance"] == 500.0
        assert balance_solver_1["earnings_balance"] == 0.0

        # 3) Create a new Task
        task_id = await buyer_server.create_task(
            job_id=job_id,
            job_iteration=1,
            status=TaskStatus.SelectingSolver,
            params="{}"
        )

        # 4) Buyer => place a Bid=100 => leftover => 900
        bid_order_id = await buyer_server.create_order(
            task_id=task_id,
            subnet_id=subnet_id,
            order_type=OrderType.Bid,
            price=100.0,
            hold_id=None
        )
        balance_buyer_2 = await buyer_server.get_balance_details_for_user()
        assert balance_buyer_2["credits_balance"] == 900.0

        # 5) Solver => place an Ask => stake=200 => leftover => 300
        ask_order_id = await solver_server.create_order(
            task_id=None,
            subnet_id=subnet_id,
            order_type=OrderType.Ask,
            price=100.0,
            hold_id=None
        )

        # Must commit after matching
        async with solver_server.get_async_session() as sess:
            await solver_server.match_bid_ask_orders(sess, subnet_id)
            await sess.commit()

        balance_solver_2 = await solver_server.get_balance_details_for_user()
        assert balance_solver_2["credits_balance"] == 300.0

        # 6) Solve => correct => solver stake freed => leftover => 500 + new earnings
        await solver_server.submit_task_result(task_id, result=json.dumps({"output": "correct"}))
        await buyer_server.finalize_sanity_check(task_id, True)

        balance_buyer_3 = await buyer_server.get_balance_details_for_user()
        # buyer leftover => 900 (no refund)
        assert balance_buyer_3["credits_balance"] == 900.0

        balance_solver_3 = await solver_server.get_balance_details_for_user()
        # Freed stake => solver leftover => 500, plus 90 in earnings
        assert balance_solver_3["credits_balance"] == 500.0
        assert balance_solver_3["earnings_balance"] == 90.0

        # 7) Another Task => resolved incorrect => solver loses stake => buyer refunded
        task_id_2 = await buyer_server.create_task(
            job_id=job_id,
            job_iteration=2,
            status=TaskStatus.SelectingSolver,
            params="{}"
        )
        bid_order_2 = await buyer_server.create_order(
            task_id=task_id_2,
            subnet_id=subnet_id,
            order_type=OrderType.Bid,
            price=150.0,
            hold_id=None
        )
        ask_order_2 = await solver_server.create_order(
            task_id=None,
            subnet_id=subnet_id,
            order_type=OrderType.Ask,
            price=150.0,
            hold_id=None
        )

        # match & commit
        async with solver_server.get_async_session() as sess2:
            await solver_server.match_bid_ask_orders(sess2, subnet_id)
            await sess2.commit()

        await solver_server.submit_task_result(task_id_2, result=json.dumps({"output": "incorrect"}))
        await buyer_server.finalize_sanity_check(task_id_2, False)

        # Buyer leftover returns from 900-150 => 750 + 150 => 900
        balance_buyer_4 = await buyer_server.get_balance_details_for_user()
        assert balance_buyer_4["credits_balance"] == 900.0

        # Solver => lost stake => leftover=200 => earnings still=90
        balance_solver_4 = await solver_server.get_balance_details_for_user()
        assert balance_solver_4["credits_balance"] == 200.0
        assert balance_solver_4["earnings_balance"] == 90.0

        print("test_earnings_and_balances_scenario passed successfully!")
