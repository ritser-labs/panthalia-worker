# In spl/master/replicate.py
# These methods remain mostly the same. The only difference is that we
# are no longer calling them from main_iteration, but from within
# wait_for_result. So the code here is mostly unchanged.

import secrets
import asyncio
import logging
from ..models import TaskStatus
logger = logging.getLogger(__name__)

async def should_replicate(prob: float = 0.1) -> bool:
    """
    Returns True with probability=prob (default=0.1 => 10%).
    We rely on `secrets.randbelow` for cryptographic randomness.
    """
    threshold = int(1000 * prob)
    return secrets.randbelow(1000) < threshold

async def spawn_replica_task(db_adapter, parent_task_id: int, replicate_price=1) -> int|None:
    """
    Clones the parent's params, creates a new (Task+Bid) with the same job_id.
    Then sets child.replicated_parent_id => parent_task_id.
    """
    parent = await db_adapter.get_task(parent_task_id)
    if not parent:
        return None

    created = await db_adapter.create_bids_and_tasks(
        job_id=parent.job_id,
        num_tasks=1,
        price=replicate_price,
        params=parent.params,
        hold_id=None
    )
    if not created or len(created) < 1:
        return None

    child_id = created[0]["task_id"]
    success = await db_adapter.update_replicated_parent(child_id, parent_task_id)
    if not success:
        return None

    return child_id

async def manage_replication_chain(db_adapter, plugin, job_id, original_task_id, first_child_id):
    """
    indefinite chain:
      1) poll the child => if child fails => chain_fail => original = CORRECT
      2) if child is correct => if child => replicate => ...
         => eventually a child is 'resolved_incorrect' => chain_fail
         => or the last child is 'resolved_correct' => chain_correct
    """
    chain_task_id = first_child_id
    while True:
        outcome = await monitor_replicate_child(db_adapter, plugin, job_id, chain_task_id)
        if outcome in ["resolved_incorrect", "resolved_correct"]:
            return "chain_fail" if outcome == "resolved_incorrect" else "chain_correct"
        elif outcome == "replicate":
            new_id = await spawn_replica_task(db_adapter, chain_task_id)
            if not new_id:
                return "chain_fail"
            chain_task_id = new_id
        else:
            return "chain_fail"

async def monitor_replicate_child(db_adapter, plugin, job_id, task_id, replicate_prob=0.1):
    """
    Poll the replicate child until it ends up in a final state or we replicate again.
    Similar to the old monitor_task_until_resolved but only used for replicate chain.
    """
    while True:
        child_task = await db_adapter.get_task(task_id)
        if not child_task:
            # forcibly deleted => resolved_incorrect
            return "resolved_incorrect"

        if child_task.status in [TaskStatus.ResolvedCorrect.name, TaskStatus.ResolvedIncorrect.name]:
            return "resolved_correct" if child_task.status == TaskStatus.ResolvedCorrect.name else "resolved_incorrect"

        if child_task.status == TaskStatus.SanityCheckPending.name:
            if not child_task.result:
                await asyncio.sleep(0.5)
                continue

            local_pass = await plugin.call_submodule("model_adapter", "run_sanity_check", child_task.result)
            if not local_pass:
                # finalize incorrect => outcome => resolved_incorrect
                await db_adapter.finalize_sanity_check(child_task.id, False)
                return "resolved_incorrect"
            else:
                # replicate or finalize correct
                import random
                if random.random() < replicate_prob:
                    return "replicate"
                await db_adapter.finalize_sanity_check(child_task.id, True)
                return "resolved_correct"

        await asyncio.sleep(0.5)

async def compare_results_locally(plugin, original_result, child_result) -> bool:
    """
    In a trivial example, we compare their 'loss' values if present.
    If |loss1 - loss2| < 0.05 => consider them matched. 
    Otherwise they mismatch.
    """
    if "loss" in original_result and "loss" in child_result:
        return abs(original_result["loss"] - child_result["loss"]) < 0.05
    # fallback => assume matched
    return True
