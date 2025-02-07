# file: spl/master/replicate.py

import secrets
import asyncio
import logging
from ..models import TaskStatus
import json
import logging

logger = logging.getLogger(__name__)


async def should_replicate(prob: float = 0.1) -> bool:
    """
    Returns True with probability = prob (default=0.1 => 10%).
    We rely on `secrets.randbelow` for cryptographic randomness.
    """
    threshold = int(1000 * prob)
    return secrets.randbelow(1000) < threshold


async def spawn_replica_task(db_adapter, parent_task_id, replicate_price=1) -> int | None:
    parent_task = await db_adapter.get_task(parent_task_id)
    if not parent_task:
        logging.warning(f"[spawn_replica_task] Parent task {parent_task_id} not found.")
        return None

    # --------------------------------------------------
    # 1) Gather the parent's partials from parent_task.result
    # --------------------------------------------------
    replicate_versions = []
    if parent_task.result:  
        try:
            # The result field might be JSON with keys as timestamps
            # e.g. { "1675449193": { "version_number": 10, "loss": ... }, ... }
            all_partials = (
                json.loads(parent_task.result)
                if isinstance(parent_task.result, str)
                else parent_task.result
            )
            # Sort by int(timestamp) so we reconstruct partials in chronological order
            for ts_str in sorted(all_partials.keys(), key=lambda x: int(x)):
                partial_data = all_partials[ts_str]
                # If each partial includes "version_number", collect it
                version = partial_data.get("version_number")
                if isinstance(version, int):
                    replicate_versions.append(version)
        except Exception as e:
            logging.error(f"[spawn_replica_task] Could not parse parent_task.result for Task {parent_task_id}: {e}")

    # Now replicate_versions might be [10, 11, 12, ...] from the parent's partials
    # Insert them in the child's params as "replicate_sequence"


    # 2) Clone parent's params, inject replicate_sequence
    try:
        parent_params = json.loads(parent_task.params)
    except:
        parent_params = {}
    if replicate_versions:
        parent_params["replicate_sequence"] = replicate_versions

    new_params_str = json.dumps(parent_params)

    # 3) Create new DB Task + Bid (as you already do)
    created = await db_adapter.create_bids_and_tasks(
        job_id=parent_task.job_id,
        num_tasks=1,
        price=replicate_price,
        params=new_params_str,
        hold_id=None
    )
    if not created or len(created) < 1:
        return None

    child_id = created[0]["task_id"]

    # 4) Mark the child's replicated_parent_id
    success = await db_adapter.update_replicated_parent(child_id, parent_task_id)
    if not success:
        return None

    logging.info(f"[spawn_replica_task] Created child {child_id} from parent {parent_task_id}. replicate_versions={replicate_versions}")
    return child_id


async def manage_replication_chain(db_adapter, plugin, job_id, original_task_id, first_child_id):
    """
    Indefinite replication chain:
      1) Wait for the child (first_child_id) to finish or replicate again.
      2) If child is resolved_incorrect => chain fails => original=CORRECT
      3) If child is correct => replicate again => ...
         => eventually a child is 'resolved_incorrect' => chain_fail
         => or a child is 'resolved_correct' => chain_correct
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


async def monitor_replicate_child(db_adapter, plugin, job_id, task_id, replicate_prob=0.25):
    """
    Poll the replicate child until it ends up in a final state or we replicate again.
    Similar to old logic but used specifically for chain-of-replication.
    """
    while True:
        child_task = await db_adapter.get_task(task_id)
        if not child_task:
            # forcibly deleted => consider it incorrect => chain fails
            return "resolved_incorrect"

        if child_task.status in [TaskStatus.ResolvedCorrect.name, TaskStatus.ResolvedIncorrect.name]:
            return "resolved_correct" if child_task.status == TaskStatus.ResolvedCorrect.name else "resolved_incorrect"

        if child_task.status == TaskStatus.SanityCheckPending.name:
            # If it has no final result => wait
            if not child_task.result:
                await asyncio.sleep(0.5)
                continue

            # local check if the child's final result is valid
            local_pass = await plugin.call_submodule("model_adapter", "run_sanity_check", child_task.result)
            if not local_pass:
                await db_adapter.finalize_sanity_check(child_task.id, False)
                return "resolved_incorrect"
            else:
                # local pass => we either replicate or finalize correct
                import random
                if random.random() < replicate_prob:
                    return "replicate"
                await db_adapter.finalize_sanity_check(child_task.id, True)
                return "resolved_correct"

        await asyncio.sleep(0.5)


# ------------------------------------------------------------------------
# Updated Comparison Logic
# ------------------------------------------------------------------------
async def compare_results_locally(plugin, original_result, child_result, rel_tolerance=1e-4):
    """
    Compare the final numeric tensors produced by the original vs. replicate tasks.

    1) If each result includes 'result_url', we:
       - Download both from SOT (via `download_file`).
       - Decode each into a Tensor (via plugin.model_adapter.decode_diff or similar).
       - Compare norms or MSE for final equality.

    2) If either is missing 'result_url', fallback to comparing 'loss'
       (the old, simpler approach).

    :param plugin: The plugin instance (gives us `call_submodule` etc.).
    :param original_result: The final result dict from the original task
                            (usually contains "result_url" and "loss").
    :param child_result:    The final result dict from the replicate child.
    :param rel_tolerance:   Tolerance for relative difference in final Tensor norms.

    :return: True if they are effectively the same; False otherwise.
    """
    # A helper to do the numeric comparison on two Tensors
    async def compare_tensors(t1, t2, rtol=1e-4):
        diff = (t1 - t2).norm().item()
        base = t1.norm().item() + 1e-12  # avoid division by zero
        rel_diff = diff / base
        logger.info(
            f"[compare_tensors] abs_diff={diff:.6f}, base_norm={base:.6f}, rel_diff={rel_diff:.6f}"
        )
        return (rel_diff < rtol)

    # 1) Check if both have "result_url"
    if "result_url" in original_result and "result_url" in child_result:
        from ..common import download_file

        original_url = original_result["result_url"]
        child_url = child_result["result_url"]

        try:
            # Download each
            odata = await download_file(original_url, download_type="tensor", chunk_timeout=300)
            cdata = await download_file(child_url, download_type="tensor", chunk_timeout=300)

            # Decode them. Typically you call `model_adapter.decode_diff` if it's a gradient
            # or you might have a dedicated decode function for final params.
            o_tensor = await plugin.call_submodule("model_adapter", "decode_diff", odata)
            c_tensor = await plugin.call_submodule("model_adapter", "decode_diff", cdata)

            # Compare
            same = await compare_tensors(o_tensor, c_tensor, rtol=rel_tolerance)
            logger.info(f"[compare_results_locally] Tensor comparison => same={same}")
            return same
        except Exception as ex:
            logger.error(f"[compare_results_locally] Error comparing downloaded tensors: {ex}", exc_info=True)
            # fallback to a simpler approach if an error arises
            pass

    # 2) If missing 'result_url' or if downloads failed => fallback to loss-based comparison
    logger.warning("[compare_results_locally] Fallback => comparing 'loss' fields only.")
    if "loss" in original_result and "loss" in child_result:
        try:
            diff_loss = abs(original_result["loss"] - child_result["loss"])
            logger.info(f"Loss difference = {diff_loss:.6f}")
            return (diff_loss < 0.05)
        except:
            pass

    # If we cannot do anything else, default to True or False as desired
    logger.warning("[compare_results_locally] No valid numeric info => default to mismatch.")
    return False
