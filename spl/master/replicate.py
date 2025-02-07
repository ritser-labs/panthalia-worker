# file: spl/master/replicate.py

import secrets
import asyncio
import logging
import json

from ..models import TaskStatus

logger = logging.getLogger(__name__)

###############################################################################
# Probability-based check for spawning a replicate child
###############################################################################
DEFAULT_REPLICATE_PROB = 0.35

async def should_replicate(prob: float = DEFAULT_REPLICATE_PROB) -> bool:
    """
    Return True with probability = prob. Uses `secrets.randbelow` for randomness.
    """
    threshold = int(1000 * prob)
    return secrets.randbelow(1000) < threshold


###############################################################################
# Spawn a new replicate child referencing the same original_task_id
###############################################################################
async def spawn_replica_task(db_adapter, parent_task_id, replicate_price=1) -> int | None:
    """
    Creates a new 'child' Task that replicates `parent_task_id`.
    The parent's `original_task_id` is carried forward so that
    each replicate child knows which solution they're verifying.

    Returns the new child task_id, or None on failure.

    We do a while-loop up the ancestor chain to find the truly original
    solver task, so grandchildren, great-grandchildren, etc. all gather
    the same 'original_task_id' + replicate_sequence if available.
    """
    parent_task = await db_adapter.get_task(parent_task_id)
    if not parent_task:
        logger.warning(f"[spawn_replica_task] Parent task {parent_task_id} not found.")
        return None

    # Walk up to the highest ancestor to find the truly original solver
    original_task = parent_task
    while original_task.replicated_parent_id is not None:
        maybe_up = await db_adapter.get_task(original_task.replicated_parent_id)
        if not maybe_up:
            break
        original_task = maybe_up

    replicate_versions = []
    if parent_task.result:
        try:
            partials_dict = (
                parent_task.result
                if isinstance(parent_task.result, dict)
                else json.loads(parent_task.result)
            )
            for tstamp in sorted(partials_dict.keys(), key=int):
                ver_num = partials_dict[tstamp].get("version_number")
                if isinstance(ver_num, int):
                    replicate_versions.append(ver_num)
        except Exception as ex:
            logger.error(f"[spawn_replica_task] Parsing parent's partial data failed: {ex}", exc_info=True)

    # Build new task params from the original ancestor
    try:
        original_params = json.loads(original_task.params)
    except:
        original_params = {}

    if replicate_versions:
        original_params["replicate_sequence"] = replicate_versions
    original_params["original_task_id"] = original_task.id

    new_params_str = json.dumps(original_params)

    # Create a new child Task + Bid order
    created = await db_adapter.create_bids_and_tasks(
        job_id=parent_task.job_id,
        num_tasks=1,
        price=replicate_price,
        params=new_params_str,
        hold_id=None
    )
    if not created:
        logger.error("[spawn_replica_task] create_bids_and_tasks returned None.")
        return None

    # Handle possible return formats: direct list OR {"created_items": [...]}
    if isinstance(created, list):
        items = created
    elif isinstance(created, dict) and "created_items" in created and isinstance(created["created_items"], list):
        items = created["created_items"]
    else:
        logger.error("[spawn_replica_task] create_bids_and_tasks returned invalid structure.")
        return None

    if not items:
        logger.error("[spawn_replica_task] No tasks created => None.")
        return None

    if not isinstance(items[0], dict) or "task_id" not in items[0]:
        logger.error("[spawn_replica_task] The first item is missing 'task_id'.")
        return None

    child_id = items[0]["task_id"]

    # Link the replicate child to its parent
    success = await db_adapter.update_replicated_parent(child_id, parent_task_id)
    if not success:
        logger.error(
            f"[spawn_replica_task] Could not set replicated_parent_id={parent_task_id} "
            f"on child_id={child_id}"
        )
        return None

    logger.info(
        f"[spawn_replica_task] Created replicate child={child_id} from parent={parent_task_id}, "
        f"original_task={original_task.id}, replicate_versions={replicate_versions}"
    )
    return child_id


###############################################################################
# Inspect a replicate child's partial outcome, but do NOT finalize it
###############################################################################
async def _monitor_replicate_child(db_adapter, child_task_id):
    """
    Examines the replicate child's partial without finalizing. Returns:
      - ('child_says_match', child_task_id) => partial has replica_status=True
      - ('child_says_mismatch', child_task_id) => partial has replica_status=False
      - ('missing_replica_status', child_task_id) => partial is final but missing that key => worthless
      - ('in_progress', child_task_id) => the child is not final yet
      - ('task_missing', child_task_id) => forcibly removed from DB
    """
    child_task = await db_adapter.get_task(child_task_id)
    if not child_task:
        return ('task_missing', child_task_id)

    # If child is still mid-solving or hasn't posted final partial
    if child_task.status in [
        TaskStatus.SelectingSolver.name,
        TaskStatus.SolverSelected.name,
        TaskStatus.Checking.name,
        TaskStatus.SanityCheckPending.name,
        TaskStatus.ReplicationPending.name
    ]:
        if not child_task.result:
            return ('in_progress', child_task_id)

        partials_dict = child_task.result
        latest_key = max(partials_dict.keys(), key=int)
        final_partial = partials_dict[latest_key]
        if 'replica_status' not in final_partial:
            return ('missing_replica_status', child_task_id)
        else:
            is_match = bool(final_partial['replica_status'])
            return ('child_says_match', child_task_id) if is_match else ('child_says_mismatch', child_task_id)

    # If child is resolved correct/incorrect, treat similarly
    if child_task.status in [TaskStatus.ResolvedCorrect.name, TaskStatus.ResolvedIncorrect.name]:
        if not child_task.result:
            return ('in_progress', child_task_id)
        partials_dict = child_task.result
        latest_key = max(partials_dict.keys(), key=int)
        final_partial = partials_dict[latest_key]
        if 'replica_status' not in final_partial:
            return ('missing_replica_status', child_task_id)
        else:
            is_match = bool(final_partial['replica_status'])
            return ('child_says_match', child_task_id) if is_match else ('child_says_mismatch', child_task_id)

    return ('in_progress', child_task_id)


async def _finalize_chain(db_adapter, plugin, job_id, original_task_id, final_replica_says: bool, child_outcomes: dict):
    """
    Once the chain is done, finalize:
      1) Parent (original task) => correct if final_replica_says=True, else incorrect
      2) Each replicate child => correct if it posted same boolean as final_replica_says
         (missing => incorrect, forcibly removed => skip, etc.)
    """
    # Finalize parent's correctness
    await db_adapter.finalize_sanity_check(original_task_id, final_replica_says)
    logger.info(f"[_finalize_chain] Parent={original_task_id} => {('CORRECT' if final_replica_says else 'INCORRECT')}")

    # Finalize each child according to whether it matched final_replica_says
    for c_id, outcome_val in child_outcomes.items():
        if outcome_val in ('task_missing', None):
            # forcibly removed or never updated => skip
            continue

        if outcome_val in ('missing_replica_status', 'in_progress'):
            # no boolean => finalize => incorrect
            await db_adapter.finalize_sanity_check(c_id, False)
            logger.info(f"[_finalize_chain] Child={c_id} => INCORRECT (missing/no partial).")
        elif outcome_val == True:
            # child posted replica_status=True => correct only if final_replica_says=True
            is_correct = (final_replica_says is True)
            await db_adapter.finalize_sanity_check(c_id, is_correct)
            logger.info(f"[_finalize_chain] Child={c_id} => {('CORRECT' if is_correct else 'INCORRECT')}.")
        elif outcome_val == False:
            # child posted replica_status=False => correct only if final_replica_says=False
            is_correct = (final_replica_says is False)
            await db_adapter.finalize_sanity_check(c_id, is_correct)
            logger.info(f"[_finalize_chain] Child={c_id} => {('CORRECT' if is_correct else 'INCORRECT')}.")
        else:
            await db_adapter.finalize_sanity_check(c_id, False)
            logger.info(f"[_finalize_chain] Child={c_id} => INCORRECT (unknown).")


###############################################################################
# Mark a replicate task as ReplicationPending if we decide to replicate it
###############################################################################
async def _mark_task_replication_pending(db_adapter, child_task_id):
    """
    A small helper to set the parent's replicate task to ReplicationPending
    so it doesn't finalize until chain is done.
    """
    child_task = await db_adapter.get_task(child_task_id)
    if child_task:
        job_id = child_task.job_id
        try:
            await db_adapter.update_task_status(child_task_id, job_id, TaskStatus.ReplicationPending.name)
            logger.info(f"[_mark_task_replication_pending] Set task={child_task_id} => ReplicationPending.")
        except Exception as ex:
            logger.warning(f"[_mark_task_replication_pending] Could not set ReplicationPending: {ex}")


###############################################################################
# The main indefinite replication orchestrator, deferring finalization
###############################################################################
async def manage_replication_chain(db_adapter, plugin, job_id, original_task_id, first_child_id):
    """
    We maintain child_outcomes: child_task_id -> one of:
      True => child_says_match
      False => child_says_mismatch
      'missing_replica_status'
      'in_progress'
      'task_missing'
      or None => not yet polled

    We do NOT finalize any tasks until the chain ends. Instead, we keep
    checking the child's partial. If it says match or mismatch, we might
    replicate further. If so, we mark that child => ReplicationPending
    and spawn the next child. If we cannot spawn a new child, or
    should_replicate() returns False, we finalize the entire chain
    using _finalize_chain.

    If the child is forcibly removed => parent's correct by default.
    If partial is missing => worthless => spawn a new child or default => correct.
    """
    child_outcomes = {}
    chain_task_id = first_child_id
    child_outcomes[chain_task_id] = None  # haven't polled yet

    while True:
        outcome, c_id = await _monitor_replicate_child(db_adapter, chain_task_id)
        child_outcomes[c_id] = (
            True if outcome == 'child_says_match'
            else False if outcome == 'child_says_mismatch'
            else outcome
        )

        if outcome == 'child_says_match':
            # child says parent's solution= correct
            if await should_replicate(prob=DEFAULT_REPLICATE_PROB):
                # set this child => ReplicationPending
                await _mark_task_replication_pending(db_adapter, chain_task_id)
                new_id = await spawn_replica_task(db_adapter, chain_task_id)
                if not new_id:
                    # cannot spawn => finalize => parent's= correct
                    await _finalize_chain(db_adapter, plugin, job_id, original_task_id, True, child_outcomes)
                    return
                chain_task_id = new_id
                child_outcomes[chain_task_id] = None
            else:
                # finalize => parent's= correct
                await _finalize_chain(db_adapter, plugin, job_id, original_task_id, True, child_outcomes)
                return

        elif outcome == 'child_says_mismatch':
            # child says parent's solution= incorrect
            if await should_replicate(prob=DEFAULT_REPLICATE_PROB):
                await _mark_task_replication_pending(db_adapter, chain_task_id)
                new_id = await spawn_replica_task(db_adapter, chain_task_id)
                if not new_id:
                    # finalize => parent's= incorrect
                    await _finalize_chain(db_adapter, plugin, job_id, original_task_id, False, child_outcomes)
                    return
                chain_task_id = new_id
                child_outcomes[chain_task_id] = None
            else:
                await _finalize_chain(db_adapter, plugin, job_id, original_task_id, False, child_outcomes)
                return

        elif outcome == 'missing_replica_status':
            # worthless partial => replicate again
            await _mark_task_replication_pending(db_adapter, chain_task_id)
            new_id = await spawn_replica_task(db_adapter, chain_task_id)
            if not new_id:
                # default parent's= correct
                await _finalize_chain(db_adapter, plugin, job_id, original_task_id, True, child_outcomes)
                return
            chain_task_id = new_id
            child_outcomes[chain_task_id] = None

        elif outcome == 'task_missing':
            # forcibly removed => parent's= correct by default
            await _finalize_chain(db_adapter, plugin, job_id, original_task_id, True, child_outcomes)
            return

        else:  # 'in_progress'
            await asyncio.sleep(0.5)
