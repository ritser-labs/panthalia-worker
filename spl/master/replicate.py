# file: spl/master/replicate.py

import secrets
import asyncio
import logging
import json

from ..models import TaskStatus, PluginReviewStatus

logger = logging.getLogger(__name__)

DEFAULT_REPLICATE_PROB = 0.35

###############################################################################
# Master-state persistence for replication chains
###############################################################################
#
# We store all replication-chain data under:
#   master_state_json["replication_chains"][str(original_task_id)] = {
#       "chain_current_task_id": <string or None>,    # storing as str(task_id)
#       "final_decision": True / False / None,
#       "child_outcomes": {
#           "1234": True/False/"missing_is_original_valid"/"task_missing"/"in_progress"/None,
#           "9999": ...
#       }
#   }
#
# By making these keys strings, we avoid JSON sorting collisions between
# int and str when `json.dumps(..., sort_keys=True)` is used.
###############################################################################


async def _get_replication_chain_state(db_adapter, job_id, original_task_id) -> dict:
    """
    Retrieves (and if necessary initializes) the replication-chain state 
    for the given original_task_id inside the master's master_state_json.
    """
    state_data = await db_adapter.get_master_state_for_job(job_id)
    if "replication_chains" not in state_data:
        state_data["replication_chains"] = {}

    chains_dict = state_data["replication_chains"]
    key = str(original_task_id)

    if key not in chains_dict:
        chains_dict[key] = {
            "chain_current_task_id": None,
            "final_decision": None,
            "child_outcomes": {}
        }
        state_data["replication_chains"] = chains_dict
        await db_adapter.update_master_state_for_job(job_id, state_data)

    return chains_dict[key]


async def _save_replication_chain_state(db_adapter, job_id, original_task_id, chain_state: dict):
    """
    Overwrites the existing chain-state sub-dict for `original_task_id` 
    in master_state_json["replication_chains"].
    """
    state_data = await db_adapter.get_master_state_for_job(job_id)
    if "replication_chains" not in state_data:
        state_data["replication_chains"] = {}

    chains_dict = state_data["replication_chains"]
    chains_dict[str(original_task_id)] = chain_state
    state_data["replication_chains"] = chains_dict
    await db_adapter.update_master_state_for_job(job_id, state_data)


###############################################################################
# Probability-based check for spawning a replicate child
###############################################################################
async def should_replicate(db_adapter, job_id: int) -> bool:
    # Retrieve the job record.
    job = await db_adapter.get_job(job_id)
    
    # Retrieve the plugin associated with the job.
    plugin = await db_adapter.get_plugin(job.plugin_id)
    
    # If the plugin is not approved, disable replication.
    if plugin.review_status != PluginReviewStatus.Approved:
        replicate_prob = 0.0
    elif not job.active:
        replicate_prob = 0.0
    else:
        replicate_prob = job.replicate_prob

    # Use secrets.randbelow for randomness.
    threshold = int(1000 * replicate_prob)
    return secrets.randbelow(1000) < threshold


###############################################################################
# Spawn a new replicate child referencing the *TOPMOST ancestor*
###############################################################################
async def spawn_replica_task(db_adapter, parent_task_id, replicate_price=1) -> int | None:
    """
    Creates a new 'child' Task that replicates the *very first (topmost) solver*.

    1) We climb up the chain from `parent_task_id` to find the truly original solver 
       (the one that is not itself a replicate).
    2) We set "original_task_id" = that top solver's ID in the new child's params.
    3) The child's `replicate_sequence` is derived from the original solver's partial 
       results (so we know exactly which aggregator versions to compare).
    4) We link the child's 'replicated_parent_id' to `parent_task_id` 
       for chain resolution logic, but the child's "original_task_id" 
       is always the top solver's ID.

    This ensures that *all* children replicate from the same top solver 
    but form a chain for correctness resolution.
    """
    parent_task = await db_adapter.get_task(parent_task_id)
    if not parent_task:
        logger.warning(f"[spawn_replica_task] Parent task {parent_task_id} not found.")
        return None

    # 1) Climb up to the highest ancestor => the true original solver
    original_task = parent_task
    while original_task.replicated_parent_id is not None:
        maybe_up = await db_adapter.get_task(original_task.replicated_parent_id)
        if not maybe_up:
            break
        original_task = maybe_up

    # 2) Gather aggregator versions from the original solver's partials
    replicate_versions = []
    if original_task.result:
        try:
            # original_task.result might be a dict or JSON string
            partials_dict = (
                original_task.result
                if isinstance(original_task.result, dict)
                else json.loads(original_task.result)
            )
            for tstamp_key in sorted(partials_dict.keys(), key=float):
                ver_num = partials_dict[tstamp_key].get("version_number")
                if isinstance(ver_num, int):
                    replicate_versions.append(ver_num)
        except Exception as ex:
            logger.error(
                f"[spawn_replica_task] Parsing original solver's partial data failed: {ex}",
                exc_info=True
            )

    # 3) Build new child params from the *original solver's* params
    try:
        original_params = json.loads(original_task.params)
    except:
        original_params = {}

    # Insert replicate info
    original_params["replicate_sequence"] = replicate_versions
    original_params["original_task_id"] = original_task.id

    new_params_str = json.dumps(original_params)

    # 4) Create child Task + Bid
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

    if isinstance(created, list):
        items = created
    elif isinstance(created, dict) and "created_items" in created:
        items = created["created_items"]
    else:
        logger.error("[spawn_replica_task] create_bids_and_tasks returned unrecognized format.")
        return None

    if not items:
        logger.error("[spawn_replica_task] No tasks created => None.")
        return None
    if not isinstance(items[0], dict) or "task_id" not in items[0]:
        logger.error("[spawn_replica_task] The new item is missing 'task_id'.")
        return None

    child_id = items[0]["task_id"]

    # 5) Link new child => parent's ID (for chain resolution)
    success = await db_adapter.update_replicated_parent(child_id, parent_task_id)
    if not success:
        logger.error(
            f"[spawn_replica_task] Could not set replicated_parent_id={parent_task_id} "
            f"on child_id={child_id}"
        )
        return None

    logger.info(
        f"[spawn_replica_task] Created replicate child={child_id} from parent={parent_task_id}, "
        f"topmost_solver={original_task.id}, replicate_versions={replicate_versions}"
    )

    # 6) Record it in the chain-state for the topmost solver using *string* key
    chain_state = await _get_replication_chain_state(db_adapter, parent_task.job_id, original_task.id)
    chain_outcomes = chain_state.get("child_outcomes", {})
    chain_outcomes[str(child_id)] = None  # newly spawned, not yet known
    chain_state["child_outcomes"] = chain_outcomes
    await _save_replication_chain_state(db_adapter, parent_task.job_id, original_task.id, chain_state)

    return child_id


###############################################################################
# Inspect a replicate child's partial outcome (but do *not* finalize it yet)
###############################################################################
async def _monitor_replicate_child(db_adapter, child_task_id: int):
    """
    Checks the replicate child's status and returns one of:
        ('child_says_match', child_task_id)
        ('child_says_mismatch', child_task_id)
        ('missing_is_original_valid', child_task_id)
        ('in_progress', child_task_id)
        ('task_missing', child_task_id)

    We'll interpret 'child_says_match' => child posted final is_original_valid=True
    We'll interpret 'child_says_mismatch' => final is_original_valid=False
    """
    child_task = await db_adapter.get_task(child_task_id)
    if not child_task:
        return ('task_missing', child_task_id)

    # If child is still mid-solving, or partially done
    if child_task.status in (
        TaskStatus.SelectingSolver.name,
        TaskStatus.SolverSelected.name,
        TaskStatus.Checking.name,
        TaskStatus.SolutionSubmitted.name,
        TaskStatus.ReplicationPending.name
    ):
        if not child_task.result:
            return ('in_progress', child_task_id)

        # There's partial data, check if it has final "is_original_valid"
        partials_dict = child_task.result
        latest_key = max(partials_dict.keys(), key=float)
        final_partial = partials_dict[latest_key]
        if 'is_original_valid' not in final_partial:
            return ('missing_is_original_valid', child_task_id)
        else:
            # child claims match or mismatch
            is_match = bool(final_partial['is_original_valid'])
            return (
                ('child_says_match', child_task_id)
                if is_match
                else ('child_says_mismatch', child_task_id)
            )

    # If child is resolved correct/incorrect, interpret the final partial:
    if child_task.status in (TaskStatus.ResolvedCorrect.name, TaskStatus.ResolvedIncorrect.name):
        if not child_task.result:
            return ('in_progress', child_task_id)
        partials_dict = child_task.result
        latest_key = max(partials_dict.keys(), key=float)
        final_partial = partials_dict[latest_key]
        if 'is_original_valid' not in final_partial:
            return ('missing_is_original_valid', child_task_id)
        is_match = bool(final_partial['is_original_valid'])
        return (
            ('child_says_match', child_task_id)
            if is_match
            else ('child_says_mismatch', child_task_id)
        )

    # Default fallback
    return ('in_progress', child_task_id)


async def _mark_task_replication_pending(db_adapter, child_task_id):
    """
    Helper to set the replicate child => ReplicationPending so it won't finalize
    prematurely. 
    """
    child_task = await db_adapter.get_task(child_task_id)
    if child_task:
        job_id = child_task.job_id
        try:
            await db_adapter.update_task_status(
                child_task_id, job_id, TaskStatus.ReplicationPending.name
            )
            logger.info(f"[_mark_task_replication_pending] Task={child_task_id} => ReplicationPending.")
        except Exception as ex:
            logger.warning(f"[_mark_task_replication_pending] Could not set ReplicationPending: {ex}")


###############################################################################
# Finalize the entire chain once we have a final decision
###############################################################################
async def _finalize_chain(db_adapter, plugin, job_id, original_task_id, final_replica_says: bool, child_outcomes: dict):
    """
    Once the chain is done, finalize:

      1) The original solver => correct if final_replica_says=True, else incorrect
      2) Each replicate child => correct if it posted *the same boolean* as final_replica_says
         (missing => finalize as incorrect, forcibly removed => skip final, etc.)

    Then store `final_decision` in the chain state so we do not re-run it 
    if the master restarts. 
    """
    # Finalize parent's correctness
    await db_adapter.finalize_sanity_check(original_task_id, final_replica_says)
    logger.info(
        f"[_finalize_chain] Top solver {original_task_id} => "
        f"{'CORRECT' if final_replica_says else 'INCORRECT'}"
    )

    # Finalize each child's correctness
    for c_id_str, outcome_val in child_outcomes.items():
        if outcome_val in ('task_missing', None):
            # forcibly removed or never posted => skip or finalize => often "INCORRECT"
            continue

        if outcome_val in ('missing_is_original_valid', 'in_progress'):
            # no final boolean => finalize => incorrect
            await db_adapter.finalize_sanity_check(int(c_id_str), False)
            logger.info(f"[_finalize_chain] Child={c_id_str} => INCORRECT (missing final is_original_valid).")

        elif outcome_val is True:
            # child posted is_original_valid=True => correct only if final_replica_says==True
            is_correct = (final_replica_says is True)
            await db_adapter.finalize_sanity_check(int(c_id_str), is_correct)
            logger.info(f"[_finalize_chain] Child={c_id_str} => {'CORRECT' if is_correct else 'INCORRECT'}.")

        elif outcome_val is False:
            # child posted is_original_valid=False => correct only if final_replica_says==False
            is_correct = (final_replica_says is False)
            await db_adapter.finalize_sanity_check(int(c_id_str), is_correct)
            logger.info(f"[_finalize_chain] Child={c_id_str} => {'CORRECT' if is_correct else 'INCORRECT'}.")

        else:
            # fallback
            await db_adapter.finalize_sanity_check(int(c_id_str), False)
            logger.info(f"[_finalize_chain] Child={c_id_str} => INCORRECT (unknown outcome).")

    # Mark chain as finalized in master_state
    chain_state = await _get_replication_chain_state(db_adapter, job_id, original_task_id)
    chain_state["final_decision"] = bool(final_replica_says)
    # If you want to clear the current child:
    chain_state["chain_current_task_id"] = None
    await _save_replication_chain_state(db_adapter, job_id, original_task_id, chain_state)


###############################################################################
# The indefinite replication orchestrator
###############################################################################
async def manage_replication_chain(db_adapter, plugin, job_id, original_task_id, first_child_id):
    """
    We keep spawning replicate children until we choose to finalize. 
    The top solver's correctness is decided only at the end, 
    based on the 'childmost' replicate result or forced fallback.

    Logic (short version):
      - If child says "match" => replicate again or finalize => parent's CORRECT
      - If child says "mismatch" => replicate again or finalize => parent's INCORRECT
      - If forcibly removed => parent's CORRECT
      - If partial is "missing_is_original_valid" => replicate again or default => parent's CORRECT
    """
    # Fetch the chain-state from the master JSON
    chain_state = await _get_replication_chain_state(db_adapter, job_id, original_task_id)
    # If final_decision is already set => chain concluded previously
    if chain_state["final_decision"] is not None:
        concluded = chain_state["final_decision"]
        logger.info(
            f"[manage_replication_chain] Chain for original={original_task_id} "
            f"already concluded => {('CORRECT' if concluded else 'INCORRECT')}."
        )
        return "chain_success" if concluded else "chain_fail"

    # If there's a child we were monitoring, pick that up;
    # else use first_child_id
    current_str = chain_state["chain_current_task_id"]
    if not current_str:
        current_str = str(first_child_id)

    chain_state["chain_current_task_id"] = current_str
    child_outcomes = chain_state["child_outcomes"]
    if str(first_child_id) not in child_outcomes:
        child_outcomes[str(first_child_id)] = None

    await _save_replication_chain_state(db_adapter, job_id, original_task_id, chain_state)

    while True:
        # Convert string -> int for monitoring
        current_child_id = int(chain_state["chain_current_task_id"])
        outcome, c_id_int = await _monitor_replicate_child(db_adapter, current_child_id)
        c_id_str = str(c_id_int)

        # Update chain outcomes
        if outcome == 'child_says_match':
            child_outcomes[c_id_str] = True
        elif outcome == 'child_says_mismatch':
            child_outcomes[c_id_str] = False
        else:
            child_outcomes[c_id_str] = outcome

        chain_state["child_outcomes"] = child_outcomes
        chain_state["chain_current_task_id"] = str(c_id_int)
        await _save_replication_chain_state(db_adapter, job_id, original_task_id, chain_state)

        if outcome == 'child_says_match':
            # Possibly replicate again, else finalize => parent's correct
            if await should_replicate(db_adapter, job_id):
                await _mark_task_replication_pending(db_adapter, c_id_int)
                new_id = await spawn_replica_task(db_adapter, c_id_int)
                if not new_id:
                    # finalize => parent's correct
                    await _finalize_chain(db_adapter, plugin, job_id, original_task_id, True, child_outcomes)
                    return "chain_success"
                chain_state["chain_current_task_id"] = str(new_id)
                await _save_replication_chain_state(db_adapter, job_id, original_task_id, chain_state)
            else:
                # finalize => parent's correct
                await _finalize_chain(db_adapter, plugin, job_id, original_task_id, True, child_outcomes)
                return "chain_success"

        elif outcome == 'child_says_mismatch':
            # Possibly replicate again, else finalize => parent's incorrect
            if await should_replicate(db_adapter, job_id):
                await _mark_task_replication_pending(db_adapter, c_id_int)
                new_id = await spawn_replica_task(db_adapter, c_id_int)
                if not new_id:
                    await _finalize_chain(db_adapter, plugin, job_id, original_task_id, False, child_outcomes)
                    return "chain_fail"
                chain_state["chain_current_task_id"] = str(new_id)
                await _save_replication_chain_state(db_adapter, job_id, original_task_id, chain_state)
            else:
                await _finalize_chain(db_adapter, plugin, job_id, original_task_id, False, child_outcomes)
                return "chain_fail"

        elif outcome == 'missing_is_original_valid':
            # child didn't provide a final status => replicate again or finalize => parent's correct
            await _mark_task_replication_pending(db_adapter, c_id_int)
            new_id = await spawn_replica_task(db_adapter, c_id_int)
            if not new_id:
                # fallback => finalize => parent's correct
                await _finalize_chain(db_adapter, plugin, job_id, original_task_id, True, child_outcomes)
                return "chain_success"
            chain_state["chain_current_task_id"] = str(new_id)
            await _save_replication_chain_state(db_adapter, job_id, original_task_id, chain_state)

        elif outcome == 'task_missing':
            # forcibly removed => parent's correct
            await _finalize_chain(db_adapter, plugin, job_id, original_task_id, True, child_outcomes)
            return "chain_success"

        else:
            # outcome == 'in_progress': child not final yet => wait & poll again
            await asyncio.sleep(0.5)
