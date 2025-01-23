#!/usr/bin/env bash
#
# add_slot_instance_prompt.sh
#
# Interactively prompts the user for instance details, including
# separate "connection info" fields (host, port, user, password,
# remote_dir) which get combined into a JSON if not empty.
#
# On success, you'll have a new DB Instance row with slot_type = SOT or WORKER,
# waiting for the Master to pick it up.
#
# This version uses $PRIVATE_TOKEN from the environment for authentication:
#   Authorization: Bearer $PRIVATE_TOKEN
# Make sure to `export PRIVATE_TOKEN=someSecretToken` before running.

set -euo pipefail

########################################
# AUTH TOKEN CHECK
########################################
if [[ -z "${PRIVATE_TOKEN:-}" ]]; then
  echo "[ERROR] PRIVATE_TOKEN environment variable is not set."
  echo "Please 'export PRIVATE_TOKEN=xxx' or see optional snippet below to prompt for it."
  exit 1
fi

AUTH_HEADER="Authorization: Bearer $PRIVATE_TOKEN"

########################################
# PROMPT THE USER FOR VALUES
########################################

# 1) DB_URL
read -r -p "Enter DB URL (e.g. http://localhost:8000) [default: http://localhost:8000]: " DB_URL
DB_URL="${DB_URL:-http://localhost:8000}"

# 2) SLOT_TYPE
#    Must be "SOT" or "WORKER"
echo "Choose slot type:"
select choice in "SOT" "WORKER"; do
  case "$choice" in
    SOT)
      SLOT_TYPE="SOT"
      break
      ;;
    WORKER)
      SLOT_TYPE="WORKER"
      break
      ;;
    *)
      echo "Invalid choice. Please pick 'SOT' or 'WORKER'."
      ;;
  esac
done

# 3) SERVICE_TYPE
#    Must match DB server's ServiceType enum: e.g. "sot" for SOT, "worker" for Worker, etc.
echo "Choose service_type (sot, worker, master, db, anvil, etc.):"
select choice2 in "sot" "worker" "master" "db" "anvil"; do
  SERVICE_TYPE="$choice2"
  break
done

# 4) NAME
read -r -p "Instance name (e.g. my-remote-sot-1): " NAME
if [ -z "$NAME" ]; then
  echo "Error: Name cannot be empty. Exiting."
  exit 1
fi

# 5) PRIVATE_KEY
read -r -p "Private key (or leave blank if not needed) [default: 0xdeadbeef]: " PRIVATE_KEY
PRIVATE_KEY="${PRIVATE_KEY:-0xdeadbeef}"

# 6) POD_ID
read -r -p "Pod ID (optional): " POD_ID
POD_ID="${POD_ID:-}"

# 7) PROCESS_ID
read -r -p "Process ID (optional; 0 or blank if none): " PROCESS_ID
PROCESS_ID="${PROCESS_ID:-0}"

# 8) GPU_ENABLED
echo "Does this instance have a GPU? (true/false) [default: false]:"
read -r GPU_ENABLED
GPU_ENABLED="${GPU_ENABLED:-false}"

########################################
# CONNECTION INFO (fields combined into JSON)
########################################
echo
echo "Optionally fill out the remote SSH connection details (press Enter to skip each)."

read -r -p "Host? " CINFO_HOST
read -r -p "Port? [default: 22] " CINFO_PORT
CINFO_PORT="${CINFO_PORT:-22}"

read -r -p "SSH user? [default: ubuntu] " CINFO_USER
CINFO_USER="${CINFO_USER:-ubuntu}"

read -r -p "SSH password? (leave blank if not needed) " CINFO_PASS

read -r -p "Remote directory? [default: /opt/spl] " CINFO_REMOTE
CINFO_REMOTE="${CINFO_REMOTE:-/opt/spl}"

# If the user left the host empty, we consider that "no connection info."
if [ -z "$CINFO_HOST" ]; then
  CONNECTION_INFO=""
else
  # Build JSON with `jq`; remove null fields if empty
  CONNECTION_INFO=$(jq -nc \
    --arg host "$CINFO_HOST" \
    --arg port "$CINFO_PORT" \
    --arg user "$CINFO_USER" \
    --arg pass "$CINFO_PASS" \
    --arg dir "$CINFO_REMOTE" '
    {
      "host": $host,
      "port": ($port | tonumber),
      "user": $user,
      "password": (if $pass == "" then null else $pass end),
      "remote_dir": $dir
    } | del(.[] | select(. == null))
    '
  )
fi


########################################
# CREATE INSTANCE (/create_instance)
########################################

REQ_BODY=$(jq -nc \
  --arg name "$NAME" \
  --arg service_type "$SERVICE_TYPE" \
  --arg priv "$PRIVATE_KEY" \
  --arg pod "$POD_ID" \
  --argjson pid "$PROCESS_ID" \
  --argjson jobid "null" \
  --argjson gpuflag $( [[ "$GPU_ENABLED" = "true" ]] && echo "true" || echo "false" ) \
  --arg cinfo "${CONNECTION_INFO}" '
{
  "name": $name,
  "service_type": $service_type,
  "job_id": $jobid,
  "private_key": $priv,
  "pod_id": $pod,
  "process_id": (if $pid == "0" then null else $pid end),
  "gpu_enabled": $gpuflag,
  "slot_type": null,
  "connection_info": (if $cinfo == "" then null else ($cinfo|fromjson) end)
}
')

echo
echo "Creating new instance at: $DB_URL/create_instance"
echo "JSON body:"
echo "$REQ_BODY"
echo

CREATE_RESP=$(
  curl -s -X POST \
    -H "Content-Type: application/json" \
    -H "$AUTH_HEADER" \
    "$DB_URL/create_instance" \
    -d "$REQ_BODY"
)

INSTANCE_ID=$(echo "$CREATE_RESP" | jq -r '.instance_id // empty')

if [ -z "$INSTANCE_ID" ] || [ "$INSTANCE_ID" = "null" ]; then
  echo "[ERROR] Failed to create instance. Response was:"
  echo "$CREATE_RESP"
  exit 1
fi

echo "Successfully created instance_id=$INSTANCE_ID"


########################################
# UPDATE SLOT_TYPE (/update_instance)
########################################
UPDATE_BODY=$(jq -nc --argjson instance_id "$INSTANCE_ID" --arg slot_type "$SLOT_TYPE" '
{
  "instance_id": $instance_id,
  "slot_type": $slot_type
}
')

echo
echo "Updating instance $INSTANCE_ID => slot_type=$SLOT_TYPE"
UPDATE_RESP=$(
  curl -s -X POST \
    -H "Content-Type: application/json" \
    -H "$AUTH_HEADER" \
    "$DB_URL/update_instance" \
    -d "$UPDATE_BODY"
)

OK=$(echo "$UPDATE_RESP" | jq -r '.success // empty')
if [ "$OK" != "true" ]; then
  echo "[ERROR] Failed to update slot_type. Response:"
  echo "$UPDATE_RESP"
  exit 1
fi

echo
echo "Instance $INSTANCE_ID is now slot_type=$SLOT_TYPE"
echo "All done!"
