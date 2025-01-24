#!/usr/bin/env bash
#
# delete_all_sot_worker.sh
#
# 1) Interactively prompts for DB_URL + private key
# 2) GET /get_all_instances to find slot_type in ["SOT", "WORKER"]
# 3) Confirm user wants to delete them
# 4) For each matching instance, calls POST /delete_instance with ECDSA-signed Authorization header
#
# Requires Python 3 + "pip install eth_account" for signing.

set -euo pipefail

########################################
# PYTHON SIGNING FUNCTION (inline)
########################################
# This function:
#   - Builds the JSON message
#   - Signs it using the Ethereum private key
#   - Prints "<messageJSON>:<signatureHex>"
sign_request() {
  local endpoint="$1"
  local data_json="$2"
  local priv_key="$3"

  python3 <<EOF
import sys, json, time, uuid
from eth_account.messages import encode_defunct
from eth_account import Account

endpoint = sys.argv[1]
raw_data  = sys.argv[2]
private_key = sys.argv[3]

try:
    data_obj = json.loads(raw_data)
except:
    data_obj = {}

nonce = str(uuid.uuid4())
timestamp = int(time.time())

msgDict = {
    "endpoint": endpoint,
    "nonce": nonce,
    "timestamp": timestamp,
    "data": data_obj
}
text = json.dumps(msgDict, sort_keys=True)
message_def = encode_defunct(text=text)
signed = Account.from_key(private_key).sign_message(message_def)

# Print the final "Authorization" value
print(f"{text}:{signed.signature.hex()}")
EOF
}


########################################
# PROMPT USER FOR DB_URL, PRIVATE_KEY
########################################
read -r -p "Enter DB URL [default: http://localhost:8000]: " DB_URL
DB_URL="${DB_URL:-http://localhost:8000}"

read -r -p "Enter your Ethereum private key (0xABC...): " PRIVATE_KEY
if [ -z "$PRIVATE_KEY" ]; then
  echo "Error: private key is required."
  exit 1
fi

########################################
# FETCH ALL INSTANCES
########################################
# GET /get_all_instances typically doesn't require auth in your code,
# but let's sign it in case your server DOES check. We do an empty data "{}".

GET_ALL_ENDPOINT="/get_all_instances"
EMPTY_DATA="{}"

AUTH_HEADER_GET=$(sign_request "$GET_ALL_ENDPOINT" "$EMPTY_DATA" "$PRIVATE_KEY")

echo "Fetching all instances from: $DB_URL$GET_ALL_ENDPOINT"

ALL_INSTANCES=$(
  curl -s -X GET \
    -H "Authorization: $AUTH_HEADER_GET" \
    "$DB_URL/get_all_instances"
)

# Should be a JSON array of instance objects
# Let's parse with jq
# We'll filter those that have .slot_type == "SOT" or "WORKER"

MATCHING_IDS=$(echo "$ALL_INSTANCES" | jq -r '
  map(select(.slot_type == "SOT" or .slot_type == "WORKER")) | map(.id) | .[]
  ' 2>/dev/null || true
)

if [ -z "$MATCHING_IDS" ]; then
  echo "No instances found with slot_type = SOT or WORKER. Nothing to delete."
  exit 0
fi

# Count them
COUNT=$(echo "$MATCHING_IDS" | wc -l | tr -d '[:space:]')
echo "Found $COUNT instance(s) with slot_type in [SOT, WORKER]:"
echo "$MATCHING_IDS"

# Confirm
read -r -p "Are you sure you want to DELETE ALL above? [y/N]: " CONFIRM
if [[ "$CONFIRM" != "y" && "$CONFIRM" != "Y" ]]; then
  echo "Aborted."
  exit 0
fi

########################################
# DELETE EACH MATCHING INSTANCE
########################################
DELETED=0
FAILED=0

while read -r INST_ID; do
  [ -z "$INST_ID" ] && continue

  # Build the JSON body: { "instance_id": 123 }
  BODY=$(jq -nc --argjson instance_id "$INST_ID" '{instance_id: $instance_id}')

  # sign => endpoint = "/delete_instance", data = BODY
  AUTH_HEADER_DEL=$(sign_request "/delete_instance" "$BODY" "$PRIVATE_KEY")

  echo "Deleting instance_id=$INST_ID ..."
  DEL_RESP=$(
    curl -s -X POST \
      -H "Content-Type: application/json" \
      -H "Authorization: $AUTH_HEADER_DEL" \
      "$DB_URL/delete_instance" \
      -d "$BODY"
  )

  OK=$(echo "$DEL_RESP" | jq -r '.success // empty')
  if [ "$OK" = "true" ]; then
    echo "  -> Deleted."
    ((DELETED++))
  else
    echo "  -> Failed. Response: $DEL_RESP"
    ((FAILED++))
  fi

done <<< "$MATCHING_IDS"

echo
echo "Done. Deleted $DELETED, failed $FAILED."
exit 0
