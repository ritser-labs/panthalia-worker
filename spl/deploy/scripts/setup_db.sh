#!/usr/bin/env bash
#
# Example usage:
#   1) [Optional] rm -f /path/to/sqlite.db  # or wipe your DB some other way
#   2) Ensure DB server is running on http://localhost:5432
#   3) export PRIVATE_KEY="0xYOUR_PRIVATE_KEY"
#   4) ./setup_db.sh

DB_URL="http://localhost:5432"

# OPTIONAL: If you want to forcibly remove local sqlite.db each run, un-comment:
# DB_PATH="/path/to/sqlite.db"
# echo "Removing old DB file at $DB_PATH"
# rm -f "$DB_PATH"

if [ -z "${PRIVATE_KEY}" ]; then
  echo "ERROR: PRIVATE_KEY environment variable not set."
  exit 1
fi

###############################################################################
# Helper function to build the Panthalia auth header:
#   "Authorization: <messageJSON>:<signatureHex>"
###############################################################################
build_auth() {
  local endpoint="$1"
  local data_json="$2"

  python <<EOF
import os, json, time, uuid
from eth_account import Account
from eth_account.messages import encode_defunct

priv = os.environ["PRIVATE_KEY"]
endpoint = "$endpoint"

data_str = '''$data_json'''
data_dict = json.loads(data_str) if data_str.strip() else {}

nonce = str(uuid.uuid4())
timestamp = int(time.time())

message_dict = {
    "endpoint": endpoint,
    "nonce": nonce,
    "timestamp": timestamp,
    "data": data_dict
}
message_json = json.dumps(message_dict, sort_keys=True)

acct = Account.from_key(priv)
msg_defunct = encode_defunct(text=message_json)
signed = acct.sign_message(msg_defunct)
sig = signed.signature.hex()

print(f"{message_json}:{sig}")
EOF
}


echo "==== 1) Create a Subnet ===="
subnet_data='{
  "dispute_period": 300,
  "solve_period": 1200,
  "stake_multiplier": 10
}'
auth_subnet=$(build_auth "/create_subnet" "$subnet_data")

subnet_resp=$(curl -s -X POST "${DB_URL}/create_subnet" \
  -H "Content-Type: application/json" \
  -H "Authorization: ${auth_subnet}" \
  -d "${subnet_data}")

echo "CreateSubnet response: $subnet_resp"

# If you want to parse out the 'subnet_id':
subnet_id=$(echo "$subnet_resp" | jq -r '.subnet_id // empty')
echo "Parsed subnet_id: $subnet_id"
echo


echo "==== 2) Create DB instance (service_type=Db) ===="
create_instance_data='{
  "name": "db",
  "service_type": "Db",
  "job_id": null,
  "private_key": "'${PRIVATE_KEY}'",
  "pod_id": "",
  "process_id": 0
}'
auth1=$(build_auth "/create_instance" "$create_instance_data")
inst_resp=$(curl -s -X POST "${DB_URL}/create_instance" \
  -H "Content-Type: application/json" \
  -H "Authorization: ${auth1}" \
  -d "${create_instance_data}")
echo "Instance creation response: $inst_resp"
echo


echo "==== 3) Create DB permission description for ModifyDb ===="
create_perm_desc_data='{
  "perm_type": "ModifyDb",
  "restricted_sot_id": null
}'
auth2=$(build_auth "/create_perm_description" "$create_perm_desc_data")
perm_desc_resp=$(curl -s -X POST "${DB_URL}/create_perm_description" \
  -H "Content-Type: application/json" \
  -H "Authorization: ${auth2}" \
  -d "${create_perm_desc_data}")
echo "PermDescription response: $perm_desc_resp"

# Possibly parse out 'perm_description_id' from the JSON:
perm_desc_id=$(echo "$perm_desc_resp" | jq -r '.perm_description_id // empty')
echo "Parsed perm_description_id: $perm_desc_id"
echo


echo "==== 4) Create Permission linking address to perm=1 (or $perm_desc_id) ===="
# If we want to use the newly created ID, do: "perm": $perm_desc_id
# For now, let's default to "1" if not found.
perm_id="${perm_desc_id:-1}"

create_perm_data='{
  "address": "0xROOT_ADDRESS",
  "perm": '"${perm_id}"'
}'
auth3=$(build_auth "/create_perm" "$create_perm_data")
perm_resp=$(curl -s -X POST "${DB_URL}/create_perm" \
  -H "Content-Type: application/json" \
  -H "Authorization: ${auth3}" \
  -d "${create_perm_data}")
echo "CreatePerm response: $perm_resp"
echo


echo "Done."
