#!/usr/bin/env bash
#
# Setup script demonstrating how to create:
#   1) DB instance (service_type="Db")
#   2) A permission description (perm_type="ModifyDb")
#   3) A permission linking an address to the newly created perm_description
#
# Expects the DB server at http://localhost:5432 and an env var $PRIVATE_KEY.
#
# Requires Python + eth_account for the signing step. 
# (pip install eth-account)

if [ -z "${PRIVATE_KEY}" ]; then
  echo "ERROR: PRIVATE_KEY environment variable not set."
  exit 1
fi

DB_URL="http://localhost:5432"

###############################################################################
# A small helper function that:
#  - Builds the JSON "message" dictionary
#  - Signs it with PRIVATE_KEY
#  - Prints out the combined "messageJSON:signatureHex"
#
# Usage:  auth_header=$(build_auth "/create_instance" '{"foo":"bar"}')
# Then pass it as a curl header: -H "Authorization: $auth_header"
###############################################################################
build_auth() {
  local endpoint="$1"
  local data_json="$2"

  # We embed a small Python snippet to do the signing.
  # This snippet:
  #   - imports eth_account
  #   - creates the message dict with endpoint, nonce, timestamp, data
  #   - signs it with PRIVATE_KEY from env
  #   - prints "messageJSON:signatureHex"
  python <<EOF
import os, json, time, uuid
from eth_account import Account
from eth_account.messages import encode_defunct

priv = os.environ["PRIVATE_KEY"]
endpoint = "$endpoint"

# If data_json is empty, we pass an empty dict, else parse it
data_str = '''$data_json'''
if not data_str.strip():
    data_dict = {}
else:
    data_dict = json.loads(data_str)

nonce = str(uuid.uuid4())
timestamp = int(time.time())

message_dict = {
    "endpoint": endpoint,
    "nonce": nonce,
    "timestamp": timestamp,
    "data": data_dict
}
message_json = json.dumps(message_dict, sort_keys=True)

# Sign
acct = Account.from_key(priv)
msg_defunct = encode_defunct(text=message_json)
signed = acct.sign_message(msg_defunct)
sig = signed.signature.hex()

# Print final "Authorization" value for curl =>  messageJSON:signatureHex
print(f"{message_json}:{sig}")
EOF
}

###############################################################################
# 1) Create DB instance
#    Data payload: name="db", service_type="Db", job_id=null, ...
###############################################################################
create_instance_data='{
  "name": "db",
  "service_type": "Db",
  "job_id": null,
  "private_key": "'${PRIVATE_KEY}'",
  "pod_id": "",
  "process_id": 0
}'

auth1=$(build_auth "/create_instance" "$create_instance_data")

echo "==== Creating DB instance ===="
curl -s -X POST "${DB_URL}/create_instance" \
  -H "Content-Type: application/json" \
  -H "Authorization: ${auth1}" \
  -d "${create_instance_data}"
echo -e "\n"

###############################################################################
# 2) Create DB permission description (perm_type="ModifyDb")
###############################################################################
create_perm_desc_data='{
  "perm_type": "ModifyDb",
  "restricted_sot_id": null
}'

auth2=$(build_auth "/create_perm_description" "$create_perm_desc_data")

echo "==== Creating DB permission description ===="
curl -s -X POST "${DB_URL}/create_perm_description" \
  -H "Content-Type: application/json" \
  -H "Authorization: ${auth2}" \
  -d "${create_perm_desc_data}"
echo -e "\n"

###############################################################################
# 3) Create a permission for the address + the perm_description_id you got above
#    For demonstration, we pass perm=1, but in reality you might parse
#    the JSON returned by create_perm_description to get the correct ID.
###############################################################################
create_perm_data='{
  "address": "0xROOT_ADDRESS",
  "perm": 1
}'

auth3=$(build_auth "/create_perm" "$create_perm_data")

echo "==== Creating Permission linking address to perm=1 ===="
curl -s -X POST "${DB_URL}/create_perm" \
  -H "Content-Type: application/json" \
  -H "Authorization: ${auth3}" \
  -d "${create_perm_data}"
echo -e "\n"

echo "Done."
