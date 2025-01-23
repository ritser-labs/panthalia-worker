#!/usr/bin/env bash

# Expecting an env var named PRIVATE_KEY, e.g.:
#   export PRIVATE_KEY="0xYOURPRIVATEKEY"
# or pass it inline:
#   PRIVATE_KEY="0xYourKey" ./setup_db.sh

if [ -z "${PRIVATE_KEY}" ]; then
  echo "ERROR: PRIVATE_KEY environment variable is not set."
  exit 1
fi

# The DB server base URL:
HOST="http://localhost:5432"

# 1) Create a DB instance (service_type="Db").
#    Set process_id to 0 (or whatever PID you have for the DB process).
#    Set job_id to null, pod_id to "", etc.

echo "==== Creating DB instance ===="
curl -s -X POST "${HOST}/create_instance" \
  -H "Content-Type: application/json" \
  -d "{
    \"name\": \"db\",
    \"service_type\": \"Db\",
    \"job_id\": null,
    \"private_key\": \"${PRIVATE_KEY}\",
    \"pod_id\": \"\",
    \"process_id\": 0
  }"
echo -e "\n"

# 2) Create a permission description for ModifyDb.
#    restricted_sot_id=null if it's a global DB permission.

echo "==== Creating DB permission description ===="
curl -s -X POST "${HOST}/create_perm_description" \
  -H "Content-Type: application/json" \
  -d '{
    "perm_type": "ModifyDb",
    "restricted_sot_id": null
  }'
echo -e "\n"

# 3) Create a permission entry linking an address to the perm_description ID (assumed to be 1).
#    If your DB returns a different perm_description_id from above,
#    replace "1" with the actual ID. Also replace 0xROOT_ADDRESS if desired.

echo "==== Creating a Permission linking PRIVATE_KEY address to perm ID 1 ===="
curl -s -X POST "${HOST}/create_perm" \
  -H "Content-Type: application/json" \
  -d '{
    "address": "0xROOT_ADDRESS",
    "perm": 1
  }'
echo -e "\n"

echo "Done."
