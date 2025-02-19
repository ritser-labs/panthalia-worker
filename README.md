panthalia repo

dependencies:
- pytorch
- docker
- nvidia container toolkit

libgl1-mesa-glx for qt

run ./build_image.sh in spl/ to build the docker image used for isolating the plugin environment

needs env variables:
- `CLOUD_KEY` (if using cloud deployment)

for db:
- `PANTHALIA_AUTH0_DOMAIN`
- `PANTHALIA_AUTH0_CLIENT_ID`
- `PANTHALIA_AUTH0_AUDIENCE`
- `STRIPE_SECRET_KEY`
- `STRIPE_WEBHOOK_SECRET`
- `STRIPE_SUCCESS_URL`
- `STRIPE_CANCEL_URL`
- `PANTHALIA_AUTH0_MGMT_CLIENT_ID`
- `PANTHALIA_AUTH0_MGMT_CLIENT_SECRET`

for master:
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `PANTHALIA_S3_BUCKET_NAME`
- `PANTHALIA_S3_FOLDER_NAME`

ensure that in auth0 dashboard the management api has delete:users permission (in the management api -> apis -> management api -> permissions)

run tests with `pytest spl/tests`

manually initialization guide:
ensure docker is set up (and current user is in docker group) and requirements.txt is installed
1. `python -m spl.db.server --host localhost --port 5432 --perm 1 --root_wallet $PUBLIC_KEY`
2. setup_db.sh
3. `python -m spl.master --private_key $PRIVATE_KEY --db_url http://localhost:5432 --num_workers 1 --deploy_type local --torch_compile`

https://arxiv.org/abs/2403.07816
https://arxiv.org/abs/2312.07987
https://arxiv.org/abs/2110.07431