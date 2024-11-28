panthalia repo

needs pytorch, docker and nvidia container toolkit

run ./build_image.sh in spl/ to build the docker image used for isolating the plugin environment

needs env variables:
- `CLOUD_KEY` (if using cloud deployment)
- `PANTHALIA_AUTH0_DOMAIN`
- `PANTHALIA_AUTH0_CLIENT_ID`
- `PANTHALIA_AUTH0_AUDIENCE`


https://arxiv.org/abs/2403.07816
https://arxiv.org/abs/2312.07987
https://arxiv.org/abs/2110.07431