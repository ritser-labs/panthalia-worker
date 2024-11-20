panthalia repo

needs pytorch, docker and nvidia container toolkit

run ./build_image.sh in spl/ to build the docker image used for isolating the plugin environment

needs env variables:
- `CLOUD_KEY` (if using cloud deployment)
- `SUPERTOKENS_CONNECTION_URI`
- `SUPERTOKENS_API_KEY`

https://arxiv.org/abs/2403.07816
https://arxiv.org/abs/2312.07987
https://arxiv.org/abs/2110.07431