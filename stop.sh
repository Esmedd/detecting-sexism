#!/bin/bash

docker container stop $(docker container ls -aq)
echo *** Docker compose down ***

# hard reset :
docker system prune -a #--filter "label!=dontPrune"

# docker build -t yns .
# docker run -e PORT=8000 -p 8000:8000 --env-file .env yns
#  docker build -t europe-west9-docker.pkg.dev/youre-not-sexist/yns/image_yns2:prod .
# docker push europe-west9-docker.pkg.dev/youre-not-sexist/yns/image_yns2:prod
# docker push europe-west9-docker.pkg.dev/youre-not-sexist/yns/image_yns:prod
# GAR_IMAGE
# europe-west9-docker.pkg.dev/youre-not-sexist/yns
# gcloud auth configure-docker europe-west9-docker.pkg.dev
