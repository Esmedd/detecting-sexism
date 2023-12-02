#!/bin/bash

docker container stop $(docker container ls -aq)
echo *** Docker compose down ***

# hard reset :
docker system prune -a #--filter "label!=dontPrune"

# docker build -t yns .
# docker run -e PORT=8000 -p 8000:8000 --env-file .env yns
