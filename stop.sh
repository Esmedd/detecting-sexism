#!/bin/bash

docker container stop $(docker container ls -aq)
echo *** Docker compose down ***

# hard reset :
docker system prune -a #--filter "label!=dontPrune"
