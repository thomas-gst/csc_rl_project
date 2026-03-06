#!/bin/bash

USER="thomas.gastellu"
BASE_PORT=8001

SERVERS=(
maserati 
ferrari  
porsche
)

for i in "${!SERVERS[@]}"; do
    SERVER=${SERVERS[$i]}
    LOCAL_PORT=$((BASE_PORT+i))

    echo "$SERVER → localhost:$LOCAL_PORT"

    ssh -f -L ${LOCAL_PORT}:localhost:8000 ${USER}@${SERVER}.polytechnique.fr \
    "cd ~/venvs/csc_rl/csc_rl_project/pokemon-showdown && node pokemon-showdown start --no-security"
done