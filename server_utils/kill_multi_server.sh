#!/bin/bash

USER="thomas.gastellu"
BASE_PORT=8001

SERVERS=(
maserati
ferrari
lamborghini
porsche
)

echo "Stopping remote showdown servers..."

for SERVER in "${SERVERS[@]}"; do
    echo "Stopping $SERVER"
    ssh ${USER}@${SERVER}.polytechnique.fr "pkill -f pokemon-showdown"
done

echo "Stopping local tunnels..."

i=0
for SERVER in "${SERVERS[@]}"; do
    PORT=$((BASE_PORT+i))
    echo "Closing tunnel on localhost:$PORT"
    lsof -ti tcp:$PORT | xargs -r kill
    ((i++))
done

echo "Done."