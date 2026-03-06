#!/bin/bash

# Kill Pokemon Showdown servers on remote machines

USER="thomas.gastellu"

SERVERS=(
maserati
ferrari
porsche
)

for SERVER in "${SERVERS[@]}"; do
    echo "Killing server on ${SERVER}.polytechnique.fr"
    
    ssh ${USER}@${SERVER}.polytechnique.fr \
    "pkill -f 'node pokemon-showdown'"
done

echo "All distributed servers killed."
