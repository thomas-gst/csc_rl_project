#!/bin/bash

# Launch Pokemon Showdown servers on remote machines for distributed Ray access
# Servers will listen on 0.0.0.0 (all interfaces) so Ray workers can connect

USER="thomas.gastellu"
PORT=8000

SERVERS=(
maserati
ferrari
porsche
)

for SERVER in "${SERVERS[@]}"; do
    echo "Starting server on ${SERVER}.polytechnique.fr:${PORT}"
    
    ssh ${USER}@${SERVER}.polytechnique.fr \
    "cd ~/venvs/csc_rl/csc_rl_project/pokemon-showdown && \
     nohup node pokemon-showdown start --no-security --host 0.0.0.0 --port ${PORT} > /tmp/showdown_${SERVER}.log 2>&1 &"
done

echo "Servers launched. Use one of these hosts in your config:"
for SERVER in "${SERVERS[@]}"; do
    echo "  - ${SERVER}.polytechnique.fr"
done
