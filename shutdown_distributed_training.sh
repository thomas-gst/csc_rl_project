#!/bin/bash

# Shutdown script for distributed training

USER="thomas.gastellu"
HEAD_NODE="ferrari"
WORKER_NODES=("maserati" "porsche")

echo "=== Shutting Down Distributed Training ==="

# Stop Ray cluster
echo ""
echo "Stopping Ray cluster..."
for node in "${HEAD_NODE}" "${WORKER_NODES[@]}"; do
    echo "  Stopping Ray on ${node}..."
    ssh ${USER}@${node}.polytechnique.fr "ray stop" || true
done

# Kill Pokemon Showdown servers
echo ""
echo "Killing Pokemon Showdown servers..."
bash server_utils/kill_multi_server.sh

echo ""
echo "=== Shutdown Complete ==="
