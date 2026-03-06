#!/bin/bash
set -e

# Configuration
USER="thomas.gastellu"
HEAD_NODE="ferrari"
WORKER_NODES=("maserati" "porsche")
RAY_PORT=6379
VENV_PATH="/users/eleves-b/2023/thomas.gastellu/venvs/csc_rl/.venv"
PROJECT_PATH="/users/eleves-b/2023/thomas.gastellu/venvs/csc_rl/csc_rl_project"

echo "=== Starting Distributed Training Setup ==="

# Step 1: Launch Pokemon Showdown servers
echo ""
echo "Step 1: Launching Pokemon Showdown servers..."
bash server_utils/launch_multi_server.sh
sleep 5

# Step 2: Stop any existing Ray instances
echo ""
echo "Step 2: Stopping any existing Ray instances..."
for node in "${HEAD_NODE}" "${WORKER_NODES[@]}"; do
    echo "  Stopping Ray on ${node}..."
    ssh ${USER}@${node}.polytechnique.fr "ray stop" || true
done
sleep 2

# Step 3: Start Ray head node
echo ""
echo "Step 3: Starting Ray head node on ${HEAD_NODE}..."
ssh ${USER}@${HEAD_NODE}.polytechnique.fr \
    "ray start --head --port=${RAY_PORT} --include-dashboard=false" &
sleep 10

# Get head node IP
HEAD_IP=$(ssh ${USER}@${HEAD_NODE}.polytechnique.fr "hostname -I | awk '{print \$1}'")
echo "  Head node IP: ${HEAD_IP}"

# Step 4: Start Ray worker nodes
echo ""
echo "Step 4: Starting Ray worker nodes..."
for node in "${WORKER_NODES[@]}"; do
    echo "  Starting worker on ${node}..."
    ssh ${USER}@${node}.polytechnique.fr \
        "ray start --address=${HEAD_IP}:${RAY_PORT}" &
done
sleep 5

# Step 5: Verify cluster
echo ""
echo "Step 5: Verifying Ray cluster status..."
ssh ${USER}@${HEAD_NODE}.polytechnique.fr "ray status"

# Step 6: Run distributed training
echo ""
echo "Step 6: Running distributed training..."
echo "  Connecting to ${HEAD_IP}:${RAY_PORT}"

ssh ${USER}@${HEAD_NODE}.polytechnique.fr \
    "cd ${PROJECT_PATH}/pokerl && \
     source ${VENV_PATH}/bin/activate && \
     export RAY_ADDRESS=${HEAD_IP}:${RAY_PORT} && \
     python distributed_train.py server=cluster distributed.env_log_level=ERROR"

echo ""
echo "=== Training Complete ==="
