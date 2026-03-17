# On lance plusieurs serveurs showdown sur des ports differents

#!/usr/bin/env bash
set -euo pipefail

# Define the ports
PORTS=(8000 8001 8002 8003 8004)
PS_DIR="./pokemon-showdown"
LOG_DIR="./logs/pokemon_showdown"
mkdir -p "$LOG_DIR"

pids=()

cleanup() {
  echo "Stopping servers..."
  for pid in "${pids[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
  wait || true
}
trap cleanup EXIT INT TERM

echo "Building Pokemon Showdown..."
(
  cd "$PS_DIR"
  node build
)

for port in "${PORTS[@]}"; do
  echo "Starting PS on :$port with auto-restart..."
  (
    # The loop keeps the port active even if a single instance crashes
    while true; do
      echo "[$(date)] Starting server on port $port" >> "$LOG_DIR/server_$port.log"
      
      # --max-old-space-size=4096 gives Node 4GB of RAM per server
      # --no-security bypasses login checks which is faster for local training
      node --max-old-space-size=4096 "$PS_DIR/pokemon-showdown" start --no-security --port "$port" \
        >> "$LOG_DIR/server_$port.out" 2>> "$LOG_DIR/server_$port.err" || true
      
      echo "Server on port $port exited. Restarting in 1s..."
      sleep 1
    done
  ) &
  pids+=("$!")
  sleep 1
done

echo "Servers running on: ${PORTS[*]}"
# wait keeps the script alive so the 'trap cleanup' can work when you Ctrl+C
wait