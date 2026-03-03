#!/bin/bash

# ==== Configuration ====
REMOTE_USER="thomas.gastellu"
REMOTE_HOST="volvo.polytechnique.fr"        
REMOTE_DIR="~/venvs/csc_rl/csc_rl_project"
LOCAL_DIR="/Users/thomasgastellu/Documents/Obsidian Vault/Cours/3A/P2/CSC_RL/csc_rl_project"           
SCRIPT_NAME=""    



# ==== Sync Local Project to Remote ====
echo "Syncing local files to remote..."
rsync -avz --exclude-from='utils/.rsyncignore'  "$LOCAL_DIR/" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR"
