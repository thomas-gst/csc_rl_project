#!/bin/bash
ssh -L 8001:localhost:8000 thomas.gastellu@maserati.polytechnique.fr \
"cd ~/venvs/csc_rl/csc_rl_project/pokemon-showdown && node pokemon-showdown start --no-security"