#!/bin/bash
# =============================================================================
# Cloud-agnostic environment setup
# =============================================================================
# Source this from any run script to set up the environment correctly
# regardless of whether you're on AWS, RunPod, or a SLURM cluster.
#
# Usage (at the top of any run script):
#   source runs/cloud/env_setup.sh

# Detect environment
if [ -n "${SLURM_JOB_ID:-}" ]; then
    # SLURM cluster (existing ESM setup)
    RUNTIME_ENV="slurm"
    export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-/mnt/main0/data/$(whoami)/gpt1900_training}"
    export PYTHONPATH="${PYTHONPATH:-$(pwd)}"
elif [ -d "/workspace" ] && [ -f "/etc/runpod.conf" ] 2>/dev/null; then
    # RunPod
    RUNTIME_ENV="runpod"
    export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-/workspace/gpt1900_data}"
elif curl -s --connect-timeout 1 http://169.254.169.254/latest/meta-data/ &>/dev/null; then
    # AWS EC2 (metadata endpoint available)
    RUNTIME_ENV="aws"
    export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-/opt/gpt1900_data}"
else
    # Local / unknown
    RUNTIME_ENV="local"
    export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
fi

export OMP_NUM_THREADS=1
mkdir -p "$NANOCHAT_BASE_DIR"

# Activate venv if not already active
if [ -z "${VIRTUAL_ENV:-}" ]; then
    if [ -d ".venv" ]; then
        source .venv/bin/activate
    fi
fi

echo "[env] Runtime: $RUNTIME_ENV | Base dir: $NANOCHAT_BASE_DIR"
