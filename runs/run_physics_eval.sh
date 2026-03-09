#!/bin/bash
set -euo pipefail

# Physics reasoning evaluation pipeline for GPT-1900 checkpoints
# Run on an interactive RunPod H100 pod.
#
# Prerequisites:
#   export ANTHROPIC_API_KEY=sk-...
#   export HF_TOKEN=hf_...
#
# Launch pod:
#   python runs/interactive_pod.py --env HF_TOKEN ANTHROPIC_API_KEY --name gpt1900-physics-eval

if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
    echo "ERROR: ANTHROPIC_API_KEY is not set"
    exit 1
fi

pip install anthropic

python -m scripts.physics_eval --output-dir results/physics_eval
