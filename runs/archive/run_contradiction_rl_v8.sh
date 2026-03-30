#!/bin/bash
set -euo pipefail

export NANOCHAT_BASE_DIR=/opt/dlami/nvme/gpt1905_training

echo "=== [$(date)] Starting contradiction RL v8 pipeline (Bedrock, empty→0, coherence 0.25) ==="

# Step 1: Download physics-pretrained base
echo "Step 1: Downloading physics SFT checkpoint..."
huggingface-cli download mhla/gpt1900-d34-physics-sft \
    --local-dir $NANOCHAT_BASE_DIR/physics_clm_checkpoints/d34

# Step 2: Discovery RL v8 — Bedrock API, empty responses score 0, coherence weight 0.25
echo "Step 2: Discovery RL v8..."
torchrun --standalone --nproc_per_node=8 -m scripts.pre1900_scripts.discovery_rl -- \
    --model-tag d34 \
    --checkpoints-dir physics_clm_checkpoints \
    --output-dir /opt/dlami/nvme/gpt1905_training/pre1900_discovery_rl_v8_checkpoints \
    --no-scaffold \
    --coherence-reward \
    --fixed-coherence-weight 0.25 \
    --device-batch-size 2 \
    --num-samples 4 \
    --examples-per-step 8 \
    --max-new-tokens 2048 \
    --num-epochs 24 \
    --judge-style contradiction \
    --train-data instruct_data/contradiction_problems/contradiction_prompts_sys_train.jsonl \
    --val-data instruct_data/contradiction_problems/contradiction_prompts_sys_val.jsonl \
    --problems-data instruct_data/contradiction_problems/contradiction_problems_train.jsonl \
    --problems-val-data instruct_data/contradiction_problems/contradiction_problems_val.jsonl \
    --eval-every 15 --save-every 35 \
    --run=pre1900_discovery_rl_v8_d34

echo "=== [$(date)] Contradiction RL v8 pipeline complete ==="
