#!/bin/bash
set -euo pipefail

echo "=== [$(date)] Starting contradiction RL v7 pipeline (no scaffold, fixed coherence) ==="

# Step 1: Download physics-pretrained base
echo "Step 1: Downloading physics SFT checkpoint..."
huggingface-cli download mhla/gpt1900-d34-physics-sft \
    --local-dir $NANOCHAT_BASE_DIR/physics_clm_checkpoints/d34

# Step 2: Discovery RL v7 — fixed coherence weighting (no EMA curriculum)
echo "Step 2: Discovery RL v7 (no scaffold, fixed coherence weight)..."
torchrun --standalone --nproc_per_node=8 -m scripts.pre1900_scripts.discovery_rl -- \
    --model-tag d34 \
    --checkpoints-dir physics_clm_checkpoints \
    --output-dir /opt/dlami/nvme/gpt1905_training/pre1900_discovery_rl_v7_checkpoints \
    --no-scaffold \
    --coherence-reward \
    --fixed-coherence-weight 0.1 \
    --device-batch-size 2 \
    --num-samples 4 \
    --examples-per-step 8 \
    --max-new-tokens 2048 \
    --num-epochs 12 \
    --judge-style contradiction \
    --train-data instruct_data/contradiction_problems/contradiction_prompts_sys_train.jsonl \
    --val-data instruct_data/contradiction_problems/contradiction_prompts_sys_val.jsonl \
    --problems-data instruct_data/contradiction_problems/contradiction_problems_train.jsonl \
    --problems-val-data instruct_data/contradiction_problems/contradiction_problems_val.jsonl \
    --eval-every 15 --save-every 35 \
    --run=pre1900_discovery_rl_v7_d34

echo "=== [$(date)] Contradiction RL v7 pipeline complete ==="
