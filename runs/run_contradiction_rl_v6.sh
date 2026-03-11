#!/bin/bash
set -euo pipefail

echo "=== [$(date)] Starting contradiction RL v6 pipeline (no scaffold) ==="

# Step 1: Download physics-pretrained base
echo "Step 1: Downloading physics SFT checkpoint..."
huggingface-cli download mhla/gpt1900-d34-physics-sft \
    --local-dir $NANOCHAT_BASE_DIR/physics_clm_checkpoints/d34

# Step 2: Discovery RL v6 — directly on physics SFT, no reasoning warmstart
echo "Step 2: Discovery RL v6 (no scaffold, coherence curriculum)..."
torchrun --standalone --nproc_per_node=8 -m scripts.pre1900_scripts.discovery_rl -- \
    --model-tag d34 \
    --checkpoints-dir physics_clm_checkpoints \
    --output-dir /opt/dlami/nvme/gpt1905_training/pre1900_discovery_rl_v6_checkpoints \
    --no-scaffold \
    --coherence-reward \
    --ema-alpha 0.05 \
    --min-coherence-weight 0.1 \
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
    --run=pre1900_discovery_rl_v6_d34

echo "=== [$(date)] Contradiction RL v6 pipeline complete ==="
