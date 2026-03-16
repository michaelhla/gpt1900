#!/bin/bash
set -euo pipefail

export NANOCHAT_BASE_DIR=/opt/dlami/nvme/gpt1905_training

echo "=== [$(date)] Starting contradiction RL v10 pipeline (EMA coherence+logical, expanded physics SFT base) ==="

# Step 1: Download expanded physics SFT checkpoint
echo "Step 1: Downloading expanded physics SFT checkpoint..."
huggingface-cli download mhla/gpt1900-d34-physicssft-expanded \
    --local-dir $NANOCHAT_BASE_DIR/physicssft_expanded_checkpoints/d34

# Step 2: Discovery RL v10 — EMA coherence with logical flow emphasis
echo "Step 2: Discovery RL v10 (EMA coherence+logical, expanded physics SFT base, 24 epochs)..."
torchrun --standalone --nproc_per_node=8 -m scripts.pre1900_scripts.discovery_rl -- \
    --model-tag d34 \
    --checkpoints-dir physicssft_expanded_checkpoints \
    --output-dir /opt/dlami/nvme/gpt1905_training/pre1900_discovery_rl_v10_checkpoints \
    --no-scaffold \
    --coherence-reward \
    --coherence-style logical \
    --ema-alpha 0.05 \
    --min-coherence-weight 0.1 \
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
    --run=pre1900_discovery_rl_v10_d34

echo "=== [$(date)] Contradiction RL v10 pipeline complete ==="
