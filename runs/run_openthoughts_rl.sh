#!/bin/bash
set -euo pipefail

export NANOCHAT_BASE_DIR=/opt/dlami/nvme/gpt1900_openthoughts

echo "=== [$(date)] Starting OpenThoughts RLVR pipeline ==="

# Verifiable RL with SymPy equivalence rewards on 1800 OpenThoughts math+science problems
# 1,800 problems / 8 per step = 225 steps per epoch.
echo "Verifiable RL (SymPy equivalence + 0.3 format bonus)..."
torchrun --standalone --nproc_per_node=8 -m scripts.pre1900_scripts.verifiable_rl -- \
    --model-tag d34 \
    --checkpoints-dir pre1900_openthoughts_sft_checkpoints \
    --output-dir pre1900_openthoughts_rl_checkpoints \
    --device-batch-size 2 \
    --num-samples 8 \
    --examples-per-step 8 \
    --max-new-tokens 1024 \
    --num-epochs 10 \
    --train-data instruct_data/openthoughts/rl_prompts_train.jsonl \
    --val-data instruct_data/openthoughts/rl_prompts_val.jsonl \
    --problems-data instruct_data/openthoughts/rl_problems_train.jsonl \
    --problems-val-data instruct_data/openthoughts/rl_problems_val.jsonl \
    --eval-every 225 --save-every 225 \
    --run=pre1900_openthoughts_rl_d34

echo "=== [$(date)] OpenThoughts RLVR complete ==="
