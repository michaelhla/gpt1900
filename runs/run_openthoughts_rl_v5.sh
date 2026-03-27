#!/bin/bash
set -euo pipefail

export NANOCHAT_BASE_DIR=/opt/dlami/nvme/gpt1900_openthoughts
export PYTHONPATH=/home/ubuntu/gpt1900
export OMP_NUM_THREADS=1
# Set these env vars before running, or use a .env file
# export WANDB_API_KEY=...
# export ANTHROPIC_API_KEY=...

echo "=== [$(date)] Starting OpenThoughts RLVR v5 (KL penalty + reduced format reward) ==="

# v5: KL penalty β=0.05, format reward 0.1 (down from 0.3), device-batch-size 1
# Starting from format-SFT checkpoint (same as v4 starting point)
# 1,800 problems / 8 per step = 225 steps per epoch, 10 epochs = 2,250 steps
torchrun --standalone --nproc_per_node=8 -m scripts.pre1900_scripts.verifiable_rl -- \
    --model-tag d34 \
    --checkpoints-dir pre1900_openthoughts_format_sft_checkpoints \
    --output-dir pre1900_openthoughts_rl_v5_checkpoints \
    --device-batch-size 1 \
    --num-samples 8 \
    --examples-per-step 8 \
    --max-new-tokens 2048 \
    --num-epochs 10 \
    --kl-coeff 0.05 \
    --train-data instruct_data/openthoughts/rl_prompts_train.jsonl \
    --val-data instruct_data/openthoughts/rl_prompts_val.jsonl \
    --problems-data instruct_data/openthoughts/rl_problems_train.jsonl \
    --problems-val-data instruct_data/openthoughts/rl_problems_val.jsonl \
    --eval-every 225 --save-every 225 \
    --run=pre1900_openthoughts_rl_d34_v5

echo "=== [$(date)] OpenThoughts RLVR v5 complete ==="
