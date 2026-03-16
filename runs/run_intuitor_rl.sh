#!/bin/bash
set -euo pipefail

export NANOCHAT_BASE_DIR=/opt/dlami/nvme/gpt1905_training

echo "=== [$(date)] Starting Intuitor RL v1 (self-certainty reward, no API calls for training) ==="

# Intuitor RL: Self-certainty reward for pre-1900 physics reasoning
# No gold answers or API calls needed for training — pure intrinsic reward
# Claude judge still used for validation eval to track actual correctness
#
# Chain: base d34 -> physics CLM expanded -> Intuitor RL

torchrun --standalone --nproc_per_node=8 \
    -m scripts.pre1900_scripts.intuitor_rl -- \
    --run=intuitor-v1 \
    --model-tag d34 \
    --checkpoints-dir physicssft_expanded_checkpoints \
    --reward-mode self-certainty \
    --num-samples 8 \
    --examples-per-step 8 \
    --max-new-tokens 512 \
    --temperature 0.9 \
    --train-data instruct_data/intuitor_prompts/intuitor_prompts_sys_train.jsonl \
    --val-data instruct_data/intuitor_prompts/intuitor_prompts_sys_val.jsonl \
    --problems-val-data instruct_data/intuitor_prompts/intuitor_problems_val.jsonl \
    --eval-every 30 \
    --save-every 60 \
    --output-dir pre1900_intuitor_rl_checkpoints

echo "=== [$(date)] Intuitor RL v1 complete ==="
