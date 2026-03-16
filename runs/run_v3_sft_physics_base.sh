#!/bin/bash
set -euo pipefail
export NANOCHAT_BASE_DIR=/opt/dlami/nvme/gpt1905_training

# SFT on safe corpus starting from the physics-expanded checkpoint
# Chain: base d34 -> physics CLM expanded (step 14400) -> v3 SFT safe -> RL
torchrun --standalone --nproc_per_node=8 -m scripts.pre1900_sft -- \
    --model-tag d34 \
    --checkpoints-dir physicssft_expanded_checkpoints \
    --output-dir /opt/dlami/nvme/gpt1905_training/v3_sft_physics_checkpoints \
    --device-batch-size=4 \
    --total-batch-size=524288 \
    --max-seq-len=2048 \
    --train-data /opt/dlami/nvme/gpt1905_training/instruct_data/v3_corpus_safe/all_train.jsonl \
    --val-data /opt/dlami/nvme/gpt1905_training/instruct_data/v3_corpus_safe/all_val.jsonl \
    --eval-every=150 \
    --run=v3_sft_physics_safe_d34
