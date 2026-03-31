#!/bin/bash
set -euo pipefail
export NANOCHAT_BASE_DIR=/opt/dlami/nvme/gpt1905_training
export HF_HOME="${NANOCHAT_BASE_DIR}/huggingface"

# Download base model if not already present
BASE_CKPT_DIR="${NANOCHAT_BASE_DIR}/base_checkpoints/d34"
if [ -z "$(find "$BASE_CKPT_DIR" -name 'model_*.pt' 2>/dev/null | head -1)" ]; then
    echo "Downloading base model mhla/gpt1900-d34-22btok..."
    mkdir -p "$BASE_CKPT_DIR"
    /home/ubuntu/gpt1900/.venv/bin/huggingface-cli download mhla/gpt1900-d34-22btok --local-dir "$BASE_CKPT_DIR" --exclude "optim_*"
fi

# Run SFT — clean corpus (predictions, race, politics, women's rights topics removed)
/home/ubuntu/gpt1900/.venv/bin/torchrun --standalone --nproc_per_node=8 -m scripts.pre1900_sft -- \
    --model-tag d34 \
    --output-dir /opt/dlami/nvme/gpt1905_training/v3_sft_checkpoints \
    --device-batch-size=4 \
    --total-batch-size=524288 \
    --max-seq-len=2048 \
    --train-data /home/ubuntu/gpt1900/data/instruct_data/v3_corpus_clean2/all_train.jsonl \
    --val-data /home/ubuntu/gpt1900/data/instruct_data/v3_corpus_clean2/all_val.jsonl \
    --eval-every=150 \
    --run=dummy
