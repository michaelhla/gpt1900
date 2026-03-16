#!/bin/bash

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR=/opt/dlami/nvme/gpt1915_7b_training
export PYTHONPATH=/root/gpt1900
export HF_TOKEN=${HF_TOKEN:?"Set HF_TOKEN env var"}

# Download 1915 corpus from HF if not already present
DATA_DIR=$NANOCHAT_BASE_DIR/base_data
mkdir -p $DATA_DIR
EXISTING=$(ls $DATA_DIR/shard_*.parquet 2>/dev/null | wc -l)
if [ "$EXISTING" -lt 1 ]; then
    echo "Downloading pre1915 corpus ($EXISTING shards present)..."
    huggingface-cli download mhla/pre1915-corpus \
        --repo-type dataset \
        --include "shard_*.parquet" \
        --local-dir $DATA_DIR
fi

# Copy tokenizer from existing nanochat cache
cp -r /root/.cache/nanochat/tokenizer $NANOCHAT_BASE_DIR/tokenizer 2>/dev/null

cd /root/gpt1900
source .venv/bin/activate

# 7B model: depth=46, n_embd=2944, 23 heads, 46 layers
# FSDP2 (ZeRO-3) shards params/grads/optimizer across GPUs
# FP8 + activation checkpointing for memory efficiency
# AdamW optimizer (Muon incompatible with FSDP's sharded params)
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --depth=46 --target-param-data-ratio=11.25 --device-batch-size=2 \
    --fsdp \
    --run=pre1915_7b_fsdp --save-every=2000 --window-pattern L \
    --core-metric-every=-1 --sample-every=-1 \
    --total-batch-size=524288
