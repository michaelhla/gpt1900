#!/bin/bash

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR=/opt/dlami/nvme/gpt1905_training
export PYTHONPATH=/root/gpt1900
export HF_TOKEN=${HF_TOKEN:?"Set HF_TOKEN env var"}

# Download 1905 corpus from HF if not already present
DATA_DIR=$NANOCHAT_BASE_DIR/base_data
mkdir -p $DATA_DIR
EXPECTED_SHARDS=445
EXISTING=$(ls $DATA_DIR/shard_*.parquet 2>/dev/null | wc -l)
if [ "$EXISTING" -lt "$EXPECTED_SHARDS" ]; then
    echo "Downloading pre1905 corpus ($EXISTING/$EXPECTED_SHARDS shards present)..."
    huggingface-cli download mhla/pre1905-corpus \
        --repo-type dataset \
        --include "shard_*.parquet" \
        --local-dir $DATA_DIR
fi

# Download checkpoint from HF if not already present
CKPT_DIR=$NANOCHAT_BASE_DIR/base_checkpoints/d34
mkdir -p $CKPT_DIR
if [ ! -f "$CKPT_DIR/model_003000.pt" ]; then
    echo "Downloading D34 1905 checkpoint..."
    huggingface-cli download mhla/gpt1905-d34 \
        model_003000.pt meta_003000.json \
        --local-dir $CKPT_DIR
fi

# Copy tokenizer from existing nanochat cache
cp -r /root/.cache/nanochat/tokenizer $NANOCHAT_BASE_DIR/tokenizer 2>/dev/null

cd /root/gpt1900
source .venv/bin/activate

torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --depth=34 --target-param-data-ratio=20 --device-batch-size=4 --fp8 \
    --run=pre1905_d34 --save-every=5000 --window-pattern L \
    --resume-from-step=10000
