#!/bin/bash

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR=/opt/dlami/nvme/gpt1964_training
export PYTHONPATH=/root/gpt1900
export HF_TOKEN=${HF_TOKEN:?"Set HF_TOKEN env var"}

# Verify corpus exists
DATA_DIR=$NANOCHAT_BASE_DIR/corpus
if [ ! -d "$DATA_DIR" ] || [ -z "$(ls $DATA_DIR/shard_*.parquet 2>/dev/null)" ]; then
    echo "ERROR: No corpus shards found at $DATA_DIR"
    echo "Run the download + reshard pipeline first:"
    echo "  python scripts/pre1900_scripts/hf_download_1900_1964.py"
    echo "  python scripts/pre1900_scripts/reshard_1900_1964.py"
    exit 1
fi

# Verify tokenizer exists
TOK_DIR=$NANOCHAT_BASE_DIR/tokenizer
if [ ! -f "$TOK_DIR/tokenizer.pkl" ]; then
    echo "ERROR: No tokenizer found at $TOK_DIR"
    echo "Train one first:"
    echo "  python -m scripts.pre1900_scripts.hf_tok_train --input $DATA_DIR --output $TOK_DIR --vocab-size 32768"
    exit 1
fi

CKPT_DIR=$NANOCHAT_BASE_DIR/base_checkpoints/d34
mkdir -p $CKPT_DIR

cd /root/gpt1900
source .venv/bin/activate

torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --depth=34 --target-param-data-ratio=20 --device-batch-size=4 --fp8 \
    --run=gpt1964_d34 --save-every=500 --resume-from-step=15500 --window-pattern L
