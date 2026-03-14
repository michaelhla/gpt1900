#!/bin/bash
set -euo pipefail
export NANOCHAT_BASE_DIR=$HOME/.cache/nanochat

# Download base model
huggingface-cli download mhla/gpt1900-d34-22btok \
    --local-dir $NANOCHAT_BASE_DIR/base_checkpoints/d34

# Copy tokenizer
cp -r $NANOCHAT_BASE_DIR/tokenizer $NANOCHAT_BASE_DIR/tokenizer 2>/dev/null || true

# Run SFT
torchrun --standalone --nproc_per_node=8 -m scripts.pre1900_sft -- \
    --model-tag d34 \
    --output-dir instruct_checkpoints \
    --device-batch-size=4 \
    --total-batch-size=524288 \
    --max-seq-len=2048 \
    --train-data instruct_data/v2_corpus/all_train.jsonl \
    --val-data instruct_data/v2_corpus/all_val.jsonl \
    --eval-every=150 \
    --run=corpus_sft_d34
