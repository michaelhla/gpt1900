#!/bin/bash
set -euo pipefail

export NANOCHAT_BASE_DIR=/opt/dlami/nvme/gpt1900_openthoughts

echo "=== [$(date)] Starting OpenThoughts SFT pipeline ==="

# Step 1: Download physics CLM expanded checkpoint (base model)
# Chain: base d34-22btok -> physics CLM expanded (pre-1900 + post-1900 physics texts)
echo "Step 1: Downloading physics CLM expanded checkpoint..."
huggingface-cli download mhla/gpt1900-d34-physicssft-expanded \
    --local-dir $NANOCHAT_BASE_DIR/physicssft_expanded_checkpoints/d34

# Also need the tokenizer
echo "Step 1b: Downloading tokenizer..."
huggingface-cli download mhla/gpt1900-d34-22btok \
    --include "tokenizer/*" \
    --local-dir $NANOCHAT_BASE_DIR/base_checkpoints/d34
# Copy tokenizer to base dir where get_tokenizer() expects it
mkdir -p $NANOCHAT_BASE_DIR/tokenizer
cp $NANOCHAT_BASE_DIR/base_checkpoints/d34/tokenizer/* $NANOCHAT_BASE_DIR/tokenizer/

# Step 2: Prepare OpenThoughts data (filter, convert, dedup)
echo "Step 2: Preparing OpenThoughts3 data..."
python -m scripts.pre1900_scripts.prepare_openthoughts \
    --output-dir instruct_data/openthoughts \
    --max-tokens 2048 \
    --val-fraction 0.05

# Step 3: SFT on filtered OpenThoughts
# Each step processes ~32 packed sequences (total-batch-size 65536 / max-seq-len 2048).
# --num-iterations -1 = full epoch over the dataset.
echo "Step 3: SFT on filtered OpenThoughts..."
torchrun --standalone --nproc_per_node=8 -m scripts.pre1900_sft -- \
    --model-tag d34 \
    --checkpoints-dir physicssft_expanded_checkpoints \
    --output-dir pre1900_openthoughts_sft_checkpoints \
    --device-batch-size=4 \
    --total-batch-size=65536 \
    --num-iterations -1 \
    --chunk-long \
    --eval-every 50 \
    --train-data instruct_data/openthoughts/sft_train.jsonl \
    --val-data instruct_data/openthoughts/sft_val.jsonl \
    --run=pre1900_openthoughts_sft_d34

echo "=== [$(date)] OpenThoughts SFT complete ==="
