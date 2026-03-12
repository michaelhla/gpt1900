#!/bin/bash
set -euo pipefail

echo "=== [$(date)] Starting generated physics format SFT ==="

# Step 1: Download physics-pretrained base
echo "Step 1: Downloading physics SFT checkpoint..."
huggingface-cli download mhla/gpt1900-d34-physics-sft \
    --local-dir $NANOCHAT_BASE_DIR/physics_clm_checkpoints/d34

# Step 2: Download generated physics dataset
echo "Step 2: Downloading generated physics dataset..."
huggingface-cli download mhla/pre1900-verifiable-physics \
    --repo-type dataset \
    --local-dir $NANOCHAT_BASE_DIR/instruct_data/generated_physics

# Step 3: Format SFT on 2031 verified traces
echo "Step 3: Format SFT (2031 traces, ~1.3M tokens)..."
torchrun --standalone --nproc_per_node=8 -m scripts.pre1900_sft -- \
    --model-tag d34 \
    --checkpoints-dir physics_clm_checkpoints \
    --output-dir pre1900_generated_physics_sft_checkpoints \
    --device-batch-size=4 \
    --total-batch-size=65536 \
    --num-iterations 200 \
    --train-data instruct_data/generated_physics/generated_format_sft.jsonl \
    --val-data instruct_data/yale_physics/sft_val.jsonl \
    --run=pre1900_generated_physics_sft_d34

echo "=== [$(date)] Generated physics format SFT complete ==="
