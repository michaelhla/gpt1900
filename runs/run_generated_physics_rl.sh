#!/bin/bash
set -euo pipefail

echo "=== [$(date)] Starting generated physics verifiable RL ==="

# Step 1: Download format SFT checkpoint (from generated physics SFT)
echo "Step 1: Downloading generated physics format SFT checkpoint..."
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

# Step 3.5: Download Yale PHYSICS problems (for combined RL set)
echo "Step 3.5: Downloading Yale PHYSICS dataset..."
huggingface-cli download mhla/pre1900-yale-physics \
    --repo-type dataset \
    --local-dir $NANOCHAT_BASE_DIR/instruct_data/yale_physics

# Step 4: Verifiable RL on 1094 problems (951 generated + 143 Yale)
echo "Step 4: Verifiable RL (SymPy equivalence + 0.3 format bonus)..."
torchrun --standalone --nproc_per_node=8 -m scripts.pre1900_scripts.verifiable_rl -- \
    --model-tag d34 \
    --checkpoints-dir pre1900_generated_physics_sft_checkpoints \
    --output-dir pre1900_generated_physics_rl_checkpoints \
    --device-batch-size 8 \
    --num-samples 8 \
    --examples-per-step 8 \
    --max-new-tokens 1024 \
    --num-epochs 3 \
    --train-data instruct_data/generated_physics/combined_prompts_sys_train.jsonl \
    --val-data instruct_data/generated_physics/combined_prompts_sys_val.jsonl \
    --problems-data instruct_data/generated_physics/combined_problems_train.jsonl \
    --problems-val-data instruct_data/generated_physics/combined_problems_val.jsonl \
    --eval-every 60 --save-every 60 \
    --run=pre1900_generated_physics_rl_d34

echo "=== [$(date)] Generated physics verifiable RL complete ==="
