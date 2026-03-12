#!/bin/bash
set -euo pipefail

echo "=== [$(date)] Starting generated physics verifiable RL ==="

# Step 1: Download physics-pretrained base (d34 physics CLM)
echo "Step 1: Downloading d34 physics CLM checkpoint..."
huggingface-cli download mhla/gpt1900-d34-physics-sft \
    --local-dir $NANOCHAT_BASE_DIR/physics_clm_checkpoints/d34

# Step 2: Download generated physics dataset (includes Yale + generated + combined)
echo "Step 2: Downloading generated physics dataset..."
huggingface-cli download mhla/pre1900-verifiable-physics \
    --repo-type dataset \
    --local-dir $NANOCHAT_BASE_DIR/instruct_data/generated_physics

# Step 3: Format SFT on 1981 verified traces
echo "Step 3: Format SFT (1981 train traces, 50 val)..."
torchrun --standalone --nproc_per_node=8 -m scripts.pre1900_sft -- \
    --model-tag d34 \
    --checkpoints-dir physics_clm_checkpoints \
    --output-dir pre1900_generated_physics_sft_checkpoints \
    --device-batch-size=4 \
    --total-batch-size=65536 \
    --num-iterations 200 \
    --train-data instruct_data/generated_physics/sft_train.jsonl \
    --val-data instruct_data/generated_physics/sft_val.jsonl \
    --run=pre1900_generated_physics_sft_d34

# Step 4: Verifiable RL on 1094 problems (951 generated + 143 Yale)
# Runs indefinitely (99999 epochs) — kill manually when satisfied.
# Checkpoints saved every epoch (136 steps = 1094 examples / 8 per step).
echo "Step 4: Verifiable RL (SymPy equivalence + 0.3 format bonus, continuous)..."
torchrun --standalone --nproc_per_node=8 -m scripts.pre1900_scripts.verifiable_rl -- \
    --model-tag d34 \
    --checkpoints-dir pre1900_generated_physics_sft_checkpoints \
    --output-dir pre1900_generated_physics_rl_checkpoints \
    --device-batch-size 8 \
    --num-samples 8 \
    --examples-per-step 8 \
    --max-new-tokens 1024 \
    --num-epochs 20 \
    --train-data instruct_data/generated_physics/combined_prompts_sys_train.jsonl \
    --val-data instruct_data/generated_physics/combined_prompts_sys_val.jsonl \
    --problems-data instruct_data/generated_physics/combined_problems_train.jsonl \
    --problems-val-data instruct_data/generated_physics/combined_problems_val.jsonl \
    --eval-every 136 --save-every 136 \
    --run=pre1900_generated_physics_rl_d34

echo "=== [$(date)] Generated physics verifiable RL complete ==="
