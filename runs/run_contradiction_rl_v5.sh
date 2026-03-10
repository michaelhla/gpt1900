#!/bin/bash
set -euo pipefail

echo "=== [$(date)] Starting contradiction RL v5 pipeline ==="

# Step 1: Download physics-pretrained base
echo "Step 1: Downloading physics SFT checkpoint..."
huggingface-cli download mhla/gpt1900-d34-physics-sft \
    --local-dir $NANOCHAT_BASE_DIR/physics_clm_checkpoints/d34

# Step 2: Generate contradiction problems (CPU, needs ANTHROPIC_API_KEY)
echo "Step 2: Generating contradiction problems..."
python -m scripts.pre1900_scripts.generate_contradiction_problems \
    --insights-file instruct_data/rl_problems/rl_insights_raw.jsonl \
    --output-dir instruct_data/contradiction_problems \
    --run phase2

# Step 3: Reasoning SFT v5 (8 GPUs)
echo "Step 3: Reasoning SFT v5..."
torchrun --standalone --nproc_per_node=8 -m scripts.pre1900_sft -- \
    --model-tag d34 \
    --checkpoints-dir physics_clm_checkpoints \
    --output-dir pre1900_reasoning_sft_v5_checkpoints \
    --device-batch-size=4 \
    --total-batch-size=65536 \
    --num-iterations 100 \
    --train-data instruct_data/reasoning/sft_train_trimmed.jsonl \
    --val-data instruct_data/reasoning/sft_val_clean.jsonl \
    --run=pre1900_reasoning_sft_v5_d34

# Step 4: Discovery RL v5 (8 GPUs, Claude judge)
echo "Step 4: Discovery RL v5..."
torchrun --standalone --nproc_per_node=8 -m scripts.pre1900_scripts.discovery_rl -- \
    --model-tag d34 \
    --checkpoints-dir pre1900_reasoning_sft_v5_checkpoints \
    --output-dir pre1900_discovery_rl_v5_checkpoints \
    --device-batch-size 2 \
    --num-samples 4 \
    --examples-per-step 8 \
    --max-new-tokens 1024 \
    --num-epochs 3 \
    --judge-style contradiction \
    --train-data instruct_data/contradiction_problems/contradiction_prompts_sys_train.jsonl \
    --val-data instruct_data/contradiction_problems/contradiction_prompts_sys_val.jsonl \
    --problems-data instruct_data/contradiction_problems/contradiction_problems_train.jsonl \
    --problems-val-data instruct_data/contradiction_problems/contradiction_problems_val.jsonl \
    --eval-every 15 --save-every 30 \
    --run=pre1900_discovery_rl_v5_d34

echo "=== [$(date)] Contradiction RL v5 pipeline complete ==="
