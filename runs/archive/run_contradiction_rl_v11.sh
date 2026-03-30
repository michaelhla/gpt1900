#!/bin/bash
set -euo pipefail

export NANOCHAT_BASE_DIR=/opt/dlami/nvme/gpt1905_training

echo "=== [$(date)] Starting contradiction RL v11 (v3 SFT physics base, fixed coherence, logical+contradiction) ==="

# Base: physics-expanded SFT -> v3 safe corpus SFT
# Uses v8's fixed coherence weight with v10's logical coherence style
# Chain: base d34 -> physics CLM expanded -> v3 SFT safe -> RL v11

# Step 1: v3 SFT on physics-expanded base (skip if already done)
V3_PHYSICS_CKPT_DIR=/opt/dlami/nvme/gpt1905_training/v3_sft_physics_checkpoints
if [ ! -d "$V3_PHYSICS_CKPT_DIR/d34" ] || [ -z "$(ls $V3_PHYSICS_CKPT_DIR/d34/model_*.pt 2>/dev/null)" ]; then
    echo "Step 1: Running v3 SFT on physics-expanded base..."
    bash runs/run_v3_sft_physics_base.sh
else
    echo "Step 1: v3 SFT physics checkpoint already exists, skipping..."
fi

# Step 2: Discovery RL v11
echo "Step 2: Discovery RL v11..."
torchrun --standalone --nproc_per_node=8 -m scripts.pre1900_scripts.discovery_rl -- \
    --model-tag d34 \
    --checkpoints-dir v3_sft_physics_checkpoints \
    --output-dir /opt/dlami/nvme/gpt1905_training/pre1900_discovery_rl_v11_checkpoints \
    --coherence-reward \
    --coherence-style logical \
    --fixed-coherence-weight 0.25 \
    --device-batch-size 2 \
    --num-samples 4 \
    --examples-per-step 8 \
    --max-new-tokens 2048 \
    --num-epochs 24 \
    --judge-style contradiction \
    --train-data instruct_data/contradiction_problems/contradiction_prompts_sys_train.jsonl \
    --val-data instruct_data/contradiction_problems/contradiction_prompts_sys_val.jsonl \
    --problems-data instruct_data/contradiction_problems/contradiction_problems_train.jsonl \
    --problems-val-data instruct_data/contradiction_problems/contradiction_problems_val.jsonl \
    --eval-every 15 --save-every 35 \
    --run=pre1900_discovery_rl_v11_d34

echo "=== [$(date)] Contradiction RL v11 pipeline complete ==="
