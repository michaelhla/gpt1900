#!/bin/bash
set -euo pipefail

export NANOCHAT_BASE_DIR=/opt/dlami/nvme/gpt1905_training

echo "=== [$(date)] Starting Math RL (GSM8K + MATH on R1 SFT model) ==="
echo ""
echo "Chain: base d34 -> physics CLM -> v3 SFT safe -> R1 reasoning SFT -> Math RL"
echo ""
echo "Key details:"
echo "  - Curriculum: Phase 1 (L1-2) -> Phase 2 (L1-3) -> Phase 3 (L1-5)"
echo "  - 200 examples per phase, 2 epochs each"
echo "  - GSM8K (30%) + MATH (70%) per phase"
echo "  - No system prompt, user prompt ends with 'Think deeply and step by step.'"
echo "  - Format reward (0.3): <think> + \\answer{} tags"
echo "  - Correctness reward (1.0): numeric (GSM8K) or SymPy (MATH)"
echo "  - Physics online eval from EVAL.json (monitor only)"
echo ""

# ---------------------------------------------------------------------------
# Step 0: Ensure R1 reasoning SFT checkpoint exists
# ---------------------------------------------------------------------------
R1_SFT_CKPT_DIR=${NANOCHAT_BASE_DIR}/r1_reasoning_sft_checkpoints
if [ ! -d "$R1_SFT_CKPT_DIR/d34" ] || [ -z "$(ls $R1_SFT_CKPT_DIR/d34/model_*.pt 2>/dev/null)" ]; then
    echo "ERROR: R1 reasoning SFT checkpoint not found at $R1_SFT_CKPT_DIR/d34"
    echo "Run the R1 reasoning SFT step first (e.g. via run_contradiction_rl_v12.sh steps 0-2)"
    exit 1
fi
echo "Step 0: R1 reasoning SFT checkpoint found, proceeding..."

# ---------------------------------------------------------------------------
# Step 1: Math RL with curriculum
# ---------------------------------------------------------------------------
echo ""
echo "=== [$(date)] Step 1: Math RL ==="
echo "  Source: r1_reasoning_sft_checkpoints/d34"
echo "  Output: math_rl_checkpoints/d34"

torchrun --standalone --nproc_per_node=8 -m scripts.math_rl -- \
    --model-tag d34 --model-step 99 \
    --checkpoints-dir r1_reasoning_sft_checkpoints \
    --output-dir math_rl_checkpoints \
    --gsm8k-ratio 0.3 \
    --examples-per-phase 200 --num-epochs 2 \
    --device-batch-size 2 --num-samples 4 --examples-per-step 8 \
    --max-new-tokens 2048 \
    --eval-every 15 --save-every 30 \
    --eval-gsm8k-examples 50 --eval-math-examples 50 \
    --run=math_rl_curriculum_d34

echo ""
echo "=== [$(date)] Math RL pipeline complete ==="
