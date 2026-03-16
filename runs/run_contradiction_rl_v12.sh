#!/bin/bash
set -euo pipefail

export NANOCHAT_BASE_DIR=/opt/dlami/nvme/gpt1905_training

echo "=== [$(date)] Starting contradiction RL v12 (R1 reasoning SFT + contradiction RL) ==="
echo ""
echo "Chain: base d34 -> physics CLM expanded -> v3 SFT safe -> R1 reasoning SFT -> contradiction RL v12"
echo ""
echo "Key changes from v11:"
echo "  - R1 distillation SFT before RL (DeepSeek R1 reasoning traces)"
echo "  - No system prompt anywhere (SFT, RL generation, physics eval)"
echo "  - User prompts end with 'Think deeply and step by step.'"
echo "  - Format reward checks <think> and \\answer{} tags"
echo "  - Coherence reward (logical style) + correctness reward (contradiction judge)"
echo "  - Physics online eval from EVAL.json"
echo ""

# ---------------------------------------------------------------------------
# Step 0: Ensure v3 SFT physics checkpoint exists (same as v11)
# ---------------------------------------------------------------------------
V3_PHYSICS_CKPT_DIR=${NANOCHAT_BASE_DIR}/v3_sft_physics_checkpoints
if [ ! -d "$V3_PHYSICS_CKPT_DIR/d34" ] || [ -z "$(ls $V3_PHYSICS_CKPT_DIR/d34/model_*.pt 2>/dev/null)" ]; then
    echo "Step 0: Running v3 SFT on physics-expanded base..."
    bash runs/run_v3_sft_physics_base.sh
else
    echo "Step 0: v3 SFT physics checkpoint already exists, skipping..."
fi

# ---------------------------------------------------------------------------
# Step 1: Prepare R1 reasoning SFT data + RL prompts (no system prompt)
# ---------------------------------------------------------------------------
echo ""
echo "=== [$(date)] Step 1: Preparing data (no system prompt, 'Think deeply' suffix) ==="

python3 -c "
import json, os, random

random.seed(42)
base = os.environ['NANOCHAT_BASE_DIR']

# --- R1 SFT data: already in correct format (no system prompt, has 'Think deeply' suffix) ---
# Just copy from repo to training dir
sft_src_train = 'instruct_data/r1_reasoning/sft_train.jsonl'
sft_src_val = 'instruct_data/r1_reasoning/sft_val.jsonl'
sft_dst_dir = os.path.join(base, 'instruct_data/r1_reasoning')
os.makedirs(sft_dst_dir, exist_ok=True)

for src, dst_name in [(sft_src_train, 'sft_train.jsonl'), (sft_src_val, 'sft_val.jsonl')]:
    dst = os.path.join(sft_dst_dir, dst_name)
    count = 0
    with open(src) as fin, open(dst, 'w') as fout:
        for line in fin:
            if line.strip():
                fout.write(line)
                count += 1
    print(f'  R1 SFT {dst_name}: {count} examples -> {dst}')

# --- RL prompts: rewrite contradiction prompts without system prompt ---
# Add 'Think deeply and step by step.' to user prompt
rl_dir = os.path.join(base, 'instruct_data/contradiction_problems')
os.makedirs(rl_dir, exist_ok=True)

for split in ['train', 'val']:
    # Read problems (has prompt + gold_answer)
    problems_src = os.path.join(rl_dir, f'contradiction_problems_{split}.jsonl')
    prompts_dst = os.path.join(rl_dir, f'contradiction_prompts_v12_{split}.jsonl')

    if not os.path.exists(problems_src):
        print(f'  Warning: {problems_src} not found, skipping')
        continue

    count = 0
    with open(problems_src) as fin, open(prompts_dst, 'w') as fout:
        for line in fin:
            if not line.strip():
                continue
            record = json.loads(line)
            # No system prompt, user prompt ends with 'Think deeply and step by step.'
            messages = [
                {'role': 'user', 'content': record['prompt'].rstrip() + '\n\nThink deeply and step by step.'},
                {'role': 'assistant', 'content': '(to be generated)'},
            ]
            fout.write(json.dumps(messages, ensure_ascii=False) + '\n')
            count += 1
    print(f'  RL prompts v12 {split}: {count} examples -> {prompts_dst}')

print('  Data preparation complete.')
"

# ---------------------------------------------------------------------------
# Step 2: R1 Reasoning SFT (v3 SFT physics -> R1 reasoning SFT)
# ---------------------------------------------------------------------------
echo ""
echo "=== [$(date)] Step 2: R1 Reasoning SFT ==="
echo "  Source: v3_sft_physics_checkpoints/d34"
echo "  Data: instruct_data/r1_reasoning/sft_train.jsonl (670 examples)"
echo "  Output: r1_reasoning_sft_checkpoints/d34"

R1_SFT_CKPT_DIR=${NANOCHAT_BASE_DIR}/r1_reasoning_sft_checkpoints
if [ ! -d "$R1_SFT_CKPT_DIR/d34" ] || [ -z "$(ls $R1_SFT_CKPT_DIR/d34/model_*.pt 2>/dev/null)" ]; then
    torchrun --standalone --nproc_per_node=8 -m scripts.pre1900_sft -- \
        --model-tag d34 \
        --checkpoints-dir v3_sft_physics_checkpoints \
        --output-dir r1_reasoning_sft_checkpoints \
        --device-batch-size=4 \
        --total-batch-size=65536 \
        --num-iterations 100 \
        --eval-every 25 \
        --train-data instruct_data/r1_reasoning/sft_train.jsonl \
        --val-data instruct_data/r1_reasoning/sft_val.jsonl \
        --run=r1_reasoning_sft_d34
else
    echo "  R1 reasoning SFT checkpoint already exists, skipping..."
fi

# ---------------------------------------------------------------------------
# Step 3: Contradiction RL v12 (R1 reasoning SFT -> RL)
# ---------------------------------------------------------------------------
echo ""
echo "=== [$(date)] Step 3: Contradiction RL v12 ==="
echo "  Source: r1_reasoning_sft_checkpoints/d34"
echo "  Data: contradiction_problems (v12 prompts, no system prompt)"
echo "  Rewards: format (0.3) + coherence (0.25 * logical) + correctness (0.75 * contradiction)"

torchrun --standalone --nproc_per_node=8 -m scripts.pre1900_scripts.discovery_rl -- \
    --model-tag d34 \
    --checkpoints-dir r1_reasoning_sft_checkpoints \
    --output-dir ${NANOCHAT_BASE_DIR}/pre1900_discovery_rl_v12_checkpoints \
    --coherence-reward \
    --coherence-style logical \
    --fixed-coherence-weight 0.25 \
    --device-batch-size 2 \
    --num-samples 4 \
    --examples-per-step 8 \
    --max-new-tokens 2048 \
    --num-epochs 24 \
    --judge-style contradiction \
    --prompt-suffix "Think deeply and step by step." \
    --train-data instruct_data/contradiction_problems/contradiction_prompts_v12_train.jsonl \
    --val-data instruct_data/contradiction_problems/contradiction_prompts_v12_val.jsonl \
    --problems-data instruct_data/contradiction_problems/contradiction_problems_train.jsonl \
    --problems-val-data instruct_data/contradiction_problems/contradiction_problems_val.jsonl \
    --eval-every 15 --save-every 35 \
    --run=pre1900_discovery_rl_v12_d34

echo ""
echo "=== [$(date)] Contradiction RL v12 pipeline complete ==="
