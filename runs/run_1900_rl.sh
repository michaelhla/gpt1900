#!/bin/bash
# 1900 RL pipeline: pre-1900 d34 → Physics CLM → v3 SFT → Contradiction RL
# Chain: base d34_pre1900 (mhla/gpt1900-d34-22btok, step 10507)
#        -> physics CLM (275M clean pre-1900 corpus, 14400 steps)
#        -> v3 SFT (safe corpus) -> contradiction RL (v11-style)

set -euo pipefail

cd /root/gpt1900

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR=/opt/dlami/nvme/gpt1905_training
export PYTHONPATH=/root/gpt1900
export HF_TOKEN=${HF_TOKEN:?"Set HF_TOKEN env var"}
export WANDB_API_KEY=${WANDB_API_KEY:?"Set WANDB_API_KEY env var"}
export ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY:?"Set ANTHROPIC_API_KEY env var"}

echo "=== [$(date)] Starting 1900 RL pipeline (pre1900 d34 → physics CLM → v3 SFT → contradiction RL) ==="

# Data: 275M clean pre-1900 physics corpus (no post-1900 texts)
DATA_DIR=${NANOCHAT_BASE_DIR}/physics_clm_data_pre1900_clean
if [ ! -f "${DATA_DIR}/train.parquet" ]; then
    echo "ERROR: ${DATA_DIR}/train.parquet not found."
    exit 1
fi

echo "=== Data ==="
python3 -c "
import pyarrow.parquet as pq
for split in ['train', 'val']:
    t = pq.read_table('${DATA_DIR}/' + split + '.parquet')
    chars = sum(len(s.as_py()) for s in t.column('text'))
    print(f'  {split}: {t.num_rows:,} docs, ~{chars/4.2/1e6:.1f}M tokens')
"

# Step 1: Physics CLM (continued pretraining, 14400 steps)
# Loads from: base_checkpoints/d34_pre1900 (gpt1900-d34-22btok, step 10507)
# Saves to: physics_clm_checkpoints/d34_pre1900/
PHYSICS_CLM_CKPT_DIR=${NANOCHAT_BASE_DIR}/physics_clm_checkpoints/d34_pre1900
if [ ! -d "$PHYSICS_CLM_CKPT_DIR" ] || [ -z "$(ls $PHYSICS_CLM_CKPT_DIR/model_*.pt 2>/dev/null)" ]; then
    echo "Step 1: Running physics CLM (14400 steps) from pre-1900 base..."
    torchrun --standalone --nproc_per_node=8 \
        -m scripts.physics_clm -- \
        --data-dir ${DATA_DIR} \
        --model-tag d34_pre1900 \
        --output-tag d34_pre1900 \
        --num-iterations 14400 \
        --device-batch-size 4 \
        --total-batch-size 65536 \
        --matrix-lr 0.005 \
        --embedding-lr 0.05 \
        --unembedding-lr 0.001 \
        --eval-every 50 \
        --save-every 4800 \
        --physics-eval \
        --run physics_clm_d34_pre1900
else
    echo "Step 1: Physics CLM checkpoint already exists, skipping..."
fi

# Step 2: v3 SFT (instruction tuning on safe corpus)
# Loads from: physics_clm_checkpoints/d34_pre1900/
# Saves to: v3_sft_physics_1900_rl_checkpoints/d34_pre1900/
V3_SFT_CKPT_DIR=${NANOCHAT_BASE_DIR}/v3_sft_physics_1900_rl_checkpoints/d34_pre1900
if [ ! -d "$V3_SFT_CKPT_DIR" ] || [ -z "$(ls $V3_SFT_CKPT_DIR/model_*.pt 2>/dev/null)" ]; then
    echo "Step 2: Running v3 SFT on physics CLM base..."
    torchrun --standalone --nproc_per_node=8 -m scripts.pre1900_sft -- \
        --model-tag d34_pre1900 \
        --checkpoints-dir physics_clm_checkpoints \
        --output-dir ${NANOCHAT_BASE_DIR}/v3_sft_physics_1900_rl_checkpoints \
        --device-batch-size=4 \
        --total-batch-size=524288 \
        --max-seq-len=2048 \
        --train-data ${NANOCHAT_BASE_DIR}/instruct_data/v3_corpus_safe/all_train.jsonl \
        --val-data ${NANOCHAT_BASE_DIR}/instruct_data/v3_corpus_safe/all_val.jsonl \
        --eval-every=150 \
        --run=v3_sft_physics_1900_rl_d34
else
    echo "Step 2: v3 SFT checkpoint already exists, skipping..."
fi

# Step 3: Contradiction RL (v11-style: logical coherence + contradiction judging)
# Loads from: v3_sft_physics_1900_rl_checkpoints/d34_pre1900/
# Saves to: pre1900_discovery_rl_1900_checkpoints/d34_pre1900/
echo "Step 3: Running contradiction RL..."
torchrun --standalone --nproc_per_node=8 -m scripts.pre1900_scripts.discovery_rl -- \
    --model-tag d34_pre1900 \
    --checkpoints-dir v3_sft_physics_1900_rl_checkpoints \
    --output-dir ${NANOCHAT_BASE_DIR}/pre1900_discovery_rl_1900_checkpoints \
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
    --run=pre1900_discovery_rl_1900_d34

echo "=== [$(date)] 1900 RL pipeline complete ==="
echo "Evaluate with: python -m scripts.physics_eval --checkpoints-dir pre1900_discovery_rl_1900_checkpoints --model-tag d34_pre1900"
