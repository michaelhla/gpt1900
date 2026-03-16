#!/bin/bash
# 1905 RL pipeline: 1905 d34 → Physics CLM → v3 SFT → Contradiction RL
# Chain: base d34 (mhla/gpt1905-d34) -> physics CLM (pre-1900 + post-1900 texts)
#        -> v3 SFT (safe corpus) -> contradiction RL (v11-style)

set -euo pipefail

cd /root/gpt1900

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR=/opt/dlami/nvme/gpt1905_training
export PYTHONPATH=/root/gpt1900
export HF_TOKEN=${HF_TOKEN:?"Set HF_TOKEN env var"}
export WANDB_API_KEY=${WANDB_API_KEY:?"Set WANDB_API_KEY env var"}
export ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY:?"Set ANTHROPIC_API_KEY env var"}

echo "=== [$(date)] Starting 1905 RL pipeline (d34 → physics CLM → v3 SFT → contradiction RL) ==="

# Step 0: Prepare physics CLM data (pre-1900 + core + 4 post-1900 texts)
DATA_DIR=${NANOCHAT_BASE_DIR}/physics_clm_data_final
if [ ! -f "${DATA_DIR}/train.parquet" ]; then
    echo "Step 0: Preparing combined physics CLM data..."
    python -m scripts.pre1900_scripts.prepare_physics_parquet \
        --input-dir data/physics_books data/core_physics_books data/post1900_physics_books \
        --output-dir ${DATA_DIR}
else
    echo "Step 0: Physics CLM data already exists, skipping..."
fi

echo "=== Data ==="
python3 -c "
import pyarrow.parquet as pq
for split in ['train', 'val']:
    t = pq.read_table('${DATA_DIR}/' + split + '.parquet')
    chars = sum(len(s.as_py()) for s in t.column('text'))
    print(f'  {split}: {t.num_rows:,} docs, ~{chars/4.2/1e6:.1f}M tokens')
"

# Step 1: Physics CLM (continued pretraining on combined physics corpus)
# Saves to: physics_clm_checkpoints/d34/
PHYSICS_CLM_CKPT_DIR=${NANOCHAT_BASE_DIR}/physics_clm_checkpoints/d34
if [ ! -d "$PHYSICS_CLM_CKPT_DIR" ] || [ -z "$(ls $PHYSICS_CLM_CKPT_DIR/model_*.pt 2>/dev/null)" ]; then
    echo "Step 1: Running physics CLM on combined corpus..."
    torchrun --standalone --nproc_per_node=8 \
        -m scripts.physics_clm -- \
        --data-dir ${DATA_DIR} \
        --model-tag d34 \
        --output-tag d34 \
        --num-epochs 10 \
        --device-batch-size 4 \
        --total-batch-size 65536 \
        --matrix-lr 0.005 \
        --embedding-lr 0.05 \
        --unembedding-lr 0.001 \
        --eval-every 50 \
        --save-every 4800 \
        --physics-eval \
        --run physics_clm_d34_1905_rl
else
    echo "Step 1: Physics CLM checkpoint already exists, skipping..."
fi

# Step 2: v3 SFT (instruction tuning on safe corpus)
# Loads from: physics_clm_checkpoints/d34/
# Saves to: v3_sft_physics_1905_rl_checkpoints/d34/
V3_SFT_CKPT_DIR=${NANOCHAT_BASE_DIR}/v3_sft_physics_1905_rl_checkpoints/d34
if [ ! -d "$V3_SFT_CKPT_DIR" ] || [ -z "$(ls $V3_SFT_CKPT_DIR/model_*.pt 2>/dev/null)" ]; then
    echo "Step 2: Running v3 SFT on physics CLM base..."
    torchrun --standalone --nproc_per_node=8 -m scripts.pre1900_sft -- \
        --model-tag d34 \
        --checkpoints-dir physics_clm_checkpoints \
        --output-dir ${NANOCHAT_BASE_DIR}/v3_sft_physics_1905_rl_checkpoints \
        --device-batch-size=4 \
        --total-batch-size=524288 \
        --max-seq-len=2048 \
        --train-data ${NANOCHAT_BASE_DIR}/instruct_data/v3_corpus_safe/all_train.jsonl \
        --val-data ${NANOCHAT_BASE_DIR}/instruct_data/v3_corpus_safe/all_val.jsonl \
        --eval-every=150 \
        --run=v3_sft_physics_1905_rl_d34
else
    echo "Step 2: v3 SFT checkpoint already exists, skipping..."
fi

# Step 3: Contradiction RL (v11-style: logical coherence + contradiction judging)
# Loads from: v3_sft_physics_1905_rl_checkpoints/d34/
# Saves to: pre1900_discovery_rl_1905_checkpoints/d34/
echo "Step 3: Running contradiction RL..."
torchrun --standalone --nproc_per_node=8 -m scripts.pre1900_scripts.discovery_rl -- \
    --model-tag d34 \
    --checkpoints-dir v3_sft_physics_1905_rl_checkpoints \
    --output-dir ${NANOCHAT_BASE_DIR}/pre1900_discovery_rl_1905_checkpoints \
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
    --run=pre1900_discovery_rl_1905_d34

echo "=== [$(date)] 1905 RL pipeline complete ==="
echo "Evaluate with: python -m scripts.physics_eval --checkpoints-dir pre1900_discovery_rl_1905_checkpoints --model-tag d34"
