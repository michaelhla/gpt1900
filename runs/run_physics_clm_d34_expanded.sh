#!/bin/bash
# Physics CLM (continued pretraining) on expanded physics dataset
# Using cleaned data from mhla/gpt1900-physics-data (anachronism-filtered)
# Saves checkpoints to /opt/dlami/nvme/

set -euo pipefail

cd /root/gpt1900

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR=/opt/dlami/nvme/gpt1905_training
export PYTHONPATH=/root/gpt1900
export HF_TOKEN=${HF_TOKEN:?"Set HF_TOKEN env var"}
export WANDB_API_KEY=${WANDB_API_KEY:?"Set WANDB_API_KEY env var"}
export ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY:?"Set ANTHROPIC_API_KEY env var"}

DATA_DIR=${NANOCHAT_BASE_DIR}/physics_clm_data

# Verify data exists
if [ ! -f "${DATA_DIR}/train.parquet" ]; then
    echo "ERROR: ${DATA_DIR}/train.parquet not found"
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

echo "=== Starting physics CLM training ==="
torchrun --standalone --nproc_per_node=8 \
    -m scripts.physics_clm -- \
    --data-dir ${DATA_DIR} \
    --num-epochs 10 \
    --device-batch-size 4 \
    --total-batch-size 65536 \
    --matrix-lr 0.005 \
    --embedding-lr 0.05 \
    --unembedding-lr 0.001 \
    --eval-every 50 \
    --save-every 4800 \
    --physics-eval \
    --output-tag pre1905_physics_clm_d34_expanded \
    --run pre1905_physics_clm_d34_expanded
