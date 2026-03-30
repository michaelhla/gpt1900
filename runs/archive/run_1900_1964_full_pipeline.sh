#!/bin/bash
set -euo pipefail

# Full pipeline: download -> reshard -> tokenizer -> pretrain
# Safe to run unattended overnight.

export OMP_NUM_THREADS=1
export PYTHONPATH=/root/gpt1900
export HF_TOKEN=${HF_TOKEN:?"Set HF_TOKEN env var"}
export HF_HOME=/opt/dlami/nvme/hf_cache

BASE=/opt/dlami/nvme/gpt1964_training
cd /root/gpt1900
source .venv/bin/activate

echo "============================================================"
echo "GPT-1964 FULL PIPELINE"
echo "Started: $(date)"
echo "Base dir: $BASE"
echo "============================================================"

# ── Step 1: Download ──────────────────────────────────────────
echo ""
echo "[Step 1/4] Downloading 1900-1964 corpus..."
echo "============================================================"
python scripts/pre1900_scripts/hf_download_1900_1964.py \
    --outdir $BASE/raw_data \
    --year-workers 16

echo ""
echo "[Step 1/4] Download complete: $(date)"
echo ""

# ── Step 2: Reshard ───────────────────────────────────────────
echo "[Step 2/4] Resharding into training corpus..."
echo "============================================================"
python scripts/pre1900_scripts/reshard_1900_1964.py \
    --input $BASE/raw_data \
    --output $BASE/corpus

echo ""
echo "[Step 2/4] Reshard complete: $(date)"
echo ""

# ── Step 3: Tokenizer ────────────────────────────────────────
echo "[Step 3/4] Training tokenizer..."
echo "============================================================"
python -m scripts.pre1900_scripts.hf_tok_train \
    --input $BASE/corpus \
    --output $BASE/tokenizer \
    --vocab-size 32768

echo ""
echo "[Step 3/4] Tokenizer complete: $(date)"
echo ""

# ── Step 4: Kill existing training + launch pretrain ──────────
echo "[Step 4/4] Launching pretraining..."
echo "============================================================"

# Kill any existing torchrun training
EXISTING_PIDS=$(pgrep -f "torchrun.*base_train" || true)
if [ -n "$EXISTING_PIDS" ]; then
    echo "Killing existing training run (PIDs: $EXISTING_PIDS)..."
    kill $EXISTING_PIDS 2>/dev/null || true
    sleep 5
    # Force kill if still alive
    kill -9 $EXISTING_PIDS 2>/dev/null || true
    sleep 2
fi

export NANOCHAT_BASE_DIR=$BASE
mkdir -p $BASE/base_checkpoints/d34

# Dataloader expects base_data/, symlink to corpus/
ln -sfn $BASE/corpus $BASE/base_data

torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --depth=34 --target-param-data-ratio=20 --device-batch-size=4 --fp8 \
    --run=gpt1964_d34 --save-every=5000 --window-pattern L
