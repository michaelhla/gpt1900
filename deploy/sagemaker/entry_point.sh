#!/bin/bash
# SageMaker training entry point for nanochat/gpt1900.
#
# SageMaker PyTorch Estimator runs this script inside the pre-built
# PyTorch DLC (Deep Learning Container).  Environment variables set
# by SageMaker:
#   SM_MODEL_DIR       — write final artifacts here (auto-uploaded to S3)
#   SM_CHANNEL_TRAINING — path to data from S3 input channel
#   SM_NUM_GPUS        — number of GPUs on the instance
#
# Custom env vars we expect:
#   TRAINING_CMD        — direct command to run (mutually exclusive w/ TRAINING_SCRIPT_B64)
#   TRAINING_SCRIPT_B64 — base64-encoded shell script
#   HF_DATA_REPO        — (optional) HuggingFace dataset repo to download
#   REPO_URL            — git repo URL (default: https://github.com/michaelhla/gpt1900)

set -euo pipefail

echo "=== SageMaker entry_point.sh ==="
echo "Instance GPUs: ${SM_NUM_GPUS:-unknown}"
echo "Model dir:     ${SM_MODEL_DIR:-/opt/ml/model}"
echo "Data channel:  ${SM_CHANNEL_TRAINING:-none}"

# ── 1. Writable workspace ──────────────────────────────────────────
export NANOCHAT_BASE_DIR="/tmp/nanochat_workdir"
mkdir -p "$NANOCHAT_BASE_DIR"

# ── 2. Symlink S3 input data into base dir ──────────────────────────
if [ -n "${SM_CHANNEL_TRAINING:-}" ] && [ -d "$SM_CHANNEL_TRAINING" ]; then
    echo "Linking S3 input data from $SM_CHANNEL_TRAINING..."
    for item in "$SM_CHANNEL_TRAINING"/*; do
        name=$(basename "$item")
        if [ ! -e "$NANOCHAT_BASE_DIR/$name" ]; then
            ln -s "$item" "$NANOCHAT_BASE_DIR/$name"
            echo "  linked: $name"
        fi
    done
fi

# ── 3. Optionally download from HuggingFace ────────────────────────
if [ -n "${HF_DATA_REPO:-}" ]; then
    echo "Downloading data from HuggingFace: $HF_DATA_REPO"
    pip install -q huggingface_hub
    huggingface-cli download "$HF_DATA_REPO" --local-dir "$NANOCHAT_BASE_DIR"
fi

# ── 4. Clone repo + install deps ───────────────────────────────────
REPO_URL="${REPO_URL:-https://github.com/michaelhla/gpt1900}"
WORK_DIR="/tmp/gpt1900"

if [ ! -d "$WORK_DIR" ]; then
    git clone "$REPO_URL" "$WORK_DIR"
fi
cd "$WORK_DIR"

# Install uv and project deps
command -v uv || (curl -LsSf https://astral.sh/uv/install.sh | sh)
export PATH="$HOME/.local/bin:$PATH"
uv sync --extra gpu
source .venv/bin/activate

export OMP_NUM_THREADS=1

# ── 5. Run training ────────────────────────────────────────────────
NUM_GPUS="${SM_NUM_GPUS:-8}"

if [ -n "${TRAINING_SCRIPT_B64:-}" ]; then
    echo "Decoding training script from TRAINING_SCRIPT_B64..."
    echo "$TRAINING_SCRIPT_B64" | base64 -d > /tmp/train.sh
    chmod +x /tmp/train.sh
    bash /tmp/train.sh
elif [ -n "${TRAINING_CMD:-}" ]; then
    echo "Running training command: $TRAINING_CMD"
    eval "$TRAINING_CMD"
else
    echo "ERROR: Neither TRAINING_SCRIPT_B64 nor TRAINING_CMD is set"
    exit 1
fi

# ── 6. Copy checkpoints to SM_MODEL_DIR for S3 upload ──────────────
MODEL_DIR="${SM_MODEL_DIR:-/opt/ml/model}"
echo "Copying checkpoints to $MODEL_DIR..."

# Copy any .pt checkpoint files
find "$NANOCHAT_BASE_DIR" -name "*.pt" -exec cp {} "$MODEL_DIR/" \; 2>/dev/null || true
# Copy config/metadata files
find "$NANOCHAT_BASE_DIR" -name "*.json" -exec cp {} "$MODEL_DIR/" \; 2>/dev/null || true
# Copy tokenizer if present
if [ -d "$NANOCHAT_BASE_DIR/tokenizer" ]; then
    cp -r "$NANOCHAT_BASE_DIR/tokenizer" "$MODEL_DIR/"
fi

echo "=== Training complete ==="
ls -lh "$MODEL_DIR/" 2>/dev/null || echo "(model dir empty)"
