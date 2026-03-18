#!/bin/bash
# Download the instruction-tuned model and launch interactive chat.
#
# Usage (on RunPod pod):
#   bash runs/chat.sh
#   bash runs/chat.sh --temperature 0.8
#   bash runs/chat.sh --top-k 100

set -e

NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
SFT_DIR="$NANOCHAT_BASE_DIR/chatsft_checkpoints"
HF_REPO="mhla/gpt1900-instruct-v3-sft"

# Download model if not already present
if [ -z "$(find "$SFT_DIR" -name 'model_*.pt' 2>/dev/null | head -1)" ]; then
    echo "Downloading model from $HF_REPO..."
    mkdir -p "$SFT_DIR"
    huggingface-cli download "$HF_REPO" --local-dir "$SFT_DIR" --exclude "optim_*"
    echo "Download complete."
else
    echo "Model already downloaded in $SFT_DIR"
fi

echo ""
echo "Launching chat..."
echo ""

# Pass through any extra args (e.g. --temperature 0.8)
python -m scripts.chat_cli -i sft --single-turn "$@"
