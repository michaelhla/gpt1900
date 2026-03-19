#!/bin/bash
# Download a model from HuggingFace and launch interactive chat or generation.
#
# Usage:
#   bash runs/chat.sh                                                 # default: sft model
#   bash runs/chat.sh -i base                                         # base pretrained model
#   bash runs/chat.sh -r mhla/gpt1900-d34-contradiction-rl-v11        # any HF repo (chat)
#   bash runs/chat.sh -i base -r mhla/gpt1900-d34-22btok --step 9000  # base + specific step
#   bash runs/chat.sh --temperature 0.8                                # extra generation args
#   bash runs/chat.sh --download-only -r mhla/gpt1900-d34-rl          # download without running

set -e

NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-/opt/dlami/nvme/hf_cache}"
export HF_HOME="${NANOCHAT_BASE_DIR}/huggingface"

# Defaults
declare -A DEFAULT_HF_MAP=(
    [base]="mhla/gpt1900-d34-22btok"
    [sft]="mhla/gpt1900-instruct-v3-sft"
)

# Parse args
SOURCE=""
HF_REPO=""
DOWNLOAD_ONLY=0
PASSTHROUGH=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        -i|--source)
            SOURCE="$2"; shift 2 ;;
        -r|--repo)
            HF_REPO="$2"; shift 2 ;;
        --download-only)
            DOWNLOAD_ONLY=1; shift ;;
        *)
            PASSTHROUGH+=("$1"); shift ;;
    esac
done

# If no -r and no -i, default to sft
if [[ -z "$HF_REPO" && -z "$SOURCE" ]]; then
    SOURCE="sft"
fi

# If -i given without -r, use the default repo for that source
if [[ -z "$HF_REPO" && -n "$SOURCE" ]]; then
    HF_REPO="${DEFAULT_HF_MAP[$SOURCE]:-}"
    if [[ -z "$HF_REPO" ]]; then
        echo "No default HF repo for source '$SOURCE'. Specify one with -r mhla/repo-name"
        exit 1
    fi
fi

# Auto-detect source from repo name if not explicitly set
if [[ -z "$SOURCE" ]]; then
    case "$HF_REPO" in
        *sft*|*rl*|*instruct*|*dialogue*) SOURCE="sft" ;;
        *) SOURCE="base" ;;
    esac
fi

# Download into a directory named after the repo
REPO_NAME="${HF_REPO##*/}"
MODEL_DIR="$NANOCHAT_BASE_DIR/$REPO_NAME"

if [ -z "$(find "$MODEL_DIR" -name 'model_*.pt' 2>/dev/null | head -1)" ]; then
    echo "Downloading from $HF_REPO..."
    mkdir -p "$MODEL_DIR"
    huggingface-cli download "$HF_REPO" --local-dir "$MODEL_DIR" --exclude "optim_*"
    echo "Download complete."
else
    echo "Model already downloaded in $MODEL_DIR"
fi

if [[ "$DOWNLOAD_ONLY" == 1 ]]; then
    echo "Download complete: $MODEL_DIR"
    exit 0
fi

echo ""
cat << 'BANNER'

       _____ _____ _______   __  ___   ___   ___
      / ____|  __ \__   __| /_ |/ _ \ / _ \ / _ \
     | |  __| |__) | | |     | | (_) | | | | | | |
     | | |_ |  ___/  | |     | |\__, | | | | | | |
     | |__| | |      | |     | |  / /| |_| | |_| |
      \_____|_|      |_|     |_| /_/  \___/ \___/

BANNER

if [[ "$SOURCE" == "base" ]]; then
    echo "  mode: base completion  |  model: $REPO_NAME"
    echo ""
    python -m scripts.generate --model-dir "$MODEL_DIR" "${PASSTHROUGH[@]}"
else
    echo "  mode: chat ($SOURCE)  |  model: $REPO_NAME"
    echo ""
    python -m scripts.chat_cli --model-dir "$MODEL_DIR" --single-turn "${PASSTHROUGH[@]}"
fi
