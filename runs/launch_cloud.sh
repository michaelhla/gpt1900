#!/bin/bash
# =============================================================================
# Cloud Training Launcher for GPT-1900
# =============================================================================
# Unified entry point for all training stages on any cloud provider.
# Run this ON the GPU instance after provisioning with aws_launch.sh or runpod_launch.sh.
#
# Usage:
#   # Pretrain d26 (single node, 8xH100, ~3 hrs):
#   DEPTH=26 TRAIN_STAGE=pretrain bash runs/launch_cloud.sh
#
#   # Pretrain d34 (single node, 8xH100, ~12 hrs):
#   DEPTH=34 TRAIN_STAGE=pretrain WANDB_RUN=pre1900_d34 bash runs/launch_cloud.sh
#
#   # SFT on pretrained d34:
#   DEPTH=34 TRAIN_STAGE=sft bash runs/launch_cloud.sh
#
#   # Reasoning SFT:
#   DEPTH=34 TRAIN_STAGE=reasoning_sft bash runs/launch_cloud.sh
#
#   # Discovery RL:
#   DEPTH=34 TRAIN_STAGE=discovery_rl bash runs/launch_cloud.sh
#
#   # Full pipeline (pretrain -> sft -> reasoning_sft -> discovery_rl):
#   DEPTH=34 TRAIN_STAGE=full_pipeline bash runs/launch_cloud.sh
#
# Environment variables (see runs/cloud/config.sh for full list):
#   CLOUD_PROVIDER  - "aws" or "runpod" (default: aws)
#   DEPTH           - transformer depth (default: 26)
#   TRAIN_STAGE     - training stage (default: pretrain)
#   NUM_NODES       - number of GPU nodes (default: 1)
#   WANDB_RUN       - wandb run name (default: dummy)
#   GPU_TYPE        - h100, a100_80g, a100_40g (default: h100)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

# Load configuration
source "$SCRIPT_DIR/cloud/config.sh"

# -----------------------------------------------------------------------------
# Environment setup (cloud-agnostic)
# -----------------------------------------------------------------------------
setup_env() {
    echo "=== [$(date)] Setting up environment ==="
    export OMP_NUM_THREADS=1
    export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR}"
    mkdir -p "$NANOCHAT_BASE_DIR"

    # Install uv if needed
    command -v uv &>/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"

    # Create venv and install deps
    [ -d ".venv" ] || uv venv
    uv sync --extra gpu
    source .venv/bin/activate

    echo "=== [$(date)] Environment ready ==="
}

# -----------------------------------------------------------------------------
# Data setup
# -----------------------------------------------------------------------------
setup_data() {
    echo "=== [$(date)] Setting up data ==="

    if [ "$TRAIN_STAGE" = "pretrain" ] || [ "$TRAIN_STAGE" = "full_pipeline" ]; then
        # Download initial shards for tokenizer training
        python -m nanochat.dataset -n 8
        # Background download remaining shards
        python -m nanochat.dataset -n "$NUM_DATA_SHARDS" &
        DATA_DOWNLOAD_PID=$!

        # Train tokenizer
        python -m scripts.tok_train
        python -m scripts.tok_eval

        echo "Waiting for data download to complete..."
        wait $DATA_DOWNLOAD_PID
    fi

    echo "=== [$(date)] Data ready ==="
}

# -----------------------------------------------------------------------------
# Build torchrun command
# -----------------------------------------------------------------------------
TORCHRUN_CMD=$(get_torchrun_cmd)
FP8_FLAG=$(get_fp8_flag)
MODEL_TAG="${MODEL_TAG:-d${DEPTH}}"

# For multi-node on AWS, MASTER_ADDR should already be set by the caller
if [ "$NUM_NODES" -gt 1 ]; then
    if [ -z "${MASTER_ADDR:-}" ]; then
        echo "ERROR: MASTER_ADDR must be set for multi-node training"
        echo "  On AWS: export MASTER_ADDR=<head_node_private_ip>"
        exit 1
    fi
    export MASTER_PORT="${MASTER_PORT:-29500}"
    # NCCL tuning for multi-node
    export NCCL_IB_DISABLE=0
    export NCCL_NET_GDR_LEVEL=5
    echo "[multi-node] MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"
fi

# -----------------------------------------------------------------------------
# Training stages
# -----------------------------------------------------------------------------
run_pretrain() {
    echo "=== [$(date)] Starting pretraining: depth=$DEPTH, nodes=$NUM_NODES ==="

    local PRETRAIN_WANDB="${WANDB_RUN}"
    if [ "$PRETRAIN_WANDB" = "dummy" ]; then
        PRETRAIN_WANDB="pre1900_d${DEPTH}_pretrain"
    fi

    $TORCHRUN_CMD -m scripts.base_train -- \
        --depth="$DEPTH" \
        --target-param-data-ratio="$TARGET_PARAM_DATA_RATIO" \
        --device-batch-size="$DEVICE_BATCH_SIZE" \
        --total-batch-size="$TOTAL_BATCH_SIZE" \
        $FP8_FLAG \
        --window-pattern "$WINDOW_PATTERN" \
        --save-every="$SAVE_EVERY" \
        --model-tag="$MODEL_TAG" \
        --run="$PRETRAIN_WANDB"

    # Evaluate
    $TORCHRUN_CMD -m scripts.base_eval -- --device-batch-size="$DEVICE_BATCH_SIZE"

    echo "=== [$(date)] Pretraining complete ==="
}

run_sft() {
    echo "=== [$(date)] Starting SFT: model_tag=$MODEL_TAG ==="

    local SFT_WANDB="${WANDB_RUN}"
    if [ "$SFT_WANDB" = "dummy" ]; then
        SFT_WANDB="pre1900_sft_${MODEL_TAG}"
    fi

    $TORCHRUN_CMD -m scripts.pre1900_sft -- \
        --model-tag "$MODEL_TAG" \
        --checkpoints-dir "$BASE_CHECKPOINTS_DIR" \
        --output-dir "$SFT_CHECKPOINTS_DIR" \
        --device-batch-size="$DEVICE_BATCH_SIZE" \
        --train-data "$SFT_TRAIN_DATA" \
        --val-data "$SFT_VAL_DATA" \
        --run="$SFT_WANDB"

    echo "=== [$(date)] SFT complete ==="
}

run_reasoning_sft() {
    echo "=== [$(date)] Starting reasoning SFT: model_tag=$MODEL_TAG ==="

    local RSFT_WANDB="${WANDB_RUN}"
    if [ "$RSFT_WANDB" = "dummy" ]; then
        RSFT_WANDB="pre1900_reasoning_sft_${MODEL_TAG}"
    fi

    $TORCHRUN_CMD -m scripts.pre1900_sft -- \
        --model-tag "$MODEL_TAG" \
        --checkpoints-dir "$BASE_CHECKPOINTS_DIR" \
        --output-dir "$REASONING_SFT_CHECKPOINTS_DIR" \
        --device-batch-size="$DEVICE_BATCH_SIZE" \
        --total-batch-size=65536 \
        --num-iterations 100 \
        --train-data "$REASONING_SFT_TRAIN_DATA" \
        --val-data "$REASONING_SFT_VAL_DATA" \
        --run="$RSFT_WANDB"

    echo "=== [$(date)] Reasoning SFT complete ==="
}

run_discovery_rl() {
    echo "=== [$(date)] Starting discovery RL: model_tag=$MODEL_TAG ==="

    local RL_WANDB="${WANDB_RUN}"
    if [ "$RL_WANDB" = "dummy" ]; then
        RL_WANDB="pre1900_discovery_rl_${MODEL_TAG}"
    fi

    $TORCHRUN_CMD -m scripts.pre1900_scripts.discovery_rl -- \
        --model-tag "$MODEL_TAG" \
        --checkpoints-dir "$REASONING_SFT_CHECKPOINTS_DIR" \
        --output-dir "$DISCOVERY_RL_CHECKPOINTS_DIR" \
        --device-batch-size 2 \
        --num-samples 4 \
        --examples-per-step 8 \
        --max-new-tokens 1024 \
        --num-epochs 3 \
        --train-data "$RL_TRAIN_DATA" \
        --val-data "$RL_VAL_DATA" \
        --problems-data "$RL_PROBLEMS_TRAIN" \
        --problems-val-data "$RL_PROBLEMS_VAL" \
        --eval-every 15 --save-every 30 \
        --run="$RL_WANDB"

    echo "=== [$(date)] Discovery RL complete ==="
}

run_full_pipeline() {
    echo "=== [$(date)] Starting full pipeline ==="
    run_pretrain
    run_sft
    run_reasoning_sft
    run_discovery_rl
    echo "=== [$(date)] Full pipeline complete ==="
}

# -----------------------------------------------------------------------------
# Checkpoint sync (optional S3 backup for AWS)
# -----------------------------------------------------------------------------
sync_checkpoints() {
    if [ "$CLOUD_PROVIDER" = "aws" ] && [ -n "${AWS_S3_BUCKET:-}" ]; then
        echo "[sync] Syncing checkpoints to s3://${AWS_S3_BUCKET}/checkpoints/"
        aws s3 sync "$NANOCHAT_BASE_DIR" "s3://${AWS_S3_BUCKET}/checkpoints/" \
            --exclude "*" --include "*/model_*.pt" --include "*/config.json" \
            --quiet
    fi
}

# Sync checkpoints periodically in the background (every 30 min)
if [ "$CLOUD_PROVIDER" = "aws" ] && [ -n "${AWS_S3_BUCKET:-}" ]; then
    (while true; do sleep 1800; sync_checkpoints; done) &
    SYNC_PID=$!
    trap "kill $SYNC_PID 2>/dev/null; sync_checkpoints" EXIT
fi

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
setup_env

case "$TRAIN_STAGE" in
    pretrain)
        setup_data
        run_pretrain
        ;;
    sft)
        run_sft
        ;;
    reasoning_sft)
        run_reasoning_sft
        ;;
    discovery_rl)
        run_discovery_rl
        ;;
    full_pipeline)
        setup_data
        run_full_pipeline
        ;;
    *)
        echo "ERROR: Unknown TRAIN_STAGE=$TRAIN_STAGE"
        echo "Valid stages: pretrain, sft, reasoning_sft, discovery_rl, full_pipeline"
        exit 1
        ;;
esac

# Final checkpoint sync
sync_checkpoints

echo ""
echo "============================================="
echo "  Training Complete!"
echo "============================================="
echo "Stage:       $TRAIN_STAGE"
echo "Model:       d${DEPTH} (${MODEL_TAG})"
echo "Checkpoints: ${NANOCHAT_BASE_DIR}/"
echo "============================================="
