#!/bin/bash
# =============================================================================
# Cloud Configuration for GPT-1900 Fine-Tuning
# =============================================================================
# This file defines all cloud-agnostic settings. Source it from launch scripts.
# Override any variable by exporting it before sourcing this file.
#
# Usage:
#   source runs/cloud/config.sh
#   # or override:
#   CLOUD_PROVIDER=runpod DEPTH=34 source runs/cloud/config.sh

# -----------------------------------------------------------------------------
# Cloud provider: "aws" or "runpod"
# -----------------------------------------------------------------------------
CLOUD_PROVIDER="${CLOUD_PROVIDER:-aws}"

# -----------------------------------------------------------------------------
# Training stage: "pretrain", "sft", "reasoning_sft", "discovery_rl"
# -----------------------------------------------------------------------------
TRAIN_STAGE="${TRAIN_STAGE:-pretrain}"

# -----------------------------------------------------------------------------
# Model configuration
# -----------------------------------------------------------------------------
DEPTH="${DEPTH:-26}"
WINDOW_PATTERN="${WINDOW_PATTERN:-L}"
FP8="${FP8:-1}"                       # 1=enable fp8 (H100 required), 0=disable
TARGET_PARAM_DATA_RATIO="${TARGET_PARAM_DATA_RATIO:-20}"

# -----------------------------------------------------------------------------
# Hardware configuration
# -----------------------------------------------------------------------------
NUM_NODES="${NUM_NODES:-1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
TOTAL_GPUS=$((NUM_NODES * GPUS_PER_NODE))

# GPU type preference (used for instance selection)
# Options: h100, a100_80g, a100_40g
GPU_TYPE="${GPU_TYPE:-h100}"

# Per-device batch size (auto-tuned based on depth if not set)
if [ -z "$DEVICE_BATCH_SIZE" ]; then
    if [ "$DEPTH" -ge 42 ]; then
        DEVICE_BATCH_SIZE=1
    elif [ "$DEPTH" -ge 34 ]; then
        DEVICE_BATCH_SIZE=4
    elif [ "$DEPTH" -ge 20 ]; then
        DEVICE_BATCH_SIZE=16
    else
        DEVICE_BATCH_SIZE=32
    fi
fi

# Total batch size (-1 = auto-compute optimal based on scaling laws)
TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:--1}"

# -----------------------------------------------------------------------------
# Data configuration
# -----------------------------------------------------------------------------
# Base directory for all training artifacts (checkpoints, data, logs)
NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"

# Number of data shards to download for pretraining
NUM_DATA_SHARDS="${NUM_DATA_SHARDS:-370}"

# SFT data paths (relative to NANOCHAT_BASE_DIR)
SFT_TRAIN_DATA="${SFT_TRAIN_DATA:-instruct_data/period/filtered_pairs.jsonl}"
SFT_VAL_DATA="${SFT_VAL_DATA:-instruct_data/period/val_pairs.jsonl}"

# Reasoning SFT data paths
REASONING_SFT_TRAIN_DATA="${REASONING_SFT_TRAIN_DATA:-instruct_data/reasoning/sft_train_trimmed.jsonl}"
REASONING_SFT_VAL_DATA="${REASONING_SFT_VAL_DATA:-instruct_data/reasoning/sft_val_clean.jsonl}"

# RL data paths
RL_TRAIN_DATA="${RL_TRAIN_DATA:-instruct_data/rl_problems/rl_prompts_sys_train_expanded.jsonl}"
RL_VAL_DATA="${RL_VAL_DATA:-instruct_data/rl_problems/rl_prompts_sys_val_clean.jsonl}"
RL_PROBLEMS_TRAIN="${RL_PROBLEMS_TRAIN:-instruct_data/rl_problems/rl_problems_train_expanded.jsonl}"
RL_PROBLEMS_VAL="${RL_PROBLEMS_VAL:-instruct_data/rl_problems/rl_problems_val.jsonl}"

# Checkpoint directories (relative to NANOCHAT_BASE_DIR)
BASE_CHECKPOINTS_DIR="${BASE_CHECKPOINTS_DIR:-base_checkpoints}"
SFT_CHECKPOINTS_DIR="${SFT_CHECKPOINTS_DIR:-pre1900_sft_checkpoints}"
REASONING_SFT_CHECKPOINTS_DIR="${REASONING_SFT_CHECKPOINTS_DIR:-pre1900_reasoning_sft_v4_checkpoints}"
DISCOVERY_RL_CHECKPOINTS_DIR="${DISCOVERY_RL_CHECKPOINTS_DIR:-pre1900_discovery_rl_v4_checkpoints}"

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
WANDB_RUN="${WANDB_RUN:-dummy}"
SAVE_EVERY="${SAVE_EVERY:-2000}"

# -----------------------------------------------------------------------------
# AWS-specific defaults
# -----------------------------------------------------------------------------
AWS_REGION="${AWS_REGION:-us-east-1}"
AWS_INSTANCE_TYPE="${AWS_INSTANCE_TYPE:-}"  # auto-selected if empty
AWS_AMI_ID="${AWS_AMI_ID:-}"               # Deep Learning AMI, auto-detected if empty
AWS_KEY_NAME="${AWS_KEY_NAME:-}"           # SSH key pair name
AWS_SECURITY_GROUP="${AWS_SECURITY_GROUP:-}"
AWS_SUBNET_ID="${AWS_SUBNET_ID:-}"
AWS_SPOT="${AWS_SPOT:-1}"                  # 1=use spot instances, 0=on-demand
AWS_SPOT_MAX_PRICE="${AWS_SPOT_MAX_PRICE:-}"  # empty=market price
AWS_EBS_SIZE="${AWS_EBS_SIZE:-500}"        # root EBS volume in GB
AWS_S3_BUCKET="${AWS_S3_BUCKET:-}"         # for checkpoint sync

# EFA (Elastic Fabric Adapter) for multi-node
AWS_EFA="${AWS_EFA:-1}"                    # 1=enable EFA for multi-node

# -----------------------------------------------------------------------------
# RunPod-specific defaults
# -----------------------------------------------------------------------------
RUNPOD_API_KEY="${RUNPOD_API_KEY:-}"
RUNPOD_GPU_TYPE="${RUNPOD_GPU_TYPE:-}"     # auto-selected if empty
RUNPOD_VOLUME_SIZE="${RUNPOD_VOLUME_SIZE:-500}"  # persistent volume in GB
RUNPOD_CLOUD_TYPE="${RUNPOD_CLOUD_TYPE:-SECURE}"  # SECURE or COMMUNITY
RUNPOD_TEMPLATE_ID="${RUNPOD_TEMPLATE_ID:-}"

# -----------------------------------------------------------------------------
# Capacity planning reference (see runs/cloud/CAPACITY.md for details)
# -----------------------------------------------------------------------------
# Model sizes and estimated training times on 8xH100:
#   d12 (   91M params,   1B tokens):  ~5 min
#   d26 (  918M params,  10B tokens):  ~3 hrs
#   d34 ( 2.1B params,   22B tokens):  ~12 hrs  (1 node)  / ~2 hrs (8 nodes)
#   d45 ( 5.2B params,   55B tokens):  ~72 hrs  (1 node)  / ~10 hrs (8 nodes)
#
# SFT stages are much cheaper: ~30 min per stage on 8xH100 for d34
# RL training: ~2-4 hrs on 8xH100 for d34

# -----------------------------------------------------------------------------
# Derived configuration helpers
# -----------------------------------------------------------------------------

get_aws_instance_type() {
    # Select AWS instance type based on GPU type and node count
    case "$GPU_TYPE" in
        h100)
            echo "p5.48xlarge"  # 8x H100 80GB, 640GB GPU memory
            ;;
        a100_80g)
            echo "p4de.24xlarge"  # 8x A100 80GB, 640GB GPU memory
            ;;
        a100_40g)
            echo "p4d.24xlarge"  # 8x A100 40GB, 320GB GPU memory
            ;;
        *)
            echo "p5.48xlarge"
            ;;
    esac
}

get_runpod_gpu_id() {
    # Select RunPod GPU type identifier
    case "$GPU_TYPE" in
        h100)
            echo "NVIDIA H100 80GB HBM3"
            ;;
        a100_80g)
            echo "NVIDIA A100 80GB PCIe"
            ;;
        a100_40g)
            echo "NVIDIA A100-SXM4-40GB"
            ;;
        *)
            echo "NVIDIA H100 80GB HBM3"
            ;;
    esac
}

get_fp8_flag() {
    if [ "$FP8" = "1" ] && [ "$GPU_TYPE" = "h100" ]; then
        echo "--fp8"
    else
        echo ""
    fi
}

# Build the torchrun command prefix for single or multi-node
get_torchrun_cmd() {
    if [ "$NUM_NODES" -eq 1 ]; then
        echo "torchrun --standalone --nproc_per_node=$GPUS_PER_NODE"
    else
        echo "torchrun --nnodes=$NUM_NODES --nproc_per_node=$GPUS_PER_NODE --rdzv_backend=c10d --rdzv_endpoint=\$MASTER_ADDR:\$MASTER_PORT"
    fi
}

echo "[config] Provider=$CLOUD_PROVIDER Stage=$TRAIN_STAGE Depth=$DEPTH Nodes=${NUM_NODES}x${GPUS_PER_NODE}gpu GPU=$GPU_TYPE FP8=$FP8"
