#!/bin/bash
# =============================================================================
# RunPod Launch Script for GPT-1900 Fine-Tuning
# =============================================================================
# Creates a RunPod GPU pod and sets it up for training.
#
# Prerequisites:
#   - RunPod API key (https://www.runpod.io/console/user/settings)
#   - curl and jq installed
#
# Usage:
#   # Single node d26 pretrain:
#   RUNPOD_API_KEY=xxx DEPTH=26 bash runs/cloud/runpod_launch.sh
#
#   # d34 with specific GPU count:
#   RUNPOD_API_KEY=xxx DEPTH=34 GPUS_PER_NODE=8 bash runs/cloud/runpod_launch.sh
#
# RunPod is best for:
#   - Single-node training (1-8 GPUs)
#   - Quick iteration / cheaper spot-like pricing
#   - When AWS capacity is unavailable

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

# -----------------------------------------------------------------------------
# Validate prerequisites
# -----------------------------------------------------------------------------
command -v curl &>/dev/null || { echo "ERROR: curl not found"; exit 1; }
command -v jq &>/dev/null || { echo "ERROR: jq not found. Install: apt-get install jq"; exit 1; }

if [ -z "$RUNPOD_API_KEY" ]; then
    echo "ERROR: RUNPOD_API_KEY not set."
    echo "  Get your key at: https://www.runpod.io/console/user/settings"
    echo "  Then: export RUNPOD_API_KEY=your_key_here"
    exit 1
fi

RUNPOD_API="https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}"

# Auto-detect GPU type for RunPod
if [ -z "$RUNPOD_GPU_TYPE" ]; then
    RUNPOD_GPU_TYPE=$(get_runpod_gpu_id)
fi
echo "[runpod] GPU type: $RUNPOD_GPU_TYPE"
echo "[runpod] GPU count: $GPUS_PER_NODE"

# -----------------------------------------------------------------------------
# Build the docker start command
# -----------------------------------------------------------------------------
REPO_URL=$(git remote get-url origin 2>/dev/null || echo "https://github.com/michaelhla/gpt1900.git")
REPO_BRANCH=$(git branch --show-current 2>/dev/null || echo "main")

# RunPod start command: clone repo, install deps, signal ready
START_CMD="bash -c '
set -e
apt-get update -qq && apt-get install -y -qq git screen
cd /workspace
git clone ${REPO_URL} gpt1900 || true
cd gpt1900
git checkout ${REPO_BRANCH} || true
git pull origin ${REPO_BRANCH} || true

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR=/workspace/gpt1900_data
mkdir -p \$NANOCHAT_BASE_DIR

command -v uv || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH=\"\$HOME/.local/bin:\$PATH\"
[ -d \".venv\" ] || uv venv
uv sync --extra gpu
echo READY > /tmp/gpt1900-ready
sleep infinity
'"

# -----------------------------------------------------------------------------
# Check GPU availability
# -----------------------------------------------------------------------------
echo "[runpod] Checking GPU availability..."
AVAIL_QUERY=$(cat <<EOF
{
  "query": "query { gpuTypes { id displayName memoryInGb securePrice communityPrice secureSpotPrice communitySpotPrice } }"
}
EOF
)

GPU_INFO=$(curl -s -X POST "$RUNPOD_API" \
    -H "Content-Type: application/json" \
    -d "$AVAIL_QUERY" | jq -r ".data.gpuTypes[] | select(.id == \"$RUNPOD_GPU_TYPE\" or .displayName == \"$RUNPOD_GPU_TYPE\")" 2>/dev/null || echo "")

if [ -n "$GPU_INFO" ]; then
    GPU_ID=$(echo "$GPU_INFO" | jq -r '.id')
    GPU_DISPLAY=$(echo "$GPU_INFO" | jq -r '.displayName')
    SECURE_PRICE=$(echo "$GPU_INFO" | jq -r '.securePrice // "N/A"')
    COMMUNITY_PRICE=$(echo "$GPU_INFO" | jq -r '.communityPrice // "N/A"')
    echo "[runpod] GPU: $GPU_DISPLAY"
    echo "[runpod] Pricing: Secure=\$${SECURE_PRICE}/hr/gpu, Community=\$${COMMUNITY_PRICE}/hr/gpu"
    echo "[runpod] Estimated cost for ${GPUS_PER_NODE} GPUs: \$$(echo "$SECURE_PRICE * $GPUS_PER_NODE" | bc 2>/dev/null || echo "?")/hr"
else
    GPU_ID="$RUNPOD_GPU_TYPE"
    echo "[runpod] Could not fetch pricing info, proceeding with GPU ID: $GPU_ID"
fi

# -----------------------------------------------------------------------------
# Create the pod
# -----------------------------------------------------------------------------
echo "[runpod] Creating pod with ${GPUS_PER_NODE}x GPU..."

# Escape the start command for JSON
ESCAPED_CMD=$(echo "$START_CMD" | python3 -c "import sys,json; print(json.dumps(sys.stdin.read()))" | sed 's/^"//;s/"$//')

CREATE_QUERY=$(cat <<EOF
{
  "query": "mutation { podFindAndDeployOnDemand(input: { name: \"gpt1900-${TRAIN_STAGE}-d${DEPTH}\", gpuTypeId: \"${GPU_ID}\", gpuCount: ${GPUS_PER_NODE}, cloudType: ${RUNPOD_CLOUD_TYPE}, volumeInGb: ${RUNPOD_VOLUME_SIZE}, containerDiskInGb: 50, dockerArgs: \"${ESCAPED_CMD}\", imageName: \"runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04\", supportPublicIp: true, ports: \"22/tcp,8000/http\" }) { id imageName desiredStatus machine { podHostId } } }"
}
EOF
)

RESPONSE=$(curl -s -X POST "$RUNPOD_API" \
    -H "Content-Type: application/json" \
    -d "$CREATE_QUERY")

POD_ID=$(echo "$RESPONSE" | jq -r '.data.podFindAndDeployOnDemand.id // empty')

if [ -z "$POD_ID" ]; then
    echo "ERROR: Failed to create pod"
    echo "Response: $RESPONSE"
    echo ""
    echo "Common issues:"
    echo "  - Insufficient funds in RunPod account"
    echo "  - GPU type unavailable in selected cloud type"
    echo "  - Try RUNPOD_CLOUD_TYPE=COMMUNITY for more availability"
    exit 1
fi

echo "[runpod] Pod created: $POD_ID"

# -----------------------------------------------------------------------------
# Wait for pod to be ready
# -----------------------------------------------------------------------------
echo "[runpod] Waiting for pod to start..."
for i in $(seq 1 60); do
    POD_STATUS=$(curl -s -X POST "$RUNPOD_API" \
        -H "Content-Type: application/json" \
        -d "{\"query\": \"{ pod(input: { podId: \\\"${POD_ID}\\\" }) { id desiredStatus runtime { uptimeInSeconds ports { ip isIpPublic privatePort publicPort type } } } }\"}" \
        | jq -r '.data.pod')

    STATUS=$(echo "$POD_STATUS" | jq -r '.desiredStatus')
    UPTIME=$(echo "$POD_STATUS" | jq -r '.runtime.uptimeInSeconds // 0')

    if [ "$STATUS" = "RUNNING" ] && [ "$UPTIME" != "0" ] && [ "$UPTIME" != "null" ]; then
        break
    fi
    echo "  Status: $STATUS (attempt $i/60)..."
    sleep 10
done

# Get connection info
SSH_INFO=$(echo "$POD_STATUS" | jq -r '.runtime.ports[] | select(.privatePort == 22) | "\(.ip):\(.publicPort)"' 2>/dev/null || echo "")

echo ""
echo "============================================="
echo "  RunPod Instance Ready"
echo "============================================="
echo "Pod ID:    $POD_ID"
echo "Status:    $STATUS"
if [ -n "$SSH_INFO" ]; then
    SSH_HOST=$(echo "$SSH_INFO" | cut -d: -f1)
    SSH_PORT=$(echo "$SSH_INFO" | cut -d: -f2)
    echo "SSH:       ssh root@${SSH_HOST} -p ${SSH_PORT} -i ~/.ssh/id_ed25519"
fi
echo ""
echo "Once the pod is ready, SSH in and start training:"
echo "  cd /workspace/gpt1900 && source .venv/bin/activate"
echo "  CLOUD_PROVIDER=runpod DEPTH=$DEPTH TRAIN_STAGE=$TRAIN_STAGE bash runs/launch_cloud.sh"
echo ""
echo "Monitor via RunPod console:"
echo "  https://www.runpod.io/console/pods"
echo ""
echo "To stop the pod when done:"
echo "  curl -s -X POST '$RUNPOD_API' -H 'Content-Type: application/json' \\"
echo "    -d '{\"query\": \"mutation { podStop(input: { podId: \\\"${POD_ID}\\\" }) { id } }\"}'"
echo ""
echo "To terminate (delete) the pod:"
echo "  curl -s -X POST '$RUNPOD_API' -H 'Content-Type: application/json' \\"
echo "    -d '{\"query\": \"mutation { podTerminate(input: { podId: \\\"${POD_ID}\\\" }) }\"}'"

# Save pod info
echo "$POD_ID" > /tmp/gpt1900_runpod_pod.txt
echo "[runpod] Pod ID saved to /tmp/gpt1900_runpod_pod.txt"
