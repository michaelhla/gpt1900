#!/bin/bash
# =============================================================================
# AWS Launch Script for GPT-1900 Fine-Tuning
# =============================================================================
# Provisions EC2 instances (spot or on-demand) and kicks off training.
#
# Prerequisites:
#   - AWS CLI v2 configured (aws configure)
#   - SSH key pair created in the target region
#   - Sufficient quota for p5/p4d instances
#
# Usage:
#   # Single node d26 pretrain (spot, ~$8/hr):
#   DEPTH=26 NUM_NODES=1 bash runs/cloud/aws_launch.sh
#
#   # Multi-node d34 pretrain (8 nodes, spot):
#   DEPTH=34 NUM_NODES=8 bash runs/cloud/aws_launch.sh
#
#   # On-demand for critical runs:
#   AWS_SPOT=0 DEPTH=34 bash runs/cloud/aws_launch.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

# -----------------------------------------------------------------------------
# Validate prerequisites
# -----------------------------------------------------------------------------
command -v aws &>/dev/null || { echo "ERROR: AWS CLI not found. Install: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html"; exit 1; }

if [ -z "$AWS_KEY_NAME" ]; then
    echo "ERROR: AWS_KEY_NAME not set. Export your SSH key pair name."
    echo "  Example: export AWS_KEY_NAME=my-gpu-key"
    exit 1
fi

# Auto-detect instance type
if [ -z "$AWS_INSTANCE_TYPE" ]; then
    AWS_INSTANCE_TYPE=$(get_aws_instance_type)
fi
echo "[aws] Instance type: $AWS_INSTANCE_TYPE"

# Auto-detect Deep Learning AMI (Ubuntu, latest)
if [ -z "$AWS_AMI_ID" ]; then
    echo "[aws] Looking up Deep Learning AMI..."
    AWS_AMI_ID=$(aws ec2 describe-images \
        --region "$AWS_REGION" \
        --owners amazon \
        --filters \
            "Name=name,Values=Deep Learning Base OSS Nvidia Driver AMI (Ubuntu 22.04)*" \
            "Name=state,Values=available" \
        --query 'Images | sort_by(@, &CreationDate) | [-1].ImageId' \
        --output text 2>/dev/null || echo "")

    if [ -z "$AWS_AMI_ID" ] || [ "$AWS_AMI_ID" = "None" ]; then
        echo "ERROR: Could not find Deep Learning AMI. Set AWS_AMI_ID manually."
        exit 1
    fi
fi
echo "[aws] AMI: $AWS_AMI_ID"

# -----------------------------------------------------------------------------
# Build user-data bootstrap script
# -----------------------------------------------------------------------------
REPO_URL=$(git remote get-url origin 2>/dev/null || echo "")
REPO_BRANCH=$(git branch --show-current 2>/dev/null || echo "main")

USER_DATA=$(cat <<'BOOTSTRAP'
#!/bin/bash
set -euo pipefail
exec > /var/log/gpt1900-setup.log 2>&1

echo "=== [$(date)] Starting GPT-1900 setup ==="

# System setup
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq git screen

# Clone repo
WORK_DIR=/opt/gpt1900
BOOTSTRAP

# Append dynamic parts
USER_DATA+="
git clone ${REPO_URL:-https://github.com/michaelhla/gpt1900.git} \$WORK_DIR
cd \$WORK_DIR
git checkout ${REPO_BRANCH}
"

USER_DATA+=$(cat <<'BOOTSTRAP2'

# Environment setup
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR=/opt/gpt1900_data
mkdir -p $NANOCHAT_BASE_DIR

# Install uv and dependencies
command -v uv &>/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate

echo "=== [$(date)] Setup complete, ready for training ==="
echo "READY" > /tmp/gpt1900-ready
BOOTSTRAP2
)

ENCODED_USER_DATA=$(echo "$USER_DATA" | base64 -w0)

# -----------------------------------------------------------------------------
# Security group (create if needed)
# -----------------------------------------------------------------------------
if [ -z "$AWS_SECURITY_GROUP" ]; then
    SG_NAME="gpt1900-training-sg"
    AWS_SECURITY_GROUP=$(aws ec2 describe-security-groups \
        --region "$AWS_REGION" \
        --filters "Name=group-name,Values=$SG_NAME" \
        --query 'SecurityGroups[0].GroupId' \
        --output text 2>/dev/null || echo "None")

    if [ "$AWS_SECURITY_GROUP" = "None" ] || [ -z "$AWS_SECURITY_GROUP" ]; then
        echo "[aws] Creating security group: $SG_NAME"
        AWS_SECURITY_GROUP=$(aws ec2 create-security-group \
            --region "$AWS_REGION" \
            --group-name "$SG_NAME" \
            --description "GPT-1900 training instances" \
            --query 'GroupId' --output text)

        # SSH access
        aws ec2 authorize-security-group-ingress \
            --region "$AWS_REGION" \
            --group-id "$AWS_SECURITY_GROUP" \
            --protocol tcp --port 22 --cidr 0.0.0.0/0

        # Inter-node NCCL communication (for multi-node)
        aws ec2 authorize-security-group-ingress \
            --region "$AWS_REGION" \
            --group-id "$AWS_SECURITY_GROUP" \
            --protocol tcp --port 29500-29600 \
            --source-group "$AWS_SECURITY_GROUP"

        # Allow all traffic within security group (NCCL, EFA)
        aws ec2 authorize-security-group-ingress \
            --region "$AWS_REGION" \
            --group-id "$AWS_SECURITY_GROUP" \
            --protocol -1 --source-group "$AWS_SECURITY_GROUP"
    fi
fi
echo "[aws] Security group: $AWS_SECURITY_GROUP"

# -----------------------------------------------------------------------------
# Launch instance(s)
# -----------------------------------------------------------------------------
COMMON_ARGS=(
    --region "$AWS_REGION"
    --image-id "$AWS_AMI_ID"
    --instance-type "$AWS_INSTANCE_TYPE"
    --key-name "$AWS_KEY_NAME"
    --security-group-ids "$AWS_SECURITY_GROUP"
    --block-device-mappings "[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"VolumeSize\":${AWS_EBS_SIZE},\"VolumeType\":\"gp3\",\"Iops\":16000,\"Throughput\":1000}}]"
    --user-data "$ENCODED_USER_DATA"
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=gpt1900-${TRAIN_STAGE}-d${DEPTH}},{Key=Project,Value=gpt1900}]"
    --count "$NUM_NODES"
)

# Add subnet if specified
if [ -n "$AWS_SUBNET_ID" ]; then
    COMMON_ARGS+=(--subnet-id "$AWS_SUBNET_ID")
fi

# Add placement group for multi-node (required for EFA and low-latency NCCL)
if [ "$NUM_NODES" -gt 1 ]; then
    PG_NAME="gpt1900-placement-group"
    aws ec2 create-placement-group \
        --region "$AWS_REGION" \
        --group-name "$PG_NAME" \
        --strategy cluster 2>/dev/null || true
    COMMON_ARGS+=(--placement "GroupName=$PG_NAME")
fi

if [ "$AWS_SPOT" = "1" ]; then
    echo "[aws] Requesting $NUM_NODES spot instance(s)..."
    SPOT_OPTS="MarketType=spot,SpotOptions={SpotInstanceType=persistent,InstanceInterruptionBehavior=stop"
    if [ -n "$AWS_SPOT_MAX_PRICE" ]; then
        SPOT_OPTS+=",MaxPrice=$AWS_SPOT_MAX_PRICE"
    fi
    SPOT_OPTS+="}"
    COMMON_ARGS+=(--instance-market-options "$SPOT_OPTS")
else
    echo "[aws] Requesting $NUM_NODES on-demand instance(s)..."
fi

INSTANCE_IDS=$(aws ec2 run-instances "${COMMON_ARGS[@]}" \
    --query 'Instances[].InstanceId' --output text)

echo "[aws] Launched instances: $INSTANCE_IDS"

# -----------------------------------------------------------------------------
# Wait for instances and get IPs
# -----------------------------------------------------------------------------
echo "[aws] Waiting for instances to be running..."
aws ec2 wait instance-running --region "$AWS_REGION" --instance-ids $INSTANCE_IDS

PUBLIC_IPS=$(aws ec2 describe-instances \
    --region "$AWS_REGION" \
    --instance-ids $INSTANCE_IDS \
    --query 'Reservations[].Instances[].PublicIpAddress' --output text)

PRIVATE_IPS=$(aws ec2 describe-instances \
    --region "$AWS_REGION" \
    --instance-ids $INSTANCE_IDS \
    --query 'Reservations[].Instances[].PrivateIpAddress' --output text)

echo ""
echo "============================================="
echo "  AWS Instances Ready"
echo "============================================="
echo "Instance IDs: $INSTANCE_IDS"
echo "Public IPs:   $PUBLIC_IPS"
echo "Private IPs:  $PRIVATE_IPS"
echo ""
echo "SSH into head node:"
HEAD_IP=$(echo "$PUBLIC_IPS" | awk '{print $1}')
echo "  ssh -i ~/.ssh/${AWS_KEY_NAME}.pem ubuntu@${HEAD_IP}"
echo ""

if [ "$NUM_NODES" -gt 1 ]; then
    HEAD_PRIVATE=$(echo "$PRIVATE_IPS" | awk '{print $1}')
    echo "For multi-node training, on the head node run:"
    echo "  export MASTER_ADDR=$HEAD_PRIVATE"
    echo "  export MASTER_PORT=29500"
    echo "  # Then launch training with srun or pdsh across all nodes"
fi

echo ""
echo "Once setup completes (check /var/log/gpt1900-setup.log), start training:"
echo "  cd /opt/gpt1900 && source .venv/bin/activate"
echo "  CLOUD_PROVIDER=aws DEPTH=$DEPTH TRAIN_STAGE=$TRAIN_STAGE bash runs/launch_cloud.sh"
echo ""

# Save instance info for later cleanup
INFO_FILE="/tmp/gpt1900_aws_instances.txt"
echo "$INSTANCE_IDS" > "$INFO_FILE"
echo "[aws] Instance IDs saved to $INFO_FILE"
echo ""
echo "To terminate when done:"
echo "  aws ec2 terminate-instances --region $AWS_REGION --instance-ids $INSTANCE_IDS"
