#!/bin/bash
#SBATCH --partition=highpri
#SBATCH --nodes=8
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --mem=0
#SBATCH --job-name=d52_8node
#SBATCH --output=/mnt/main0/home/michaelhla/pre1915_train_%j.log

cd /mnt/main0/home/michaelhla/evolutionaryscale/gpt1900
export PATH="/mnt/main0/home/michaelhla/evolutionaryscale/.pixi/envs/default/bin:$PATH"
export CONDA_PREFIX="/mnt/main0/home/michaelhla/evolutionaryscale/.pixi/envs/default"

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR=/mnt/main0/data/michaelhla/gpt1915_training
export PYTHONPATH=/mnt/main0/home/michaelhla/evolutionaryscale/gpt1900

# Multi-node rendezvous
HEAD_NODE=$(scontrol show hostname $SLURM_NODELIST | head -1)
export MASTER_ADDR=$HEAD_NODE
export MASTER_PORT=29500

# NCCL tuning for multi-node
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=5

srun torchrun \
    --nnodes=8 \
    --nproc_per_node=8 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$HEAD_NODE:29500 \
    -m scripts.base_train -- \
    --depth=52 --aspect-ratio=64 --head-dim=128 \
    --target-param-data-ratio=11 --device-batch-size=1 --fp8 --activation-checkpointing \
    --run=pre1915_d52_combined --save-every=3000 --window-pattern L
