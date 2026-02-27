#!/bin/bash
#SBATCH --partition=highpri
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=0
#SBATCH --job-name=d52_memprofile
#SBATCH --output=/mnt/main0/home/michaelhla/d52_memprofile_%j.log

cd /mnt/main0/home/michaelhla/evolutionaryscale/gpt1900
export PATH="/mnt/main0/home/michaelhla/evolutionaryscale/.pixi/envs/default/bin:$PATH"
export CONDA_PREFIX="/mnt/main0/home/michaelhla/evolutionaryscale/.pixi/envs/default"
export PYTHONPATH=/mnt/main0/home/michaelhla/evolutionaryscale/gpt1900
export NANOCHAT_BASE_DIR=/mnt/main0/data/michaelhla/gpt1915_training
export TORCH_COMPILE_DISABLE=1

python -m scripts.base_train \
    --depth=52 --aspect-ratio=64 --head-dim=128 \
    --num-iterations=2 --device-batch-size=2 --max-seq-len=2048 \
    --window-pattern L
