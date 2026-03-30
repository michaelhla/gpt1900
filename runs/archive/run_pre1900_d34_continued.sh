#!/bin/bash
#SBATCH --partition=highpri
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --mem=0
#SBATCH --job-name=pre1900_d34_cont
#SBATCH --output=/mnt/main0/home/michaelhla/pre1900_train_%j.log

cd /mnt/main0/home/michaelhla/evolutionaryscale/gpt1900
export PATH="/mnt/main0/home/michaelhla/evolutionaryscale/.pixi/envs/default/bin:$PATH"
export CONDA_PREFIX="/mnt/main0/home/michaelhla/evolutionaryscale/.pixi/envs/default"

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR=/mnt/main0/data/michaelhla/gpt1900_training
export PYTHONPATH=/mnt/main0/home/michaelhla/evolutionaryscale/gpt1900

torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --depth=34 --target-param-data-ratio=20 --device-batch-size=4 --fp8 \
    --run=pre1900_d34_continued --save-every=1000 --window-pattern L \
    --model-tag=d34-8b-subset \
    --resume-from-step=10507
