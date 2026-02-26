#!/bin/bash
#SBATCH --partition=highpri
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --mem=0
#SBATCH --job-name=physics_clm_d34
#SBATCH --output=/mnt/main0/home/michaelhla/physics_clm_d34_%j.log

cd /mnt/main0/home/michaelhla/evolutionaryscale/gpt1900
export PATH="/mnt/main0/home/michaelhla/evolutionaryscale/.pixi/envs/default/bin:$PATH"
export CONDA_PREFIX="/mnt/main0/home/michaelhla/evolutionaryscale/.pixi/envs/default"

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR=/mnt/main0/data/michaelhla/gpt1900_training
export PYTHONPATH=/mnt/main0/home/michaelhla/evolutionaryscale/gpt1900

DATA_DIR=${NANOCHAT_BASE_DIR}/physics_clm_data

# Step 1: Prepare data (if not already done)
if [ ! -f "${DATA_DIR}/train.parquet" ]; then
    echo "=== Preparing physics parquet data ==="
    python -m scripts.pre1900_scripts.prepare_physics_parquet \
        --input-dir data/physics_books data/core_physics_books \
        --output-dir ${DATA_DIR}
fi

# Step 2: Train
echo "=== Starting physics CLM training ==="
torchrun --standalone --nproc_per_node=8 \
    -m scripts.physics_clm -- \
    --data-dir ${DATA_DIR} \
    --num-epochs 5 \
    --device-batch-size 8 \
    --total-batch-size 65536 \
    --matrix-lr 0.005 \
    --embedding-lr 0.05 \
    --unembedding-lr 0.001 \
    --eval-every 50 \
    --save-every 200 \
    --output-tag pre1900_physics_clm_d34 \
    --run pre1900_physics_clm_d34
