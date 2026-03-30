#!/bin/bash
#SBATCH --partition=highpri
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --mem=0
#SBATCH --job-name=reasoning_sft_d34
#SBATCH --output=/mnt/main0/home/michaelhla/reasoning_sft_d34_%j.log

cd /mnt/main0/home/michaelhla/evolutionaryscale/gpt1900
export PATH="/mnt/main0/home/michaelhla/evolutionaryscale/.pixi/envs/default/bin:$PATH"
export CONDA_PREFIX="/mnt/main0/home/michaelhla/evolutionaryscale/.pixi/envs/default"

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR=/mnt/main0/data/michaelhla/gpt1900_training
export PYTHONPATH=/mnt/main0/home/michaelhla/evolutionaryscale/gpt1900

DATA_DIR=${NANOCHAT_BASE_DIR}/instruct_data

# Step 1: Prepare data (prepend system prompt)
echo "=== Preparing reasoning SFT data ==="
python -m scripts.pre1900_scripts.prepare_reasoning_data --base-dir ${NANOCHAT_BASE_DIR}

# Step 2: SFT on reasoning traces (from base d34)
echo "=== Starting reasoning SFT ==="
torchrun --standalone --nproc_per_node=8 -m scripts.pre1900_sft -- \
    --model-tag d34 \
    --device-batch-size=4 \
    --total-batch-size=65536 \
    --num-iterations 50 \
    --output-dir pre1900_reasoning_sft_checkpoints \
    --train-data instruct_data/reasoning/sft_train.jsonl \
    --val-data instruct_data/reasoning/sft_val.jsonl \
    --run=pre1900_reasoning_sft_d34
