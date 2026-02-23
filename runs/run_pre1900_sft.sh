#!/bin/bash
#SBATCH --partition=highpri
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --mem=0
#SBATCH --job-name=pre1900_sft
#SBATCH --output=/mnt/main0/home/michaelhla/pre1900_sft_%j.log

cd /mnt/main0/home/michaelhla/evolutionaryscale/gpt1900
export PATH="/mnt/main0/home/michaelhla/evolutionaryscale/.pixi/envs/default/bin:$PATH"
export CONDA_PREFIX="/mnt/main0/home/michaelhla/evolutionaryscale/.pixi/envs/default"

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR=/mnt/main0/data/michaelhla/gpt1900_training
export PYTHONPATH=/mnt/main0/home/michaelhla/evolutionaryscale/gpt1900

MODEL_TAG=d26
NUM_SAMPLES=50000
DATA_DIR=${NANOCHAT_BASE_DIR}/instruct_data

# Stage 1: Generate unconditional text from base model
echo "=== Stage 1: Generating unconditional samples ==="
python -m scripts.pre1900_scripts.generate_unconditional \
    --model-tag ${MODEL_TAG} \
    --num-samples ${NUM_SAMPLES} \
    --batch-size 32 \
    --output ${DATA_DIR}/raw_generations.jsonl

# Stage 2: Craft instruction pairs using Claude
echo "=== Stage 2: Crafting instruction pairs ==="
python -m scripts.pre1900_scripts.craft_instruct_pairs \
    --input ${DATA_DIR}/raw_generations.jsonl \
    --output ${DATA_DIR}/crafted_pairs.jsonl \
    --max-concurrent 20 \
    --multi-turn-ratio 0.3

# Stage 3: Filter for anachronisms
echo "=== Stage 3: Filtering instruction pairs ==="
python -m scripts.pre1900_scripts.filter_instruct_pairs \
    --input ${DATA_DIR}/crafted_pairs.jsonl \
    --output-dir ${DATA_DIR}

# Stage 4: SFT training
echo "=== Stage 4: SFT training ==="
torchrun --standalone --nproc_per_node=8 -m scripts.pre1900_sft -- \
    --model-tag ${MODEL_TAG} \
    --device-batch-size=16 \
    --train-data instruct_data/filtered_pairs.jsonl \
    --val-data instruct_data/val_pairs.jsonl \
    --run=pre1900_sft_${MODEL_TAG}
