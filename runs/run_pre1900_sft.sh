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

MODEL_TAG=d34
NUM_SAMPLES=400000
DATA_DIR=${NANOCHAT_BASE_DIR}/instruct_data
START_STAGE=${1:-1}  # Pass start stage as first arg, default 1

# Stage 1: Generate unconditional text from base model (8 GPUs)
if [ ${START_STAGE} -le 1 ]; then
echo "=== Stage 1: Generating unconditional samples ==="
torchrun --standalone --nproc_per_node=8 -m scripts.pre1900_scripts.generate_unconditional -- \
    --model-tag ${MODEL_TAG} \
    --num-samples ${NUM_SAMPLES} \
    --batch-size 128 \
    --output ${DATA_DIR}/raw_generations.jsonl
fi

# Stage 2: Craft instruction pairs using Claude (CPU only)
if [ ${START_STAGE} -le 2 ]; then
echo "=== Stage 2: Crafting instruction pairs ==="
python -m scripts.pre1900_scripts.craft_instruct_pairs \
    --input ${DATA_DIR}/raw_generations.jsonl \
    --output-dir ${DATA_DIR} \
    --num-samples 100000 \
    --multi-turn-ratio 0.3 \
    --modern-ratio 0.5
fi

# Stage 3: Filter for anachronisms (CPU only) — both files separately
if [ ${START_STAGE} -le 3 ]; then
echo "=== Stage 3a: Filtering period-style pairs ==="
python -m scripts.pre1900_scripts.filter_instruct_pairs \
    --input ${DATA_DIR}/period_pairs.jsonl \
    --output-dir ${DATA_DIR}/period

echo "=== Stage 3b: Filtering modern-style pairs ==="
python -m scripts.pre1900_scripts.filter_instruct_pairs \
    --input ${DATA_DIR}/modern_pairs.jsonl \
    --output-dir ${DATA_DIR}/modern
fi

# Stage 4a: SFT on period-style pairs (8 GPUs)
if [ ${START_STAGE} -le 4 ]; then
echo "=== Stage 4a: SFT on period-style pairs ==="
torchrun --standalone --nproc_per_node=8 -m scripts.pre1900_sft -- \
    --model-tag ${MODEL_TAG} \
    --device-batch-size=16 \
    --train-data instruct_data/period/filtered_pairs.jsonl \
    --val-data instruct_data/period/val_pairs.jsonl \
    --run=pre1900_sft_period_${MODEL_TAG}

# Stage 4b: SFT on modern-style pairs (continues from 4a)
echo "=== Stage 4b: SFT on modern-style pairs ==="
torchrun --standalone --nproc_per_node=8 -m scripts.pre1900_sft -- \
    --model-tag pre1900_sft_period_${MODEL_TAG} \
    --device-batch-size=16 \
    --train-data instruct_data/modern/filtered_pairs.jsonl \
    --val-data instruct_data/modern/val_pairs.jsonl \
    --run=pre1900_sft_modern_${MODEL_TAG}
fi
