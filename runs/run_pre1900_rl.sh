#!/bin/bash
#SBATCH --partition=highpri
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --mem=0
#SBATCH --job-name=pre1900_rl
#SBATCH --output=/mnt/main0/home/michaelhla/pre1900_rl_%j.log

cd /mnt/main0/home/michaelhla/evolutionaryscale/gpt1900
export PATH="/mnt/main0/home/michaelhla/evolutionaryscale/.pixi/envs/default/bin:$PATH"
export CONDA_PREFIX="/mnt/main0/home/michaelhla/evolutionaryscale/.pixi/envs/default"

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR=/mnt/main0/data/michaelhla/gpt1900_training
export PYTHONPATH=/mnt/main0/home/michaelhla/evolutionaryscale/gpt1900
export ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}

MODEL_TAG=pre1900_sft_period_d34

torchrun --standalone --nproc_per_node=8 -m scripts.pre1900_scripts.coherence_rl -- \
    --model-tag ${MODEL_TAG} \
    --checkpoints-dir pre1900_sft_checkpoints \
    --device-batch-size 2 \
    --num-samples 4 \
    --examples-per-step 8 \
    --train-data instruct_data/all_filtered_pairs.jsonl \
    --val-data instruct_data/all_val_pairs.jsonl \
    --run=pre1900_rl_d34
