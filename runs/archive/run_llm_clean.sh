#!/bin/bash
#SBATCH --partition=midpri
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --job-name=mhla_llm_clean
#SBATCH --output=/mnt/main0/home/michaelhla/llm_clean_%j.log

export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PATH="/mnt/main0/home/michaelhla/.pixi/bin:$PATH"

cd /mnt/main0/home/michaelhla/evolutionaryscale/gpt1900

uv run --extra gpu python -m scripts.pre1900_scripts.llm_clean \
    --input /mnt/main0/data/michaelhla/pre1900_raw \
    --output /mnt/main0/data/michaelhla/pre1900_llm_cleaned \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --num-gpus 8 \
    --newspapers-first
