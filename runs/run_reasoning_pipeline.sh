#!/bin/bash
#SBATCH --partition=highpri
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --mem=0
#SBATCH --job-name=reasoning_pipeline_d34
#SBATCH --output=/mnt/main0/home/michaelhla/reasoning_pipeline_d34_%j.log

set -e  # exit on first failure

cd /mnt/main0/home/michaelhla/evolutionaryscale/gpt1900
export PATH="/mnt/main0/home/michaelhla/evolutionaryscale/.pixi/envs/default/bin:$PATH"
export CONDA_PREFIX="/mnt/main0/home/michaelhla/evolutionaryscale/.pixi/envs/default"

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR=/mnt/main0/data/michaelhla/gpt1900_training
export PYTHONPATH=/mnt/main0/home/michaelhla/evolutionaryscale/gpt1900

# ---- Step 1: Prepare data (prepend system prompt) ----
echo "=== [$(date)] Preparing reasoning data ==="
python -m scripts.pre1900_scripts.prepare_reasoning_data --base-dir ${NANOCHAT_BASE_DIR}

# ---- Step 2: Reasoning trace SFT (base d34 -> reasoning SFT) ----
echo "=== [$(date)] Starting reasoning SFT ==="
torchrun --standalone --nproc_per_node=8 -m scripts.pre1900_sft -- \
    --model-tag d34 \
    --device-batch-size=4 \
    --total-batch-size=65536 \
    --num-iterations 50 \
    --output-dir pre1900_reasoning_sft_checkpoints \
    --train-data instruct_data/reasoning/sft_train.jsonl \
    --val-data instruct_data/reasoning/sft_val.jsonl \
    --run=pre1900_reasoning_sft_d34

echo "=== [$(date)] Reasoning SFT complete ==="

# ---- Step 3: Discovery RL (reasoning SFT -> discovery RL) ----
echo "=== [$(date)] Starting discovery RL ==="
torchrun --standalone --nproc_per_node=8 -m scripts.pre1900_scripts.discovery_rl -- \
    --model-tag d34 \
    --checkpoints-dir pre1900_reasoning_sft_checkpoints \
    --device-batch-size 2 \
    --num-samples 4 \
    --examples-per-step 8 \
    --max-new-tokens 512 \
    --num-epochs 3 \
    --train-data instruct_data/rl_problems/rl_prompts_sys_train.jsonl \
    --val-data instruct_data/rl_problems/rl_prompts_sys_val.jsonl \
    --problems-data instruct_data/rl_problems/rl_problems_train.jsonl \
    --problems-val-data instruct_data/rl_problems/rl_problems_val.jsonl \
    --eval-every 15 --save-every 30 \
    --run=pre1900_discovery_rl_d34

echo "=== [$(date)] Pipeline complete ==="
