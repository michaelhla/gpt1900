#!/bin/bash
#SBATCH --partition=highpri
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --mem=0
#SBATCH --job-name=discovery_rl_v3_d34
#SBATCH --output=/mnt/main0/home/michaelhla/discovery_rl_v3_d34_%j.log

cd /mnt/main0/home/michaelhla/evolutionaryscale/gpt1900
export PATH="/mnt/main0/home/michaelhla/evolutionaryscale/.pixi/envs/default/bin:$PATH"
export CONDA_PREFIX="/mnt/main0/home/michaelhla/evolutionaryscale/.pixi/envs/default"

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR=/mnt/main0/data/michaelhla/gpt1900_training
export PYTHONPATH=/mnt/main0/home/michaelhla/evolutionaryscale/gpt1900
export OPENAI_API_KEY=${OPENAI_API_KEY}

echo "=== [$(date)] Starting discovery RL v3 ==="
echo "Loading from: pre1900_reasoning_sft_v3_checkpoints/d34"
echo "Saving to:    pre1900_discovery_rl_v3_checkpoints/d34"
echo "Data:         expanded RL dataset (800 train problems)"
echo "Judge:        gpt-4.1-mini"

torchrun --standalone --nproc_per_node=8 -m scripts.pre1900_scripts.discovery_rl -- \
    --model-tag d34 \
    --checkpoints-dir pre1900_reasoning_sft_v3_checkpoints \
    --output-dir pre1900_discovery_rl_v3_checkpoints \
    --device-batch-size 2 \
    --num-samples 4 \
    --examples-per-step 8 \
    --max-new-tokens 1024 \
    --num-epochs 3 \
    --train-data instruct_data/rl_problems/rl_prompts_sys_train_expanded.jsonl \
    --val-data instruct_data/rl_problems/rl_prompts_sys_val_clean.jsonl \
    --problems-data instruct_data/rl_problems/rl_problems_train_expanded.jsonl \
    --problems-val-data instruct_data/rl_problems/rl_problems_val.jsonl \
    --eval-every 15 --save-every 30 \
    --run=pre1900_discovery_rl_v3_d34

echo "=== [$(date)] Discovery RL v3 complete ==="
