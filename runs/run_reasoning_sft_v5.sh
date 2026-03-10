#!/bin/bash
#SBATCH --partition=highpri
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --mem=0
#SBATCH --job-name=reasoning_sft_v5_d34
#SBATCH --output=/mnt/main0/home/michaelhla/reasoning_sft_v5_d34_%j.log

cd /mnt/main0/home/michaelhla/evolutionaryscale/gpt1900
export PATH="/mnt/main0/home/michaelhla/evolutionaryscale/.pixi/envs/default/bin:$PATH"
export CONDA_PREFIX="/mnt/main0/home/michaelhla/evolutionaryscale/.pixi/envs/default"

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR=/mnt/main0/data/michaelhla/gpt1900_training
export PYTHONPATH=/mnt/main0/home/michaelhla/evolutionaryscale/gpt1900

echo "=== [$(date)] Starting reasoning SFT v5 ==="
echo "Loading from: physics_clm_checkpoints/d34 (d34-physics-sft from HF)"
echo "Saving to:    pre1900_reasoning_sft_v5_checkpoints/d34"
echo "Data:         sft_train_trimmed.jsonl (817 examples)"
echo ""
echo "This teaches chat tokens + <think>...\\answer{} format to the physics-CLM base."
echo "Download the d34-physics-sft checkpoint first:"
echo "  huggingface-cli download mhla/gpt1900-d34-physics-sft --local-dir \$NANOCHAT_BASE_DIR/physics_clm_checkpoints/d34"

torchrun --standalone --nproc_per_node=8 -m scripts.pre1900_sft -- \
    --model-tag d34 \
    --checkpoints-dir physics_clm_checkpoints \
    --output-dir pre1900_reasoning_sft_v5_checkpoints \
    --device-batch-size=4 \
    --total-batch-size=65536 \
    --num-iterations 100 \
    --train-data instruct_data/reasoning/sft_train_trimmed.jsonl \
    --val-data instruct_data/reasoning/sft_val_clean.jsonl \
    --run=pre1900_reasoning_sft_v5_d34

echo "=== [$(date)] Reasoning SFT v5 complete ==="
