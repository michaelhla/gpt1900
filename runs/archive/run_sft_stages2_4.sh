#!/bin/bash
# Runs stages 2-4 end-to-end: craft pairs (batch API) -> filter -> sbatch SFT
set -e

cd /mnt/main0/home/michaelhla/evolutionaryscale/gpt1900
export PATH="/mnt/main0/home/michaelhla/evolutionaryscale/.pixi/envs/default/bin:$PATH"
export CONDA_PREFIX="/mnt/main0/home/michaelhla/evolutionaryscale/.pixi/envs/default"
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR=/mnt/main0/data/michaelhla/gpt1900_training
export PYTHONPATH=/mnt/main0/home/michaelhla/evolutionaryscale/gpt1900

MODEL_TAG=d34
DATA_DIR=${NANOCHAT_BASE_DIR}/instruct_data

# === Stage 2: Craft instruction pairs via batch API ===
echo "=== Stage 2: Crafting instruction pairs ==="
python -m scripts.pre1900_scripts.craft_instruct_pairs \
    --input ${DATA_DIR}/raw_generations.jsonl \
    --output-dir ${DATA_DIR} \
    --num-samples 100000 \
    --multi-turn-ratio 0.3 \
    --modern-ratio 0.5

echo "=== Stage 2 complete ==="

# === Stage 3: Filter both files ===
echo "=== Stage 3a: Filtering period-style pairs ==="
python -m scripts.pre1900_scripts.filter_instruct_pairs \
    --input ${DATA_DIR}/period_pairs.jsonl \
    --output-dir ${DATA_DIR}/period

echo "=== Stage 3b: Filtering modern-style pairs ==="
python -m scripts.pre1900_scripts.filter_instruct_pairs \
    --input ${DATA_DIR}/modern_pairs.jsonl \
    --output-dir ${DATA_DIR}/modern

echo "=== Stage 3 complete ==="

# === Stage 4: Submit SFT jobs via sbatch ===
echo "=== Stage 4: Submitting SFT sbatch jobs ==="

# Create a temporary sbatch script for period SFT
cat > /tmp/sft_period.sh << 'SBATCH_EOF'
#!/bin/bash
#SBATCH --partition=highpri
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --mem=0
#SBATCH --job-name=sft_period
#SBATCH --output=/mnt/main0/home/michaelhla/sft_period_%j.log

cd /mnt/main0/home/michaelhla/evolutionaryscale/gpt1900
export PATH="/mnt/main0/home/michaelhla/evolutionaryscale/.pixi/envs/default/bin:$PATH"
export CONDA_PREFIX="/mnt/main0/home/michaelhla/evolutionaryscale/.pixi/envs/default"
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR=/mnt/main0/data/michaelhla/gpt1900_training
export PYTHONPATH=/mnt/main0/home/michaelhla/evolutionaryscale/gpt1900

torchrun --standalone --nproc_per_node=8 -m scripts.pre1900_sft -- \
    --model-tag d34 \
    --device-batch-size=4 \
    --train-data instruct_data/period/filtered_pairs.jsonl \
    --val-data instruct_data/period/val_pairs.jsonl \
    --run=pre1900_sft_period_d34
SBATCH_EOF

# Create a temporary sbatch script for modern SFT (depends on period finishing)
cat > /tmp/sft_modern.sh << 'SBATCH_EOF'
#!/bin/bash
#SBATCH --partition=highpri
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --mem=0
#SBATCH --job-name=sft_modern
#SBATCH --output=/mnt/main0/home/michaelhla/sft_modern_%j.log

cd /mnt/main0/home/michaelhla/evolutionaryscale/gpt1900
export PATH="/mnt/main0/home/michaelhla/evolutionaryscale/.pixi/envs/default/bin:$PATH"
export CONDA_PREFIX="/mnt/main0/home/michaelhla/evolutionaryscale/.pixi/envs/default"
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR=/mnt/main0/data/michaelhla/gpt1900_training
export PYTHONPATH=/mnt/main0/home/michaelhla/evolutionaryscale/gpt1900

torchrun --standalone --nproc_per_node=8 -m scripts.pre1900_sft -- \
    --model-tag pre1900_sft_period_d34 \
    --device-batch-size=4 \
    --train-data instruct_data/modern/filtered_pairs.jsonl \
    --val-data instruct_data/modern/val_pairs.jsonl \
    --run=pre1900_sft_modern_d34
SBATCH_EOF

# Submit period SFT, then modern SFT with dependency
PERIOD_JOB=$(sbatch --parsable /tmp/sft_period.sh)
echo "Submitted period SFT job: ${PERIOD_JOB}"

MODERN_JOB=$(sbatch --parsable --dependency=afterok:${PERIOD_JOB} /tmp/sft_modern.sh)
echo "Submitted modern SFT job: ${MODERN_JOB} (depends on ${PERIOD_JOB})"

echo "=== All stages launched ==="
