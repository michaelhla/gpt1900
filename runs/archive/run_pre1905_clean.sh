#!/bin/bash
#SBATCH --partition=midpri
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --job-name=pre1905_clean
#SBATCH --output=/mnt/main0/home/michaelhla/pre1905_clean_%j.log

cd /mnt/main0/home/michaelhla/evolutionaryscale/gpt1900
export PATH="/mnt/main0/home/michaelhla/evolutionaryscale/.pixi/envs/default/bin:$PATH"
export CONDA_PREFIX="/mnt/main0/home/michaelhla/evolutionaryscale/.pixi/envs/default"
export PYTHONPATH=/mnt/main0/home/michaelhla/evolutionaryscale/gpt1900

python -m scripts.pre1900_scripts.clean_year_split \
    --end-year 1904 \
    --pre1900-staging /mnt/main0/data/michaelhla/pre1900_full_clean/_staging \
    --supplement-raw /mnt/main0/data/michaelhla/pre1915_supplement_raw \
    --output /mnt/main0/data/michaelhla/pre1905_full_clean \
    --workers 64 --world-size 64
