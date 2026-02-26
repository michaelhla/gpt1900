#!/bin/bash
# Generate reasoning trace training data from pre-1900 physics books.
#
# Three-pass pipeline:
#   1a: Whole-book insight extraction (major insights + thesis)
#   1b: Chunk-level insight extraction (detailed technical insights)
#   2:  Reasoning trace generation from extracted insights
#   Filter: Remove anachronisms using existing filter
#
# Usage:
#   bash runs/run_reasoning_traces.sh          # Run all passes
#   bash runs/run_reasoning_traces.sh 1a       # Run only pass 1a
#   bash runs/run_reasoning_traces.sh 1b       # Run only pass 1b
#   bash runs/run_reasoning_traces.sh 2        # Run only pass 2
#   bash runs/run_reasoning_traces.sh filter   # Run only the filter

cd /mnt/main0/home/michaelhla/evolutionaryscale/gpt1900
export PATH="/mnt/main0/home/michaelhla/evolutionaryscale/.pixi/envs/default/bin:$PATH"
export CONDA_PREFIX="/mnt/main0/home/michaelhla/evolutionaryscale/.pixi/envs/default"
export PYTHONPATH=/mnt/main0/home/michaelhla/evolutionaryscale/gpt1900

export NANOCHAT_BASE_DIR=/mnt/main0/data/michaelhla/gpt1900_training
DATA_DIR=${NANOCHAT_BASE_DIR}/instruct_data

PASS=${1:-all}

# Pass 1a: Whole-book insight extraction (major insights + thesis)
if [ "$PASS" = "1a" ] || [ "$PASS" = "all" ]; then
echo "=== Pass 1a: Whole-book insight extraction ==="
python -m scripts.pre1900_scripts.generate_reasoning_traces \
    --books-dir data/physics_books --output-dir ${DATA_DIR} \
    --run-pass 1a --pass1-model claude-opus-4-6 \
    --max-concurrent-pass1 5 --resume
fi

# Pass 1b: Chunk-level insight extraction (detailed insights)
if [ "$PASS" = "1b" ] || [ "$PASS" = "all" ]; then
echo "=== Pass 1b: Chunk-level insight extraction ==="
python -m scripts.pre1900_scripts.generate_reasoning_traces \
    --books-dir data/physics_books --output-dir ${DATA_DIR} \
    --run-pass 1b --pass1-model claude-opus-4-6 \
    --max-concurrent-chunks 20 --chunk-tokens 100000 --resume
fi

# Pass 2: Generate reasoning traces from all insights
if [ "$PASS" = "2" ] || [ "$PASS" = "all" ]; then
echo "=== Pass 2: Generate reasoning traces ==="
python -m scripts.pre1900_scripts.generate_reasoning_traces \
    --books-dir data/physics_books --output-dir ${DATA_DIR} \
    --run-pass 2 --pass2-model claude-sonnet-4-20250514 \
    --max-concurrent-pass2 80 --resume
fi

# Filter for anachronisms
if [ "$PASS" = "filter" ] || [ "$PASS" = "all" ]; then
echo "=== Filtering reasoning traces for anachronisms ==="
python -m scripts.pre1900_scripts.filter_instruct_pairs \
    --input ${DATA_DIR}/reasoning_traces_clean.jsonl \
    --output-dir ${DATA_DIR}/reasoning
fi

echo "=== Done ==="
