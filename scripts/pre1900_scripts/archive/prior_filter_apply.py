#!/usr/bin/env python3
"""
Apply prior-based filtering to cleaned parquet shards (Dask-parallelized).

Loads a precomputed prior table (from prior_filter.py --save-priors), tokenizes
each document, computes mean/std of token log-priors, and keeps only documents
within the specified thresholds.

Batches multiple small shards per worker to amortize tokenizer loading overhead
(e.g. 6,910 newspaper shards × 12MB each → ~108 batches of 64 shards).

Usage:
    # 1. First, compute and save the prior table (only need to do this once):
    python scripts/pre1900_scripts/prior_filter.py \
        --datadir /mnt/main0/data/michaelhla/pre1900_raw \
        --save-priors /mnt/main0/data/michaelhla/prior_table.npy

    # 2. Run cleaning:
    python scripts/pre1900_scripts/hf_clean.py \
        --input /mnt/main0/data/michaelhla/pre1900_raw \
        --output /mnt/main0/data/michaelhla/pre1900_clean

    # 3. Apply prior filter to cleaned data (parallelized):
    python scripts/pre1900_scripts/prior_filter_apply.py \
        --input /mnt/main0/data/michaelhla/pre1900_clean \
        --output /mnt/main0/data/michaelhla/pre1900_filtered \
        --priors /mnt/main0/data/michaelhla/prior_table.npy \
        --workers 64
"""

from __future__ import annotations
import argparse
import glob
import json
import os
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from evolutionaryscale.utils.executor import DaskExecutorContext, TaskGranularity


def process_shard_batch(
    batch_json: str,
    priors_path: str,
    mean_lo: float,
    mean_hi: float,
    std_max: float | None,
) -> tuple[int, int, int, int, int]:
    """
    Process a batch of (input_path, output_path) pairs.

    Loads tokenizer and prior table once per batch, then iterates over all shards.
    Uses JSON-serialized batch list to avoid Dask nested-list serialization issues.

    Returns:
        (n_in, n_kept, n_filtered_lo, n_filtered_hi, n_filtered_std)
    """
    from transformers import GPT2TokenizerFast

    pairs = json.loads(batch_json)

    # Load once per worker invocation
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    log_priors = np.load(priors_path)

    total_in = 0
    total_kept = 0
    total_lo = 0
    total_hi = 0
    total_std = 0

    for input_path, output_path in pairs:
        try:
            table = pq.read_table(input_path)
        except Exception:
            continue  # skip corrupt files
        n_rows = table.num_rows
        total_in += n_rows

        kept_rows = []
        for i in range(n_rows):
            text = table.column("text")[i].as_py()
            if not text:
                total_lo += 1
                continue

            ids = tokenizer.encode(text[:50_000], add_special_tokens=False)
            if len(ids) == 0:
                total_lo += 1
                continue

            token_lps = log_priors[ids]
            mean_p = float(token_lps.mean())
            std_p = float(token_lps.std())

            if mean_p < mean_lo:
                total_lo += 1
                continue
            if mean_p > mean_hi:
                total_hi += 1
                continue
            if std_max is not None and std_p > std_max:
                total_std += 1
                continue

            kept_rows.append(i)

        n_kept = len(kept_rows)
        total_kept += n_kept
        if n_kept > 0:
            import pyarrow as pa
            # Cast string columns to large_string to avoid 2GB offset overflow
            schema = table.schema
            new_fields = []
            for field in schema:
                if field.type == pa.string():
                    new_fields.append(field.with_type(pa.large_string()))
                else:
                    new_fields.append(field)
            table = table.cast(pa.schema(new_fields))
            filtered_table = table.take(kept_rows)
            # Write in chunks to avoid Parquet 2GB page/array limits
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            writer = pq.ParquetWriter(output_path, filtered_table.schema,
                                      compression="snappy")
            chunk_size = 200  # small enough that no chunk exceeds 2GB
            for start in range(0, n_kept, chunk_size):
                writer.write_table(filtered_table.slice(start, chunk_size))
            writer.close()

    return total_in, total_kept, total_lo, total_hi, total_std


def main():
    parser = argparse.ArgumentParser(
        description="Apply prior-based filtering to cleaned parquet shards (Dask-parallelized)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", "-i", type=Path, required=True,
                        help="Directory of cleaned parquet shards (from hf_clean.py)")
    parser.add_argument("--output", "-o", type=Path, required=True,
                        help="Directory for filtered parquet shards")
    parser.add_argument("--priors", "-p", type=Path, required=True,
                        help="Path to saved prior table (.npy from prior_filter.py)")
    parser.add_argument("--mean-lo", type=float, default=-11.36,
                        help="Lower bound on mean(log2 prior) — docs below this are filtered (default: p10)")
    parser.add_argument("--mean-hi", type=float, default=-9.91,
                        help="Upper bound on mean(log2 prior) — docs above this are filtered (default: p99)")
    parser.add_argument("--std-max", type=float, default=None,
                        help="Max std(log2 prior) — docs above this are filtered (default: no std filter)")
    parser.add_argument("--workers", type=int, default=64,
                        help="Number of Dask workers")
    parser.add_argument("--shards-per-batch", type=int, default=64,
                        help="Number of input shards per worker batch (amortizes tokenizer load)")
    parser.add_argument("--mem-per-worker", type=int, default=16,
                        help="Memory per worker in GB")
    parser.add_argument("--partition", type=str, default="midpri",
                        help="Slurm partition")

    args = parser.parse_args()

    # Validate inputs
    priors_path = str(args.priors.resolve())
    if not os.path.exists(priors_path):
        print(f"ERROR: Prior table not found: {priors_path}")
        return

    # Find input shards
    input_files = sorted(glob.glob(str(args.input / "**" / "*.parquet"), recursive=True))
    if not input_files:
        print(f"ERROR: No parquet files found in {args.input}")
        return
    print(f"Found {len(input_files)} input shards")

    # Setup output — mirror input directory structure
    args.output.mkdir(parents=True, exist_ok=True)

    # Build (input_path, output_path) pairs, preserving subdirectory structure
    all_pairs = []
    for fpath in input_files:
        rel = os.path.relpath(fpath, args.input)
        out_path = str(args.output / rel)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        all_pairs.append((fpath, out_path))

    # Check for already-processed shards (resume support)
    remaining = [(inp, out) for inp, out in all_pairs if not os.path.exists(out)]
    if len(remaining) < len(all_pairs):
        print(f"Resuming: {len(all_pairs) - len(remaining)} shards already processed, {len(remaining)} remaining")

    if not remaining:
        print("All shards already processed!")
        return

    # Group into batches — each batch is processed by one worker with one tokenizer load
    batches = []
    for i in range(0, len(remaining), args.shards_per_batch):
        batches.append(remaining[i:i + args.shards_per_batch])

    n_shards = len(remaining)
    print(f"Processing {n_shards} shards in {len(batches)} batches ({args.shards_per_batch} shards/batch)")
    print(f"Using {args.workers} workers")
    print(f"Filter: mean_prior in [{args.mean_lo}, {args.mean_hi}]"
          + (f", std_prior <= {args.std_max}" if args.std_max else ""))

    t0 = time.time()

    # Serialize batches as JSON strings (avoids Dask nested-list serialization issues)
    batch_jsons = [json.dumps(batch) for batch in batches]

    with DaskExecutorContext(
        task_granularity=TaskGranularity.CPU,
        slurm_partition=args.partition,
        num_jobs=min(args.workers, len(batches)),
        cpus_per_worker=1,
        mem_per_worker_gb=args.mem_per_worker,
    ) as executor:
        results = executor.map(
            process_shard_batch,
            batch_jsons,
            [priors_path] * len(batches),
            [args.mean_lo] * len(batches),
            [args.mean_hi] * len(batches),
            [args.std_max] * len(batches),
            progress=True,
            errors="skip",
        )

    # Aggregate stats
    total_in = 0
    total_kept = 0
    total_filtered_lo = 0
    total_filtered_hi = 0
    total_filtered_std = 0
    n_failed = 0

    for result in results:
        if result is None:
            n_failed += 1
            continue
        n_in, n_kept, n_lo, n_hi, n_std = result
        total_in += n_in
        total_kept += n_kept
        total_filtered_lo += n_lo
        total_filtered_hi += n_hi
        total_filtered_std += n_std

    elapsed = time.time() - t0

    print()
    print("=" * 60)
    print("PRIOR FILTER RESULTS")
    print("=" * 60)
    print(f"  Input docs:        {total_in:,}")
    print(f"  Kept:              {total_kept:,} ({100*total_kept/max(total_in,1):.1f}%)")
    print(f"  Filtered (low):    {total_filtered_lo:,} (mean_prior < {args.mean_lo})")
    print(f"  Filtered (high):   {total_filtered_hi:,} (mean_prior > {args.mean_hi})")
    if args.std_max is not None:
        print(f"  Filtered (std):    {total_filtered_std:,} (std_prior > {args.std_max})")
    if n_failed:
        print(f"  Failed batches:    {n_failed}")
    print(f"  Output dir:        {args.output}")
    print(f"  Elapsed:           {elapsed:.1f}s ({elapsed/3600:.2f}h)")


if __name__ == "__main__":
    main()
