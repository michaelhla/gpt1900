#!/usr/bin/env python3
"""
Reshard parquet files into uniformly-sized shards.

Reads existing parquet shards and rewrites them into smaller, uniform shards
to avoid I/O stalls during training when hitting large shards.

Usage:
    python scripts/pre1900_scripts/reshard.py --input ./data/pre1900_parquet --output ./data/pre1900_resharded
    python scripts/pre1900_scripts/reshard.py --input ./data/pre1900_parquet --output ./data/pre1900_resharded --docs-per-shard 5000
"""

import argparse
import gc
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(
        description='Reshard parquet files into uniform sizes',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--input', '-i', type=Path, required=True,
                        help='Input directory containing parquet shards')
    parser.add_argument('--output', '-o', type=Path, required=True,
                        help='Output directory for resharded files')
    parser.add_argument('--docs-per-shard', type=int, default=5000,
                        help='Target documents per output shard')
    parser.add_argument('--row-group-size', type=int, default=500,
                        help='Row group size within each parquet file (must produce >= nproc row groups per shard)')
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    input_files = sorted(args.input.glob("shard_*.parquet"))
    if not input_files:
        print(f"No shard_*.parquet files found in {args.input}")
        return

    # Identify the last shard as validation (same convention as hf_prepare.py)
    train_files = input_files[:-1]
    val_file = input_files[-1]

    print(f"Found {len(input_files)} shards ({len(train_files)} train + 1 val)")
    print(f"Target: {args.docs_per_shard} docs per output shard")

    # Reshard training files
    out_idx = 0
    buffer = []
    total_docs = 0

    for filepath in tqdm(train_files, desc="Reading train shards"):
        pf = pq.ParquetFile(filepath)
        for rg_idx in range(pf.num_row_groups):
            rg = pf.read_row_group(rg_idx, columns=["text"])
            texts = rg.column("text").to_pylist()
            buffer.extend(texts)
            total_docs += len(texts)

            while len(buffer) >= args.docs_per_shard:
                chunk = buffer[:args.docs_per_shard]
                buffer = buffer[args.docs_per_shard:]
                out_path = args.output / f"shard_{out_idx:05d}.parquet"
                out_table = pa.table({"text": chunk})
                pq.write_table(out_table, out_path,
                               row_group_size=args.row_group_size,
                               compression='snappy')
                del out_table
                out_idx += 1

        gc.collect()

    # Write remaining train buffer
    if buffer:
        out_path = args.output / f"shard_{out_idx:05d}.parquet"
        out_table = pa.table({"text": buffer})
        pq.write_table(out_table, out_path,
                       row_group_size=args.row_group_size,
                       compression='snappy')
        del out_table
        out_idx += 1
        buffer = []
        gc.collect()

    n_train_shards = out_idx

    # Copy val shard as-is (just reshard if needed)
    print(f"\nResharding validation shard...")
    val_pf = pq.ParquetFile(val_file)
    val_docs = []
    for rg_idx in range(val_pf.num_row_groups):
        rg = val_pf.read_row_group(rg_idx, columns=["text"])
        val_docs.extend(rg.column("text").to_pylist())

    val_path = args.output / f"shard_{out_idx:05d}.parquet"
    out_table = pa.table({"text": val_docs})
    pq.write_table(out_table, val_path,
                   row_group_size=args.row_group_size,
                   compression='snappy')
    del out_table
    out_idx += 1

    print(f"\n{'='*60}")
    print(f"Resharding complete!")
    print(f"  Train: {n_train_shards} shards, {total_docs:,} docs")
    print(f"  Val: 1 shard, {len(val_docs):,} docs")
    print(f"  Total: {out_idx} shards")
    print(f"  Output: {args.output}")


if __name__ == "__main__":
    main()
