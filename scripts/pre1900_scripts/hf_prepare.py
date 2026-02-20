#!/usr/bin/env python3
"""
Prepare Historical Corpus for Training (streaming, low-memory)

Reads cleaned parquet shards and creates train/val split for nanochat.
Instead of loading all docs into memory, this:
1. First pass: counts docs per input shard
2. Shuffles shard order + assigns each shard to an output shard
3. Second pass: streams docs through, writing output shards

Usage:
    python scripts/pre1900_scripts/hf_prepare.py --input ./data/pre1900_filtered --output ./data/pre1900_parquet
"""

import argparse
import gc
import random
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(
        description='Prepare historical corpus for nanochat training',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--input', '-i', type=Path, required=True,
                        help='Input directory containing cleaned parquet shards')
    parser.add_argument('--output', '-o', type=Path, required=True,
                        help='Output directory for training parquet files')
    parser.add_argument('--docs-per-shard', type=int, default=50000,
                        help='Number of documents per output parquet shard')
    parser.add_argument('--row-group-size', type=int, default=1000,
                        help='Row group size within each parquet file')
    parser.add_argument('--val-fraction', type=float, default=0.01,
                        help='Fraction of data for validation (last shard)')
    parser.add_argument('--min-length', type=int, default=1000,
                        help='Minimum document length in characters')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for shuffling')
    args = parser.parse_args()

    random.seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)

    # Find all input files
    pq_files = sorted(args.input.rglob("*.parquet"))
    if not pq_files:
        print(f"No .parquet files found in {args.input}")
        return
    print(f"Found {len(pq_files)} input shards")

    # Pass 1: count docs per shard (cheap — just reads metadata)
    print("Counting documents...")
    shard_counts = []
    total_docs = 0
    for pq_path in tqdm(pq_files, desc="Counting"):
        try:
            meta = pq.read_metadata(pq_path)
            n = meta.num_rows
        except Exception:
            n = 0
        shard_counts.append(n)
        total_docs += n
    print(f"Total documents: {total_docs:,}")

    # Determine val split — reserve some input shards for val
    n_val_target = max(1, int(total_docs * args.val_fraction))

    # Shuffle input shard order for mixing sources
    indices = list(range(len(pq_files)))
    random.shuffle(indices)

    # Assign last shards (by shuffled order) to val until we have enough
    val_indices = set()
    val_count = 0
    for idx in reversed(indices):
        if val_count >= n_val_target:
            break
        val_indices.add(idx)
        val_count += shard_counts[idx]

    train_indices = [i for i in indices if i not in val_indices]
    val_indices_list = [i for i in indices if i in val_indices]

    print(f"Train shards: {len(train_indices)}, Val shards: {len(val_indices_list)}")
    print(f"Approx train docs: {total_docs - val_count:,}, val docs: {val_count:,}")

    # Pass 2: stream through train shards, writing output
    print(f"\nWriting training shards ({args.docs_per_shard} docs each)...")
    out_shard_idx = 0
    buffer = []
    train_doc_count = 0
    train_chars = 0

    for i, src_idx in enumerate(tqdm(train_indices, desc="Processing train")):
        pq_path = pq_files[src_idx]
        try:
            table = pq.read_table(pq_path, columns=["text"])
        except Exception as e:
            print(f"Warning: Could not read {pq_path}: {e}")
            continue

        texts = table.column("text").to_pylist()
        del table
        gc.collect()

        # Shuffle within shard
        random.shuffle(texts)

        for text in texts:
            if len(text) < args.min_length:
                continue
            buffer.append(text)
            train_chars += len(text)
            train_doc_count += 1

            if len(buffer) >= args.docs_per_shard:
                random.shuffle(buffer)
                out_path = args.output / f"shard_{out_shard_idx:05d}.parquet"
                out_table = pa.table({"text": buffer})
                pq.write_table(out_table, out_path, row_group_size=args.row_group_size,
                               compression='snappy')
                del out_table
                buffer = []
                out_shard_idx += 1
                gc.collect()

    # Write remaining train buffer
    if buffer:
        random.shuffle(buffer)
        out_path = args.output / f"shard_{out_shard_idx:05d}.parquet"
        out_table = pa.table({"text": buffer})
        pq.write_table(out_table, out_path, row_group_size=args.row_group_size,
                       compression='snappy')
        del out_table
        out_shard_idx += 1
        buffer = []
        gc.collect()

    n_train_shards = out_shard_idx

    # Pass 3: write val shard(s)
    print(f"\nWriting validation shard (shard_{out_shard_idx:05d})...")
    val_docs = []
    val_chars = 0
    for src_idx in val_indices_list:
        pq_path = pq_files[src_idx]
        try:
            table = pq.read_table(pq_path, columns=["text"])
        except Exception:
            continue
        texts = table.column("text").to_pylist()
        del table
        for text in texts:
            if len(text) < args.min_length:
                continue
            val_docs.append(text)
            val_chars += len(text)

    random.shuffle(val_docs)
    val_path = args.output / f"shard_{out_shard_idx:05d}.parquet"
    out_table = pa.table({"text": val_docs})
    pq.write_table(out_table, val_path, row_group_size=args.row_group_size,
                   compression='snappy')
    del out_table, val_docs
    gc.collect()

    total_shards = out_shard_idx + 1

    print(f"\n{'='*60}")
    print("Dataset preparation complete!")
    print(f"  Total shards: {total_shards}")
    print(f"  Training: {n_train_shards} shards, {train_doc_count:,} docs ({train_chars / 1e9:.2f} GB)")
    print(f"  Validation: 1 shard, {val_chars / 1e6:.1f} MB")
    print(f"  Output: {args.output}")

    print(f"\nVerifying parquet files...")
    parquet_files = sorted(args.output.glob("shard_*.parquet"))
    for pf_path in parquet_files[:2]:
        pf = pq.ParquetFile(pf_path)
        print(f"  {pf_path.name}: {pf.metadata.num_rows} docs, {pf.metadata.num_row_groups} row groups")

    print(f"\nTo use with nanochat:")
    print(f"  ln -s {args.output.absolute()} ~/.cache/nanochat/base_data")
    print(f"  # or: export NANOCHAT_BASE_DIR=<parent_of_base_data>")


if __name__ == "__main__":
    main()
