#!/usr/bin/env python3
"""
Reshard 1900-1964 raw data into balanced, chunked parquet shards for pretraining.

Single-pass streaming script that:
1. Reads all parquets from raw_data/ (institutional/ and newspapers/ subdirs)
2. Chunks long documents (max_chars=8000, min_chars=200) at natural boundaries
3. Reservoir-samples validation data (val_fraction=0.005)
4. Writes balanced shards via length-sorted round-robin distribution
5. Aligns row groups to world_size (8 GPUs)

No filtering or cleaning -- just chunking and resharding.

Usage:
    python scripts/pre1900_scripts/reshard_1900_1964.py \
        --input /opt/dlami/nvme/gpt1964_training/raw_data \
        --output /opt/dlami/nvme/gpt1964_training/corpus
"""

from __future__ import annotations
import argparse
import gc
import os
import random
import re
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


CHARS_PER_TOKEN = 4.2
CHUNK_SENTENCE_RE = re.compile(r'[.!?]\s')


def chunk_document(text: str, max_chars: int = 8000, min_chars: int = 200) -> list[str]:
    """Split a long document into chunks of roughly max_chars at natural boundaries."""
    if len(text) <= max_chars:
        return [text]

    chunks = []
    remaining = text

    while len(remaining) > max_chars:
        window = remaining[:max_chars]
        split_pos = window.rfind('\n\n')

        if split_pos > max_chars / 4:
            split_pos += 2
        else:
            match = None
            for m in CHUNK_SENTENCE_RE.finditer(window):
                match = m
            if match and match.end() > max_chars / 4:
                split_pos = match.end()
            else:
                split_pos = max_chars

        chunk = remaining[:split_pos].rstrip()
        if len(chunk) >= min_chars:
            chunks.append(chunk)
        remaining = remaining[split_pos:].lstrip()

    if len(remaining.strip()) >= min_chars:
        chunks.append(remaining.strip())

    return chunks


def flush_buffer(buf: list[str], start_shard: int, docs_per_shard: int,
                 row_group_size: int, output: Path,
                 shard_char_counts: list[int]) -> int:
    """Sort + round-robin distribute buffer into shards, write them."""
    if not buf:
        return 0
    n_shards = len(buf) // docs_per_shard
    if n_shards == 0:
        return 0
    n_use = n_shards * docs_per_shard
    to_shard = buf[:n_use]

    to_shard.sort(key=len)
    buckets: list[list[str]] = [[] for _ in range(n_shards)]
    for i, chunk in enumerate(to_shard):
        buckets[i % n_shards].append(chunk)

    for j, bucket in enumerate(buckets):
        random.shuffle(bucket)
        out_path = output / f"shard_{start_shard + j:05d}.parquet"
        table = pa.table({"text": bucket})
        pq.write_table(table, out_path,
                       row_group_size=row_group_size,
                       compression="zstd",
                       write_statistics=False,
                       use_dictionary=False)
        shard_char_counts.append(sum(len(c) for c in bucket))
        del table

    written = n_shards
    if (start_shard + written) % 50 < written or start_shard == 0:
        print(f"    Written shards {start_shard:,}-{start_shard + written - 1:,}")

    del buf[:n_use]
    gc.collect()
    return written


def main():
    parser = argparse.ArgumentParser(
        description="Reshard 1900-1964 raw data into balanced parquet shards",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", "-i", type=Path,
                        default=Path("/opt/dlami/nvme/gpt1964_training/raw_data"))
    parser.add_argument("--output", "-o", type=Path,
                        default=Path("/opt/dlami/nvme/gpt1964_training/corpus"))
    parser.add_argument("--max-chars", type=int, default=8000)
    parser.add_argument("--min-chars", type=int, default=200)
    parser.add_argument("--row-group-size", type=int, default=1024)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--docs-per-shard", type=int, default=65536,
                        help="Target docs per shard (will be aligned to row_group_size * world_size)")
    parser.add_argument("--val-fraction", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    random.seed(args.seed)

    args.output.mkdir(parents=True, exist_ok=True)

    # Align docs_per_shard to world_size
    rgs_per_gpu = max(1, round(args.docs_per_shard / (args.row_group_size * args.world_size)))
    docs_per_shard = args.row_group_size * rgs_per_gpu * args.world_size
    n_rg_per_shard = docs_per_shard // args.row_group_size

    # Collect input files from both subdirs
    all_input_files = []
    for subdir in ["institutional", "newspapers"]:
        subpath = args.input / subdir
        if subpath.exists():
            files = sorted(subpath.glob("*.parquet"))
            all_input_files.extend(files)
            print(f"  {subdir}/: {len(files)} files")

    random.shuffle(all_input_files)

    print(f"\n{'=' * 60}")
    print(f"RESHARD 1900-1964 CORPUS")
    print(f"{'=' * 60}")
    print(f"  Input files: {len(all_input_files)}")
    print(f"  Chunking: max_chars={args.max_chars}, min_chars={args.min_chars}")
    print(f"  Target: {docs_per_shard} docs/shard, {n_rg_per_shard} row_groups/shard "
          f"({rgs_per_gpu} per GPU x {args.world_size} GPUs)")

    if not all_input_files:
        print("  ERROR: No input files found.")
        return

    t0 = time.time()

    # Streaming one-pass
    BUFFER_SHARDS = 10
    buffer_capacity = BUFFER_SHARDS * docs_per_shard

    buffer: list[str] = []
    val_reservoir: list[str] = []
    total_chunks_seen = 0
    total_chars = 0
    total_raw_docs = 0
    total_chunked_docs = 0
    shard_idx = 0
    shard_char_counts: list[int] = []

    print(f"\n  Streaming through all input files (buffer={BUFFER_SHARDS} shards)...")

    for file_idx, sf in enumerate(all_input_files):
        try:
            pf = pq.ParquetFile(sf)
            for rg_idx in range(pf.num_row_groups):
                table = pf.read_row_group(rg_idx, columns=["text"])
                texts = table.column("text").to_pylist()
                del table
                for text in texts:
                    if not text:
                        continue
                    total_raw_docs += 1
                    doc_chunks = chunk_document(text, max_chars=args.max_chars, min_chars=args.min_chars)
                    if len(doc_chunks) > 1:
                        total_chunked_docs += 1
                    for c in doc_chunks:
                        total_chunks_seen += 1
                        total_chars += len(c)
                        if random.random() < args.val_fraction:
                            val_reservoir.append(c)
                        else:
                            buffer.append(c)
        except Exception:
            continue

        if len(buffer) >= buffer_capacity:
            n_written = flush_buffer(buffer, shard_idx, docs_per_shard,
                                     args.row_group_size, args.output,
                                     shard_char_counts)
            shard_idx += n_written

        if (file_idx + 1) % 500 == 0:
            est_tok = total_chars / CHARS_PER_TOKEN
            print(f"    Processed {file_idx + 1}/{len(all_input_files)} files, "
                  f"{total_chunks_seen:,} chunks, ~{est_tok / 1e9:.2f}B tokens, "
                  f"{shard_idx} shards written")

    # Final flush of remaining buffer
    if len(buffer) >= docs_per_shard:
        n_written = flush_buffer(buffer, shard_idx, docs_per_shard,
                                 args.row_group_size, args.output,
                                 shard_char_counts)
        shard_idx += n_written

    # Any leftover (< docs_per_shard) goes into val
    val_reservoir.extend(buffer)
    buffer.clear()
    gc.collect()

    n_train_shards = shard_idx

    # Write val shard (always last, as expected by dataloader)
    print(f"  Writing validation shard (shard_{shard_idx:05d}), {len(val_reservoir):,} chunks...")
    val_path = args.output / f"shard_{shard_idx:05d}.parquet"
    random.shuffle(val_reservoir)
    table = pa.table({"text": val_reservoir})
    pq.write_table(table, val_path,
                    row_group_size=args.row_group_size,
                    compression="zstd",
                    write_statistics=False,
                    use_dictionary=False)
    del table
    shard_idx += 1
    del val_reservoir
    gc.collect()

    elapsed = time.time() - t0
    est_tokens = total_chars / CHARS_PER_TOKEN

    # Report token balance
    if shard_char_counts:
        avg_c = sum(shard_char_counts) / len(shard_char_counts)
        min_c = min(shard_char_counts)
        max_c = max(shard_char_counts)
        imbalance = (max_c - min_c) / avg_c * 100
        print(f"\n  Token balance: avg={avg_c / CHARS_PER_TOKEN / 1e6:.1f}M tok/shard, "
              f"min={min_c / CHARS_PER_TOKEN / 1e6:.1f}M, "
              f"max={max_c / CHARS_PER_TOKEN / 1e6:.1f}M, "
              f"imbalance={imbalance:.2f}%")

    # Report file sizes
    shard_sizes = []
    for i in range(n_train_shards):
        sp = args.output / f"shard_{i:05d}.parquet"
        if sp.exists():
            shard_sizes.append(os.path.getsize(sp))

    print(f"\n{'=' * 60}")
    print(f"DONE")
    print(f"{'=' * 60}")
    print(f"  Docs: {total_raw_docs:,} raw -> {total_chunks_seen:,} chunks ({total_chunked_docs:,} split)")
    print(f"  Train: {n_train_shards} shards")
    print(f"    {docs_per_shard} docs/shard, {n_rg_per_shard} row_groups (÷ {args.world_size} = {rgs_per_gpu} per GPU)")
    if shard_sizes:
        print(f"    File sizes: avg={sum(shard_sizes) / len(shard_sizes) / 1e6:.1f}MB, "
              f"min={min(shard_sizes) / 1e6:.1f}MB, max={max(shard_sizes) / 1e6:.1f}MB")
    print(f"  Val:   1 shard (shard_{n_train_shards:05d})")
    print(f"  Total: {shard_idx} shards")
    print(f"  Est. tokens: {est_tokens / 1e9:.2f}B")
    print(f"  Output: {args.output}")
    print(f"  Elapsed: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
