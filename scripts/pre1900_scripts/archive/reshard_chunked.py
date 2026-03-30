#!/usr/bin/env python3
"""
Reshard parquet files, chunking long documents to avoid dataloader waste.

Long documents (>max_chars) are split at paragraph boundaries into chunks
of ~max_chars characters (~2000 tokens). This prevents the BOS-aligned
bestfit dataloader from discarding 90%+ of long documents.

Usage:
    python scripts/pre1900_scripts/reshard_chunked.py \
        --input /mnt/main0/data/michaelhla/pre1900_resharded_v2 \
        --output /mnt/main0/data/michaelhla/pre1900_resharded_v3
"""

import argparse
import gc
import re
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


def chunk_document(text, max_chars=8000):
    """Split a long document into chunks of roughly max_chars at natural boundaries.

    Priority: split at paragraph breaks (\n\n), then sentence boundaries (. ! ?),
    then hard-split as a last resort.
    """
    if len(text) <= max_chars:
        return [text]

    chunks = []
    remaining = text

    while len(remaining) > max_chars:
        # Try to find a paragraph break within the window
        window = remaining[:max_chars]
        split_pos = window.rfind('\n\n')

        if split_pos > max_chars // 4:  # found a reasonable paragraph break
            split_pos += 2  # include the \n\n with the first chunk
        else:
            # Try sentence boundary (. or ! or ? followed by space/newline)
            # Search from the end of the window backwards
            match = None
            for m in re.finditer(r'[.!?]\s', window):
                match = m  # keep updating to get the last match
            if match and match.end() > max_chars // 4:
                split_pos = match.end()
            else:
                # Hard split at max_chars
                split_pos = max_chars

        chunk = remaining[:split_pos].rstrip()
        if chunk:  # skip empty chunks
            chunks.append(chunk)
        remaining = remaining[split_pos:].lstrip()

    if remaining.strip():
        chunks.append(remaining.strip())

    return chunks


def main():
    parser = argparse.ArgumentParser(
        description='Reshard parquets with long-document chunking',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--input', '-i', type=Path, required=True,
                        help='Input directory containing parquet shards')
    parser.add_argument('--output', '-o', type=Path, required=True,
                        help='Output directory for resharded files')
    parser.add_argument('--max-chars', type=int, default=8000,
                        help='Max characters per document chunk (~2000 tokens at 4 chars/tok)')
    parser.add_argument('--min-chars', type=int, default=200,
                        help='Discard chunks shorter than this')
    parser.add_argument('--docs-per-shard', type=int, default=50000,
                        help='Target documents per output shard')
    parser.add_argument('--row-group-size', type=int, default=500,
                        help='Row group size within each parquet file')
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    input_files = sorted(args.input.glob("shard_*.parquet"))
    if not input_files:
        print(f"No shard_*.parquet files found in {args.input}")
        return

    # Last shard is validation (same convention as hf_prepare.py / reshard.py)
    train_files = input_files[:-1]
    val_file = input_files[-1]

    print(f"Found {len(input_files)} shards ({len(train_files)} train + 1 val)")
    print(f"Max chars per chunk: {args.max_chars} (~{args.max_chars // 4} tokens)")
    print(f"Target docs per output shard: {args.docs_per_shard}")

    # --- Process training shards ---
    out_idx = 0
    buffer = []
    stats = {"input_docs": 0, "output_docs": 0, "chunked_docs": 0, "total_input_chars": 0, "total_output_chars": 0}

    def flush_buffer():
        nonlocal out_idx, buffer
        if not buffer:
            return
        out_path = args.output / f"shard_{out_idx:05d}.parquet"
        out_table = pa.table({"text": buffer})
        pq.write_table(out_table, out_path,
                       row_group_size=args.row_group_size,
                       compression='snappy')
        del out_table
        out_idx += 1
        buffer = []

    for filepath in tqdm(train_files, desc="Processing train shards"):
        pf = pq.ParquetFile(filepath)
        for rg_idx in range(pf.num_row_groups):
            rg = pf.read_row_group(rg_idx, columns=["text"])
            texts = rg.column("text").to_pylist()

            for text in texts:
                stats["input_docs"] += 1
                stats["total_input_chars"] += len(text)

                chunks = chunk_document(text, max_chars=args.max_chars)
                if len(chunks) > 1:
                    stats["chunked_docs"] += 1

                for chunk in chunks:
                    if len(chunk) >= args.min_chars:
                        buffer.append(chunk)
                        stats["output_docs"] += 1
                        stats["total_output_chars"] += len(chunk)

                        if len(buffer) >= args.docs_per_shard:
                            flush_buffer()

        gc.collect()

    # Write remaining train buffer
    flush_buffer()
    n_train_shards = out_idx

    # --- Process validation shard ---
    print(f"\nProcessing validation shard...")
    val_pf = pq.ParquetFile(val_file)
    val_docs = []
    for rg_idx in range(val_pf.num_row_groups):
        rg = val_pf.read_row_group(rg_idx, columns=["text"])
        texts = rg.column("text").to_pylist()
        for text in texts:
            chunks = chunk_document(text, max_chars=args.max_chars)
            for chunk in chunks:
                if len(chunk) >= args.min_chars:
                    val_docs.append(chunk)

    val_path = args.output / f"shard_{out_idx:05d}.parquet"
    out_table = pa.table({"text": val_docs})
    pq.write_table(out_table, val_path,
                   row_group_size=args.row_group_size,
                   compression='snappy')
    del out_table
    out_idx += 1

    # --- Report ---
    retention = stats["total_output_chars"] / stats["total_input_chars"] * 100
    print(f"\n{'='*60}")
    print(f"Resharding complete!")
    print(f"  Input:  {stats['input_docs']:,} docs, {stats['total_input_chars']/1e9:.2f}B chars")
    print(f"  Output: {stats['output_docs']:,} docs, {stats['total_output_chars']/1e9:.2f}B chars ({retention:.1f}% retained)")
    print(f"  Chunked: {stats['chunked_docs']:,} long docs were split")
    print(f"  Train: {n_train_shards} shards")
    print(f"  Val: 1 shard, {len(val_docs):,} docs")
    print(f"  Total: {out_idx} shards")
    print(f"  Output: {args.output}")
    est_tokens = stats["total_output_chars"] / 4.19
    print(f"  Estimated tokens: {est_tokens/1e9:.1f}B")


if __name__ == "__main__":
    main()
