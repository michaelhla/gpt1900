#!/usr/bin/env python3
"""Process remaining institutional shards that are too large for Dask workers.

Usage:
    python scripts/pre1900_scripts/clean_remaining.py --workers 8
"""
import argparse
import glob
import gc
import os
import sys
import time

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hf_clean import process_document, normalize_ocr_score

RAW_DIR = "/mnt/main0/data/michaelhla/pre1900_raw"
CLEAN_DIR = "/mnt/main0/data/michaelhla/pre1900_clean"


def process_shard(inp, out):
    ts = time.time()
    shard = os.path.basename(inp)
    sz = os.path.getsize(inp) / 1e9
    print(f"  {shard} ({sz:.1f} GB)...", flush=True)

    table = pq.read_table(inp)
    texts = table.column("text").to_pylist()
    sources = table.column("source").to_pylist() if "source" in table.column_names else [""] * len(texts)
    doc_ids = table.column("doc_id").to_pylist() if "doc_id" in table.column_names else [""] * len(texts)
    years = table.column("year").to_pylist() if "year" in table.column_names else [0] * len(texts)
    ocr_scores_raw = table.column("ocr_score").to_pylist() if "ocr_score" in table.column_names else [-1.0] * len(texts)
    legibilities = table.column("legibility").to_pylist() if "legibility" in table.column_names else [-1.0] * len(texts)
    del table
    gc.collect()
    print(f"  loaded {time.time()-ts:.0f}s", flush=True)

    output_rows = []
    n_kept = 0
    for j, (text, source, doc_id, year, raw_ocr, legibility) in enumerate(
        zip(texts, sources, doc_ids, years, ocr_scores_raw, legibilities)
    ):
        ocr_score = normalize_ocr_score(raw_ocr if raw_ocr is not None else -1.0, source)
        cleaned, reason = process_document(
            text, 0.0, 0.85, 50, 5000,
            ocr_score=ocr_score,
            legibility=legibility if legibility is not None else -1.0,
            source=source,
        )
        if cleaned is not None:
            n_kept += 1
            output_rows.append({
                "text": cleaned, "source": source, "doc_id": doc_id,
                "year": year, "ocr_score": ocr_score,
                "legibility": legibility if legibility is not None else -1.0,
            })
        if (j + 1) % 2000 == 0:
            elapsed = time.time() - ts
            rate = (j + 1) / elapsed
            eta = (len(texts) - j - 1) / rate
            print(f"  {j+1}/{len(texts)} ({rate:.1f} docs/s, ETA {eta:.0f}s)", flush=True)

    del texts, sources, doc_ids, years, ocr_scores_raw, legibilities

    if output_rows:
        columns = {}
        for row in output_rows:
            for k, v in row.items():
                columns.setdefault(k, []).append(v)
        out_table = pa.table(columns)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        pq.write_table(out_table, out, compression="snappy")
        del out_table
    del output_rows
    gc.collect()

    print(f"  done: {n_kept} kept, {time.time()-ts:.0f}s", flush=True)
    return n_kept


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of shards to process in parallel (each needs ~40GB RAM)")
    parser.add_argument("--raw-dir", type=str, default=RAW_DIR)
    parser.add_argument("--clean-dir", type=str, default=CLEAN_DIR)
    args = parser.parse_args()

    raw_files = sorted(glob.glob(os.path.join(args.raw_dir, "**", "*.parquet"), recursive=True))
    pairs = []
    for f in raw_files:
        rel = os.path.relpath(f, args.raw_dir)
        out = os.path.join(args.clean_dir, rel)
        if not os.path.exists(out):
            pairs.append((f, out))

    if not pairs:
        print("All shards already processed!")
        return

    print(f"Processing {len(pairs)} remaining shards")
    t0 = time.time()

    if args.workers <= 1:
        for i, (inp, out) in enumerate(pairs):
            print(f"[{i+1}/{len(pairs)}]", flush=True)
            process_shard(inp, out)
    else:
        from multiprocessing import Pool
        with Pool(args.workers) as pool:
            pool.starmap(process_shard, pairs)

    print(f"\nAll done in {time.time()-t0:.0f}s ({(time.time()-t0)/3600:.1f}h)")


if __name__ == "__main__":
    main()
