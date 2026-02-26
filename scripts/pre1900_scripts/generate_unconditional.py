#!/usr/bin/env python3
"""
Generate unconditional text samples from the pre-1900 base model.

Generates from [BOS] token with temperature 1.0, producing 512-token samples.
Filters out samples that are too short or mostly non-ASCII.

Supports multi-GPU via torchrun: each rank generates its share independently,
then rank 0 concatenates all per-rank files into the final output.

Usage:
    # Single GPU:
    python -m scripts.pre1900_scripts.generate_unconditional --model-tag d26 --num-samples 200000

    # 8 GPUs (recommended):
    torchrun --standalone --nproc_per_node=8 -m scripts.pre1900_scripts.generate_unconditional -- --model-tag d26 --num-samples 200000
"""

import os
import argparse
import json

import torch
import torch.distributed as dist
from contextlib import nullcontext

from nanochat.common import compute_init, compute_cleanup, print0, autodetect_device_type
from nanochat.checkpoint_manager import load_model
from nanochat.engine import Engine


def fraction_non_ascii(text: str) -> float:
    if not text:
        return 1.0
    non_ascii = sum(1 for c in text if ord(c) > 127)
    return non_ascii / len(text)


def main():
    parser = argparse.ArgumentParser(description="Generate unconditional text from base model")
    parser.add_argument("--model-tag", type=str, default=None, help="Model tag (e.g. d26)")
    parser.add_argument("--step", type=int, default=None, help="Model step to load (default = last)")
    parser.add_argument("--num-samples", type=int, default=200000, help="Total number of samples to generate")
    parser.add_argument("--batch-size", type=int, default=32, help="Samples per batch per GPU")
    parser.add_argument("--output", type=str, default="instruct_data/raw_generations.jsonl", help="Output JSONL path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
    args = parser.parse_args()

    # Compute init
    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()

    # Each rank generates its share
    samples_per_rank = args.num_samples // ddp_world_size
    if ddp_rank < args.num_samples % ddp_world_size:
        samples_per_rank += 1  # distribute remainder

    # Load model (each rank loads independently onto its own GPU)
    model, tokenizer, meta = load_model("base", device, phase="eval", model_tag=args.model_tag, step=args.step)
    engine = Engine(model, tokenizer)

    # BOS-only prompt
    bos_tokens = tokenizer("", prepend="<|bos|>")

    max_tokens = 512
    min_chars = 100
    max_non_ascii_frac = 0.5

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # Each rank writes to its own temp file
    rank_output = f"{args.output}.rank{ddp_rank}"

    total_generated = 0
    total_kept = 0
    total_filtered = 0

    print0(f"Generating {args.num_samples} total samples across {ddp_world_size} GPUs")
    print0(f"  Per-rank: ~{samples_per_rank} samples, batch_size={args.batch_size}, max_tokens={max_tokens}")
    print0(f"Output: {args.output}")

    with open(rank_output, "w", encoding="utf-8") as f:
        while total_kept < samples_per_rank:
            batch_num_samples = min(args.batch_size, samples_per_rank - total_kept + 100)
            # Each rank gets a unique seed: base_seed * world_size + rank offset
            seed = args.seed + ddp_rank * args.num_samples + total_generated

            with autocast_ctx:
                results, _ = engine.generate_batch(
                    bos_tokens,
                    num_samples=batch_num_samples,
                    max_tokens=max_tokens,
                    temperature=1.0,
                    seed=seed,
                )

            for token_ids in results:
                total_generated += 1
                text = tokenizer.decode(token_ids)

                # Filter
                if len(text) < min_chars:
                    total_filtered += 1
                    continue
                if fraction_non_ascii(text) > max_non_ascii_frac:
                    total_filtered += 1
                    continue

                record = {"text": text, "num_tokens": max_tokens}
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_kept += 1

                if total_kept >= samples_per_rank:
                    break

            if total_kept % 1000 < args.batch_size:
                print(f"  [Rank {ddp_rank}] Generated: {total_generated}, Kept: {total_kept}/{samples_per_rank}, Filtered: {total_filtered}")

    print(f"  [Rank {ddp_rank}] Done. Generated: {total_generated}, Kept: {total_kept}, Filtered: {total_filtered}")

    # Synchronize all ranks before concatenation
    if ddp:
        dist.barrier()

    # Rank 0 concatenates all per-rank files into the final output
    if ddp_rank == 0:
        print0(f"Concatenating {ddp_world_size} rank files into {args.output}...")
        total_lines = 0
        with open(args.output, "w", encoding="utf-8") as out_f:
            for rank in range(ddp_world_size):
                rank_file = f"{args.output}.rank{rank}"
                with open(rank_file, "r", encoding="utf-8") as in_f:
                    for line in in_f:
                        out_f.write(line)
                        total_lines += 1
                os.remove(rank_file)
        print0(f"Final output: {args.output} ({total_lines} samples)")

    # Clean up non-rank-0 temp files after barrier
    if ddp:
        dist.barrier()

    compute_cleanup()


if __name__ == "__main__":
    main()
