#!/usr/bin/env python3
"""
Generate unconditional text samples from the pre-1900 base model.

Generates from [BOS] token with temperature 1.0, producing 1024-token samples.
Filters out samples that are too short or mostly non-ASCII.

Usage:
    python -m scripts.pre1900_scripts.generate_unconditional --model-tag d26 --num-samples 50000 --output instruct_data/raw_generations.jsonl
"""

import os
import argparse
import json

import torch
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
    parser.add_argument("--num-samples", type=int, default=50000, help="Total number of samples to generate")
    parser.add_argument("--batch-size", type=int, default=32, help="Samples per batch")
    parser.add_argument("--output", type=str, default="instruct_data/raw_generations.jsonl", help="Output JSONL path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
    args = parser.parse_args()

    # Compute init
    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()

    # Load model
    model, tokenizer, meta = load_model("base", device, phase="eval", model_tag=args.model_tag, step=args.step)
    engine = Engine(model, tokenizer)

    # BOS-only prompt
    bos_tokens = tokenizer("", prepend="<|bos|>")

    max_tokens = 1024
    min_chars = 100
    max_non_ascii_frac = 0.5

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    total_generated = 0
    total_kept = 0
    total_filtered = 0

    print0(f"Generating {args.num_samples} samples, batch_size={args.batch_size}, max_tokens={max_tokens}")
    print0(f"Output: {args.output}")

    with open(args.output, "w", encoding="utf-8") as f:
        while total_kept < args.num_samples:
            batch_num_samples = min(args.batch_size, args.num_samples - total_kept + 100)  # overshoot slightly to account for filtering
            seed = args.seed + total_generated

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

                if total_kept >= args.num_samples:
                    break

            if total_kept % 1000 < args.batch_size:
                print0(f"  Generated: {total_generated}, Kept: {total_kept}/{args.num_samples}, Filtered: {total_filtered}")

    print0(f"\nDone. Generated: {total_generated}, Kept: {total_kept}, Filtered: {total_filtered}")
    print0(f"Output: {args.output}")

    compute_cleanup()


if __name__ == "__main__":
    main()
