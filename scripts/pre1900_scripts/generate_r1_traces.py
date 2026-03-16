#!/usr/bin/env python3
"""
Generate reasoning traces using DeepSeek R1 on Bedrock.

Sends contradiction + discovery RL problems to R1, extracts reasoning_content
and content, and formats as SFT training data with <think>...</think> and
\\answer{} tags.

Output is CustomJSON format (one JSON array of messages per line) compatible
with the nanochat data loader.

Usage:
    python -m scripts.pre1900_scripts.generate_r1_traces \
        --output-dir instruct_data/r1_reasoning \
        --max-concurrent 10
"""

import json
import argparse
import time
import re
from pathlib import Path

import boto3
from concurrent.futures import ThreadPoolExecutor

from scripts.pre1900_scripts.constants import REASONING_SYSTEM_PROMPT

# ---------------------------------------------------------------------------
# Bedrock R1 client
# ---------------------------------------------------------------------------

MODEL_ID = "us.deepseek.r1-v1:0"
REGION = "us-east-1"


def call_r1_sync(
    client,
    prompt: str,
    max_tokens: int = 4096,
    temperature: float = 0.6,
    max_retries: int = 5,
) -> dict | None:
    """Call DeepSeek R1 on Bedrock synchronously with retry.

    Returns {"reasoning": ..., "content": ...} or None.
    """
    body = json.dumps({
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    })

    for attempt in range(max_retries):
        try:
            response = client.invoke_model(
                modelId=MODEL_ID,
                body=body,
                contentType="application/json",
                accept="application/json",
            )
            result = json.loads(response["body"].read())
            msg = result["choices"][0]["message"]
            return {
                "reasoning": msg.get("reasoning_content", ""),
                "content": msg.get("content", ""),
            }
        except client.exceptions.ThrottlingException:
            wait = min(2 ** attempt * 2, 30)
            time.sleep(wait)
        except Exception as e:
            if "throttl" in str(e).lower() or "rate" in str(e).lower():
                wait = min(2 ** attempt * 2, 30)
                time.sleep(wait)
            else:
                print(f"  Error: {e}")
                return None
    print(f"  Exhausted retries")
    return None


# ---------------------------------------------------------------------------
# Format R1 output into SFT training format
# ---------------------------------------------------------------------------

def format_as_sft(reasoning: str, content: str) -> str:
    """Convert R1's reasoning_content + content into <think>...</think> + \\answer{} format."""
    reasoning = (reasoning or "").strip()
    content = (content or "").strip()

    parts = []
    if reasoning:
        parts.append(f"<think>\n{reasoning}\n</think>")

    if content:
        parts.append(f"\n\\answer{{{content}}}")

    return "\n".join(parts)


def validate_trace(assistant_msg: str) -> bool:
    """Check that the formatted trace has the required structure."""
    if "<think>" not in assistant_msg or "</think>" not in assistant_msg:
        return False
    if "\\answer{" not in assistant_msg:
        return False
    think_match = re.search(r"<think>(.*?)</think>", assistant_msg, re.DOTALL)
    if not think_match or len(think_match.group(1).strip()) < 100:
        return False
    return True


# ---------------------------------------------------------------------------
# Load all problems (contradiction + discovery RL), deduplicated
# ---------------------------------------------------------------------------

HF_SNAP = "/root/.cache/huggingface/hub/datasets--mhla--gpt1900-instruct-data/snapshots/581647be149471056962a9c900c9166b407fce15"


def load_all_problems() -> list[dict]:
    """Load and deduplicate contradiction + discovery RL problems."""
    sources = [
        # (path, source_tag, split)
        (Path(HF_SNAP) / "contradiction_problems" / "contradiction_problems_train.jsonl", "contradiction", "train"),
        (Path(HF_SNAP) / "contradiction_problems" / "contradiction_problems_val.jsonl", "contradiction", "val"),
        (Path(HF_SNAP) / "rl_problems" / "rl_problems_train.jsonl", "discovery", "train"),
        (Path(HF_SNAP) / "rl_problems" / "rl_problems_val.jsonl", "discovery", "val"),
        (Path(HF_SNAP) / "rl_problems" / "rl_problems_train_expanded.jsonl", "discovery_expanded", "train"),
    ]

    # Deduplicate by prompt prefix (first 200 chars)
    seen_prompts = set()
    problems = []

    for path, source, split in sources:
        if not path.exists():
            print(f"  Warning: {path} not found, skipping")
            continue
        added = 0
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                key = record["prompt"][:200]
                if key in seen_prompts:
                    continue
                seen_prompts.add(key)
                record["_source"] = source
                record["_split"] = split
                problems.append(record)
                added += 1
        print(f"  {source} ({split}): +{added} problems")

    return problems


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------

def generate_all(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    traces_path = output_dir / "r1_traces_raw.jsonl"

    problems = load_all_problems()
    print(f"Total unique problems: {len(problems)}")

    # Resume state
    already_done = set()
    if args.resume and traces_path.exists():
        with open(traces_path) as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    already_done.add(record.get("_idx", -1))
        print(f"  Resume: {len(already_done)} already done")

    to_process = [(i, p) for i, p in enumerate(problems) if i not in already_done]
    print(f"  To process: {len(to_process)}")

    if not to_process:
        print("Nothing to generate. Building SFT dataset from existing traces...")
        build_sft_dataset(traces_path, output_dir)
        return

    client = boto3.client("bedrock-runtime", region_name=REGION)
    executor = ThreadPoolExecutor(max_workers=args.max_concurrent)

    mode = "a" if args.resume else "w"
    t0 = time.time()
    success = 0
    failed = 0
    invalid = 0

    def process_one(idx_problem):
        idx, problem = idx_problem
        result = call_r1_sync(
            client, problem["prompt"],
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        return idx, problem, result

    batch_size = args.max_concurrent * 2
    with open(traces_path, mode) as raw_f:
        for batch_start in range(0, len(to_process), batch_size):
            batch = to_process[batch_start:batch_start + batch_size]
            futures = list(executor.map(process_one, batch))

            for idx, problem, result in futures:
                if result is None:
                    failed += 1
                    continue

                assistant_msg = format_as_sft(result["reasoning"], result["content"])

                if not validate_trace(assistant_msg):
                    invalid += 1
                    raw_record = {
                        "_idx": idx,
                        "_valid": False,
                        "_source": problem.get("_source", ""),
                        "reasoning": result["reasoning"],
                        "content": result["content"],
                        "prompt": problem["prompt"],
                    }
                    raw_f.write(json.dumps(raw_record, ensure_ascii=False) + "\n")
                    continue

                raw_record = {
                    "_idx": idx,
                    "_valid": True,
                    "_source": problem.get("_source", ""),
                    "_split": problem.get("_split", "train"),
                    "reasoning": result["reasoning"],
                    "content": result["content"],
                    "prompt": problem["prompt"],
                    "gold_answer": problem.get("gold_answer", ""),
                    "domain": problem.get("domain", ""),
                    "book_filename": problem.get("book_filename", ""),
                    "title": problem.get("title", ""),
                    "year": problem.get("year", ""),
                }
                raw_f.write(json.dumps(raw_record, ensure_ascii=False) + "\n")
                success += 1

            raw_f.flush()
            total_done = batch_start + len(batch)
            elapsed = time.time() - t0
            rate = total_done / elapsed if elapsed > 0 else 0
            print(f"  Progress: {total_done}/{len(to_process)} ({rate:.1f}/s) | "
                  f"success={success} invalid={invalid} failed={failed} | "
                  f"{elapsed:.0f}s elapsed")

    elapsed = time.time() - t0
    print(f"\nGeneration done in {elapsed:.0f}s: {success} valid, {invalid} invalid, {failed} failed")

    print("\nBuilding SFT dataset...")
    build_sft_dataset(traces_path, output_dir)


def build_sft_dataset(traces_path: Path, output_dir: Path):
    """Build clean SFT dataset from raw traces."""
    system_message = {"role": "system", "content": REASONING_SYSTEM_PROMPT}

    train_records = []
    val_records = []

    source_counts = {}

    with open(traces_path) as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            if not record.get("_valid", False):
                continue

            assistant_msg = format_as_sft(record["reasoning"], record["content"])

            messages = [
                system_message,
                {"role": "user", "content": record["prompt"]},
                {"role": "assistant", "content": assistant_msg},
            ]

            src = record.get("_source", "unknown")
            source_counts[src] = source_counts.get(src, 0) + 1

            if record.get("_split") == "val":
                val_records.append(messages)
            else:
                train_records.append(messages)

    # Write split files
    train_path = output_dir / "sft_train.jsonl"
    val_path = output_dir / "sft_val.jsonl"

    with open(train_path, "w") as f:
        for msgs in train_records:
            f.write(json.dumps(msgs, ensure_ascii=False) + "\n")

    with open(val_path, "w") as f:
        for msgs in val_records:
            f.write(json.dumps(msgs, ensure_ascii=False) + "\n")

    print(f"  SFT train:  {train_path} ({len(train_records)} examples)")
    print(f"  SFT val:    {val_path} ({len(val_records)} examples)")
    print(f"  Total:      {len(train_records) + len(val_records)}")
    print(f"  By source:  {source_counts}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate R1 reasoning traces for contradiction + discovery problems"
    )
    parser.add_argument("--output-dir", type=str, default="instruct_data/r1_reasoning",
                        help="Output directory")
    parser.add_argument("--max-concurrent", type=int, default=10,
                        help="Max concurrent Bedrock requests")
    parser.add_argument("--max-tokens", type=int, default=4096,
                        help="Max tokens for R1 response")
    parser.add_argument("--temperature", type=float, default=0.6,
                        help="Sampling temperature")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing output")
    args = parser.parse_args()

    generate_all(args)


if __name__ == "__main__":
    main()
