#!/usr/bin/env python3
"""
Generate reasoning traces for format SFT from Yale PHYSICS problems.

Uses Claude to solve Yale physics problems in the target format
(<think>...</think>\\answer{LaTeX}), then SymPy-verifies against gold answers.
Only keeps traces where the answer is verified correct.

This produces high-quality SFT data for teaching the model the quantitative
reasoning format without suppressing mathematical expressions.

Usage:
    python -m scripts.pre1900_scripts.prepare_yale_format_sft \
        --base-dir $NANOCHAT_BASE_DIR \
        --max-problems 150 \
        --max-concurrent 10
"""

import argparse
import asyncio
import json
import os
import random
from pathlib import Path

import anthropic

from scripts.pre1900_scripts.constants import QUANTITATIVE_REASONING_SYSTEM_PROMPT
from tasks.yale_physics import extract_answer_latex, is_symbolically_equivalent


# ---------------------------------------------------------------------------
# Trace generation via Claude
# ---------------------------------------------------------------------------

GENERATION_SYSTEM_PROMPT = QUANTITATIVE_REASONING_SYSTEM_PROMPT

async def generate_trace(
    client: anthropic.AsyncAnthropic,
    question: str,
    model: str,
    semaphore: asyncio.Semaphore,
    max_tokens: int = 4096,
) -> str | None:
    """Generate a reasoning trace for a physics problem."""
    async with semaphore:
        try:
            result = await client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=GENERATION_SYSTEM_PROMPT,
                messages=[
                    {"role": "user", "content": question},
                ],
            )
            return result.content[0].text
        except Exception as e:
            print(f"  API error: {e}")
            return None


async def generate_and_verify_batch(
    problems: list[dict],
    model: str,
    max_concurrent: int,
) -> list[dict]:
    """Generate traces for a batch of problems and verify against gold answers."""
    client = anthropic.AsyncAnthropic()
    semaphore = asyncio.Semaphore(max_concurrent)

    # Generate all traces concurrently
    tasks = [
        generate_trace(client, p["questions"], model, semaphore)
        for p in problems
    ]
    traces = await asyncio.gather(*tasks)

    # Verify each trace
    verified = []
    stats = {"total": len(problems), "api_fail": 0, "no_answer": 0,
             "parse_fail": 0, "wrong": 0, "correct": 0}

    for problem, trace in zip(problems, traces):
        if trace is None:
            stats["api_fail"] += 1
            continue

        # Extract answer from trace
        pred = extract_answer_latex(trace)
        if pred is None:
            stats["no_answer"] += 1
            continue

        # Verify against first gold answer
        gold = problem["gold_answers"][0]
        result = is_symbolically_equivalent(pred, gold)

        if result is None:
            stats["parse_fail"] += 1
            continue
        elif result is False:
            stats["wrong"] += 1
            continue
        else:
            stats["correct"] += 1
            verified.append({
                "id": problem["id"],
                "domain": problem.get("domain", problem.get("_domain", "")),
                "questions": problem["questions"],
                "gold_answers": problem["gold_answers"],
                "trace": trace,
                "predicted_answer": pred,
            })

    return verified, stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate format SFT traces from Yale PHYSICS")
    parser.add_argument("--base-dir", type=str, default=None)
    parser.add_argument("--model", type=str, default="claude-sonnet-4-20250514",
                        help="Claude model for trace generation")
    parser.add_argument("--max-problems", type=int, default=150,
                        help="Max problems to attempt")
    parser.add_argument("--max-concurrent", type=int, default=10,
                        help="Max concurrent API requests")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    base_dir = args.base_dir or os.environ.get("NANOCHAT_BASE_DIR")
    if not base_dir:
        base_dir = str(Path(__file__).resolve().parent.parent.parent)

    random.seed(args.seed)

    # Load Yale problems (train split)
    problems_path = Path(base_dir) / "instruct_data" / "yale_physics" / "yale_problems_train.jsonl"
    if not problems_path.exists():
        print(f"Error: {problems_path} not found. Run prepare_yale_physics.py first.")
        return

    problems = []
    with open(problems_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            problems.append(json.loads(line))

    print(f"Loaded {len(problems)} problems from {problems_path}")

    # Sample a subset
    if len(problems) > args.max_problems:
        problems = random.sample(problems, args.max_problems)
        print(f"Sampled {args.max_problems} problems for trace generation")

    # Generate and verify traces
    print(f"\nGenerating traces with {args.model} (max_concurrent={args.max_concurrent})...")
    verified, stats = asyncio.run(
        generate_and_verify_batch(problems, args.model, args.max_concurrent)
    )

    print(f"\nGeneration stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"  yield: {len(verified)}/{stats['total']} ({100*len(verified)/max(stats['total'],1):.0f}%)")

    if not verified:
        print("No verified traces generated. Check API key and problem format.")
        return

    # Write SFT output in CustomJSON format (system + user + assistant)
    output_dir = Path(base_dir) / "instruct_data" / "yale_physics"
    output_dir.mkdir(parents=True, exist_ok=True)

    sft_path = output_dir / "yale_format_sft.jsonl"
    system_message = {"role": "system", "content": QUANTITATIVE_REASONING_SYSTEM_PROMPT}

    with open(sft_path, "w", encoding="utf-8") as f:
        for item in verified:
            messages = [
                system_message,
                {"role": "user", "content": item["questions"]},
                {"role": "assistant", "content": item["trace"]},
            ]
            f.write(json.dumps(messages, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(verified)} verified SFT traces to {sft_path}")

    # Also write a metadata file for reference
    meta_path = output_dir / "yale_format_sft_meta.jsonl"
    with open(meta_path, "w", encoding="utf-8") as f:
        for item in verified:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Wrote metadata to {meta_path}")

    # Summary by domain
    by_domain: dict[str, int] = {}
    for item in verified:
        d = item["domain"]
        by_domain[d] = by_domain.get(d, 0) + 1
    print(f"\nVerified traces by domain:")
    for domain, count in sorted(by_domain.items()):
        print(f"  {domain}: {count}")

    print("\nDone.")


if __name__ == "__main__":
    main()
