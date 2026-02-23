#!/usr/bin/env python3
"""
Use Claude (Anthropic API) to create standalone multi-turn conversations
from unconditionally generated text samples.

Each raw generation serves as a "knowledge anchor" — Claude creates natural
conversations using only that knowledge, in the style of the source text.

Usage:
    python -m scripts.pre1900_scripts.craft_instruct_pairs \
        --input instruct_data/raw_generations.jsonl \
        --output instruct_data/crafted_pairs.jsonl
"""

import os
import json
import random
import asyncio
import argparse
import time

import anthropic

# System prompt prepended to all output conversations
SYSTEM_PROMPT = "You are a knowledgeable assistant with expertise in history, science, literature, and the affairs of the world up to the year 1900."

SINGLE_TURN_PROMPT = """\
You are creating a single-turn question-answer pair for a language model that only has
knowledge from before the year 1900.

You will be given a text excerpt that represents what this model knows. Create a natural,
standalone question-answer pair using ONLY the facts, topics, and knowledge in this text.

RULES:
1. The conversation must be STANDALONE — do NOT reference "the text" or "the passage".
2. Use ONLY knowledge explicitly present in the provided text. Do NOT add external facts.
3. Never mention anything from after 1900.
4. The assistant's response should match the STYLE of the provided text — same tone,
   vocabulary, and register (typically educated 19th century English prose).
5. When reusing words or phrases from the text, copy them EXACTLY as they appear —
   preserve any unusual spacing, number-for-letter substitutions (e.g. "0" for "O"),
   or unusual apostrophe placement. These are authentic period artifacts.
6. If the text is too garbled or incoherent, return: {"rejected": true, "reason": "..."}

Return JSON:
{"user": "...", "assistant": "..."}"""

MULTI_TURN_PROMPT = """\
You are creating a multi-turn conversation for a language model that only has knowledge
from before the year 1900.

You will be given a text excerpt that represents what this model knows. Create a natural,
standalone conversation with 2-3 exchanges using ONLY the facts, topics, and knowledge
in this text.

RULES:
1. The conversation must be STANDALONE — do NOT reference "the text" or "the passage".
2. Use ONLY knowledge explicitly present in the provided text. Do NOT add external facts.
3. Never mention anything from after 1900.
4. The assistant's responses should match the STYLE of the provided text — same tone,
   vocabulary, and register (typically educated 19th century English prose).
5. When reusing words or phrases from the text, copy them EXACTLY as they appear —
   preserve any unusual spacing, number-for-letter substitutions (e.g. "0" for "O"),
   or unusual apostrophe placement. These are authentic period artifacts.
6. If the text is too garbled or incoherent, return: {"rejected": true, "reason": "..."}

Return JSON:
{"turns": [
  {"role": "user", "content": "..."},
  {"role": "assistant", "content": "..."},
  {"role": "user", "content": "..."},
  {"role": "assistant", "content": "..."}
]}"""


def parse_response(text: str, is_multi_turn: bool) -> dict | None:
    """Parse Claude's JSON response. Returns None on parse failure."""
    # Strip markdown code fences if present
    text = text.strip()
    if text.startswith("```"):
        # Remove first line (```json or ```) and last line (```)
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None

    # Check for rejection
    if data.get("rejected"):
        return {"rejected": True, "reason": data.get("reason", "unknown")}

    if is_multi_turn:
        turns = data.get("turns")
        if not turns or not isinstance(turns, list) or len(turns) < 4:
            return None
        # Validate alternating roles
        for i, turn in enumerate(turns):
            expected = "user" if i % 2 == 0 else "assistant"
            if turn.get("role") != expected or not turn.get("content"):
                return None
        # Build conversation with system prompt
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for turn in turns:
            messages.append({"role": turn["role"], "content": turn["content"]})
        return {"messages": messages}
    else:
        user = data.get("user")
        assistant = data.get("assistant")
        if not user or not assistant:
            return None
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
        return {"messages": messages}


async def process_sample(
    client: anthropic.AsyncAnthropic,
    text: str,
    model: str,
    multi_turn_ratio: float,
    semaphore: asyncio.Semaphore,
    sample_idx: int,
) -> tuple[int, dict | None]:
    """Process a single sample through Claude. Returns (index, result_or_None)."""
    is_multi_turn = random.random() < multi_turn_ratio
    prompt = MULTI_TURN_PROMPT if is_multi_turn else SINGLE_TURN_PROMPT

    async with semaphore:
        try:
            response = await client.messages.create(
                model=model,
                max_tokens=2048,
                messages=[{"role": "user", "content": f"{prompt}\n\n---\n\n{text}"}],
            )
            response_text = response.content[0].text
            result = parse_response(response_text, is_multi_turn)
            return sample_idx, result
        except Exception as e:
            print(f"  Error on sample {sample_idx}: {e}")
            return sample_idx, None


async def main_async(args):
    # Load input samples
    samples = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            samples.append(record["text"])
    print(f"Loaded {len(samples)} samples from {args.input}")

    # Check for resume
    already_done = set()
    if args.resume and os.path.exists(args.output):
        with open(args.output, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    already_done.add(len(already_done))
        print(f"Resuming: {len(already_done)} samples already processed")

    # Determine which samples to process
    to_process = [(i, samples[i]) for i in range(len(samples)) if i not in already_done]
    print(f"Processing {len(to_process)} samples (multi_turn_ratio={args.multi_turn_ratio})")

    client = anthropic.AsyncAnthropic()
    semaphore = asyncio.Semaphore(args.max_concurrent)

    # Open output files
    rejections_path = args.output.replace(".jsonl", "_rejections.jsonl")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    mode = "a" if args.resume else "w"

    n_success = 0
    n_rejected = 0
    n_failed = 0
    t0 = time.time()

    # Process in chunks to avoid too many concurrent tasks
    chunk_size = args.max_concurrent * 4
    with open(args.output, mode, encoding="utf-8") as out_f, \
         open(rejections_path, mode, encoding="utf-8") as rej_f:

        for chunk_start in range(0, len(to_process), chunk_size):
            chunk = to_process[chunk_start:chunk_start + chunk_size]
            tasks = [
                process_sample(client, text, args.model, args.multi_turn_ratio, semaphore, idx)
                for idx, text in chunk
            ]

            results = await asyncio.gather(*tasks)

            for sample_idx, result in results:
                if result is None:
                    n_failed += 1
                elif result.get("rejected"):
                    n_rejected += 1
                    rej_f.write(json.dumps({"index": sample_idx, "reason": result["reason"]}, ensure_ascii=False) + "\n")
                else:
                    n_success += 1
                    # Write as JSON array matching CustomJSON format
                    out_f.write(json.dumps(result["messages"], ensure_ascii=False) + "\n")

            out_f.flush()
            rej_f.flush()

            total_done = chunk_start + len(chunk)
            elapsed = time.time() - t0
            rate = total_done / elapsed if elapsed > 0 else 0
            print(f"  Progress: {total_done}/{len(to_process)} ({rate:.1f}/s) | "
                  f"success={n_success} rejected={n_rejected} failed={n_failed}")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s")
    print(f"  Success: {n_success}")
    print(f"  Rejected: {n_rejected}")
    print(f"  Failed: {n_failed}")
    print(f"  Output: {args.output}")
    print(f"  Rejections: {rejections_path}")


def main():
    parser = argparse.ArgumentParser(description="Craft instruction pairs from raw generations using Claude")
    parser.add_argument("--input", type=str, required=True, help="Input JSONL with raw generations")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL with crafted conversations")
    parser.add_argument("--model", type=str, default="claude-sonnet-4-20250514", help="Anthropic model to use")
    parser.add_argument("--max-concurrent", type=int, default=20, help="Max concurrent API requests")
    parser.add_argument("--multi-turn-ratio", type=float, default=0.3, help="Fraction of multi-turn conversations")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
