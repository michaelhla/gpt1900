#!/usr/bin/env python3
"""
Prepare the OpenThoughts3-1.2M dataset for SFT training.

Downloads the dataset, filters by domain (math + science), applies anachronism
filtering on science examples, converts to CustomJSON format, chunks long
assistant responses so the model sees all tokens (including answers), and
deduplicates to 1 annotation per unique question.

Each long conversation is split into multiple chunks, each with the same
system+user prefix but a different slice of the assistant response. This way
the model sees the full reasoning trace including the final \\answer{}.

Usage:
    python -m scripts.pre1900_scripts.prepare_openthoughts \
        --output-dir instruct_data/openthoughts \
        --max-chars 6500 \
        --val-fraction 0.05
"""

import argparse
import json
import os
import random
import re
from collections import defaultdict

from datasets import load_dataset

from scripts.pre1900_scripts.filter_instruct_pairs import (
    compile_patterns,
    check_conversation,
)
from scripts.pre1900_scripts.constants import (
    REASONING_SYSTEM_PROMPT,
    QUANTITATIVE_REASONING_SYSTEM_PROMPT,
)


def has_quantitative_answer(text: str) -> bool:
    """Detect if the response contains a quantitative/LaTeX answer."""
    m = re.search(r"\\(?:boxed|answer)\{(.+?)\}", text)
    if m:
        inner = m.group(1)
        if re.search(r"[\d\\frac\\sqrt\\pi\\int\\sum]", inner):
            return True
    return False


def chunk_assistant(assistant_msg: str, chunk_chars: int) -> list[str]:
    """Split assistant message into chunks of approximately chunk_chars characters.

    Tries to cut at paragraph or sentence boundaries to keep chunks coherent.
    """
    if len(assistant_msg) <= chunk_chars:
        return [assistant_msg]

    chunks = []
    remaining = assistant_msg
    while len(remaining) > chunk_chars:
        # Take a chunk_chars slice
        candidate = remaining[:chunk_chars]

        # Try to cut at a paragraph boundary (double newline)
        cut = candidate.rfind("\n\n")
        # Fall back to single newline
        if cut < chunk_chars * 0.5:
            cut = candidate.rfind("\n")
        # Fall back to sentence end
        if cut < chunk_chars * 0.5:
            cut = candidate.rfind(". ")
        # Last resort: hard cut
        if cut < chunk_chars * 0.5:
            cut = chunk_chars

        chunks.append(remaining[:cut + 1].rstrip())
        remaining = remaining[cut + 1:].lstrip()

    if remaining.strip():
        chunks.append(remaining.strip())

    return chunks


def convert_and_chunk(example: dict, max_chars: int) -> list[list[dict]] | None:
    """Convert OpenThoughts3 format to chunked CustomJSON conversations.

    Returns a list of conversations (each is [system, user, assistant]) or None.
    Long assistant responses are split into multiple conversations, each with
    the same system+user prefix but a different chunk of the assistant response.
    """
    convs = example.get("conversations", [])
    if len(convs) < 2:
        return None

    human_msg = None
    assistant_msg = None
    for turn in convs:
        if turn["from"] == "human":
            human_msg = turn["value"]
        elif turn["from"] in ("assistant", "gpt"):
            assistant_msg = turn["value"]

    if not human_msg or not assistant_msg:
        return None

    # Replace \boxed{} -> \answer{} in assistant response
    assistant_msg = re.sub(r"\\boxed\{", r"\\answer{", assistant_msg)

    # Choose system prompt based on answer type
    if has_quantitative_answer(assistant_msg):
        system_prompt = QUANTITATIVE_REASONING_SYSTEM_PROMPT
    else:
        system_prompt = REASONING_SYSTEM_PROMPT

    # Calculate char budget for assistant per chunk
    # Overhead: system prompt + user message + special token chars (~50 chars)
    prefix_chars = len(system_prompt) + len(human_msg) + 50
    assistant_budget = max_chars - prefix_chars
    if assistant_budget < 200:
        return None  # question too long, no room for assistant

    # Chunk the assistant response
    assistant_chunks = chunk_assistant(assistant_msg, assistant_budget)

    # Build one conversation per chunk
    result = []
    for chunk in assistant_chunks:
        result.append([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": human_msg},
            {"role": "assistant", "content": chunk},
        ])

    return result


def main():
    parser = argparse.ArgumentParser(description="Prepare OpenThoughts3-1.2M for SFT")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory relative to NANOCHAT_BASE_DIR")
    parser.add_argument("--max-chars", type=int, default=6500,
                        help="Max total chars per conversation chunk (~2048 tokens at ~3.2 chars/token)")
    parser.add_argument("--val-fraction", type=float, default=0.05,
                        help="Fraction for validation split")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only print statistics, don't write files")
    args = parser.parse_args()

    random.seed(args.seed)

    # Resolve output directory
    base_dir = os.environ.get("NANOCHAT_BASE_DIR")
    if not base_dir:
        from nanochat.common import get_base_dir
        base_dir = get_base_dir()
    output_dir = os.path.join(base_dir, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Compile anachronism filters
    always_reject, context_pats, meta_reject = compile_patterns()

    # -------------------------------------------------------------------------
    # Phase 1: Stream dataset and filter
    # -------------------------------------------------------------------------
    print("Loading OpenThoughts3-1.2M (streaming)...")
    ds = load_dataset("open-thoughts/OpenThoughts3-1.2M", split="train", streaming=True)

    stats = {
        "total": 0,
        "domain_code_skip": 0,
        "domain_math": 0,
        "domain_science": 0,
        "domain_other": 0,
        "anachronism_rejected": 0,
        "conversion_failed": 0,
        "kept_examples": 0,
        "total_chunks": 0,
    }
    rejection_reasons = defaultdict(int)

    # Group by question text for deduplication
    # Key: question text -> list of (char_count, list_of_chunk_conversations)
    question_groups: dict[str, list[tuple[int, list[list[dict]]]]] = defaultdict(list)

    for example in ds:
        stats["total"] += 1
        domain = example.get("domain", "")

        # Domain filter: skip code
        if domain == "code":
            stats["domain_code_skip"] += 1
            continue

        if domain == "math":
            stats["domain_math"] += 1
        elif domain == "science":
            stats["domain_science"] += 1
        else:
            stats["domain_other"] += 1

        # Anachronism filter: only for science
        if domain == "science":
            combined = " ".join(
                turn.get("value", "") for turn in example.get("conversations", [])
            )
            reason = check_conversation(combined, always_reject, context_pats, meta_reject)
            if reason is not None:
                stats["anachronism_rejected"] += 1
                bucket = reason.split(":")[0]
                rejection_reasons[bucket] += 1
                continue

        # Convert and chunk
        chunked_convs = convert_and_chunk(example, max_chars=args.max_chars)
        if chunked_convs is None:
            stats["conversion_failed"] += 1
            continue

        stats["kept_examples"] += 1
        stats["total_chunks"] += len(chunked_convs)

        # Original total chars (for dedup: prefer shorter originals)
        orig_chars = sum(len(t.get("value", "")) for t in example.get("conversations", []))

        # Group by question for dedup
        question_text = chunked_convs[0][1]["content"]  # user message from first chunk
        question_groups[question_text].append((orig_chars, chunked_convs))

        # Progress
        if stats["total"] % 50000 == 0:
            print(f"  Processed {stats['total']:,} examples, kept {stats['kept_examples']:,}, "
                  f"unique questions {len(question_groups):,}, "
                  f"total chunks {stats['total_chunks']:,}")

    # -------------------------------------------------------------------------
    # Phase 2: Deduplicate - keep shortest annotation per question
    # -------------------------------------------------------------------------
    print(f"\nDeduplicating: {len(question_groups):,} unique questions "
          f"from {stats['kept_examples']:,} kept examples")

    deduped_chunks = []
    for question, annotations in question_groups.items():
        annotations.sort(key=lambda x: x[0])  # shortest original first
        _, chunked_convs = annotations[0]
        deduped_chunks.extend(chunked_convs)

    n_unique = len(question_groups)
    print(f"After dedup: {n_unique:,} unique questions -> {len(deduped_chunks):,} total chunks")

    # -------------------------------------------------------------------------
    # Phase 3: Train/val split and write
    # -------------------------------------------------------------------------
    # Split by question first, then expand to chunks
    questions = list(question_groups.keys())
    random.shuffle(questions)
    n_val_q = max(1, int(len(questions) * args.val_fraction))
    val_questions = set(questions[:n_val_q])

    train_chunks = []
    val_chunks = []
    for question, annotations in question_groups.items():
        annotations.sort(key=lambda x: x[0])
        _, chunked_convs = annotations[0]
        if question in val_questions:
            val_chunks.extend(chunked_convs)
        else:
            train_chunks.extend(chunked_convs)

    # Shuffle chunks within each split
    random.shuffle(train_chunks)
    random.shuffle(val_chunks)

    # Print statistics
    print(f"\n{'='*60}")
    print(f"OpenThoughts3-1.2M Preparation Statistics")
    print(f"{'='*60}")
    print(f"Total examples streamed: {stats['total']:,}")
    print(f"  Code (dropped):        {stats['domain_code_skip']:,}")
    print(f"  Math:                  {stats['domain_math']:,}")
    print(f"  Science:               {stats['domain_science']:,}")
    print(f"  Other:                 {stats['domain_other']:,}")
    print(f"  Anachronism rejected:  {stats['anachronism_rejected']:,}")
    if rejection_reasons:
        for reason, count in sorted(rejection_reasons.items(), key=lambda x: -x[1]):
            print(f"    {reason}: {count}")
    print(f"  Conversion failed:     {stats['conversion_failed']:,}")
    print(f"  Kept examples:         {stats['kept_examples']:,}")
    print(f"  Unique questions:      {n_unique:,}")
    print(f"  Total chunks:          {len(deduped_chunks):,}")
    print(f"  Avg chunks/question:   {len(deduped_chunks) / max(n_unique, 1):.1f}")
    print(f"    Train:               {len(train_chunks):,} chunks ({len(questions) - n_val_q:,} questions)")
    print(f"    Val:                 {len(val_chunks):,} chunks ({n_val_q:,} questions)")

    if args.dry_run:
        print("\n[DRY RUN] No files written.")
        if deduped_chunks:
            # Show chunks from one question
            q = questions[0]
            annotations = question_groups[q]
            annotations.sort(key=lambda x: x[0])
            _, chunks = annotations[0]
            print(f"\nSample question ({len(chunks)} chunks):")
            print(f"  User: {chunks[0][1]['content'][:200]}")
            for i, c in enumerate(chunks[:3]):
                total = sum(len(m["content"]) for m in c)
                print(f"\n  Chunk {i+1}/{len(chunks)} ({total:,} chars):")
                print(f"    Assistant start: {c[2]['content'][:150]}...")
                print(f"    Assistant end:   ...{c[2]['content'][-150:]}")
        return

    # Write output files
    train_path = os.path.join(output_dir, "sft_train.jsonl")
    val_path = os.path.join(output_dir, "sft_val.jsonl")

    with open(train_path, "w", encoding="utf-8") as f:
        for messages in train_chunks:
            f.write(json.dumps(messages, ensure_ascii=False) + "\n")

    with open(val_path, "w", encoding="utf-8") as f:
        for messages in val_chunks:
            f.write(json.dumps(messages, ensure_ascii=False) + "\n")

    print(f"\nOutput:")
    print(f"  Train: {train_path} ({len(train_chunks):,} chunks)")
    print(f"  Val:   {val_path} ({len(val_chunks):,} chunks)")


if __name__ == "__main__":
    main()
