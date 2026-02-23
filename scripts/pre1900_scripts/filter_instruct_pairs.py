#!/usr/bin/env python3
"""
Filter crafted instruction pairs for anachronisms and meta-references.

Reuses the anachronism detection patterns from clean_full_corpus.py and adds
additional filters for AI/ML meta-references.

Splits 5% of passing conversations into a validation set.

Usage:
    python -m scripts.pre1900_scripts.filter_instruct_pairs \
        --input instruct_data/crafted_pairs.jsonl \
        --output-dir instruct_data
"""

import os
import re
import json
import random
import argparse

from scripts.pre1900_scripts.clean_full_corpus import (
    ALWAYS_REJECT_PATTERNS,
    CONTEXT_PATTERNS,
    CONTEXT_THRESHOLD,
    POST_1900_YEAR_RE,
)

# Additional meta-reference patterns — things that should never appear
# in a pre-1900 assistant's responses
META_REFERENCE_PATTERNS = [
    r"\bartificial\s+intelligence\b",
    r"\blanguage\s+model\b",
    r"\bmachine\s+learning\b",
    r"\bdeep\s+learning\b",
    r"\bneural\s+network\b",
    r"\bcomputer(?:s)?\b",
    r"\binternet\b",
    r"\bemail\b",
    r"\bwebsite\b",
    r"\bonline\b",
    r"\bdigital\b",
    r"\bsoftware\b",
    r"\bhardware\b",
    r"\balgorithm(?:s)?\b",
    r"\bdatabase\b",
    r"\bprogramming\b",
    r"\b(?:AI|ML|NLP|GPT|LLM)\b",
    r"\bchatbot\b",
    r"\bvirtual\s+assistant\b",
    r"\bas\s+an?\s+(?:AI|language\s+model|assistant\s+trained)\b",
    r"\bI\s+(?:was|am)\s+(?:trained|programmed|designed)\b",
    r"\baccording\s+to\s+(?:the\s+)?(?:text|passage|excerpt|source)\b",
    r"\bthe\s+(?:text|passage|excerpt)\s+(?:says|states|mentions|describes|indicates)\b",
    r"\bin\s+the\s+(?:provided|given)\s+(?:text|passage|excerpt)\b",
]


def compile_patterns():
    """Compile all filter patterns."""
    always_reject = re.compile("|".join(ALWAYS_REJECT_PATTERNS), re.IGNORECASE)
    context_pats = [re.compile(p, re.IGNORECASE) for p in CONTEXT_PATTERNS]
    meta_reject = re.compile("|".join(META_REFERENCE_PATTERNS), re.IGNORECASE)
    return always_reject, context_pats, meta_reject


def get_combined_text(messages: list[dict]) -> str:
    """Extract combined text from all messages in a conversation."""
    parts = []
    for msg in messages:
        if msg.get("content"):
            parts.append(msg["content"])
    return " ".join(parts)


def check_conversation(text: str, always_reject, context_pats, meta_reject) -> str | None:
    """Check a conversation's combined text for anachronisms.

    Returns None if clean, or a rejection reason string.
    """
    # 1. Always-reject patterns (post-1900 physics/science)
    m = always_reject.search(text)
    if m:
        return f"always_reject:{m.group()[:40]}"

    # 2. Post-1900 year references
    year_matches = POST_1900_YEAR_RE.findall(text)
    if len(year_matches) >= 2:  # stricter threshold for short conversations
        return f"post_1900_years:{len(year_matches)}"

    # 3. Context pattern co-occurrence
    context_hits = 0
    for pat in context_pats:
        if pat.search(text):
            context_hits += 1
            if context_hits >= CONTEXT_THRESHOLD:
                return "context_accumulation"

    # 4. Meta-references (AI, language model, passage references, etc.)
    m = meta_reject.search(text)
    if m:
        return f"meta_reference:{m.group()[:40]}"

    return None


def main():
    parser = argparse.ArgumentParser(description="Filter crafted instruction pairs for anachronisms")
    parser.add_argument("--input", type=str, required=True, help="Input JSONL with crafted conversations")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for filtered files")
    parser.add_argument("--val-fraction", type=float, default=0.05, help="Fraction for validation split")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    always_reject, context_pats, meta_reject = compile_patterns()

    # Read all conversations
    conversations = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                messages = json.loads(line)
                conversations.append((line_num, messages))
            except json.JSONDecodeError:
                print(f"  Warning: invalid JSON on line {line_num + 1}, skipping")

    print(f"Loaded {len(conversations)} conversations from {args.input}")

    # Filter
    passed = []
    rejected = []
    reason_counts = {}

    for line_num, messages in conversations:
        combined = get_combined_text(messages)
        reason = check_conversation(combined, always_reject, context_pats, meta_reject)

        if reason is None:
            passed.append(messages)
        else:
            rejected.append({"line": line_num, "reason": reason, "messages": messages})
            bucket = reason.split(":")[0]
            reason_counts[bucket] = reason_counts.get(bucket, 0) + 1

    print(f"\nFiltering results:")
    print(f"  Passed:   {len(passed)} ({100 * len(passed) / max(len(conversations), 1):.1f}%)")
    print(f"  Rejected: {len(rejected)} ({100 * len(rejected) / max(len(conversations), 1):.1f}%)")

    if reason_counts:
        print(f"\n  Rejection reasons:")
        for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
            print(f"    {reason}: {count}")

    # Split into train and validation
    random.shuffle(passed)
    n_val = max(1, int(len(passed) * args.val_fraction))
    val_set = passed[:n_val]
    train_set = passed[n_val:]

    print(f"\n  Train: {len(train_set)}")
    print(f"  Val:   {len(val_set)}")

    # Write outputs
    train_path = os.path.join(args.output_dir, "filtered_pairs.jsonl")
    val_path = os.path.join(args.output_dir, "val_pairs.jsonl")
    rejected_path = os.path.join(args.output_dir, "rejected_pairs.jsonl")

    with open(train_path, "w", encoding="utf-8") as f:
        for messages in train_set:
            f.write(json.dumps(messages, ensure_ascii=False) + "\n")

    with open(val_path, "w", encoding="utf-8") as f:
        for messages in val_set:
            f.write(json.dumps(messages, ensure_ascii=False) + "\n")

    with open(rejected_path, "w", encoding="utf-8") as f:
        for item in rejected:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n  Output:")
    print(f"    Train:    {train_path}")
    print(f"    Val:      {val_path}")
    print(f"    Rejected: {rejected_path}")


if __name__ == "__main__":
    main()
