#!/usr/bin/env python3
"""
Clean generated SFT data:
  1. Remove overused filler phrases ("I confess", "indeed", etc.)
  2. Filter post-1900 anachronisms (years, tech terms, physics)
  3. Unicode cleanup (curly quotes, em dashes -> ASCII)
  4. Train/val split

Usage:
    python -m scripts.pre1900_scripts.clean_sft_data \
        --input-dir instruct_data/v3_generated/ \
        --output-dir instruct_data/v3_cleaned/
"""

import argparse
import json
import os
import random
import re

from scripts.pre1900_scripts.clean_full_corpus import (
    ALWAYS_REJECT_PATTERNS,
    CONTEXT_PATTERNS,
    CONTEXT_THRESHOLD,
    POST_1900_YEAR_RE,
)

from scripts.pre1900_scripts.filter_instruct_pairs import (
    META_REFERENCE_PATTERNS,
)

# ---------------------------------------------------------------------------
# Phrase filtering
# ---------------------------------------------------------------------------

# Phrases to strip from assistant text (case-insensitive replacement)
STRIP_PHRASES = [
    # Overused fillers
    (r"\bI confess(?:\s+that)?\b", ""),
    (r"\bIndeed,\s*", ""),        # "Indeed, " sentence-initial filler only
    # Prediction/opinion markers
    (r"\bmark my words\b", ""),
    (r"\bI foresee\b", ""),
    (r"\bI am firmly convinced\b", "I believe"),
    (r"\bI am quite convinced\b", "I believe"),
    (r"\bI am persuaded\b", "I believe"),
    (r"\bI hold it to be\b", "I consider it"),
    (r"\bthe coming century shall\b", "the future may"),
    (r"\bwithin fifty years\b", "in time"),
    (r"\bbeyond our present imagining\b", "remarkable"),
    (r"\bannihilated distance\b", "shortened distances"),
]

STRIP_COMPILED = [(re.compile(pat, re.IGNORECASE), repl) for pat, repl in STRIP_PHRASES]


def strip_phrases(text: str) -> str:
    """Remove overused filler phrases from text."""
    for pat, repl in STRIP_COMPILED:
        text = pat.sub(repl, text)
    # Clean up double spaces and leading/trailing punctuation artifacts
    text = re.sub(r"  +", " ", text)
    text = re.sub(r"^\s*[,;]\s*", "", text)  # leading comma after removal
    text = re.sub(r"\s+([,;.!?])", r"\1", text)  # space before punctuation
    return text.strip()


# ---------------------------------------------------------------------------
# Unicode cleanup
# ---------------------------------------------------------------------------

def clean_unicode(text: str) -> str:
    text = text.replace("\u2014", ".")
    text = text.replace("\u2013", ".")
    text = text.replace("\u2018", "'")
    text = text.replace("\u2019", "'")
    text = text.replace("\u201c", '"')
    text = text.replace("\u201d", '"')
    return text


# ---------------------------------------------------------------------------
# Anachronism detection
# ---------------------------------------------------------------------------

_always_reject_re = re.compile("|".join(ALWAYS_REJECT_PATTERNS), re.IGNORECASE)
_context_pats = [re.compile(p, re.IGNORECASE) for p in CONTEXT_PATTERNS]
_meta_reject_re = re.compile("|".join(META_REFERENCE_PATTERNS), re.IGNORECASE)


def is_anachronistic(messages: list[dict]) -> str | None:
    """Returns rejection reason or None if clean."""
    text = " ".join(m["content"] for m in messages)

    # Always-reject physics/science terms
    m = _always_reject_re.search(text)
    if m:
        return f"always_reject:{m.group()[:40]}"

    # Post-1900 year refs
    year_matches = POST_1900_YEAR_RE.findall(text)
    if len(year_matches) >= 2:
        return f"post_1900_years:{len(year_matches)}"

    # Context pattern accumulation
    hits = 0
    for pat in _context_pats:
        if pat.search(text):
            hits += 1
            if hits >= CONTEXT_THRESHOLD:
                return "context_accumulation"

    # Meta-references
    m = _meta_reject_re.search(text)
    if m:
        return f"meta_reference:{m.group()[:40]}"

    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load all conversations from period and modern pairs
    all_convos = []
    for fname in ["period_pairs.jsonl", "modern_pairs.jsonl"]:
        path = os.path.join(args.input_dir, fname)
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    all_convos.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        print(f"Loaded {fname}")

    print(f"Total input conversations: {len(all_convos)}")

    # Process
    cleaned = []
    rejected = []
    reason_counts = {}
    phrase_strip_count = 0

    for msgs in all_convos:
        # 1. Unicode cleanup
        for m in msgs:
            m["content"] = clean_unicode(m["content"])

        # 2. Strip overused phrases from assistant turns
        changed = False
        for m in msgs:
            if m["role"] == "assistant":
                original = m["content"]
                m["content"] = strip_phrases(m["content"])
                if m["content"] != original:
                    changed = True
        if changed:
            phrase_strip_count += 1

        # 3. Check for empty assistant responses after stripping
        if any(not m["content"].strip() for m in msgs):
            rejected.append(("empty_after_strip", msgs))
            reason_counts["empty_after_strip"] = reason_counts.get("empty_after_strip", 0) + 1
            continue

        # 4. Anachronism filter
        reason = is_anachronistic(msgs)
        if reason:
            rejected.append((reason, msgs))
            bucket = reason.split(":")[0]
            reason_counts[bucket] = reason_counts.get(bucket, 0) + 1
            continue

        cleaned.append(msgs)

    print(f"\nCleaning results:")
    print(f"  Passed:   {len(cleaned)} ({100 * len(cleaned) / len(all_convos):.1f}%)")
    print(f"  Rejected: {len(rejected)} ({100 * len(rejected) / len(all_convos):.1f}%)")
    print(f"  Phrase-stripped: {phrase_strip_count} conversations modified")

    if reason_counts:
        print(f"\n  Rejection reasons:")
        for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
            print(f"    {reason}: {count}")

    # Verify "indeed" and "confess" rates after cleaning
    indeed_count = 0
    confess_count = 0
    for msgs in cleaned:
        asst = " ".join(m["content"].lower() for m in msgs if m["role"] == "assistant")
        if "indeed" in asst:
            indeed_count += 1
        if "i confess" in asst:
            confess_count += 1
    print(f"\n  Post-clean rates:")
    print(f"    'indeed': {indeed_count} ({100*indeed_count/max(len(cleaned),1):.1f}%)")
    print(f"    'i confess': {confess_count} ({100*confess_count/max(len(cleaned),1):.1f}%)")

    # Shuffle and split
    rng.shuffle(cleaned)
    n_val = max(1, int(len(cleaned) * args.val_fraction))
    val_set = cleaned[:n_val]
    train_set = cleaned[n_val:]

    # Write
    train_path = os.path.join(args.output_dir, "all_train.jsonl")
    val_path = os.path.join(args.output_dir, "all_val.jsonl")
    rejected_path = os.path.join(args.output_dir, "rejected.jsonl")

    for path, data in [(train_path, train_set), (val_path, val_set)]:
        with open(path, "w", encoding="utf-8") as f:
            for msgs in data:
                f.write(json.dumps(msgs, ensure_ascii=False) + "\n")

    with open(rejected_path, "w", encoding="utf-8") as f:
        for reason, msgs in rejected:
            f.write(json.dumps({"reason": reason, "messages": msgs}, ensure_ascii=False) + "\n")

    print(f"\nOutput:")
    print(f"  Train: {len(train_set)} -> {train_path}")
    print(f"  Val:   {len(val_set)} -> {val_path}")
    print(f"  Rejected: {len(rejected)} -> {rejected_path}")


if __name__ == "__main__":
    main()
