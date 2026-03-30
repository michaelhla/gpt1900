#!/usr/bin/env python3
"""
Build final v3 training corpus by combining:
  - New v3 cleaned data (explain, conversation, creative, question) at full weight
  - Old v2 data: prediction/opinion heavily downsampled, neutral at 50%
  - Leaky rows (post-1900 knowledge) excluded from both

Also applies: unicode cleanup (em dashes -> periods, curly quotes -> ASCII),
"Indeed," stripping, "I confess" stripping.

Usage:
    python -m scripts.pre1900_scripts.build_v3_corpus \
        --new-dir instruct_data/v3_cleaned/ \
        --old-dir instruct_data/v2_corpus/ \
        --leaky-rows instruct_data/leaky_rows_all.json \
        --output-dir instruct_data/v3_corpus/
"""

import argparse
import json
import os
import random
import re


# ---------------------------------------------------------------------------
# Phrase detection
# ---------------------------------------------------------------------------

PREDICTION_MARKERS = [
    "coming century shall",
    "mark my words",
    "within fifty years",
    "beyond our present imagining",
    "annihilated distance",
    "i predict that by",
    "i foresee",
    "the twentieth century shall",
]

OPINION_MARKERS = [
    "i hold it to be",
    "i am firmly convinced",
    "i am quite convinced",
    "i am persuaded",
]

# Phrases to strip from assistant text
STRIP_PATTERNS = [
    (r"\bI confess(?:\s+that)?\b", ""),
    (r"\bIndeed,\s*", ""),  # sentence-initial "Indeed," only
]

STRIP_COMPILED = [(re.compile(p, re.IGNORECASE), r) for p, r in STRIP_PATTERNS]


def get_assistant_text(msgs):
    return " ".join(m["content"].lower() for m in msgs if m["role"] == "assistant")


def is_prediction_heavy(msgs):
    text = get_assistant_text(msgs)
    return sum(1 for p in PREDICTION_MARKERS if p in text) >= 2


def is_opinion_heavy(msgs):
    text = get_assistant_text(msgs)
    return sum(1 for p in OPINION_MARKERS if p in text) >= 1


# ---------------------------------------------------------------------------
# Text cleanup
# ---------------------------------------------------------------------------

def clean_text(text):
    """Unicode cleanup + phrase stripping."""
    # Em/en dashes -> periods
    text = text.replace("\u2014", ".")
    text = text.replace("\u2013", ".")
    # Curly quotes -> ASCII
    text = text.replace("\u2018", "'")
    text = text.replace("\u2019", "'")
    text = text.replace("\u201c", '"')
    text = text.replace("\u201d", '"')
    return text


def strip_phrases(text):
    for pat, repl in STRIP_COMPILED:
        text = pat.sub(repl, text)
    text = re.sub(r"  +", " ", text)
    text = re.sub(r"^\s*[,;]\s*", "", text)
    text = re.sub(r"\s+([,;.!?])", r"\1", text)
    return text.strip()


def clean_conversation(msgs):
    for m in msgs:
        m["content"] = clean_text(m["content"])
        if m["role"] == "assistant":
            m["content"] = strip_phrases(m["content"])
    return msgs


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_jsonl(path):
    conversations = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                conversations.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return conversations


def load_jsonl_with_indices(path):
    """Load JSONL returning list of (index, conversation)."""
    conversations = []
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                conversations.append((idx, json.loads(line)))
            except json.JSONDecodeError:
                continue
    return conversations


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--new-dir", type=str, required=True,
                        help="Directory with cleaned v3 data (all_train.jsonl, all_val.jsonl)")
    parser.add_argument("--old-dir", type=str, required=True,
                        help="Directory with old v2 data (modern_pairs.jsonl, period_pairs.jsonl)")
    parser.add_argument("--leaky-rows", type=str, default=None,
                        help="JSON file mapping filenames to lists of leaky row indices")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for final corpus")
    parser.add_argument("--old-keep-ratio", type=float, default=0.15,
                        help="Fraction of old prediction/opinion examples to keep (default: 0.15)")
    parser.add_argument("--old-neutral-keep-ratio", type=float, default=0.5,
                        help="Fraction of old neutral examples to keep (default: 0.5)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load leaky row indices
    leaky = {}
    if args.leaky_rows and os.path.exists(args.leaky_rows):
        with open(args.leaky_rows) as f:
            raw = json.load(f)
        for key, indices in raw.items():
            leaky[key] = set(indices)
        print(f"Loaded leaky rows: {sum(len(v) for v in leaky.values())} total")

    # -----------------------------------------------------------------------
    # Load new v3 cleaned data (already cleaned by clean_sft_data.py)
    # -----------------------------------------------------------------------
    new_convos = []
    v3_leaky = leaky.get("v3_cleaned/all_train", set())

    new_train_path = os.path.join(args.new_dir, "all_train.jsonl")
    if os.path.exists(new_train_path):
        for idx, msgs in load_jsonl_with_indices(new_train_path):
            if idx not in v3_leaky:
                new_convos.append(msgs)
        print(f"New v3 train: {len(new_convos)} (excluded {len(v3_leaky)} leaky)")

    new_val_path = os.path.join(args.new_dir, "all_val.jsonl")
    new_val = load_jsonl(new_val_path) if os.path.exists(new_val_path) else []
    print(f"New v3 val: {len(new_val)}")

    # -----------------------------------------------------------------------
    # Load and filter old v2 data
    # -----------------------------------------------------------------------
    old_kept = []
    stats = {
        "pred_total": 0, "pred_kept": 0,
        "opin_total": 0, "opin_kept": 0,
        "neutral_total": 0, "neutral_kept": 0,
        "leaky_excluded": 0,
    }

    for fname in ["modern_pairs.jsonl", "period_pairs.jsonl"]:
        path = os.path.join(args.old_dir, fname)
        if not os.path.exists(path):
            continue

        leaky_set = leaky.get(f"v2/{fname}", set())
        file_total = 0
        file_leaky = 0

        for idx, msgs in load_jsonl_with_indices(path):
            file_total += 1

            # Exclude leaky rows
            if idx in leaky_set:
                file_leaky += 1
                stats["leaky_excluded"] += 1
                continue

            # Classify and downsample
            if is_prediction_heavy(msgs):
                stats["pred_total"] += 1
                if rng.random() < args.old_keep_ratio:
                    old_kept.append(msgs)
                    stats["pred_kept"] += 1
            elif is_opinion_heavy(msgs):
                stats["opin_total"] += 1
                if rng.random() < args.old_keep_ratio:
                    old_kept.append(msgs)
                    stats["opin_kept"] += 1
            else:
                stats["neutral_total"] += 1
                if rng.random() < args.old_neutral_keep_ratio:
                    old_kept.append(msgs)
                    stats["neutral_kept"] += 1

        print(f"Old v2 {fname}: {file_total} total, {file_leaky} leaky excluded")

    print(f"\nOld v2 breakdown (after leaky exclusion):")
    print(f"  Leaky excluded:   {stats['leaky_excluded']}")
    print(f"  Prediction-heavy: {stats['pred_kept']}/{stats['pred_total']} kept ({args.old_keep_ratio:.0%})")
    print(f"  Opinion-heavy:    {stats['opin_kept']}/{stats['opin_total']} kept ({args.old_keep_ratio:.0%})")
    print(f"  Neutral:          {stats['neutral_kept']}/{stats['neutral_total']} kept ({args.old_neutral_keep_ratio:.0%})")
    print(f"  Total kept:       {len(old_kept)}")

    # -----------------------------------------------------------------------
    # Combine, clean, and split
    # -----------------------------------------------------------------------
    # Clean old v2 data (v3 is already cleaned)
    old_cleaned = [clean_conversation(msgs) for msgs in old_kept]

    all_convos = new_convos + old_cleaned
    rng.shuffle(all_convos)

    # Train/val split (add new_val to val set)
    n_val = max(1, int(len(all_convos) * 0.05))
    val_set = all_convos[:n_val] + new_val
    train_set = all_convos[n_val:]

    # -----------------------------------------------------------------------
    # Write
    # -----------------------------------------------------------------------
    train_path = os.path.join(args.output_dir, "all_train.jsonl")
    val_path = os.path.join(args.output_dir, "all_val.jsonl")

    with open(train_path, "w", encoding="utf-8") as f:
        for msgs in train_set:
            f.write(json.dumps(msgs, ensure_ascii=False) + "\n")

    with open(val_path, "w", encoding="utf-8") as f:
        for msgs in val_set:
            f.write(json.dumps(msgs, ensure_ascii=False) + "\n")

    # -----------------------------------------------------------------------
    # Stats
    # -----------------------------------------------------------------------
    confess_count = sum(1 for msgs in train_set if "confess" in get_assistant_text(msgs))
    indeed_count = sum(1 for msgs in train_set if "indeed" in get_assistant_text(msgs))

    print(f"\n{'='*50}")
    print(f"FINAL DATASET")
    print(f"{'='*50}")
    print(f"  New v3 data:   {len(new_convos)}")
    print(f"  Old v2 kept:   {len(old_kept)}")
    print(f"  Total:         {len(all_convos)}")
    print(f"  Train: {len(train_set)} -> {train_path}")
    print(f"  Val:   {len(val_set)} -> {val_path}")
    print(f"\n  Quality checks:")
    print(f"    'confess' rate: {100*confess_count/len(train_set):.1f}%")
    print(f"    'indeed' rate:  {100*indeed_count/len(train_set):.1f}%")


if __name__ == "__main__":
    main()
