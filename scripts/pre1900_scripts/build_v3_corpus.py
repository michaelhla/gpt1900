#!/usr/bin/env python3
"""
Build v3 training corpus by combining:
  - New generation data (explain, conversation, creative, question) at full weight
  - Old v2 data (history, prediction, opinion) heavily downsampled

Also applies text cleanup: replaces unicode dashes, filters repetitive phrases.

Usage:
    python -m scripts.pre1900_scripts.build_v3_corpus \
        --new-dir instruct_data/v3_generated/ \
        --old-dir instruct_data/v2_corpus/ \
        --output-dir instruct_data/v3_corpus/ \
        --old-keep-ratio 0.15
"""

import argparse
import json
import os
import random
import re


# Phrases that indicate prediction/opinion template leakage
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

OVERUSED_PHRASES = [
    "i confess",
]


def get_assistant_text(msgs):
    return " ".join(m["content"].lower() for m in msgs if m["role"] == "assistant")


def is_prediction_heavy(msgs):
    text = get_assistant_text(msgs)
    return sum(1 for p in PREDICTION_MARKERS if p in text) >= 2


def is_opinion_heavy(msgs):
    text = get_assistant_text(msgs)
    return sum(1 for p in OPINION_MARKERS if p in text) >= 1


def clean_text(text):
    """Replace unicode dashes and curly quotes with ASCII equivalents."""
    text = text.replace("\u2014", "--")  # em dash
    text = text.replace("\u2013", "--")  # en dash
    text = text.replace("\u2018", "'")   # left single quote
    text = text.replace("\u2019", "'")   # right single quote
    text = text.replace("\u201c", '"')   # left double quote
    text = text.replace("\u201d", '"')   # right double quote
    return text


def clean_conversation(msgs):
    for m in msgs:
        m["content"] = clean_text(m["content"])
    return msgs


def load_jsonl(path):
    conversations = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                msgs = json.loads(line)
                conversations.append(msgs)
            except json.JSONDecodeError:
                continue
    return conversations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--new-dir", type=str, required=True,
                        help="Directory with new generation data (period_pairs.jsonl, modern_pairs.jsonl)")
    parser.add_argument("--old-dir", type=str, required=True,
                        help="Directory with old v2 data (all_train.jsonl)")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for v3 corpus")
    parser.add_argument("--old-keep-ratio", type=float, default=0.15,
                        help="Fraction of old prediction/opinion examples to keep (default: 0.15)")
    parser.add_argument("--old-neutral-keep-ratio", type=float, default=0.5,
                        help="Fraction of old neutral (non-prediction/opinion) examples to keep (default: 0.5)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load new data
    new_convos = []
    for fname in ["period_pairs.jsonl", "modern_pairs.jsonl", "all_filtered_pairs.jsonl"]:
        path = os.path.join(args.new_dir, fname)
        if os.path.exists(path):
            loaded = load_jsonl(path)
            print(f"New data: {fname}: {len(loaded)} conversations")
            new_convos.extend(loaded)

    # Load old data
    old_path = os.path.join(args.old_dir, "all_train.jsonl")
    old_convos = load_jsonl(old_path)
    print(f"Old data: {len(old_convos)} conversations")

    # Classify and downsample old data
    old_kept = []
    old_pred_total = 0
    old_opinion_total = 0
    old_neutral_total = 0
    old_pred_kept = 0
    old_opinion_kept = 0
    old_neutral_kept = 0

    for msgs in old_convos:
        if is_prediction_heavy(msgs):
            old_pred_total += 1
            if rng.random() < args.old_keep_ratio:
                old_kept.append(msgs)
                old_pred_kept += 1
        elif is_opinion_heavy(msgs):
            old_opinion_total += 1
            if rng.random() < args.old_keep_ratio:
                old_kept.append(msgs)
                old_opinion_kept += 1
        else:
            old_neutral_total += 1
            if rng.random() < args.old_neutral_keep_ratio:
                old_kept.append(msgs)
                old_neutral_kept += 1

    print(f"\nOld data breakdown:")
    print(f"  Prediction-heavy: {old_pred_kept}/{old_pred_total} kept")
    print(f"  Opinion-heavy:    {old_opinion_kept}/{old_opinion_total} kept")
    print(f"  Neutral:          {old_neutral_kept}/{old_neutral_total} kept")
    print(f"  Total kept:       {len(old_kept)}")

    # Combine and clean
    all_convos = [clean_conversation(msgs) for msgs in new_convos + old_kept]
    rng.shuffle(all_convos)

    # Train/val split
    n_val = max(1, int(len(all_convos) * 0.05))
    val_set = all_convos[:n_val]
    train_set = all_convos[n_val:]

    # Write
    train_path = os.path.join(args.output_dir, "all_train.jsonl")
    val_path = os.path.join(args.output_dir, "all_val.jsonl")

    with open(train_path, "w", encoding="utf-8") as f:
        for msgs in train_set:
            f.write(json.dumps(msgs, ensure_ascii=False) + "\n")

    with open(val_path, "w", encoding="utf-8") as f:
        for msgs in val_set:
            f.write(json.dumps(msgs, ensure_ascii=False) + "\n")

    # Stats
    confess_count = sum(1 for msgs in train_set if "confess" in get_assistant_text(msgs))

    print(f"\nFinal dataset:")
    print(f"  Train: {len(train_set)} -> {train_path}")
    print(f"  Val:   {len(val_set)} -> {val_path}")
    print(f"  'confess' rate: {100*confess_count/len(train_set):.1f}%")


if __name__ == "__main__":
    main()
