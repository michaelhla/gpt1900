#!/usr/bin/env python3
"""
Post-process modern-style user prompts to make them more realistic:
  - Randomly drop capitalization
  - Remove/reduce punctuation
  - Add occasional typos
  - Shorten some prompts
  - Make phrasing more casual

Only modifies user messages in conversations that already have modern-style
prompts (detected by casual markers). Leaves period-style untouched.

Usage:
    python -m scripts.pre1900_scripts.casualize_prompts \
        --input instruct_data/v3_corpus/all_train.jsonl \
        --output instruct_data/v3_corpus/all_train.jsonl
"""

import argparse
import json
import random
import re


# Common typos: swap adjacent keys, double letters, missing letters
TYPO_SWAPS = {
    'the': ['teh', 'hte', 'th'],
    'that': ['taht', 'tht'],
    'what': ['waht', 'wht'],
    'with': ['wiht', 'wth'],
    'about': ['aobut', 'abut', 'aboit'],
    'have': ['ahve', 'hve'],
    'this': ['thsi', 'tihs'],
    'from': ['form', 'fomr'],
    'they': ['tehy', 'thye'],
    'were': ['wre', 'weer'],
    'been': ['ben', 'bene'],
    'some': ['soem', 'sone'],
    'would': ['woudl', 'wuold'],
    'could': ['coudl', 'cuold'],
    'their': ['thier', 'tehir'],
    'there': ['thre', 'tehre'],
    'think': ['thnk', 'thikn'],
    'know': ['knwo', 'konw'],
    'really': ['realy', 'relly'],
    'because': ['becuase', 'becasue', 'bc'],
    'people': ['poeple', 'ppl'],
    'something': ['somethign', 'somethin'],
    'actually': ['acutally', 'actualy'],
    'different': ['diffrent', 'diferent'],
    'interesting': ['interesing', 'intresting'],
    'question': ['quesiton', 'questoin'],
}


def is_modern_prompt(user_msg):
    """Detect if a user message is modern-style (vs period-style)."""
    lower = user_msg.lower()
    casual_markers = [
        "hey", "so ", "like,", "like ", "gonna", "kinda", "pretty ",
        "i'm ", "i've ", "i was ", "what's ", "how's ", "that's ",
        "don't ", "can't ", "won't ", "isn't ", "didn't ",
        "cool", "stuff", "thing is", "right?", "you know",
    ]
    formal_markers = [
        "esteemed", "good sir", "pray tell", "i beseech",
        "i have lately", "i have often", "permit me",
        "most learned", "dear sir", "venerable",
    ]
    casual_score = sum(1 for m in casual_markers if m in lower)
    formal_score = sum(1 for m in formal_markers if m in lower)
    return casual_score > formal_score and casual_score >= 1


def add_typos(text, rng, rate=0.15):
    """Randomly introduce typos into ~rate of eligible words."""
    words = text.split()
    result = []
    for word in words:
        clean = word.strip(".,!?;:\"'()[]")
        if clean.lower() in TYPO_SWAPS and rng.random() < rate:
            typo = rng.choice(TYPO_SWAPS[clean.lower()])
            # Preserve any trailing punctuation
            suffix = word[len(clean):]
            result.append(typo + suffix)
        else:
            result.append(word)
    return " ".join(result)


def drop_capitalization(text, rng):
    """Lowercase the entire message."""
    return text.lower()


def drop_punctuation(text, rng):
    """Remove or reduce punctuation, keeping ? marks."""
    # Remove periods at end of sentences (but keep ?)
    text = re.sub(r'\.\s+([A-Za-z])', lambda m: ' ' + m.group(1).lower(), text)
    # Remove trailing period
    text = text.rstrip('.')
    # Randomly drop commas
    if rng.random() < 0.5:
        text = re.sub(r',\s+', lambda m: ' ' if rng.random() < 0.6 else m.group(), text)
    return text


def shorten_prompt(text, rng):
    """Truncate long prompts to just the core question."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    if len(sentences) <= 2:
        return text

    # Keep first 1-2 sentences and last sentence (usually the question)
    if len(sentences) >= 4:
        n_keep = rng.randint(1, 2)
        shortened = sentences[:n_keep] + [sentences[-1]]
        return " ".join(shortened)
    elif len(sentences) == 3:
        return " ".join([sentences[0], sentences[-1]])
    return text


def casualize_message(text, rng):
    """Apply random casual transformations to a user message."""
    transforms = []

    # Always lowercase (70% chance)
    if rng.random() < 0.70:
        transforms.append("lowercase")

    # Drop punctuation (50% chance)
    if rng.random() < 0.50:
        transforms.append("drop_punct")

    # Add typos (30% chance, subtle)
    if rng.random() < 0.30:
        transforms.append("typos")

    # Shorten long prompts (25% chance, only if > 200 chars)
    if len(text) > 200 and rng.random() < 0.25:
        transforms.append("shorten")

    # Apply in order
    if "shorten" in transforms:
        text = shorten_prompt(text, rng)
    if "drop_punct" in transforms:
        text = drop_punctuation(text, rng)
    if "lowercase" in transforms:
        text = drop_capitalization(text, rng)
    if "typos" in transforms:
        text = add_typos(text, rng, rate=0.12)

    return text.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--ratio", type=float, default=0.6,
                        help="Fraction of modern prompts to casualize (default: 0.6)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    conversations = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    conversations.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    print(f"Loaded {len(conversations)} conversations")

    modified = 0
    modern_count = 0

    for msgs in conversations:
        # Check if first user message is modern-style
        user_msgs = [m for m in msgs if m["role"] == "user"]
        if not user_msgs:
            continue

        if not is_modern_prompt(user_msgs[0]["content"]):
            continue

        modern_count += 1

        if rng.random() > args.ratio:
            continue

        # Casualize all user messages in this conversation
        for m in msgs:
            if m["role"] == "user":
                m["content"] = casualize_message(m["content"], rng)

        modified += 1

    print(f"Modern-style detected: {modern_count}")
    print(f"Casualized: {modified} ({100*modified/max(modern_count,1):.0f}%)")

    # Sample some before/after
    with open(args.output, "w", encoding="utf-8") as f:
        for msgs in conversations:
            f.write(json.dumps(msgs, ensure_ascii=False) + "\n")

    print(f"Written to {args.output}")

    # Show examples
    print(f"\nSample casualized prompts:")
    shown = 0
    for msgs in conversations:
        user_msg = msgs[0]["content"]
        if any(c in user_msg for c in ['teh', 'hte', 'wht']) or (user_msg == user_msg.lower() and '?' in user_msg):
            print(f"  -> {user_msg[:120]}")
            shown += 1
            if shown >= 10:
                break


if __name__ == "__main__":
    main()
