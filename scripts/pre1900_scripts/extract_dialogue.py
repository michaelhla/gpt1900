#!/usr/bin/env python3
"""
Extract real dialogue from the pre-1900 corpus parquet shards.

Finds attributed quoted speech in books/newspapers, segments consecutive
quotes into two-speaker conversations, and outputs user/assistant pairs
in the format expected by tasks/customjson.py.

Stages:
  A. OCR normalization (smart quotes, hyphenated breaks, double spaces)
  B. Quote extraction via regex (pre/post/mid-break attribution)
  C. Conversation segmentation (consecutive quotes, 2 speakers)
  D. Quality filters (length, OCR quality, dedup)
  E. Output to JSONL with train/val split

Usage:
    # Test on 1 shard
    python -m scripts.pre1900_scripts.extract_dialogue \
        --data-dir /opt/dlami/nvme/gpt1905_training/pre1900_data \
        --output-dir instruct_data/ \
        --max-shards 1

    # Full run
    python -m scripts.pre1900_scripts.extract_dialogue \
        --data-dir /opt/dlami/nvme/gpt1905_training/pre1900_data \
        --output-dir instruct_data/ \
        --workers 64
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import pyarrow.parquet as pq


# ============================================================================
# Stage A: OCR Normalization
# ============================================================================

# Smart/curly quotes → ASCII
_SMART_QUOTES = str.maketrans({
    "\u2018": "'", "\u2019": "'",  # single curly
    "\u201C": '"', "\u201D": '"',  # double curly
    "\u201E": '"', "\u201F": '"',  # low-9, high-reversed-9
    "\u00AB": '"', "\u00BB": '"',  # guillemets
    "\u2039": "'", "\u203A": "'",  # single guillemets
})

_HYPHEN_BREAK_RE = re.compile(r"(\w)-\n(\w)")
_DOUBLE_SPACE_RE = re.compile(r"  +")


def normalize_ocr(text: str) -> str:
    """Normalize OCR artifacts: smart quotes, hyphenated line breaks, double spaces."""
    text = text.translate(_SMART_QUOTES)
    text = _HYPHEN_BREAK_RE.sub(r"\1\2", text)
    text = _DOUBLE_SPACE_RE.sub(" ", text)
    return text


# ============================================================================
# Stage B: Quote Extraction
# ============================================================================

SPEECH_VERBS = (
    r"said|replied|answered|asked|cried|exclaimed|observed|remarked|"
    r"continued|added|responded|inquired|demanded|declared|murmured|"
    r"whispered|shouted|returned|rejoined|repeated|interrupted|insisted|"
    r"objected|explained|admitted|entreated|pronounced|announced"
)

# Speaker: proper names (Mr. Darcy, Mrs. Bennet, Dr. Watson, the Doctor),
# pronouns (he, she, I), or Title + Name
SPEAKER = (
    r"(?:"
    r"(?:Mr\.|Mrs\.|Miss|Dr\.|Sir|Lady|Lord|Captain|Colonel|General|"
    r"Major|Professor|Rev\.|Father|Mother|Uncle|Aunt|the)\s+"
    r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?"
    r"|[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]+)?"
    r"|[Hh]e|[Ss]he|I"
    r")"
)

# Pattern 1: Post-attribution — "quoted text," said Speaker
_POST_ATTR_RE = re.compile(
    r'"([^"]{10,}?)[,.!?]"\s*'
    rf"(?:{SPEECH_VERBS})\s+"
    rf"({SPEAKER})",
    re.DOTALL,
)

# Pattern 2: Pre-attribution — Speaker said, "quoted text"
_PRE_ATTR_RE = re.compile(
    rf"({SPEAKER})\s+"
    rf"(?:{SPEECH_VERBS}),?\s*"
    r'"([^"]{10,}?)[,.!?]"',
    re.DOTALL,
)

# Pattern 3: Mid-break — "first part," said Speaker, "second part"
_MID_BREAK_RE = re.compile(
    r'"([^"]{5,}?)[,.]"\s*'
    rf"(?:{SPEECH_VERBS})\s+"
    rf"({SPEAKER}),?\s*"
    r'"([^"]{5,}?)[,.!?]"',
    re.DOTALL,
)


def extract_quotes(text: str) -> list[tuple[str, str, int]]:
    """Extract (speaker, quote_text, char_position) from text.

    Returns quotes sorted by position in the original text.
    """
    quotes: list[tuple[str, str, int]] = []

    # Mid-break (check first — it's the most specific pattern)
    for m in _MID_BREAK_RE.finditer(text):
        speaker = m.group(2).strip()
        combined = m.group(1).strip() + " " + m.group(3).strip()
        quotes.append((speaker, combined, m.start()))

    # Post-attribution
    for m in _POST_ATTR_RE.finditer(text):
        quote_text = m.group(1).strip()
        speaker = m.group(2).strip()
        quotes.append((speaker, quote_text, m.start()))

    # Pre-attribution
    for m in _PRE_ATTR_RE.finditer(text):
        speaker = m.group(1).strip()
        quote_text = m.group(2).strip()
        quotes.append((speaker, quote_text, m.start()))

    # Deduplicate by position (overlapping patterns)
    if not quotes:
        return []

    quotes.sort(key=lambda x: x[2])

    deduped: list[tuple[str, str, int]] = [quotes[0]]
    for q in quotes[1:]:
        # Skip if position overlaps with previous (within 10 chars)
        if abs(q[2] - deduped[-1][2]) > 10:
            deduped.append(q)

    return deduped


# ============================================================================
# Stage B2: Unattributed Consecutive Quote Extraction
# ============================================================================

# Pattern for a standalone quoted paragraph/line (no attribution needed).
# Matches a quote that starts near a line boundary.
_STANDALONE_QUOTE_RE = re.compile(
    r'(?:^|\n)\s*"([^"]{20,}?)[,.!?;]"'
    r'\s*(?=\n|$)',
    re.MULTILINE,
)


def extract_unattributed_conversations(text: str) -> list[list[dict[str, str]]]:
    """Find runs of consecutive quoted paragraphs with no attribution.

    These are very common in 19th-century fiction — each new paragraph is a
    different speaker, with the reader expected to track who is who.

    Returns conversations directly as message lists (alternating user/assistant).
    """
    matches = list(_STANDALONE_QUOTE_RE.finditer(text))
    if len(matches) < 2:
        return []

    # Group into consecutive runs (quotes within 200 chars of each other)
    runs: list[list[re.Match]] = []
    current_run: list[re.Match] = [matches[0]]

    for m in matches[1:]:
        gap = m.start() - current_run[-1].end()
        if gap <= 200:
            current_run.append(m)
        else:
            if len(current_run) >= 2:
                runs.append(current_run)
            current_run = [m]

    if len(current_run) >= 2:
        runs.append(current_run)

    # Convert each run to alternating user/assistant
    conversations: list[list[dict[str, str]]] = []
    for run in runs:
        messages = []
        for i, m in enumerate(run):
            role = "user" if i % 2 == 0 else "assistant"
            messages.append({"role": role, "content": m.group(1).strip()})

        # Ensure ends with assistant
        if messages[-1]["role"] != "assistant":
            messages = messages[:-1]

        if len(messages) >= 2:
            conversations.append(messages)

    return conversations


# ============================================================================
# Stage C: Conversation Segmentation (for attributed quotes)
# ============================================================================

# Max chars between consecutive quotes to be considered part of same conversation
MAX_GAP_CHARS = 500


def _normalize_speaker(speaker: str) -> str:
    """Normalize speaker for comparison (lowercase pronouns, strip titles)."""
    s = speaker.strip()
    if s.lower() in ("he", "she", "i"):
        return s.lower()
    return s


def segment_conversations(
    quotes: list[tuple[str, str, int]],
) -> list[list[tuple[str, str]]]:
    """Group consecutive quotes into conversation segments with exactly 2 speakers.

    Returns list of conversations, each being [(speaker, text), ...].
    """
    if len(quotes) < 2:
        return []

    # Split into segments by gap
    segments: list[list[tuple[str, str, int]]] = []
    current: list[tuple[str, str, int]] = [quotes[0]]

    for q in quotes[1:]:
        gap = q[2] - current[-1][2]
        if gap <= MAX_GAP_CHARS:
            current.append(q)
        else:
            if len(current) >= 2:
                segments.append(current)
            current = [q]

    if len(current) >= 2:
        segments.append(current)

    # Filter to exactly 2 distinct speakers per segment
    conversations: list[list[tuple[str, str]]] = []

    for seg in segments:
        speakers_raw = [_normalize_speaker(s) for s, _, _ in seg]
        unique = list(dict.fromkeys(speakers_raw))  # preserve order

        if len(unique) == 2:
            # Direct 2-speaker conversation
            conv = [(s, t) for s, t, _ in seg]
            conversations.append(conv)
        elif len(unique) == 1 and unique[0] in ("he", "she"):
            # Single pronoun — could be alternating speakers, use alternation heuristic
            conv = [(s, t) for s, t, _ in seg]
            conversations.append(conv)
        # Skip segments with >2 speakers or single named speaker

    return conversations


def conversation_to_messages(
    conv: list[tuple[str, str]],
) -> list[dict[str, str]] | None:
    """Convert a segmented conversation to user/assistant message format.

    First speaker → user, second → assistant, alternating.
    Returns None if the conversation doesn't alternate properly.
    """
    if len(conv) < 2:
        return None

    speakers_norm = [_normalize_speaker(s) for s, _ in conv]
    unique = list(dict.fromkeys(speakers_norm))

    if len(unique) == 1:
        # Pronoun-only: alternate artificially
        messages = []
        for i, (_, text) in enumerate(conv):
            role = "user" if i % 2 == 0 else "assistant"
            messages.append({"role": role, "content": text})
        return messages

    # Map speakers to roles
    speaker_a, speaker_b = unique[0], unique[1]
    role_map = {speaker_a: "user", speaker_b: "assistant"}

    messages = []
    last_role = None
    for speaker_raw, text in conv:
        role = role_map[_normalize_speaker(speaker_raw)]
        if role == last_role:
            # Same speaker consecutive — merge into previous turn
            messages[-1]["content"] += " " + text
        else:
            messages.append({"role": role, "content": text})
            last_role = role

    # Must start with user
    if messages and messages[0]["role"] != "user":
        messages = messages[1:]

    # Must end with assistant
    if messages and messages[-1]["role"] != "assistant":
        messages = messages[:-1]

    # Must have at least 2 messages
    if len(messages) < 2:
        return None

    return messages


# ============================================================================
# Stage D: Quality Filters
# ============================================================================

_ALPHA_PUNCT_RE = re.compile(r"[a-zA-Z0-9\s.,;:!?'\"\-\(\)]")
_CHAPTER_RE = re.compile(
    r"^(?:CHAPTER|BOOK|PART|SECTION|VOL(?:UME)?\.?)\s+[IVXLCDM\d]+",
    re.IGNORECASE,
)
_PAGE_NUM_RE = re.compile(r"^\s*\d{1,4}\s*$")
_TOC_RE = re.compile(r"\.{3,}\s*\d+\s*$")
# OCR artifacts: "66" and "99" are common misreads of curly quotes
_OCR_QUOTE_ARTIFACT_RE = re.compile(r"(?:^|\s)(?:66|99|6 6|9 9)(?:\s|$)")


def passes_quality(messages: list[dict[str, str]]) -> bool:
    """Apply quality filters to a conversation."""
    for msg in messages:
        text = msg["content"]

        # Min 20 chars per turn
        if len(text) < 20:
            return False

        # Max 2000 chars per turn
        if len(text) > 2000:
            return False

        # OCR quote artifacts (66/99 = misread curly quotes)
        if _OCR_QUOTE_ARTIFACT_RE.search(text):
            return False

        # OCR quality: ≥85% standard chars
        if len(text) > 0:
            good = sum(1 for c in text if _ALPHA_PUNCT_RE.match(c))
            if good / len(text) < 0.85:
                return False

        # Excessive uppercase (>50%)
        alpha = [c for c in text if c.isalpha()]
        if alpha and sum(1 for c in alpha if c.isupper()) / len(alpha) > 0.50:
            return False

        # Chapter headings, page numbers, TOC artifacts
        if _CHAPTER_RE.search(text):
            return False
        if _PAGE_NUM_RE.match(text):
            return False
        if _TOC_RE.search(text):
            return False

        # Embedded page headers/footers (e.g. "77 THE IRON COUSIN.")
        if re.search(r"\d{1,3}\s+[A-Z]{2,}(?:\s+[A-Z]{2,}){1,5}\.?\s", text):
            return False

    return True


def content_hash(messages: list[dict[str, str]]) -> str:
    """Hash conversation content for deduplication."""
    combined = "||".join(m["content"] for m in messages)
    return hashlib.md5(combined.encode("utf-8")).hexdigest()


# ============================================================================
# Stage E: Process a single shard
# ============================================================================

def process_shard(shard_path: str) -> list[list[dict[str, str]]]:
    """Process one parquet shard and return list of conversations."""
    try:
        pf = pq.ParquetFile(shard_path)
    except Exception:
        return []

    all_conversations: list[list[dict[str, str]]] = []

    for rg_idx in range(pf.num_row_groups):
        try:
            table = pf.read_row_group(rg_idx, columns=["text"])
        except Exception:
            continue
        texts = table.column("text").to_pylist()
        del table

        for text in texts:
            if not text or len(text) < 200:
                continue

            # Stage A: normalize
            text = normalize_ocr(text)

            # Stage B1: extract attributed quotes
            quotes = extract_quotes(text)
            if len(quotes) >= 2:
                # Stage C: segment into conversations
                convs = segment_conversations(quotes)
                for conv in convs:
                    messages = conversation_to_messages(conv)
                    if messages is not None and passes_quality(messages):
                        all_conversations.append(messages)

            # Stage B2: extract unattributed consecutive quotes
            unattr_convs = extract_unattributed_conversations(text)
            for messages in unattr_convs:
                if passes_quality(messages):
                    all_conversations.append(messages)

    return all_conversations


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Extract real dialogue from pre-1900 corpus parquet shards",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir", type=Path, required=True,
        help="Directory containing shard_*.parquet files",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("instruct_data"),
        help="Output directory for dialogue JSONL files",
    )
    parser.add_argument(
        "--workers", type=int, default=64,
        help="Number of parallel workers",
    )
    parser.add_argument(
        "--max-shards", type=int, default=0,
        help="Max shards to process (0 = all)",
    )
    parser.add_argument(
        "--val-fraction", type=float, default=0.05,
        help="Fraction of conversations for validation split",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Discover shards
    shard_paths = sorted(args.data_dir.glob("shard_*.parquet"))
    if not shard_paths:
        # Try without shard_ prefix
        shard_paths = sorted(args.data_dir.glob("*.parquet"))

    if not shard_paths:
        print(f"ERROR: No parquet files found in {args.data_dir}")
        sys.exit(1)

    if args.max_shards > 0:
        shard_paths = shard_paths[: args.max_shards]

    print(f"Processing {len(shard_paths)} shards with {args.workers} workers")
    print(f"Output: {args.output_dir}")

    t0 = time.time()

    # Process shards in parallel
    shard_str_paths = [str(p) for p in shard_paths]

    n_workers = min(args.workers, len(shard_str_paths))
    if n_workers <= 1:
        # Single-process mode for debugging
        results = [process_shard(p) for p in shard_str_paths]
    else:
        with Pool(processes=n_workers) as pool:
            results = pool.map(process_shard, shard_str_paths)

    # Collect all conversations
    all_conversations: list[list[dict[str, str]]] = []
    for shard_convs in results:
        all_conversations.extend(shard_convs)

    print(f"\nExtracted {len(all_conversations)} raw conversations")

    # Deduplication
    seen_hashes: set[str] = set()
    deduped: list[list[dict[str, str]]] = []
    for conv in all_conversations:
        h = content_hash(conv)
        if h not in seen_hashes:
            seen_hashes.add(h)
            deduped.append(conv)

    n_dupes = len(all_conversations) - len(deduped)
    print(f"Deduplicated: {n_dupes} duplicates removed, {len(deduped)} unique")

    if not deduped:
        print("No conversations extracted. Try checking the data or patterns.")
        sys.exit(0)

    # Train/val split
    random.shuffle(deduped)
    n_val = max(1, int(len(deduped) * args.val_fraction))
    val_set = deduped[:n_val]
    train_set = deduped[n_val:]

    # Write outputs
    train_path = args.output_dir / "dialogue_pairs.jsonl"
    val_path = args.output_dir / "dialogue_val_pairs.jsonl"

    with open(train_path, "w", encoding="utf-8") as f:
        for conv in train_set:
            f.write(json.dumps(conv, ensure_ascii=False) + "\n")

    with open(val_path, "w", encoding="utf-8") as f:
        for conv in val_set:
            f.write(json.dumps(conv, ensure_ascii=False) + "\n")

    elapsed = time.time() - t0

    # Stats
    turn_counts = [len(c) for c in deduped]
    turn_lengths = [len(m["content"]) for c in deduped for m in c]

    print(f"\nResults:")
    print(f"  Train: {len(train_set)} conversations → {train_path}")
    print(f"  Val:   {len(val_set)} conversations → {val_path}")
    print(f"  Turns/conversation: min={min(turn_counts)}, "
          f"median={sorted(turn_counts)[len(turn_counts)//2]}, "
          f"max={max(turn_counts)}")
    print(f"  Chars/turn: min={min(turn_lengths)}, "
          f"median={sorted(turn_lengths)[len(turn_lengths)//2]}, "
          f"max={max(turn_lengths)}")
    print(f"  Elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
