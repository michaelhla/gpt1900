#!/usr/bin/env python3
"""
Generate contradiction-resolution RL training problems from physics book insights.

Two-phase pipeline:
  Phase 1 — Reuse or extract insights from physics books
  Phase 2 — Reframe insights as contradiction-resolution problems:
            present observations + classical assumptions that conflict,
            ask "which assumption fails and what replaces it?"

The output trains the RL loop to resolve contradictions — the skill tested
by the physics eval (EVAL.json).

Usage:
    # Use existing insights, generate contradiction problems
    python -m scripts.pre1900_scripts.generate_contradiction_problems \
        --insights-file instruct_data/rl_problems/rl_insights_raw.jsonl \
        --output-dir instruct_data/contradiction_problems \
        --run phase2

    # Extract fresh insights + generate problems
    python -m scripts.pre1900_scripts.generate_contradiction_problems \
        --books-dir data/physics_books \
        --existing-insights instruct_data/insights_raw.jsonl \
        --output-dir instruct_data/contradiction_problems \
        --run all
"""

import json
import asyncio
import argparse
import time
import random
from pathlib import Path
from difflib import SequenceMatcher

import anthropic

from scripts.pre1900_scripts.collect_physics_books import CURATED_WORKS
from scripts.pre1900_scripts.constants import REASONING_SYSTEM_PROMPT
from scripts.pre1900_scripts.generate_rl_problems import (
    WORK_BY_FILENAME,
    get_work_meta,
    get_era_description,
    estimate_tokens,
    chunk_text,
    parse_json_response,
    deduplicate_insights,
    validate_insight,
    call_api,
    INSIGHT_EXTRACTION_PROMPT,
)


# ---------------------------------------------------------------------------
# EVAL topics to exclude (prevent overfitting to eval set)
# ---------------------------------------------------------------------------

EVAL_EXCLUSION_KEYWORDS = [
    # UV catastrophe / blackbody
    "blackbody", "black body", "black-body", "ultraviolet catastrophe",
    "uv catastrophe", "cavity radiation", "spectral energy density",
    "rayleigh-jeans", "equipartition",
    # Photoelectric effect
    "photoelectric", "photo-electric", "electrons ejected by light",
    # Special relativity thought experiments
    "chasing a light beam", "frozen light", "light wave stationary",
    "approaching the speed of light", "speed limit",
    "train and lightning", "train lightning", "simultaneity",
    "michelson-morley", "michelson morley", "aether wind",
    "interferometer",
    # General relativity thought experiments
    "accelerating elevator", "elevator light", "elevator in space",
    "free fall equivalence", "equivalence principle",
    "falling freely", "weightlessness in free fall",
]


def is_eval_topic(insight: dict) -> bool:
    """Check if an insight overlaps with one of the 8 EVAL topics."""
    text = " ".join([
        insight.get("setup", ""),
        insight.get("insight", ""),
        insight.get("excerpt", ""),
        insight.get("domain", ""),
    ]).lower()
    return any(kw in text for kw in EVAL_EXCLUSION_KEYWORDS)


# ---------------------------------------------------------------------------
# Phase 2 prompt: contradiction-resolution framing
# ---------------------------------------------------------------------------

CONTRADICTION_PROMPT = """\
You are generating a CONTRADICTION-RESOLUTION problem for a language model that \
thinks like a {era_description}.

Given the following scientific insight from "{title}" by {author} ({year}), \
reframe it as a contradiction between observations and classical assumptions.

The problem must have this structure:
1. **Observations**: Numbered list of experimental facts or well-established observations \
(written in period-appropriate language, no post-1900 concepts)
2. **Classical assumptions**: Numbered list of plausible assumptions that a scientist of \
the era would hold (at least one of which is wrong or incomplete)
3. **Contradiction**: A brief statement showing how the assumptions + observations lead \
to a prediction that conflicts with what is actually observed
4. **Question**: Ask "Which assumption is most likely wrong? How can we reconcile \
[the contradiction]?"

The gold_answer must:
- Identify which specific numbered assumption fails
- Explain WHY it fails
- Propose what replaces it (the correct physical principle)
- Show how the replacement resolves the contradiction

CRITICAL CONSTRAINTS:
- Use ONLY knowledge available before 1900. No post-1900 concepts or terminology.
- Do NOT include equations or mathematical symbols. Frame everything conceptually.
- The problem must be self-contained — do not reference "the text" or "the passage".
- The observations must be real and scientifically correct.
- The contradiction must be genuine — not a strawman.
- Write as if you truly exist in {year} with no knowledge of the future.

SETUP (the evidence/constraints):
{setup}

INSIGHT (the conclusion — use this to construct the gold_answer):
{insight}

RELEVANT EXCERPT FROM THE TEXT:
{excerpt}

DOMAIN: {domain}

Return a JSON object with exactly two fields:
- "prompt": the full contradiction-resolution problem (string containing observations, \
assumptions, contradiction, and question)
- "gold_answer": the answer identifying which assumption fails and what replaces it (string)

Return ONLY valid JSON."""


# ---------------------------------------------------------------------------
# Validation for contradiction problems
# ---------------------------------------------------------------------------

def validate_contradiction_problem(result: dict) -> bool:
    """Validate that a generated problem has the required structure."""
    if not isinstance(result, dict):
        return False
    prompt = result.get("prompt", "")
    gold_answer = result.get("gold_answer", "")
    if not isinstance(prompt, str) or len(prompt) < 100:
        return False
    if not isinstance(gold_answer, str) or len(gold_answer) < 50:
        return False
    # Check for key structural elements
    prompt_lower = prompt.lower()
    has_observations = any(kw in prompt_lower for kw in ["observation", "observed", "experimental"])
    has_assumptions = "assumption" in prompt_lower
    has_question = "which assumption" in prompt_lower or "reconcile" in prompt_lower or "?" in prompt
    return has_observations and has_assumptions and has_question


# ---------------------------------------------------------------------------
# Phase 1: Extract insights (reuse from generate_rl_problems.py)
# ---------------------------------------------------------------------------

def identify_unused_books(books_dir: Path, existing_insights_path: Path | None) -> list[str]:
    """Identify books that have no insights in the existing insights file."""
    all_books = sorted([f.name for f in books_dir.glob("*.txt")])
    books_with_insights = set()
    if existing_insights_path and existing_insights_path.exists():
        with open(existing_insights_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    books_with_insights.add(record.get("book_filename", ""))
    unused = [b for b in all_books if b not in books_with_insights]
    return unused


async def extract_insights_from_books(
    client: anthropic.AsyncAnthropic,
    books_dir: Path,
    unused_books: list[str],
    model: str,
    semaphore: asyncio.Semaphore,
    output_path: Path,
    chunk_tokens: int,
    resume: bool,
):
    """Extract insights from unused books by chunking and sending to Claude."""
    already_done = set()
    if resume and output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        key = (record.get("book_filename", ""), record.get("chunk_idx", -1))
                        already_done.add(key)
                    except json.JSONDecodeError:
                        pass
        print(f"  Resume: {len(already_done)} chunks already processed")

    all_chunks = []
    for filename in unused_books:
        book_path = books_dir / filename
        if not book_path.exists():
            print(f"  [{filename}] File not found, skipping")
            continue
        text = book_path.read_text(encoding="utf-8")
        if estimate_tokens(text) < 100:
            print(f"  [{filename}] Too short, skipping")
            continue
        chunks = chunk_text(text, target_tokens=chunk_tokens)
        for chunk_idx, chunk in enumerate(chunks):
            if (filename, chunk_idx) not in already_done:
                all_chunks.append((filename, chunk_idx, chunk, len(chunks)))

    print(f"Phase 1: {len(all_chunks)} chunks to process from "
          f"{len(unused_books)} books ({len(already_done)} already done)")

    if not all_chunks:
        return

    mode = "a" if resume else "w"
    t0 = time.time()
    success = 0
    failed = 0
    total_insights = 0

    batch_size = semaphore._value * 2
    with open(output_path, mode, encoding="utf-8") as f:
        for batch_start in range(0, len(all_chunks), batch_size):
            batch = all_chunks[batch_start:batch_start + batch_size]
            tasks = [
                _extract_chunk_insights(client, fn, ci, tc, chunk, model, semaphore)
                for fn, ci, chunk, tc in batch
            ]
            results = await asyncio.gather(*tasks)

            for (filename, chunk_idx, _, total_chunks), insights in zip(batch, results):
                if insights is None:
                    failed += 1
                    continue
                meta = get_work_meta(filename)
                for idx, insight in enumerate(insights):
                    record = {
                        "book_filename": filename,
                        "author": meta["author"],
                        "title": meta["title"],
                        "year": meta["year"],
                        "chunk_idx": chunk_idx,
                        "insight_idx": idx,
                        **insight,
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    total_insights += 1
                success += 1

            f.flush()
            total_done = batch_start + len(batch)
            elapsed = time.time() - t0
            rate = total_done / elapsed if elapsed > 0 else 0
            print(f"  Progress: {total_done}/{len(all_chunks)} ({rate:.1f}/s) | "
                  f"chunks_ok={success} failed={failed} insights={total_insights}")

    elapsed = time.time() - t0
    print(f"\nPhase 1 done in {elapsed:.0f}s: {success} chunks, "
          f"{total_insights} insights, {failed} failed")

    # Deduplicate per book
    print("Deduplicating insights...")
    _deduplicate_insights_file(output_path)


async def _extract_chunk_insights(
    client: anthropic.AsyncAnthropic,
    filename: str,
    chunk_idx: int,
    total_chunks: int,
    chunk: str,
    model: str,
    semaphore: asyncio.Semaphore,
) -> list[dict] | None:
    meta = get_work_meta(filename)
    label = f"extract:{filename}:chunk{chunk_idx}/{total_chunks}"
    prompt = INSIGHT_EXTRACTION_PROMPT.format(
        title=meta["title"], author=meta["author"], year=meta["year"],
    )
    messages = [{"role": "user", "content": [
        {"type": "text", "text": prompt, "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": f"--- SECTION (chunk {chunk_idx + 1}/{total_chunks}) ---\n\n{chunk}"},
    ]}]
    response_text = await call_api(
        client, model, messages, max_tokens=8192, semaphore=semaphore, label=label,
    )
    if not response_text:
        return None
    data = parse_json_response(response_text)
    if not isinstance(data, dict):
        print(f"  [{label}] Failed to parse JSON")
        return None
    insights = data.get("insights", [])
    if not isinstance(insights, list):
        return None
    valid = [i for i in insights if validate_insight(i)]
    token_est = estimate_tokens(chunk)
    print(f"  [{label}] ~{token_est:,}tok -> {len(valid)}/{len(insights)} valid insights")
    return valid


def _deduplicate_insights_file(path: Path):
    by_book: dict[str, list[dict]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                fn = record["book_filename"]
                if fn not in by_book:
                    by_book[fn] = []
                by_book[fn].append(record)

    total_before = sum(len(v) for v in by_book.values())
    total_after = 0
    with open(path, "w", encoding="utf-8") as f:
        for fn in sorted(by_book.keys()):
            insights = by_book[fn]
            deduped = deduplicate_insights(insights)
            total_after += len(deduped)
            for record in deduped:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            if len(insights) != len(deduped):
                print(f"  [{fn}] {len(insights)} -> {len(deduped)} after dedup")
    print(f"  Dedup: {total_before} -> {total_after} insights")


# ---------------------------------------------------------------------------
# Phase 2: Generate contradiction-resolution problems from insights
# ---------------------------------------------------------------------------

async def generate_contradiction_problems(
    client: anthropic.AsyncAnthropic,
    insights: list[dict],
    model: str,
    semaphore: asyncio.Semaphore,
    output_path: Path,
    resume: bool,
):
    """Generate contradiction-resolution problems from extracted insights."""
    # Filter out EVAL topics
    filtered = [ins for ins in insights if not is_eval_topic(ins)]
    excluded = len(insights) - len(filtered)
    print(f"Filtered out {excluded} insights overlapping with EVAL topics")
    print(f"Remaining: {len(filtered)} insights")

    # Resume state
    already_done = set()
    if resume and output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        already_done.add(record.get("_source_idx", -1))
                    except json.JSONDecodeError:
                        pass
        print(f"  Resume: {len(already_done)} problems already generated")

    to_process = [(i, ins) for i, ins in enumerate(filtered) if i not in already_done]
    print(f"Phase 2: {len(to_process)} contradiction problems to generate "
          f"({len(already_done)} already done)")

    if not to_process:
        return

    mode = "a" if resume else "w"
    t0 = time.time()
    success = 0
    failed = 0

    batch_size = semaphore._value * 4
    with open(output_path, mode, encoding="utf-8") as f:
        for batch_start in range(0, len(to_process), batch_size):
            batch = to_process[batch_start:batch_start + batch_size]
            tasks = [
                _generate_single_contradiction(client, idx, insight, model, semaphore)
                for idx, insight in batch
            ]
            results = await asyncio.gather(*tasks)

            for (idx, insight), result in zip(batch, results):
                if result is None:
                    failed += 1
                    continue
                if not validate_contradiction_problem(result):
                    failed += 1
                    continue

                record = {
                    "_source_idx": idx,
                    "prompt": result["prompt"],
                    "gold_answer": result["gold_answer"],
                    "book_filename": insight.get("book_filename", ""),
                    "author": insight.get("author", ""),
                    "title": insight.get("title", ""),
                    "year": insight.get("year", ""),
                    "domain": insight.get("domain", ""),
                    "difficulty": insight.get("difficulty", ""),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                success += 1

            f.flush()
            total_done = batch_start + len(batch)
            elapsed = time.time() - t0
            rate = total_done / elapsed if elapsed > 0 else 0
            print(f"  Progress: {total_done}/{len(to_process)} ({rate:.1f}/s) | "
                  f"success={success} failed={failed}")

    elapsed = time.time() - t0
    print(f"\nPhase 2 done in {elapsed:.0f}s: {success} problems generated, {failed} failed")


async def _generate_single_contradiction(
    client: anthropic.AsyncAnthropic,
    idx: int,
    insight: dict,
    model: str,
    semaphore: asyncio.Semaphore,
) -> dict | None:
    """Generate a single contradiction-resolution problem from an insight."""
    prompt = CONTRADICTION_PROMPT.format(
        era_description=get_era_description(insight.get("year")),
        title=insight.get("title", ""),
        author=insight.get("author", ""),
        year=insight.get("year", ""),
        setup=insight.get("setup", ""),
        insight=insight.get("insight", ""),
        excerpt=insight.get("excerpt", ""),
        domain=insight.get("domain", ""),
    )

    messages = [{"role": "user", "content": [
        {"type": "text", "text": prompt, "cache_control": {"type": "ephemeral"}},
    ]}]

    label = f"contradiction:{insight.get('book_filename', '')}:{idx}"
    response_text = await call_api(
        client, model, messages, max_tokens=2048,
        semaphore=semaphore, label=label,
    )

    if not response_text:
        return None

    return parse_json_response(response_text)


# ---------------------------------------------------------------------------
# Post-process: deduplicate, validate, create train/val splits
# ---------------------------------------------------------------------------

def deduplicate_problems(problems: list[dict], threshold: float = 0.5) -> list[dict]:
    """Remove near-duplicate problems based on prompt text overlap."""
    if not problems:
        return problems
    kept = []
    for candidate in problems:
        is_dup = False
        cand_prompt = candidate.get("prompt", "")
        for existing in kept:
            existing_prompt = existing.get("prompt", "")
            ratio = SequenceMatcher(None, cand_prompt, existing_prompt).ratio()
            if ratio > threshold:
                is_dup = True
                break
        if not is_dup:
            kept.append(candidate)
    return kept


def create_contradiction_dataset(problems_path: Path, output_dir: Path, val_fraction: float = 0.05):
    """Create train/val splits with system prompt prepended.

    Outputs:
      - contradiction_problems_train.jsonl: training problems (full metadata)
      - contradiction_problems_val.jsonl: validation problems
      - contradiction_prompts_sys_train.jsonl: CustomJSON with system prompt for engine
      - contradiction_prompts_sys_val.jsonl: CustomJSON with system prompt for engine
    """
    problems = []
    with open(problems_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                problems.append(json.loads(line))

    print(f"Loaded {len(problems)} problems from {problems_path}")

    # Deduplicate
    before = len(problems)
    problems = deduplicate_problems(problems)
    print(f"  After dedup: {before} -> {len(problems)}")

    # Shuffle and split
    random.shuffle(problems)
    n_val = max(1, int(len(problems) * val_fraction))
    val_set = problems[:n_val]
    train_set = problems[n_val:]

    print(f"  Train: {len(train_set)}")
    print(f"  Val:   {len(val_set)}")

    system_message = {"role": "system", "content": REASONING_SYSTEM_PROMPT}

    # Write problem files (full metadata for RL judge)
    for split_name, split_data in [("train", train_set), ("val", val_set)]:
        path = output_dir / f"contradiction_problems_{split_name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for record in split_data:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"  Problems: {path}")

    # Write CustomJSON format with system prompt prepended (for RL engine)
    for split_name, split_data in [("train", train_set), ("val", val_set)]:
        path = output_dir / f"contradiction_prompts_sys_{split_name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for record in split_data:
                messages = [
                    system_message,
                    {"role": "user", "content": record["prompt"]},
                    {"role": "assistant", "content": "(to be generated)"},
                ]
                f.write(json.dumps(messages, ensure_ascii=False) + "\n")
        print(f"  Prompts (sys): {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main_async(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    phases = args.run
    if "all" in phases:
        phases = ["phase1", "phase2"]

    client = anthropic.AsyncAnthropic()
    semaphore = asyncio.Semaphore(args.max_concurrent)

    insights_path = output_dir / "contradiction_insights_raw.jsonl"
    problems_path = output_dir / "contradiction_problems_raw.jsonl"

    # Phase 1: Extract or load insights
    if "phase1" in phases:
        print("=" * 60)
        print("PHASE 1: Extract insights from unused books")
        print("=" * 60)

        books_dir = Path(args.books_dir)
        existing_path = Path(args.existing_insights) if args.existing_insights else None
        unused_books = identify_unused_books(books_dir, existing_path)
        unused_books = [b for b in unused_books
                        if estimate_tokens((books_dir / b).read_text(encoding="utf-8")) >= 100]

        print(f"Found {len(unused_books)} unused books:")
        for b in unused_books:
            token_est = estimate_tokens((books_dir / b).read_text(encoding="utf-8"))
            print(f"  {b:60s} ~{token_est:>8,} tokens")

        await extract_insights_from_books(
            client, books_dir, unused_books, args.extract_model,
            semaphore, insights_path, args.chunk_tokens, args.resume,
        )

    # Phase 2: Generate contradiction-resolution problems
    if "phase2" in phases:
        print("\n" + "=" * 60)
        print("PHASE 2: Generate contradiction-resolution problems")
        print("=" * 60)

        # Load insights — prefer existing RL insights if available
        source_path = None
        if args.insights_file:
            source_path = Path(args.insights_file)
        elif insights_path.exists():
            source_path = insights_path

        if source_path is None or not source_path.exists():
            print(f"Error: No insights file found. Provide --insights-file or run phase1 first.")
            return

        insights = []
        with open(source_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    insights.append(json.loads(line))
        print(f"Loaded {len(insights)} insights from {source_path}")

        await generate_contradiction_problems(
            client, insights, args.problem_model, semaphore,
            problems_path, args.resume,
        )

        # Create train/val splits
        print("\nCreating train/val splits...")
        random.seed(args.seed)
        create_contradiction_dataset(problems_path, output_dir, val_fraction=args.val_fraction)


def main():
    parser = argparse.ArgumentParser(
        description="Generate contradiction-resolution RL problems from physics insights"
    )
    parser.add_argument("--books-dir", type=str, default="data/physics_books",
                        help="Directory containing physics book .txt files")
    parser.add_argument("--existing-insights", type=str,
                        default="instruct_data/insights_raw.jsonl",
                        help="Path to insights_raw.jsonl from SFT pipeline")
    parser.add_argument("--insights-file", type=str, default=None,
                        help="Path to pre-extracted insights JSONL to reuse "
                             "(e.g. instruct_data/rl_problems/rl_insights_raw.jsonl)")
    parser.add_argument("--output-dir", type=str,
                        default="instruct_data/contradiction_problems",
                        help="Output directory for contradiction problems")
    parser.add_argument("--extract-model", type=str,
                        default="claude-sonnet-4-20250514",
                        help="Model for insight extraction (phase 1)")
    parser.add_argument("--problem-model", type=str,
                        default="claude-sonnet-4-20250514",
                        help="Model for generating contradiction problems (phase 2)")
    parser.add_argument("--max-concurrent", type=int, default=80,
                        help="Max concurrent API requests")
    parser.add_argument("--chunk-tokens", type=int, default=100_000,
                        help="Target chunk size in tokens for book chunking")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing output")
    parser.add_argument("--val-fraction", type=float, default=0.05,
                        help="Fraction for validation split")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for split")
    parser.add_argument("--run", type=str, nargs="+",
                        default=["all"],
                        choices=["phase1", "phase2", "all"],
                        help="Which phases to run")
    args = parser.parse_args()

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
