#!/usr/bin/env python3
"""
Generate RL training problems from the 26 physics books NOT used for SFT.

Two-phase pipeline:
  Phase 1 — Extract insights from books by chunking and sending to Claude
  Phase 2 — Generate problem prompts paired with gold answers for RL training

The output is used by the RL training loop, where:
  1. The model receives the problem prompt
  2. The model generates a response with <think> reasoning and \\answer{}
  3. Claude judges the response using the gold answer as context,
     scoring on reasoning quality and correctness of the conclusion

Usage:
    # Extract insights from unused books, then generate problems
    python -m scripts.pre1900_scripts.generate_rl_problems \
        --books-dir data/physics_books \
        --existing-insights instruct_data/insights_raw.jsonl \
        --output-dir instruct_data/rl_problems \
        --extract-model claude-sonnet-4-20250514 \
        --problem-model claude-sonnet-4-20250514 \
        --max-concurrent 80 \
        --run phase1 phase2

    # Or run all phases at once
    python -m scripts.pre1900_scripts.generate_rl_problems \
        --books-dir data/physics_books \
        --existing-insights instruct_data/insights_raw.jsonl \
        --output-dir instruct_data/rl_problems \
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

# ---------------------------------------------------------------------------
# Metadata lookup
# ---------------------------------------------------------------------------

WORK_BY_FILENAME = {w["filename"]: w for w in CURATED_WORKS}


def get_work_meta(filename: str) -> dict:
    """Return metadata dict for a book filename, with sensible defaults."""
    meta = WORK_BY_FILENAME.get(filename, {})
    return {
        "author": meta.get("author", "Unknown"),
        "title": meta.get("title", filename.replace(".txt", "").replace("_", " ").title()),
        "year": meta.get("year", "unknown"),
    }


def get_era_description(year) -> str:
    if isinstance(year, int):
        if year < 1700:
            return "seventeenth-century natural philosopher"
        elif year < 1800:
            return "eighteenth-century natural philosopher"
        else:
            return "nineteenth-century natural philosopher"
    return "pre-1900 natural philosopher"


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token for English prose."""
    return len(text) // 4


# ---------------------------------------------------------------------------
# Text chunking (reused from generate_reasoning_traces.py)
# ---------------------------------------------------------------------------

def chunk_text(text: str, target_tokens: int = 100_000, overlap_tokens: int = 2000) -> list[str]:
    """Split text into chunks at paragraph boundaries with overlap."""
    target_chars = target_tokens * 4
    overlap_chars = overlap_tokens * 4

    if estimate_tokens(text) <= target_tokens + overlap_tokens:
        return [text]

    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = []
    current_len = 0

    for para in paragraphs:
        para_len = len(para) + 2
        if current_len + para_len > target_chars and current_chunk:
            chunks.append("\n\n".join(current_chunk))
            overlap_parts = []
            overlap_len = 0
            for p in reversed(current_chunk):
                if overlap_len + len(p) > overlap_chars:
                    break
                overlap_parts.insert(0, p)
                overlap_len += len(p) + 2
            current_chunk = overlap_parts
            current_len = overlap_len

        current_chunk.append(para)
        current_len += para_len

    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    return chunks


# ---------------------------------------------------------------------------
# JSON parsing helpers
# ---------------------------------------------------------------------------

def parse_json_response(text: str) -> dict | list | None:
    """Parse Claude's JSON response, stripping markdown fences if present."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        for start_char, end_char in [("{", "}"), ("[", "]")]:
            start = text.find(start_char)
            end = text.rfind(end_char)
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(text[start:end + 1])
                except json.JSONDecodeError:
                    continue
        return None


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def deduplicate_insights(insights: list[dict], threshold: float = 0.6) -> list[dict]:
    """Remove near-duplicate insights based on excerpt text overlap."""
    if not insights:
        return insights

    kept = []
    for candidate in insights:
        is_dup = False
        cand_excerpt = candidate.get("excerpt", "")
        for i, existing in enumerate(kept):
            existing_excerpt = existing.get("excerpt", "")
            ratio = SequenceMatcher(None, cand_excerpt, existing_excerpt).ratio()
            if ratio > threshold:
                if len(candidate.get("setup", "")) > len(existing.get("setup", "")):
                    kept[i] = candidate
                is_dup = True
                break
        if not is_dup:
            kept.append(candidate)

    return kept


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

INSIGHT_EXTRACTION_PROMPT = """\
You are a historian of science analyzing a section from a pre-1900 physics text.

CRITICAL CONSTRAINT: This book was written in {year}. You must ONLY extract insights \
using concepts and language available before 1900. Do NOT reference or frame anything \
using post-1900 physics (no relativity, no quantum mechanics, no modern atomic theory). \
Present each insight as it would have been understood by the author and their contemporaries.

Book: "{title}" by {author} ({year})

Your task: find the key scientific insights in this section — conceptual breakthroughs, \
experimental results, thought experiments, and physical reasoning.

IMPORTANT — Focus on SCIENTIFICALLY CORRECT and CONCEPTUAL insights:
- Prioritize conceptual understanding and physical reasoning over mathematical derivations
- SKIP insights premised on theories later proven wrong (e.g., aether, caloric, phlogiston) \
UNLESS the insight refutes such a theory
- AVOID purely mathematical insights — no equations or symbolic derivations. \
Extract the physical reasoning instead
- Conceptual arguments, experimental interpretations, and physical analogies are \
especially valuable

Look for:
- Conceptual explanations of physical phenomena
- Experimental procedures and the qualitative conclusions drawn from them
- Physical arguments and analogies that illuminate why something works
- Connections between different physical phenomena
- Landmark thought experiments

For each insight, provide:
1. **setup**: The experimental evidence or theoretical constraints that LEAD TO the insight. \
Must NOT contain the answer. Use only concepts available in {year}. Must be detailed enough \
that a knowledgeable reader could reason toward the conclusion.
2. **insight**: The specific scientific conclusion or result, stated as the author would have \
understood it.
3. **excerpt**: A verbatim passage (1-5 sentences) capturing this insight.
4. **domain**: Physics subdomain (e.g., "thermodynamics", "optics", "electromagnetism", \
"mechanics", "acoustics", "astronomy", "kinetic theory").
5. **difficulty**: "introductory", "intermediate", or "advanced".

Return a JSON object with a single field:
"insights": A JSON array of insight objects, each with: setup, insight, excerpt, domain, difficulty.

Return ONLY valid JSON. If this section contains no worthwhile insights, return {{"insights": []}}."""


PROBLEM_PROMPT = """\
You are generating a problem for a language model that thinks like a {era_description}.

Given the following scientific setup and insight from "{title}" by {author} ({year}), \
create a PROBLEM PROMPT that:
- Presents experimental evidence, observations, or theoretical constraints as a \
period-appropriate question
- Is written as an educated person of the era would — formal prose, period vocabulary
- Does NOT reveal the conclusion — the model must reason toward it
- Is self-contained and standalone — do not reference "the text" or "the passage"
- Contains enough information that a knowledgeable reader could reason toward the answer

CRITICAL CONSTRAINT: The year is {year}. Use ONLY knowledge available before 1900.
Do NOT include any post-1900 concepts, terminology, or scientists.
Do NOT include equations or mathematical symbols. Frame everything conceptually.
Write as if you truly exist in {year} with no knowledge of the future.

SETUP (the evidence/constraints to present):
{setup}

INSIGHT (the conclusion the model should reason toward — do NOT include this in the prompt):
{insight}

RELEVANT EXCERPT FROM THE TEXT:
{excerpt}

Return a JSON object with a single field:
- "prompt": the problem prompt (string)

Return ONLY valid JSON."""

# ---------------------------------------------------------------------------
# Judge prompt template (exported for use by RL training script)
# ---------------------------------------------------------------------------

RL_JUDGE_PROMPT = """\
You are evaluating a language model's response to a pre-1900 physics problem. \
The model should reason like a pre-1900 natural philosopher — using conceptual arguments, \
physical intuition, and period-appropriate scientific prose. Do NOT penalize archaic style \
or pre-1900 vocabulary.

PROBLEM:
{prompt}

CORRECT ANSWER (for your reference — the model does not see this):
{gold_answer}

MODEL'S RESPONSE:
{response}

Rate the response on these 5 criteria (each 0 or 1):
1. Correct conclusion: Does the model arrive at a conclusion that is substantively correct, \
capturing the key physical insight? It does not need to match the reference answer \
word-for-word, but the core idea must be right.
2. Sound reasoning: Does the model demonstrate genuine step-by-step reasoning rather than \
just asserting the answer? Are the logical steps valid?
3. Physical insight: Does the model show understanding of the underlying physical principles, \
causes, and mechanisms?
4. No anachronisms: Is the response free of post-1900 concepts, terminology, and knowledge?
5. Coherence: Is the response well-structured, internally consistent, and free of \
contradictions or garbled text?

Return ONLY a single integer from 1 to 5 representing the total score. Nothing else."""


# ---------------------------------------------------------------------------
# Async API call with retry
# ---------------------------------------------------------------------------

async def call_api(
    client: anthropic.AsyncAnthropic,
    model: str,
    messages: list[dict],
    max_tokens: int,
    semaphore: asyncio.Semaphore,
    label: str = "",
) -> str | None:
    max_retries = 8
    async with semaphore:
        for attempt in range(max_retries):
            try:
                response = await client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    messages=messages,
                )
                return response.content[0].text
            except anthropic.RateLimitError:
                wait = min(2 ** attempt * 5, 120)
                print(f"  Rate limited on {label}, retrying in {wait}s "
                      f"(attempt {attempt+1}/{max_retries})")
                await asyncio.sleep(wait)
            except Exception as e:
                print(f"  Error on {label}: {e}")
                return None
    print(f"  Exhausted retries for {label}")
    return None


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_insight(item: dict) -> bool:
    """Validate a single insight."""
    if not isinstance(item, dict):
        return False
    setup = item.get("setup", "")
    insight = item.get("insight", "")
    excerpt = item.get("excerpt", "")
    if not isinstance(setup, str) or len(setup) < 50:
        return False
    if not isinstance(insight, str) or len(insight) < 20:
        return False
    if not isinstance(excerpt, str) or len(excerpt) < 10:
        return False
    return True


# ---------------------------------------------------------------------------
# Phase 1: Extract insights from unused books
# ---------------------------------------------------------------------------

def identify_unused_books(
    books_dir: Path,
    existing_insights_path: Path | None,
) -> list[str]:
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

    # Resume state
    already_done = set()
    if resume and output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        key = (record.get("book_filename", ""),
                               record.get("chunk_idx", -1))
                        already_done.add(key)
                    except json.JSONDecodeError:
                        pass
        print(f"  Resume: {len(already_done)} chunks already processed")

    # Build list of chunks to process
    all_chunks = []
    for filename in unused_books:
        book_path = books_dir / filename
        if not book_path.exists():
            print(f"  [{filename}] File not found, skipping")
            continue

        text = book_path.read_text(encoding="utf-8")
        token_est = estimate_tokens(text)

        # Skip near-empty files
        if token_est < 100:
            print(f"  [{filename}] Only ~{token_est} tokens, skipping")
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
                _extract_chunk_insights(
                    client, filename, chunk_idx, total_chunks,
                    chunk, model, semaphore
                )
                for filename, chunk_idx, chunk, total_chunks in batch
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
    """Extract insights from a single chunk. Returns list of valid insights or None."""
    meta = get_work_meta(filename)
    label = f"extract:{filename}:chunk{chunk_idx}/{total_chunks}"

    prompt = INSIGHT_EXTRACTION_PROMPT.format(
        title=meta["title"],
        author=meta["author"],
        year=meta["year"],
    )

    messages = [{"role": "user", "content": [
        {"type": "text", "text": prompt, "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": f"--- SECTION (chunk {chunk_idx + 1}/{total_chunks}) ---"
                                 f"\n\n{chunk}"},
    ]}]

    response_text = await call_api(
        client, model, messages, max_tokens=8192,
        semaphore=semaphore, label=label,
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
    """Deduplicate insights in-place, grouped by book."""
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
# Phase 2: Generate RL problem prompts from insights
# ---------------------------------------------------------------------------

async def generate_problems(
    client: anthropic.AsyncAnthropic,
    insights: list[dict],
    model: str,
    semaphore: asyncio.Semaphore,
    output_path: Path,
    resume: bool,
):
    """Generate RL problem prompts from extracted insights."""

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

    to_process = [(i, ins) for i, ins in enumerate(insights) if i not in already_done]
    print(f"Phase 2: {len(to_process)} problems to generate "
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
                _generate_single_problem(client, idx, insight, model, semaphore)
                for idx, insight in batch
            ]

            results = await asyncio.gather(*tasks)

            for (idx, insight), result in zip(batch, results):
                if result is None:
                    failed += 1
                    continue

                prompt_text = result.get("prompt", "")
                if not prompt_text or len(prompt_text) < 50:
                    failed += 1
                    continue

                record = {
                    "_source_idx": idx,
                    "prompt": prompt_text,
                    "gold_answer": insight.get("insight", ""),
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


async def _generate_single_problem(
    client: anthropic.AsyncAnthropic,
    idx: int,
    insight: dict,
    model: str,
    semaphore: asyncio.Semaphore,
) -> dict | None:
    """Generate a single problem prompt from an insight."""
    prompt = PROBLEM_PROMPT.format(
        era_description=get_era_description(insight.get("year")),
        title=insight.get("title", ""),
        author=insight.get("author", ""),
        year=insight.get("year", ""),
        setup=insight.get("setup", ""),
        insight=insight.get("insight", ""),
        excerpt=insight.get("excerpt", ""),
    )

    messages = [{"role": "user", "content": [
        {"type": "text", "text": prompt, "cache_control": {"type": "ephemeral"}},
    ]}]

    label = f"problem:{insight.get('book_filename', '')}:{idx}"
    response_text = await call_api(
        client, model, messages, max_tokens=1024,
        semaphore=semaphore, label=label,
    )

    if not response_text:
        return None

    return parse_json_response(response_text)


# ---------------------------------------------------------------------------
# Post-process: create train/val splits
# ---------------------------------------------------------------------------

def create_rl_dataset(problems_path: Path, output_dir: Path, val_fraction: float = 0.05):
    """Create train/val splits in formats usable by the RL training loop.

    Outputs:
      - rl_problems_train.jsonl: training problems (one JSON object per line)
      - rl_problems_val.jsonl: validation problems
      - rl_prompts_train.jsonl: CustomJSON format for engine (user-only, dummy assistant)
      - rl_prompts_val.jsonl: CustomJSON format for engine
    """
    problems = []
    with open(problems_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                problems.append(json.loads(line))

    print(f"Loaded {len(problems)} problems from {problems_path}")

    # Shuffle and split
    random.shuffle(problems)
    n_val = max(1, int(len(problems) * val_fraction))
    val_set = problems[:n_val]
    train_set = problems[n_val:]

    print(f"  Train: {len(train_set)}")
    print(f"  Val:   {len(val_set)}")

    # Write problem files (full metadata for RL judge)
    for split_name, split_data in [("train", train_set), ("val", val_set)]:
        path = output_dir / f"rl_problems_{split_name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for record in split_data:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"  Problems: {path}")

    # Write CustomJSON format (for loading into RL engine)
    for split_name, split_data in [("train", train_set), ("val", val_set)]:
        path = output_dir / f"rl_prompts_{split_name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for record in split_data:
                messages = [
                    {"role": "user", "content": record["prompt"]},
                    {"role": "assistant", "content": "(to be generated)"},
                ]
                f.write(json.dumps(messages, ensure_ascii=False) + "\n")
        print(f"  Prompts: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main_async(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    books_dir = Path(args.books_dir)

    phases = args.run
    if "all" in phases:
        phases = ["phase1", "phase2"]

    client = anthropic.AsyncAnthropic()
    semaphore = asyncio.Semaphore(args.max_concurrent)

    rl_insights_path = output_dir / "rl_insights_raw.jsonl"
    problems_path = output_dir / "rl_problems_raw.jsonl"

    # Phase 1: Extract insights from unused books
    if "phase1" in phases:
        print("=" * 60)
        print("PHASE 1: Extract insights from unused books")
        print("=" * 60)

        existing_path = Path(args.existing_insights) if args.existing_insights else None
        unused_books = identify_unused_books(books_dir, existing_path)

        # Filter out the empty file
        unused_books = [b for b in unused_books
                        if estimate_tokens((books_dir / b).read_text(encoding="utf-8")) >= 100]

        print(f"Found {len(unused_books)} unused books:")
        for b in unused_books:
            token_est = estimate_tokens((books_dir / b).read_text(encoding="utf-8"))
            print(f"  {b:60s} ~{token_est:>8,} tokens")

        await extract_insights_from_books(
            client, books_dir, unused_books, args.extract_model,
            semaphore, rl_insights_path, args.chunk_tokens, args.resume,
        )

    # Phase 2: Generate RL problem prompts
    if "phase2" in phases:
        print("\n" + "=" * 60)
        print("PHASE 2: Generate RL problem prompts")
        print("=" * 60)

        if not rl_insights_path.exists():
            print(f"Error: {rl_insights_path} does not exist. Run phase1 first.")
            return

        insights = []
        with open(rl_insights_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    insights.append(json.loads(line))
        print(f"Loaded {len(insights)} insights from {rl_insights_path}")

        await generate_problems(
            client, insights, args.problem_model, semaphore,
            problems_path, args.resume,
        )

        # Create train/val splits
        print("\nCreating train/val splits...")
        random.seed(args.seed)
        create_rl_dataset(problems_path, output_dir, val_fraction=args.val_fraction)


def main():
    parser = argparse.ArgumentParser(
        description="Generate RL problems from unused physics books"
    )
    parser.add_argument("--books-dir", type=str, default="data/physics_books",
                        help="Directory containing physics book .txt files")
    parser.add_argument("--existing-insights", type=str,
                        default="instruct_data/insights_raw.jsonl",
                        help="Path to insights_raw.jsonl from SFT pipeline "
                             "(to identify unused books)")
    parser.add_argument("--output-dir", type=str,
                        default="instruct_data/rl_problems",
                        help="Output directory for RL problems")
    parser.add_argument("--extract-model", type=str,
                        default="claude-sonnet-4-20250514",
                        help="Model for insight extraction (phase 1)")
    parser.add_argument("--problem-model", type=str,
                        default="claude-sonnet-4-20250514",
                        help="Model for generating problem prompts (phase 2)")
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
