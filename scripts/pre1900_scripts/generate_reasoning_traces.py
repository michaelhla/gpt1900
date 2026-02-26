#!/usr/bin/env python3
"""
Generate reasoning trace training data from pre-1900 physics books.

Three-pass pipeline:
  Pass 1a — Whole-book insight extraction (major insights + thesis)
  Pass 1b — Chunk-level insight extraction (detailed technical insights)
  Pass 2  — Reasoning trace generation from extracted insights

Output is compatible with the CustomJSON data loader.

Usage:
    # Pass 1a: whole-book insights
    python -m scripts.pre1900_scripts.generate_reasoning_traces \
        --books-dir data/physics_books --output-dir instruct_data \
        --run-pass 1a --pass1-model claude-opus-4-6

    # Pass 1b: chunk-level insights
    python -m scripts.pre1900_scripts.generate_reasoning_traces \
        --books-dir data/physics_books --output-dir instruct_data \
        --run-pass 1b --pass1-model claude-opus-4-6

    # Pass 2: generate reasoning traces
    python -m scripts.pre1900_scripts.generate_reasoning_traces \
        --books-dir data/physics_books --output-dir instruct_data \
        --run-pass 2 --pass2-model claude-sonnet-4-20250514
"""

import os
import json
import asyncio
import argparse
import time
import re
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


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------

def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token for English prose."""
    return len(text) // 4


# ---------------------------------------------------------------------------
# Text chunking
# ---------------------------------------------------------------------------

def chunk_text(text: str, target_tokens: int = 100_000, overlap_tokens: int = 2000) -> list[str]:
    """Split text into chunks at paragraph boundaries with overlap.

    Returns list of text chunks, each approximately target_tokens long.
    """
    target_chars = target_tokens * 4
    overlap_chars = overlap_tokens * 4

    if estimate_tokens(text) <= target_tokens + overlap_tokens:
        return [text]

    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = []
    current_len = 0

    for para in paragraphs:
        para_len = len(para) + 2  # account for \n\n
        if current_len + para_len > target_chars and current_chunk:
            chunks.append("\n\n".join(current_chunk))
            # Build overlap from the end of the current chunk
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
        # Try to find JSON within the response
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
# Prompt templates
# ---------------------------------------------------------------------------

PASS_1A_PROMPT = """\
You are a historian of science analyzing a pre-1900 physics text. Your task is to extract the key scientific insights, derivations, and discoveries from this book.

CRITICAL CONSTRAINT: This book was written before 1900. You must ONLY extract insights, concepts, and knowledge that existed before 1900. Do NOT interpret any finding through the lens of post-1900 physics (no relativity, no quantum mechanics, no atomic models beyond Thomson's, no radioactivity beyond Becquerel's initial discovery). Frame everything in the scientific language and conceptual framework of the era.

Book: "{title}" by {author} ({year})

For each major insight, provide:
1. **setup**: The experimental evidence, observations, or theoretical constraints that LEAD TO the insight. This must NOT contain the answer — it should present the conditions from which the insight can be derived through reasoning. Include enough context that a knowledgeable reader could, in principle, derive the conclusion. Use only concepts and terminology available in {year}.
2. **insight**: The key scientific conclusion, derivation result, or discovery — stated as the author would have understood it, not with modern hindsight.
3. **excerpt**: A verbatim passage from the text (1-5 sentences) that best captures this insight.
4. **domain**: The physics subdomain (e.g., "thermodynamics", "optics", "electromagnetism", "mechanics", "acoustics", "astronomy", "kinetic theory").
5. **difficulty**: One of "introductory", "intermediate", or "advanced".

IMPORTANT — Focus on SCIENTIFICALLY CORRECT and CONCEPTUAL insights:
- Prioritize conceptual understanding, physical intuition, and qualitative reasoning over mathematical derivations
- Focus on insights where the core physical reasoning is sound and still considered correct (e.g., conservation of energy, thermodynamic laws, wave phenomena, electromagnetic induction, Newtonian mechanics in its domain)
- SKIP insights based on theories later shown to be fundamentally wrong (e.g., luminiferous aether as a real substance, caloric theory, phlogiston) UNLESS the insight is the argument that REFUTED such a theory
- AVOID insights that are primarily mathematical — no equations, no derivations that require symbolic manipulation. Instead, extract the physical reasoning and conceptual arguments behind such results
- Thought experiments, analogies, and physical arguments are especially valuable

Focus on:
- Conceptual breakthroughs and the physical reasoning behind them
- Landmark thought experiments and what they reveal
- Key experimental results and the qualitative conclusions drawn from them
- How ideas and physical concepts build on each other within the book
- Physical intuition — how the author explains WHY things work the way they do

Return a JSON object with exactly two fields:
1. "thesis": A 3-5 sentence summary of the book's central argument and narrative arc.
2. "insights": A JSON array of insight objects, each with fields: setup, insight, excerpt, domain, difficulty.

Return ONLY valid JSON. Do not include any text before or after the JSON object."""

PASS_1B_PROMPT = """\
You are a historian of science performing a detailed analysis of a section from a pre-1900 physics text.

CRITICAL CONSTRAINT: This book was written in {year}. You must ONLY extract insights using concepts and language available before 1900. Do NOT reference or frame anything using post-1900 physics (no relativity, no quantum mechanics, no modern atomic theory, no concepts developed after 1900). Present each insight as it would have been understood by the author and their contemporaries.

Book: "{title}" by {author} ({year})
Book thesis: {thesis}

The following major insights have ALREADY been extracted from a whole-book analysis. Do NOT repeat these:
{existing_insight_titles}

Your task: find ADDITIONAL detailed, technical insights specific to this section that were missed in the whole-book scan.

IMPORTANT — Focus on SCIENTIFICALLY CORRECT and CONCEPTUAL insights:
- Prioritize conceptual understanding and physical reasoning over mathematical derivations
- SKIP insights premised on theories later proven wrong (e.g., aether, caloric, phlogiston) UNLESS the insight refutes such a theory
- AVOID purely mathematical insights — no equations or symbolic derivations. Extract the physical reasoning instead
- Conceptual arguments, experimental interpretations, and physical analogies are especially valuable

Look for:
- Conceptual explanations of physical phenomena
- Experimental procedures and the qualitative conclusions drawn from them
- Physical arguments and analogies that illuminate why something works
- Subtle conceptual points, caveats, or physical conditions that qualify the major results
- Connections between different physical phenomena

For each insight, provide:
1. **setup**: The experimental evidence or theoretical constraints that LEAD TO the insight. Must NOT contain the answer. Use only concepts available in {year}.
2. **insight**: The specific scientific conclusion or result, stated as the author would have understood it.
3. **excerpt**: A verbatim passage (1-5 sentences) capturing this insight.
4. **domain**: Physics subdomain.
5. **difficulty**: "introductory", "intermediate", or "advanced".

Return a JSON object with a single field:
"insights": A JSON array of insight objects, each with: setup, insight, excerpt, domain, difficulty.

Return ONLY valid JSON. If this section contains no additional insights worth extracting, return {{"insights": []}}."""

PASS_2_PROMPT = """\
You are generating a training example for a language model that thinks like a {era_description}.

CRITICAL CONSTRAINT: The year is {year}. The model being trained has NO knowledge of anything after 1900.
You must NOT include ANY of the following in either the user prompt or the assistant response:
- Relativity (special or general), quantum mechanics, radioactivity (beyond Becquerel if after 1896), electrons (beyond cathode rays if before 1897)
- Any scientist, discovery, or concept from after 1900
- Modern terminology that did not exist before 1900 (e.g., "photon", "quantum", "spacetime", "entropy increase" stated as a universal law if before Clausius)
- Retrospective framing (e.g., "this would later be known as..." or "this anticipated...")
Everything must be stated as it was understood AT THE TIME, not with historical hindsight.

Given the following scientific setup and insight from "{title}" by {author} ({year}), create a training pair where:

1. The **user message** poses the experimental evidence or theoretical constraints as a period-appropriate prompt. Write as an educated person of the era would — formal prose, period vocabulary. Do NOT reveal the conclusion in the prompt. The prompt should make it possible to reason toward the insight.

2. The **assistant message** contains:
   - A `<think>` block with step-by-step CONCEPTUAL reasoning in the scientific prose style of the era. The reasoning should:
     * Work through the evidence using physical intuition and qualitative arguments — NOT mathematical equations
     * Reason about causes, effects, and physical mechanisms in plain scientific prose
     * Consider alternative explanations where appropriate and explain why they fail on physical grounds
     * Build the argument step-by-step: each conceptual step should follow naturally from the previous one
     * Use analogies, thought experiments, and appeals to known physical principles where helpful
     * Be substantial (at least several paragraphs)
     * Reference only theories, scientists, and concepts known before {year}
     * Do NOT include equations, mathematical symbols, or symbolic derivations
   - An `\\answer{{}}` block with the key conclusion stated clearly and concisely.

SETUP (the evidence/constraints to present):
{setup}

INSIGHT (the conclusion to reason toward):
{insight}

RELEVANT EXCERPT FROM THE TEXT:
{excerpt}

RULES:
1. The user prompt must be STANDALONE — do not reference "the text" or "the passage".
2. Use ONLY knowledge available before {year}. No post-1900 concepts, terminology, or scientists.
3. The reasoning in <think> must be genuine step-by-step CONCEPTUAL reasoning, not just restating the answer. Use physical arguments and intuition, NOT mathematical equations.
4. Match the scientific prose style of the era — measured, formal, with period-appropriate terminology.
5. The \\answer{{}} must be a clear, concise statement of the conclusion (1-3 sentences).
6. Do NOT use any anachronistic language. Write as if you truly exist in {year} with no knowledge of the future.
7. Do NOT include equations, mathematical symbols, or formal derivations. Reason entirely in words about physical concepts.

Return a JSON object with exactly two fields:
- "user": the user message (string)
- "assistant": the assistant message (string, containing <think>...</think> and \\answer{{}})

Return ONLY valid JSON."""


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def validate_insight(item: dict) -> bool:
    """Validate a single insight from pass 1."""
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


def validate_trace(data: dict) -> bool:
    """Validate a reasoning trace from pass 2."""
    if not isinstance(data, dict):
        return False
    user = data.get("user", "")
    assistant = data.get("assistant", "")
    if not isinstance(user, str) or len(user) < 20:
        return False
    if not isinstance(assistant, str):
        return False
    if "<think>" not in assistant or "</think>" not in assistant:
        return False
    if "\\answer{" not in assistant:
        return False
    # Check that think content is substantial
    think_match = re.search(r"<think>(.*?)</think>", assistant, re.DOTALL)
    if not think_match or len(think_match.group(1).strip()) < 100:
        return False
    return True


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def deduplicate_insights(insights: list[dict], threshold: float = 0.6) -> list[dict]:
    """Remove near-duplicate insights based on excerpt text overlap.

    If two insights share >threshold of their excerpt text, keep the one with
    the longer setup (more context is better for training).
    """
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
                # Keep the one with the longer setup
                if len(candidate.get("setup", "")) > len(existing.get("setup", "")):
                    kept[i] = candidate
                is_dup = True
                break
        if not is_dup:
            kept.append(candidate)

    return kept


# ---------------------------------------------------------------------------
# Async API calls with retry
# ---------------------------------------------------------------------------

async def call_api(
    client: anthropic.AsyncAnthropic,
    model: str,
    messages: list[dict],
    max_tokens: int,
    semaphore: asyncio.Semaphore,
    label: str = "",
    betas: list[str] | None = None,
) -> str | None:
    """Call the Anthropic API with exponential backoff on rate limits."""
    max_retries = 8
    async with semaphore:
        for attempt in range(max_retries):
            try:
                if betas:
                    response = await client.messages.create(
                        model=model,
                        max_tokens=max_tokens,
                        messages=messages,
                        betas=betas,
                    )
                else:
                    response = await client.messages.create(
                        model=model,
                        max_tokens=max_tokens,
                        messages=messages,
                    )
                return response.content[0].text
            except anthropic.RateLimitError:
                wait = min(2 ** attempt * 5, 120)
                print(f"  Rate limited on {label}, retrying in {wait}s (attempt {attempt+1}/{max_retries})")
                await asyncio.sleep(wait)
            except Exception as e:
                print(f"  Error on {label}: {e}")
                return None
    print(f"  Exhausted retries for {label}")
    return None


# ---------------------------------------------------------------------------
# Pass 1a: Whole-book insight extraction
# ---------------------------------------------------------------------------

async def run_pass_1a(
    client: anthropic.AsyncAnthropic,
    books_dir: Path,
    output_dir: Path,
    model: str,
    semaphore: asyncio.Semaphore,
    resume: bool,
    only: list[str] | None,
):
    """Extract major insights and thesis from each book (full book context)."""
    insights_path = output_dir / "insights_1a.jsonl"
    thesis_path = output_dir / "theses.jsonl"

    # Load resume state
    already_done = set()
    if resume:
        if insights_path.exists():
            with open(insights_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        record = json.loads(line)
                        already_done.add(record.get("book_filename", ""))
        print(f"  Resume: {len(already_done)} books already processed for pass 1a")

    # Collect books to process
    book_files = sorted(books_dir.glob("*.txt"))
    if only:
        book_files = [f for f in book_files if f.name in only]

    to_process = [f for f in book_files if f.name not in already_done]
    print(f"Pass 1a: {len(to_process)} books to process ({len(already_done)} already done)")

    if not to_process:
        return

    mode = "a" if resume else "w"
    t0 = time.time()
    success = 0
    failed = 0

    # Process in chunks
    chunk_size = semaphore._value * 2
    with open(insights_path, mode, encoding="utf-8") as insights_f, \
         open(thesis_path, mode, encoding="utf-8") as thesis_f:

        for chunk_start in range(0, len(to_process), chunk_size):
            chunk = to_process[chunk_start:chunk_start + chunk_size]
            tasks = []

            for book_path in chunk:
                tasks.append(_process_book_1a(
                    client, book_path, model, semaphore,
                    insights_f, thesis_f,
                ))

            results = await asyncio.gather(*tasks)
            for ok in results:
                if ok:
                    success += 1
                else:
                    failed += 1

            insights_f.flush()
            thesis_f.flush()

            total_done = chunk_start + len(chunk)
            elapsed = time.time() - t0
            print(f"  Progress: {total_done}/{len(to_process)} | "
                  f"success={success} failed={failed} | {elapsed:.0f}s elapsed")

    elapsed = time.time() - t0
    print(f"\nPass 1a done in {elapsed:.0f}s: {success} succeeded, {failed} failed")


async def _process_book_1a(
    client: anthropic.AsyncAnthropic,
    book_path: Path,
    model: str,
    semaphore: asyncio.Semaphore,
    insights_f,
    thesis_f,
) -> bool:
    """Process a single book for pass 1a. Returns True on success."""
    filename = book_path.name
    meta = get_work_meta(filename)
    text = book_path.read_text(encoding="utf-8")
    token_est = estimate_tokens(text)

    print(f"  [{filename}] ~{token_est:,} tokens, sending to {model}...")

    prompt = PASS_1A_PROMPT.format(
        title=meta["title"],
        author=meta["author"],
        year=meta["year"],
    )

    # Use beta header for 1M context if book is large
    betas = None
    if token_est > 180_000:
        betas = ["context-1m-2025-08-07"]

    messages = [{"role": "user", "content": [
        {"type": "text", "text": prompt, "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": f"--- FULL TEXT ---\n\n{text}"},
    ]}]

    response_text = await call_api(
        client, model, messages, max_tokens=8192,
        semaphore=semaphore, label=f"1a:{filename}", betas=betas,
    )

    if not response_text:
        print(f"  [{filename}] No response")
        return False

    data = parse_json_response(response_text)
    if not isinstance(data, dict):
        print(f"  [{filename}] Failed to parse JSON response")
        return False

    thesis = data.get("thesis", "")
    insights = data.get("insights", [])

    if not isinstance(insights, list):
        print(f"  [{filename}] insights is not a list")
        return False

    # Validate and filter insights
    valid_insights = [i for i in insights if validate_insight(i)]
    print(f"  [{filename}] Extracted {len(valid_insights)}/{len(insights)} valid insights (thesis: {len(thesis)} chars)")

    # Write thesis
    thesis_record = {
        "book_filename": filename,
        "author": meta["author"],
        "title": meta["title"],
        "year": meta["year"],
        "thesis": thesis,
        "num_insights": len(valid_insights),
    }
    thesis_f.write(json.dumps(thesis_record, ensure_ascii=False) + "\n")

    # Write insights
    for idx, insight in enumerate(valid_insights):
        record = {
            "book_filename": filename,
            "author": meta["author"],
            "title": meta["title"],
            "year": meta["year"],
            "pass": "1a",
            "insight_idx": idx,
            **insight,
        }
        insights_f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return True


# ---------------------------------------------------------------------------
# Pass 1b: Chunk-level insight extraction
# ---------------------------------------------------------------------------

async def run_pass_1b(
    client: anthropic.AsyncAnthropic,
    books_dir: Path,
    output_dir: Path,
    model: str,
    semaphore: asyncio.Semaphore,
    chunk_tokens: int,
    resume: bool,
    only: list[str] | None,
):
    """Extract detailed insights from book chunks."""
    insights_path = output_dir / "insights_1b.jsonl"
    thesis_path = output_dir / "theses.jsonl"
    insights_1a_path = output_dir / "insights_1a.jsonl"

    # Load theses from pass 1a
    theses = {}
    if thesis_path.exists():
        with open(thesis_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    theses[record["book_filename"]] = record.get("thesis", "")

    # Load 1a insights for dedup context
    insights_1a_by_book = {}
    if insights_1a_path.exists():
        with open(insights_1a_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    fn = record["book_filename"]
                    if fn not in insights_1a_by_book:
                        insights_1a_by_book[fn] = []
                    insights_1a_by_book[fn].append(record)

    # Load resume state
    already_done = set()
    if resume and insights_path.exists():
        with open(insights_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    key = (record.get("book_filename", ""), record.get("chunk_idx", -1))
                    already_done.add(key)
        print(f"  Resume: {len(already_done)} chunks already processed for pass 1b")

    # Collect books
    book_files = sorted(books_dir.glob("*.txt"))
    if only:
        book_files = [f for f in book_files if f.name in only]

    # Build list of (book_path, chunk_idx, chunk_text) to process
    all_chunks = []
    for book_path in book_files:
        filename = book_path.name
        if filename not in theses:
            print(f"  [{filename}] No thesis found from pass 1a, skipping (run pass 1a first)")
            continue

        text = book_path.read_text(encoding="utf-8")
        chunks = chunk_text(text, target_tokens=chunk_tokens)

        for chunk_idx, chunk in enumerate(chunks):
            if (filename, chunk_idx) not in already_done:
                all_chunks.append((book_path, chunk_idx, chunk, len(chunks)))

    print(f"Pass 1b: {len(all_chunks)} chunks to process ({len(already_done)} already done)")

    if not all_chunks:
        return

    mode = "a" if resume else "w"
    t0 = time.time()
    success = 0
    failed = 0

    chunk_size = semaphore._value * 2
    with open(insights_path, mode, encoding="utf-8") as f:
        for batch_start in range(0, len(all_chunks), chunk_size):
            batch = all_chunks[batch_start:batch_start + chunk_size]
            tasks = []

            for book_path, chunk_idx, chunk, total_chunks in batch:
                filename = book_path.name
                meta = get_work_meta(filename)
                thesis = theses.get(filename, "")
                existing_insights = insights_1a_by_book.get(filename, [])

                # Build list of existing insight summaries
                existing_titles = "\n".join(
                    f"- {ins.get('insight', '')[:100]}"
                    for ins in existing_insights
                ) or "(none)"

                tasks.append(_process_chunk_1b(
                    client, filename, meta, chunk_idx, total_chunks,
                    chunk, thesis, existing_titles, model, semaphore, f,
                ))

            results = await asyncio.gather(*tasks)
            for ok in results:
                if ok:
                    success += 1
                else:
                    failed += 1

            f.flush()
            total_done = batch_start + len(batch)
            elapsed = time.time() - t0
            print(f"  Progress: {total_done}/{len(all_chunks)} | "
                  f"success={success} failed={failed} | {elapsed:.0f}s elapsed")

    elapsed = time.time() - t0
    print(f"\nPass 1b done in {elapsed:.0f}s: {success} succeeded, {failed} failed")


async def _process_chunk_1b(
    client: anthropic.AsyncAnthropic,
    filename: str,
    meta: dict,
    chunk_idx: int,
    total_chunks: int,
    chunk_text_content: str,
    thesis: str,
    existing_insight_titles: str,
    model: str,
    semaphore: asyncio.Semaphore,
    output_f,
) -> bool:
    """Process a single chunk for pass 1b. Returns True on success."""
    token_est = estimate_tokens(chunk_text_content)
    label = f"1b:{filename}:chunk{chunk_idx}/{total_chunks}"

    prompt = PASS_1B_PROMPT.format(
        title=meta["title"],
        author=meta["author"],
        year=meta["year"],
        thesis=thesis,
        existing_insight_titles=existing_insight_titles,
    )

    betas = None
    if token_est > 180_000:
        betas = ["context-1m-2025-08-07"]

    messages = [{"role": "user", "content": [
        {"type": "text", "text": prompt, "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": f"--- SECTION (chunk {chunk_idx + 1}/{total_chunks}) ---\n\n{chunk_text_content}"},
    ]}]

    response_text = await call_api(
        client, model, messages, max_tokens=8192,
        semaphore=semaphore, label=label, betas=betas,
    )

    if not response_text:
        return False

    data = parse_json_response(response_text)
    if not isinstance(data, dict):
        print(f"  [{label}] Failed to parse JSON")
        return False

    insights = data.get("insights", [])
    if not isinstance(insights, list):
        return False

    valid_insights = [i for i in insights if validate_insight(i)]
    print(f"  [{label}] ~{token_est:,}tok -> {len(valid_insights)}/{len(insights)} valid insights")

    for idx, insight in enumerate(valid_insights):
        record = {
            "book_filename": filename,
            "author": meta["author"],
            "title": meta["title"],
            "year": meta["year"],
            "pass": "1b",
            "chunk_idx": chunk_idx,
            "insight_idx": idx,
            **insight,
        }
        output_f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return True


# ---------------------------------------------------------------------------
# Merge & deduplicate pass 1a + 1b
# ---------------------------------------------------------------------------

def merge_and_deduplicate(output_dir: Path):
    """Merge insights_1a.jsonl + insights_1b.jsonl, deduplicate, write insights_raw.jsonl."""
    insights_1a_path = output_dir / "insights_1a.jsonl"
    insights_1b_path = output_dir / "insights_1b.jsonl"
    output_path = output_dir / "insights_raw.jsonl"

    # Load all insights grouped by book
    by_book = {}
    for path in [insights_1a_path, insights_1b_path]:
        if not path.exists():
            print(f"  Warning: {path} does not exist, skipping")
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    fn = record["book_filename"]
                    if fn not in by_book:
                        by_book[fn] = []
                    by_book[fn].append(record)

    total_before = sum(len(v) for v in by_book.values())
    print(f"Merging: {total_before} total insights across {len(by_book)} books")

    # Deduplicate per book
    total_after = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for fn in sorted(by_book.keys()):
            insights = by_book[fn]
            deduped = deduplicate_insights(insights)
            total_after += len(deduped)
            for record in deduped:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            if len(insights) != len(deduped):
                print(f"  [{fn}] {len(insights)} -> {len(deduped)} after dedup")

    print(f"Deduplication: {total_before} -> {total_after} insights")
    print(f"Output: {output_path}")


# ---------------------------------------------------------------------------
# Pass 2: Reasoning trace generation
# ---------------------------------------------------------------------------

async def run_pass_2(
    client: anthropic.AsyncAnthropic,
    output_dir: Path,
    model: str,
    semaphore: asyncio.Semaphore,
    resume: bool,
):
    """Generate reasoning traces from extracted insights."""
    insights_path = output_dir / "insights_raw.jsonl"
    traces_path = output_dir / "reasoning_traces.jsonl"

    if not insights_path.exists():
        print(f"Error: {insights_path} does not exist. Run passes 1a and 1b first.")
        return

    # Load insights
    insights = []
    with open(insights_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f):
            if line.strip():
                record = json.loads(line)
                record["_line_num"] = line_num
                insights.append(record)

    print(f"Loaded {len(insights)} insights from {insights_path}")

    # Resume state
    already_done = set()
    if resume and traces_path.exists():
        with open(traces_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    # Each output line is [{"role":"user",...}, {"role":"assistant",...}]
                    # We also write metadata lines, so check
                    try:
                        data = json.loads(line)
                        if isinstance(data, dict) and "_source_line" in data:
                            already_done.add(data["_source_line"])
                    except json.JSONDecodeError:
                        pass
        print(f"  Resume: {len(already_done)} traces already generated")

    to_process = [ins for ins in insights if ins["_line_num"] not in already_done]
    print(f"Pass 2: {len(to_process)} insights to process")

    if not to_process:
        return

    mode = "a" if resume else "w"
    t0 = time.time()
    success = 0
    failed = 0
    invalid = 0

    chunk_size = semaphore._value * 4
    with open(traces_path, mode, encoding="utf-8") as f:
        for batch_start in range(0, len(to_process), chunk_size):
            batch = to_process[batch_start:batch_start + chunk_size]
            tasks = [
                _generate_trace(client, insight, model, semaphore)
                for insight in batch
            ]

            results = await asyncio.gather(*tasks)

            for insight, result in zip(batch, results):
                if result is None:
                    failed += 1
                    continue

                if not validate_trace(result):
                    invalid += 1
                    continue

                # Write the training pair as a JSON array of messages (CustomJSON format)
                messages = [
                    {"role": "user", "content": result["user"]},
                    {"role": "assistant", "content": result["assistant"]},
                ]
                f.write(json.dumps(messages, ensure_ascii=False) + "\n")

                # Write source metadata on a separate line for resume tracking
                meta_record = {
                    "_source_line": insight["_line_num"],
                    "_book": insight["book_filename"],
                    "_pass": insight.get("pass", ""),
                }
                f.write(json.dumps(meta_record, ensure_ascii=False) + "\n")

                success += 1

            f.flush()
            total_done = batch_start + len(batch)
            elapsed = time.time() - t0
            rate = total_done / elapsed if elapsed > 0 else 0
            print(f"  Progress: {total_done}/{len(to_process)} ({rate:.1f}/s) | "
                  f"success={success} invalid={invalid} failed={failed}")

    elapsed = time.time() - t0
    print(f"\nPass 2 done in {elapsed:.0f}s: {success} traces, {invalid} invalid, {failed} failed")
    print(f"Output: {traces_path}")

    # Post-process: write a clean version without metadata lines
    clean_path = output_dir / "reasoning_traces_clean.jsonl"
    count = 0
    with open(traces_path, "r", encoding="utf-8") as fin, \
         open(clean_path, "w", encoding="utf-8") as fout:
        for line in fin:
            if line.strip():
                try:
                    data = json.loads(line)
                    if isinstance(data, list):
                        fout.write(line)
                        count += 1
                except json.JSONDecodeError:
                    pass
    print(f"Clean output: {clean_path} ({count} traces)")


async def _generate_trace(
    client: anthropic.AsyncAnthropic,
    insight: dict,
    model: str,
    semaphore: asyncio.Semaphore,
) -> dict | None:
    """Generate a single reasoning trace. Returns {"user": ..., "assistant": ...} or None."""
    year = insight.get("year", "unknown")
    if isinstance(year, int):
        if year < 1700:
            era = "seventeenth-century natural philosopher"
        elif year < 1800:
            era = "eighteenth-century natural philosopher"
        else:
            era = "nineteenth-century natural philosopher"
    else:
        era = "pre-1900 natural philosopher"

    prompt = PASS_2_PROMPT.format(
        era_description=era,
        title=insight.get("title", ""),
        author=insight.get("author", ""),
        year=year,
        setup=insight.get("setup", ""),
        insight=insight.get("insight", ""),
        excerpt=insight.get("excerpt", ""),
    )

    label = f"2:{insight.get('book_filename', '')}:{insight.get('insight_idx', '')}"

    messages = [{"role": "user", "content": [
        {"type": "text", "text": prompt, "cache_control": {"type": "ephemeral"}},
    ]}]

    response_text = await call_api(
        client, model, messages, max_tokens=4096,
        semaphore=semaphore, label=label,
    )

    if not response_text:
        return None

    data = parse_json_response(response_text)
    if not isinstance(data, dict):
        return None

    return data


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main_async(args):
    os.makedirs(args.output_dir, exist_ok=True)
    output_dir = Path(args.output_dir)
    books_dir = Path(args.books_dir)

    client = anthropic.AsyncAnthropic()

    only = args.only if args.only else None

    if args.run_pass in ("1a", "all"):
        print("=" * 60)
        print("PASS 1a: Whole-book insight extraction")
        print("=" * 60)
        sem = asyncio.Semaphore(args.max_concurrent_pass1)
        await run_pass_1a(
            client, books_dir, output_dir, args.pass1_model,
            sem, args.resume, only,
        )

    if args.run_pass in ("1b", "all"):
        print("=" * 60)
        print("PASS 1b: Chunk-level insight extraction")
        print("=" * 60)
        sem = asyncio.Semaphore(args.max_concurrent_chunks)
        await run_pass_1b(
            client, books_dir, output_dir, args.pass1_model,
            sem, args.chunk_tokens, args.resume, only,
        )

    if args.run_pass in ("1a", "1b", "all"):
        # Always merge after extraction passes
        print("=" * 60)
        print("Merging and deduplicating insights")
        print("=" * 60)
        merge_and_deduplicate(output_dir)

    if args.run_pass in ("2", "all"):
        print("=" * 60)
        print("PASS 2: Reasoning trace generation")
        print("=" * 60)
        sem = asyncio.Semaphore(args.max_concurrent_pass2)
        await run_pass_2(client, output_dir, args.pass2_model, sem, args.resume)


def main():
    parser = argparse.ArgumentParser(
        description="Generate reasoning traces from pre-1900 physics books"
    )
    parser.add_argument("--books-dir", type=str, default="data/physics_books",
                        help="Directory containing physics book .txt files")
    parser.add_argument("--output-dir", type=str, default="instruct_data",
                        help="Output directory for generated data")
    parser.add_argument("--run-pass", type=str, choices=["1a", "1b", "2", "all"], default="all",
                        help="Which pass to run")
    parser.add_argument("--pass1-model", type=str, default="claude-opus-4-6",
                        help="Model for insight extraction (passes 1a/1b)")
    parser.add_argument("--pass2-model", type=str, default="claude-sonnet-4-20250514",
                        help="Model for trace generation (pass 2)")
    parser.add_argument("--max-concurrent-pass1", type=int, default=5,
                        help="Max concurrent API requests for pass 1a (whole-book)")
    parser.add_argument("--max-concurrent-chunks", type=int, default=20,
                        help="Max concurrent API requests for pass 1b (chunks)")
    parser.add_argument("--max-concurrent-pass2", type=int, default=80,
                        help="Max concurrent API requests for pass 2 (traces)")
    parser.add_argument("--chunk-tokens", type=int, default=100_000,
                        help="Target chunk size in tokens for pass 1b")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing output files")
    parser.add_argument("--only", type=str, nargs="+",
                        help="Process only specific book filenames")
    args = parser.parse_args()

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
