#!/usr/bin/env python3
"""
Generate verifiable physics problems from pre-1900 textbooks.

Three-phase pipeline:
  Phase 1 — Extract/generate candidate problems with gold LaTeX answers from book chunks
  Phase 2 — Verify by re-solving (no gold visible) + SymPy cross-check → keep verified traces as SFT data
  Phase 3 — Split and format output files

Output files:
  - generated_problems_{train,val}.jsonl    — RL problems (id, domain, questions, gold_answers)
  - generated_prompts_sys_{train,val}.jsonl  — RL prompts (CustomJSON with system prompt)
  - generated_format_sft.jsonl               — verified SFT traces from Phase 2

Usage:
    python -m scripts.pre1900_scripts.generate_verifiable_problems \
        --books-dir data/physics_books \
        --output-dir instruct_data/generated_physics \
        --model claude-sonnet-4-20250514 \
        --max-concurrent 50 \
        --chunk-tokens 50000 \
        --max-books 200 \
        --val-fraction 0.05 \
        --seed 42 \
        --resume \
        --run phase1 phase2 phase3
"""

import json
import asyncio
import argparse
import os
import random
import re
import time
from pathlib import Path

import anthropic
from sympy.parsing.latex import parse_latex

from scripts.pre1900_scripts.collect_physics_books import CURATED_WORKS
from scripts.pre1900_scripts.constants import QUANTITATIVE_REASONING_SYSTEM_PROMPT
from scripts.pre1900_scripts.generate_rl_problems import (
    call_api,
    chunk_text,
    estimate_tokens,
    parse_json_response,
)
from tasks.yale_physics import extract_answer_latex, is_symbolically_equivalent

# ---------------------------------------------------------------------------
# Book selection: 1900-era teaching relevance
# ---------------------------------------------------------------------------

# Tier 1 curated works that are genuinely canonical (not hindsight-selected)
CANONICAL_CURATED_FILENAMES = {
    # Core textbooks and treatises every 1900 physicist knew
    "maxwell_treatise_em_full.txt",
    "maxwell_theory_of_heat.txt",
    "thomson_tait_natural_philosophy.txt",
    "clausius_mechanical_theory_of_heat.txt",
    "rayleigh_theory_of_sound.txt",
    "hertz_electric_waves.txt",
    "tyndall_heat_mode_of_motion.txt",
    "tyndall_sound.txt",
    "helmholtz_sensations_of_tone.txt",
    "helmholtz_conservation_of_force.txt",
    "gibbs_scientific_papers_vol1.txt",
    "gibbs_scientific_papers_vol2.txt",
    "newton_opticks.txt",
    "galileo_two_new_sciences.txt",
    "faraday_experimental_researches_electricity.txt",
    "carnot_motive_power_of_heat.txt",
    "gilbert_on_the_magnet.txt",
    "laplace_system_of_the_world.txt",
    "fourier_analytical_theory_heat.txt",
    "huygens_treatise_on_light.txt",
    "thomson_baltimore_lectures.txt",
    "kelvin_popular_lectures.txt",
    "stokes_mathematical_physical_papers.txt",
    "lorentz_theory_of_electrons.txt",
    "boltzmann_kinetic_theory_vol1.txt",
    "boltzmann_kinetic_theory_vol2.txt",
    "jj_thomson_corpuscular_theory.txt",
    "poynting_thomson_electricity_magnetism.txt",
    "lodge_modern_views_electricity.txt",
    "lamb_hydrodynamics.txt",
    "lamb_treatise_motion_fluids.txt",
    "mach_science_of_mechanics.txt",
    "preston_theory_of_light.txt",
    "thompson_elementary_lessons_electricity.txt",
    "stewart_conservation_of_energy.txt",
    "joule_scientific_papers.txt",
}

TEXTBOOK_INDICATORS = [
    "textbook", "text-book", "text book",
    "elementary", "beginners", "beginner",
    "school", "academy", "academies",
    "course of", "lessons", "lesson",
    "exercises", "problems", "examples",
    "compendium", "primer", "manual",
    "introduction to", "introductory",
    "class-book", "class book",
]

TREATISE_INDICATORS = [
    "treatise on", "elements of", "principles of", "theory of",
    "lectures on", "handbook of", "outlines of",
]

PHYSICS_DOMAIN_WORDS = [
    "physics", "electricity", "magnetism", "optics", "mechanics",
    "thermodynamics", "heat", "light", "sound", "dynamics",
    "statics", "hydrostatics", "hydrodynamics", "pneumatics",
    "acoustics", "natural philosophy", "energy", "force",
    "motion", "waves", "gravitation", "radiation",
    "electromagnetism", "electromagnetic",
]

NON_PHYSICS_EXCLUSIONS = [
    "dictionary", "catalog", "catalogue", "newspaper", "gazette",
    "fiction", "novel", "poetry", "sermon", "hymn", "prayer",
    "philosophy of mind", "metaphysics", "occult", "spiritualism",
    "astrology", "alchemy", "phrenology", "mesmerism",
    "political", "law", "jurisprudence", "medical", "surgery",
    "anatomy", "botany", "zoology", "agriculture",
]


def score_book_for_teaching(title: str, physics_score: int, downloads: int) -> float:
    """Score a book for 1900-era teaching relevance."""
    title_lower = title.lower()

    score = float(physics_score)

    # Textbook indicators (strong bonus)
    for indicator in TEXTBOOK_INDICATORS:
        if indicator in title_lower:
            score += 5
            break

    # Treatise + physics domain (moderate bonus)
    for indicator in TREATISE_INDICATORS:
        if indicator in title_lower:
            for domain in PHYSICS_DOMAIN_WORDS:
                if domain in title_lower:
                    score += 3
                    break
            break

    # Downloads as popularity proxy (log-scaled)
    if downloads and downloads > 0:
        import math
        score += math.log10(max(downloads, 1)) * 1.5

    return score


def is_excluded(title: str) -> bool:
    """Check if title matches non-physics exclusion patterns."""
    title_lower = title.lower()
    return any(excl in title_lower for excl in NON_PHYSICS_EXCLUSIONS)


def select_books(
    books_dir: Path,
    manifest_path: Path,
    max_books: int,
    min_size_kb: int = 50,
    min_ocr_quality: float = 0.8,
) -> list[dict]:
    """Select top books by 1900-era teaching relevance.

    Returns list of dicts with: filename, title, author, year, teaching_score
    """
    candidates = []

    # 1. Load IA-discovered books from manifest
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for entry in manifest:
            filename = entry["filename"]
            filepath = books_dir / filename
            if not filepath.exists():
                continue

            size_kb = entry.get("size_bytes", 0) / 1024
            if size_kb < min_size_kb:
                continue

            ocr = entry.get("ocr_quality", 0)
            if ocr < min_ocr_quality:
                continue

            title = entry.get("title", "")
            if is_excluded(title):
                continue

            teaching_score = score_book_for_teaching(
                title,
                entry.get("physics_score", 0),
                entry.get("downloads", 0),
            )

            candidates.append({
                "filename": filename,
                "title": title,
                "author": entry.get("creator", "Unknown"),
                "year": entry.get("year", "unknown"),
                "teaching_score": teaching_score,
                "source": "ia_discovery",
            })

    # 2. Add canonical curated works
    curated_by_fn = {w["filename"]: w for w in CURATED_WORKS}
    for filename in CANONICAL_CURATED_FILENAMES:
        if filename in curated_by_fn:
            meta = curated_by_fn[filename]
            filepath = books_dir / filename
            if not filepath.exists():
                continue

            size_kb = filepath.stat().st_size / 1024
            if size_kb < min_size_kb:
                continue

            # Canonical works get a bonus
            teaching_score = score_book_for_teaching(
                meta.get("title", ""),
                5,  # base physics score for canonical
                10000,  # high assumed popularity
            )
            teaching_score += 20  # canonical bonus

            # Skip if already in candidates (from IA manifest duplicate)
            if any(c["filename"] == filename for c in candidates):
                # Update score if canonical
                for c in candidates:
                    if c["filename"] == filename:
                        c["teaching_score"] = max(c["teaching_score"], teaching_score)
                        break
                continue

            candidates.append({
                "filename": filename,
                "title": meta.get("title", ""),
                "author": meta.get("author", "Unknown"),
                "year": meta.get("year", "unknown"),
                "teaching_score": teaching_score,
                "source": "curated",
            })

    # 3. Sort by teaching score, take top N
    candidates.sort(key=lambda x: -x["teaching_score"])
    selected = candidates[:max_books]

    print(f"Book selection: {len(candidates)} total candidates, selected top {len(selected)}")
    print(f"  IA-discovered: {sum(1 for c in selected if c['source'] == 'ia_discovery')}")
    print(f"  Curated canonical: {sum(1 for c in selected if c['source'] == 'curated')}")

    return selected


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

PROBLEM_EXTRACTION_PROMPT = """\
You are a physics professor creating quantitative practice problems from a pre-1900 physics textbook.

Book: "{title}" by {author} ({year})

Your task: analyze this section and create 1-5 quantitative physics problems inspired by the content. \
Each problem must have a definite numerical or symbolic answer that can be verified by computer algebra.

REQUIREMENTS for each problem:
1. **Quantitative only**: The answer must be a computable value — a number, fraction, algebraic expression, \
or formula. NO textual/conceptual answers.
2. **Self-contained**: The problem must include all necessary information. Do NOT reference "the text", \
"the passage", "the author", or "the above". A student should be able to solve it from the problem alone.
3. **SymPy-parseable answer**: The gold_answer must be valid LaTeX that SymPy can parse:
   - OK: "42", "\\frac{{3}}{{4}}", "\\sqrt{{2}}", "\\frac{{mv^2}}{{2}}", "\\pi r^2", "2\\pi\\sqrt{{\\frac{{l}}{{g}}}}"
   - NOT OK: "42 m/s" (no units), "approximately 3.14" (no text), "the speed increases" (not quantitative)
4. **Pre-1900 physics**: Only use concepts available before 1900.
5. **Clean OCR**: Fix any OCR artifacts in the source material when formulating the problem.

TYPES of problems to extract or generate:
- Worked examples from the text (reformulate as a problem with clean answer)
- Exercises or "examples" sections (common in 19th-century textbooks)
- Derivations that yield a clean formula (ask for the final result)
- Numerical calculations described in the text
- Problems inspired by the physics content (using the same concepts and methods)

For each problem provide:
- "question": The complete, self-contained problem statement
- "gold_answer": The answer as a SymPy-parseable LaTeX expression (NO units, NO text)
- "domain": Physics subdomain (mechanics, thermodynamics, optics, electromagnetism, acoustics, \
fluid_mechanics, waves, gravitation, kinetic_theory)
- "solution_sketch": Brief explanation of the solution method (1-2 sentences)

Return a JSON object: {{"problems": [...]}}
If this section has no suitable quantitative content, return {{"problems": []}}
Return ONLY valid JSON."""

# ---------------------------------------------------------------------------
# SymPy parse check
# ---------------------------------------------------------------------------


def _check_sympy_available() -> bool:
    """Check if SymPy LaTeX parsing is available (requires antlr4)."""
    try:
        parse_latex("1")
        return True
    except Exception:
        return False


_SYMPY_AVAILABLE = _check_sympy_available()

# Patterns that indicate non-parseable LaTeX (text, units, etc.)
_BAD_ANSWER_PATTERNS = [
    re.compile(r"^\s*$"),                    # empty
    re.compile(r"^[a-zA-Z\s]{10,}$"),        # pure text
    re.compile(r"\b(m/s|kg|cm|mm|cal|joule|watt|newton|ohm|volt|amp)\b", re.I),  # units
    re.compile(r"\b(approximately|about|roughly|nearly|around)\b", re.I),  # hedging
    re.compile(r"\b(increases|decreases|remains|the|is|are|it)\b", re.I),  # sentences
]

# Patterns that indicate valid LaTeX math
_GOOD_ANSWER_PATTERNS = [
    re.compile(r"^\d+(\.\d+)?$"),                  # plain number
    re.compile(r"\\frac\s*\{"),                     # fraction
    re.compile(r"\\sqrt"),                          # square root
    re.compile(r"\\pi"),                            # pi
    re.compile(r"\\(sin|cos|tan|log|ln|exp)\b"),    # trig/log
    re.compile(r"[a-zA-Z]\s*\^"),                   # exponent
    re.compile(r"\\(alpha|beta|gamma|theta|omega|lambda|mu|sigma|rho|tau|phi|epsilon|delta)\b"),
]


def sympy_parse_check(latex_str: str) -> bool:
    """Check if a LaTeX string is parseable by SymPy.

    Falls back to heuristic check if antlr4 is not installed.
    """
    if not latex_str or not latex_str.strip():
        return False

    s = latex_str.strip()

    # Reject obvious non-math
    for pat in _BAD_ANSWER_PATTERNS:
        if pat.search(s):
            return False

    # If SymPy parsing available, use it
    if _SYMPY_AVAILABLE:
        try:
            pp = s
            pp = re.sub(r"_\{.*?\}", "", pp)
            pp = re.sub(r"_\\?\w", "", pp)
            pp = pp.replace(r"\left", "").replace(r"\right", "")
            pp = pp.replace(r"\cdot", "*").replace(r"\times", "*")
            pp = pp.replace(r"\mathrm", "").replace(r"\text", "")
            if "=" in pp:
                pp = pp.split("=")[-1].strip()
            expr = parse_latex(pp)
            return expr is not None
        except Exception:
            return False

    # Heuristic fallback: accept if it looks like math
    # Accept plain numbers
    if re.match(r"^-?\d+(\.\d+)?$", s):
        return True

    # Accept if contains any good LaTeX pattern
    for pat in _GOOD_ANSWER_PATTERNS:
        if pat.search(s):
            return True

    # Accept short algebraic expressions (variables, operators)
    if re.match(r"^[a-zA-Z0-9\s\+\-\*\/\^\(\)\{\}\\,\.]+$", s) and len(s) < 200:
        return True

    return False


# ---------------------------------------------------------------------------
# Phase 1: Problem generation from book chunks
# ---------------------------------------------------------------------------


async def phase1_generate_candidates(
    client: anthropic.AsyncAnthropic,
    books_dir: Path,
    selected_books: list[dict],
    model: str,
    semaphore: asyncio.Semaphore,
    output_path: Path,
    chunk_tokens: int,
    resume: bool,
):
    """Phase 1: Extract/generate candidate problems from book chunks."""

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

    # Build chunk list
    all_chunks = []
    for book in selected_books:
        filename = book["filename"]
        book_path = books_dir / filename
        if not book_path.exists():
            continue

        text = book_path.read_text(encoding="utf-8")
        if estimate_tokens(text) < 500:
            continue

        chunks = chunk_text(text, target_tokens=chunk_tokens)
        for chunk_idx, chunk in enumerate(chunks):
            if (filename, chunk_idx) not in already_done:
                all_chunks.append({
                    "filename": filename,
                    "chunk_idx": chunk_idx,
                    "chunk": chunk,
                    "total_chunks": len(chunks),
                    "title": book["title"],
                    "author": book["author"],
                    "year": book["year"],
                })

    print(f"Phase 1: {len(all_chunks)} chunks to process from "
          f"{len(selected_books)} books ({len(already_done)} already done)")

    if not all_chunks:
        return

    mode = "a" if resume else "w"
    t0 = time.time()
    success = 0
    failed = 0
    total_problems = 0
    parse_rejected = 0

    batch_size = semaphore._value * 2
    with open(output_path, mode, encoding="utf-8") as f:
        for batch_start in range(0, len(all_chunks), batch_size):
            batch = all_chunks[batch_start:batch_start + batch_size]
            tasks = [
                _extract_problems_from_chunk(client, item, model, semaphore)
                for item in batch
            ]

            results = await asyncio.gather(*tasks)

            for item, problems in zip(batch, results):
                if problems is None:
                    failed += 1
                    continue

                for prob in problems:
                    gold = prob.get("gold_answer", "")
                    if not sympy_parse_check(gold):
                        parse_rejected += 1
                        continue

                    record = {
                        "book_filename": item["filename"],
                        "chunk_idx": item["chunk_idx"],
                        "title": item["title"],
                        "author": item["author"],
                        "year": item["year"],
                        "question": prob["question"],
                        "gold_answer": gold,
                        "domain": prob.get("domain", "physics"),
                        "solution_sketch": prob.get("solution_sketch", ""),
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    total_problems += 1

                success += 1

            f.flush()
            total_done = batch_start + len(batch)
            elapsed = time.time() - t0
            rate = total_done / elapsed if elapsed > 0 else 0
            print(f"  Progress: {total_done}/{len(all_chunks)} ({rate:.1f}/s) | "
                  f"chunks_ok={success} failed={failed} problems={total_problems} "
                  f"parse_rejected={parse_rejected}")

    elapsed = time.time() - t0
    print(f"\nPhase 1 done in {elapsed:.0f}s: {success} chunks, "
          f"{total_problems} problems, {parse_rejected} parse-rejected, {failed} failed")


async def _extract_problems_from_chunk(
    client: anthropic.AsyncAnthropic,
    item: dict,
    model: str,
    semaphore: asyncio.Semaphore,
) -> list[dict] | None:
    """Extract problems from a single book chunk."""
    label = f"gen:{item['filename']}:chunk{item['chunk_idx']}/{item['total_chunks']}"

    prompt = PROBLEM_EXTRACTION_PROMPT.format(
        title=item["title"],
        author=item["author"],
        year=item["year"],
    )

    messages = [{"role": "user", "content": [
        {"type": "text", "text": prompt, "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": f"--- SECTION (chunk {item['chunk_idx'] + 1}/"
                                 f"{item['total_chunks']}) ---\n\n{item['chunk']}"},
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

    problems = data.get("problems", [])
    if not isinstance(problems, list):
        return None

    # Validate each problem
    valid = []
    for p in problems:
        if not isinstance(p, dict):
            continue
        q = p.get("question", "")
        a = p.get("gold_answer", "")
        if isinstance(q, str) and len(q) >= 30 and isinstance(a, str) and len(a) >= 1:
            valid.append(p)

    tok = estimate_tokens(item["chunk"])
    print(f"  [{label}] ~{tok:,}tok -> {len(valid)}/{len(problems)} valid problems")
    return valid


# ---------------------------------------------------------------------------
# Phase 2: Independent verification + SFT trace generation
# ---------------------------------------------------------------------------


async def phase2_verify_candidates(
    client: anthropic.AsyncAnthropic,
    candidates_path: Path,
    model: str,
    semaphore: asyncio.Semaphore,
    output_path: Path,
    resume: bool,
):
    """Phase 2: Verify candidates by re-solving without gold, then SymPy-check."""

    # Load candidates
    candidates = []
    with open(candidates_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if line.strip():
                record = json.loads(line)
                record["_candidate_idx"] = i
                candidates.append(record)

    print(f"Loaded {len(candidates)} candidates from {candidates_path}")

    # Resume state
    already_done = set()
    if resume and output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        already_done.add(record.get("_candidate_idx", -1))
                    except json.JSONDecodeError:
                        pass
        print(f"  Resume: {len(already_done)} candidates already verified")

    to_process = [c for c in candidates if c["_candidate_idx"] not in already_done]
    print(f"Phase 2: {len(to_process)} candidates to verify "
          f"({len(already_done)} already done)")

    if not to_process:
        return

    mode = "a" if resume else "w"
    t0 = time.time()
    stats = {
        "verified": 0, "wrong": 0, "no_answer": 0,
        "parse_fail": 0, "api_fail": 0,
    }

    batch_size = semaphore._value * 2
    with open(output_path, mode, encoding="utf-8") as f:
        for batch_start in range(0, len(to_process), batch_size):
            batch = to_process[batch_start:batch_start + batch_size]
            tasks = [
                _verify_single(client, cand, model, semaphore)
                for cand in batch
            ]

            results = await asyncio.gather(*tasks)

            for cand, result in zip(batch, results):
                if result is None:
                    stats["api_fail"] += 1
                    continue

                status = result["status"]
                stats[status] = stats.get(status, 0) + 1

                # Keep all problems with a trace, regardless of match
                if result.get("trace"):
                    record = {
                        "_candidate_idx": cand["_candidate_idx"],
                        "book_filename": cand["book_filename"],
                        "title": cand["title"],
                        "author": cand["author"],
                        "year": cand["year"],
                        "question": cand["question"],
                        "gold_answer": cand["gold_answer"],
                        "domain": cand["domain"],
                        "trace": result["trace"],
                        "predicted_answer": result.get("predicted_answer", ""),
                        "verified": status == "verified",
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

            f.flush()
            total_done = batch_start + len(batch)
            elapsed = time.time() - t0
            rate = total_done / elapsed if elapsed > 0 else 0
            print(f"  Progress: {total_done}/{len(to_process)} ({rate:.1f}/s) | "
                  f"verified={stats['verified']} wrong={stats['wrong']} "
                  f"no_answer={stats['no_answer']} parse_fail={stats['parse_fail']} "
                  f"api_fail={stats['api_fail']}")

    elapsed = time.time() - t0
    total = sum(stats.values())
    print(f"\nPhase 2 done in {elapsed:.0f}s:")
    for k, v in stats.items():
        pct = 100 * v / max(total, 1)
        print(f"  {k}: {v} ({pct:.0f}%)")
    yield_pct = 100 * stats["verified"] / max(total, 1)
    print(f"  Verification yield: {stats['verified']}/{total} ({yield_pct:.0f}%)")


async def _verify_single(
    client: anthropic.AsyncAnthropic,
    candidate: dict,
    model: str,
    semaphore: asyncio.Semaphore,
) -> dict | None:
    """Verify a single candidate by re-solving without gold answer."""
    label = f"verify:{candidate['book_filename']}:{candidate['_candidate_idx']}"

    # Use system prompt to get proper <think> + \answer{} format
    messages = [
        {"role": "user", "content": candidate["question"]},
    ]

    # call_api doesn't support system param, so prepend as user context
    # Instead, use client directly with system prompt
    max_retries = 8
    response_text = None
    async with semaphore:
        for attempt in range(max_retries):
            try:
                response = await client.messages.create(
                    model=model,
                    max_tokens=4096,
                    system=QUANTITATIVE_REASONING_SYSTEM_PROMPT,
                    messages=messages,
                )
                response_text = response.content[0].text
                break
            except anthropic.RateLimitError:
                wait = min(2 ** attempt * 5, 120)
                print(f"  Rate limited on {label}, retrying in {wait}s "
                      f"(attempt {attempt+1}/{max_retries})")
                await asyncio.sleep(wait)
            except Exception as e:
                print(f"  Error on {label}: {e}")
                break
    if not response_text:
        return None

    # Extract predicted answer
    pred = extract_answer_latex(response_text)
    if pred is None:
        return {"status": "no_answer", "trace": response_text}

    # SymPy equivalence check
    gold = candidate["gold_answer"]
    equiv = is_symbolically_equivalent(pred, gold)

    if equiv is None:
        status = "parse_fail"
    elif equiv is False:
        status = "wrong"
    else:
        status = "verified"

    return {
        "status": status,
        "trace": response_text,
        "predicted_answer": pred,
    }


# ---------------------------------------------------------------------------
# Phase 3: Dedup, split, format output
# ---------------------------------------------------------------------------


def _classify_gold(gold: str) -> str:
    """Classify a gold answer as 'number', 'fraction', or 'symbolic'."""
    s = gold.strip()
    if re.match(r"^-?\d+(\.\d+)?$", s):
        return "number"
    if re.match(r"^-?\\?d?frac\s*\{-?\d+(\.\d+)?\}\s*\{-?\d+(\.\d+)?\}$", s):
        return "fraction"
    return "symbolic"


def _has_correct_sft_format(trace: str) -> bool:
    """Check if a trace has the correct <think>...</think>\\answer{...} format."""
    if not trace:
        return False
    has_think_open = "<think>" in trace
    has_think_close = "</think>" in trace
    has_answer = bool(re.search(r"\\answer\s*\{", trace))
    return has_think_open and has_think_close and has_answer


def phase3_format_output(
    verified_path: Path,
    output_dir: Path,
    val_fraction: float,
    seed: int,
    rl_count: int = 1000,
):
    """Phase 3: Split and format output files.

    RL split: rl_count verified problems, prioritized by answer type
              (number > fraction > symbolic) for most reliable reward signal.
    SFT split: all entries with correct <think>...</think>\\answer{} format.
    """

    # Load all entries
    all_entries = []
    with open(verified_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                all_entries.append(json.loads(line))

    print(f"Loaded {len(all_entries)} total entries")

    # Re-verify all entries with improved checker
    print("\nRe-verifying with improved equivalence checker...")
    for entry in all_entries:
        gold = entry.get("gold_answer", "")
        pred = entry.get("predicted_answer", "")
        if not pred:
            trace = entry.get("trace", "")
            pred = extract_answer_latex(trace) if trace else None
            if pred:
                entry["predicted_answer"] = pred

        if pred:
            result = is_symbolically_equivalent(pred, gold)
            entry["verified"] = result is True
        else:
            entry["verified"] = False

    verified_all = [e for e in all_entries if e.get("verified")]
    print(f"  Verified after re-check: {len(verified_all)}/{len(all_entries)} "
          f"({100*len(verified_all)/len(all_entries):.0f}%)")

    # --- RL split: verified problems, prioritized by gold answer type ---
    random.seed(seed)

    # Bucket verified by answer type
    verified_number = [e for e in verified_all if _classify_gold(e["gold_answer"]) == "number"]
    verified_fraction = [e for e in verified_all if _classify_gold(e["gold_answer"]) == "fraction"]
    verified_symbolic = [e for e in verified_all if _classify_gold(e["gold_answer"]) == "symbolic"]

    random.shuffle(verified_number)
    random.shuffle(verified_fraction)
    random.shuffle(verified_symbolic)

    print(f"\n  Verified by type: {len(verified_number)} number, "
          f"{len(verified_fraction)} fraction, {len(verified_symbolic)} symbolic")

    # Take in priority order, but reserve spots for symbolic
    symbolic_min = min(100, len(verified_symbolic))
    remaining_budget = rl_count - symbolic_min

    rl_set = []
    for bucket in [verified_number, verified_fraction]:
        take = min(len(bucket), remaining_budget - len(rl_set))
        rl_set.extend(bucket[:take])
        if len(rl_set) >= remaining_budget:
            break

    # Add symbolic (guaranteed minimum + any remaining slots)
    remaining = rl_count - len(rl_set)
    rl_set.extend(verified_symbolic[:remaining])

    random.shuffle(rl_set)

    n_num = sum(1 for e in rl_set if _classify_gold(e["gold_answer"]) == "number")
    n_frac = sum(1 for e in rl_set if _classify_gold(e["gold_answer"]) == "fraction")
    n_sym = sum(1 for e in rl_set if _classify_gold(e["gold_answer"]) == "symbolic")
    print(f"\nRL set: {len(rl_set)} verified problems "
          f"({n_num} number + {n_frac} fraction + {n_sym} symbolic)")

    # --- SFT split: entries with correct <think>...</think>\answer{} format ---
    sft_set = [e for e in all_entries if _has_correct_sft_format(e.get("trace", ""))]
    random.shuffle(sft_set)

    print(f"SFT set: {len(sft_set)} traces with correct format")

    # --- Stratified train/val for RL ---
    by_domain: dict[str, list[dict]] = {}
    for v in rl_set:
        d = v.get("domain", "physics")
        if d not in by_domain:
            by_domain[d] = []
        by_domain[d].append(v)

    rl_train = []
    rl_val = []
    for domain, items in by_domain.items():
        random.shuffle(items)
        n_val = max(1, int(len(items) * val_fraction))
        if len(items) <= 2:
            rl_train.extend(items)
        else:
            rl_val.extend(items[:n_val])
            rl_train.extend(items[n_val:])

    print(f"\nRL split: {len(rl_train)} train, {len(rl_val)} val")
    print(f"Domains: {sorted(by_domain.keys())}")
    for d in sorted(by_domain.keys()):
        n_train = sum(1 for t in rl_train if t.get("domain") == d)
        n_val = sum(1 for t in rl_val if t.get("domain") == d)
        print(f"  {d}: {n_train} train, {n_val} val")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Write output files
    _write_rl_problems(rl_train, rl_val, output_dir)
    _write_rl_prompts(rl_train, rl_val, output_dir)
    _write_sft_traces(sft_set, output_dir)


def _make_id(domain: str, idx: int) -> str:
    return f"gen_{domain}_{idx:04d}"


def _write_rl_problems(train_set: list[dict], val_set: list[dict], output_dir: Path):
    """Write RL problem files (id, domain, questions, gold_answers)."""
    for split_name, split_data in [("train", train_set), ("val", val_set)]:
        path = output_dir / f"generated_problems_{split_name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for idx, item in enumerate(split_data):
                record = {
                    "id": _make_id(item.get("domain", "physics"), idx),
                    "domain": item.get("domain", "physics"),
                    "questions": item["question"],
                    "gold_answers": [item["gold_answer"]],
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"  RL problems: {path} ({len(split_data)} records)")


def _write_rl_prompts(train_set: list[dict], val_set: list[dict], output_dir: Path):
    """Write RL prompts in CustomJSON format (system + user + empty assistant)."""
    system_message = {"role": "system", "content": QUANTITATIVE_REASONING_SYSTEM_PROMPT}
    for split_name, split_data in [("train", train_set), ("val", val_set)]:
        path = output_dir / f"generated_prompts_sys_{split_name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for item in split_data:
                messages = [
                    system_message,
                    {"role": "user", "content": item["question"]},
                    {"role": "assistant", "content": ""},
                ]
                f.write(json.dumps(messages, ensure_ascii=False) + "\n")
        print(f"  RL prompts: {path} ({len(split_data)} records)")


def _write_sft_traces(sft_set: list[dict], output_dir: Path):
    """Write SFT traces with correct \\answer{} formatting."""
    system_message = {"role": "system", "content": QUANTITATIVE_REASONING_SYSTEM_PROMPT}

    all_with_traces = [item for item in sft_set if item.get("trace")]

    if not all_with_traces:
        print("  No SFT traces to write (run Phase 2 first)")
        return

    path = output_dir / "generated_format_sft.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for item in all_with_traces:
            messages = [
                system_message,
                {"role": "user", "content": item["question"]},
                {"role": "assistant", "content": item["trace"]},
            ]
            f.write(json.dumps(messages, ensure_ascii=False) + "\n")
    print(f"  SFT traces: {path} ({len(all_with_traces)} records)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Bedrock model+region cascade (no Anthropic API fallback for Phase 2)
# ---------------------------------------------------------------------------

_BEDROCK_REGIONS = ["us-east-1", "us-west-2", "us-east-2"]

# Model cascade: priority order. Each model has separate rate limits.
_MODEL_CASCADE = [
    {"bedrock_id": "us.anthropic.claude-opus-4-6-v1",              "regions": ["us-east-1", "us-west-2", "us-east-2"]},
    {"bedrock_id": "us.anthropic.claude-opus-4-5-20251101-v1:0",   "regions": ["us-east-1", "us-west-2", "us-east-2"]},
    {"bedrock_id": "us.anthropic.claude-opus-4-1-20250805-v1:0",   "regions": ["us-east-1", "us-west-2", "us-east-2"]},
    {"bedrock_id": "us.anthropic.claude-sonnet-4-6",               "regions": ["us-east-1", "us-west-2", "us-east-2"]},
    {"bedrock_id": "us.anthropic.claude-sonnet-4-5-20250929-v1:0", "regions": ["us-east-1", "us-west-2", "us-east-2"]},
    {"bedrock_id": "us.anthropic.claude-sonnet-4-20250514-v1:0",   "regions": ["us-east-1", "us-west-2", "us-east-2"]},
]

# Anthropic model ID → default Bedrock model ID (for Phase 1 compat)
_ANTHROPIC_TO_BEDROCK = {
    "claude-sonnet-4-20250514": "us.anthropic.claude-sonnet-4-20250514-v1:0",
    "claude-opus-4-6": "us.anthropic.claude-opus-4-6-v1",
    "claude-opus-4-20250514": "us.anthropic.claude-opus-4-20250514-v1:0",
    "claude-haiku-4-5-20251001": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
}

_BEDROCK_TO_ANTHROPIC = {v: k for k, v in _ANTHROPIC_TO_BEDROCK.items()}


class _MessagesProxy:
    """Proxy for client.messages.create() with model+region cascade."""

    def __init__(self, parent: "FallbackClient"):
        self._parent = parent

    async def create(self, **kwargs):
        p = self._parent
        model = kwargs.get("model", "")
        bedrock_model = _ANTHROPIC_TO_BEDROCK.get(model, model)

        if not p._cascade:
            # Simple mode: single set of region clients
            last_err = None
            for _ in range(len(p._region_clients)):
                idx = p._current_region_idx
                client = p._region_clients[idx]
                region = p._regions[idx]
                try:
                    kwargs["model"] = bedrock_model
                    result = await client.messages.create(**kwargs)
                    return result
                except anthropic.RateLimitError as e:
                    last_err = e
                    p._current_region_idx = (idx + 1) % len(p._region_clients)
                    print(f"  Rate limited on {region}, rotating to {p._regions[p._current_region_idx]}")
                except Exception as e:
                    last_err = e
                    p._current_region_idx = (idx + 1) % len(p._region_clients)
                    print(f"  Error on {region}, rotating to {p._regions[p._current_region_idx]}")

            # Fall back to Anthropic API if available
            if p._anthropic_client is not None:
                print(f"  All Bedrock regions exhausted, falling back to Anthropic API")
                anthropic_model = _BEDROCK_TO_ANTHROPIC.get(bedrock_model, model)
                kwargs["model"] = anthropic_model
                return await p._anthropic_client.messages.create(**kwargs)
            raise last_err

        # Cascade mode: try each model across its regions
        last_err = None
        for tier in p._cascade:
            tier_model = tier["bedrock_id"]
            clients = tier["clients"]
            regions = tier["regions"]
            region_idx = tier["_idx"]

            for _ in range(len(clients)):
                client = clients[region_idx]
                region = regions[region_idx]
                try:
                    kwargs["model"] = tier_model
                    result = await client.messages.create(**kwargs)
                    # Success — remember this region for next call
                    tier["_idx"] = region_idx
                    return result
                except anthropic.RateLimitError as e:
                    last_err = e
                    region_idx = (region_idx + 1) % len(clients)
                    tier["_idx"] = region_idx
                except Exception as e:
                    last_err = e
                    region_idx = (region_idx + 1) % len(clients)
                    tier["_idx"] = region_idx

            # All regions exhausted for this model — cascade to next
            short_name = tier_model.split("anthropic.")[-1]
            print(f"  All regions exhausted for {short_name}, cascading to next model")

        # Everything exhausted
        raise last_err


class FallbackClient:
    """Client wrapper: model+region cascade for Bedrock, optional Anthropic API fallback."""

    def __init__(
        self,
        regions: list[str] | None = None,
        anthropic_api_key: str | None = None,
        bedrock: bool = True,
        cascade: bool = False,
    ):
        self._cascade = None
        self._region_clients = []
        self._regions = []
        self._current_region_idx = 0

        if bedrock and cascade:
            # Build cascade: each model tier gets clients for its regions
            self._cascade = []
            for tier in _MODEL_CASCADE:
                tier_regions = regions or tier["regions"]
                clients = [anthropic.AsyncAnthropicBedrock(aws_region=r) for r in tier_regions]
                self._cascade.append({
                    "bedrock_id": tier["bedrock_id"],
                    "clients": clients,
                    "regions": tier_regions,
                    "_idx": 0,
                })
        elif bedrock:
            # Simple region rotation (for Phase 1)
            for region in (regions or _BEDROCK_REGIONS[:1]):
                self._region_clients.append(
                    anthropic.AsyncAnthropicBedrock(aws_region=region)
                )
                self._regions.append(region)

        # Optional Anthropic API client (last-resort fallback)
        self._anthropic_client = None
        if anthropic_api_key:
            self._anthropic_client = anthropic.AsyncAnthropic(api_key=anthropic_api_key)
        elif not bedrock:
            self._anthropic_client = anthropic.AsyncAnthropic()

        self.messages = _MessagesProxy(self)


def _make_client(bedrock: bool, region: str | None = None, anthropic_api_key: str | None = None,
                 cascade: bool = False):
    """Create the appropriate async Anthropic client."""
    if bedrock:
        regions = region.split(",") if region else _BEDROCK_REGIONS
        return FallbackClient(
            regions=regions,
            anthropic_api_key=anthropic_api_key,
            bedrock=True,
            cascade=cascade,
        )
    return FallbackClient(bedrock=False, anthropic_api_key=anthropic_api_key)


async def main_async(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    books_dir = Path(args.books_dir)
    manifest_path = books_dir / "ia_discovery_manifest.json"

    phases = args.run
    if "all" in phases:
        phases = ["phase1", "phase2", "phase3"]

    api_key = args.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")
    semaphore = asyncio.Semaphore(args.max_concurrent)

    candidates_path = output_dir / "candidates_raw.jsonl"
    verified_path = output_dir / "verified_raw.jsonl"

    # Select books (needed for phase1)
    if "phase1" in phases:
        # Phase 1: Bedrock with Anthropic API fallback allowed
        client = _make_client(args.bedrock, args.aws_region, api_key)

        print("=" * 60)
        print("BOOK SELECTION")
        print("=" * 60)
        selected_books = select_books(
            books_dir, manifest_path,
            max_books=args.max_books,
        )
        for i, b in enumerate(selected_books[:20]):
            print(f"  {i+1:>3}. [{b['teaching_score']:.1f}] {b['title'][:70]}")
        if len(selected_books) > 20:
            print(f"  ... and {len(selected_books) - 20} more")

        print("\n" + "=" * 60)
        print(f"PHASE 1: Generate candidate problems ({args.phase1_model})")
        print("=" * 60)
        await phase1_generate_candidates(
            client, books_dir, selected_books, args.phase1_model,
            semaphore, candidates_path, args.chunk_tokens, args.resume,
        )

    if "phase2" in phases:
        # Phase 2: Bedrock only — model+region cascade, no Anthropic API fallback
        client = _make_client(args.bedrock, args.aws_region, anthropic_api_key=None, cascade=True)

        print("\n" + "=" * 60)
        print(f"PHASE 2: Verify candidates ({args.phase2_model})")
        print("=" * 60)

        if not candidates_path.exists():
            print(f"Error: {candidates_path} does not exist. Run phase1 first.")
            return

        await phase2_verify_candidates(
            client, candidates_path, args.phase2_model,
            semaphore, verified_path, args.resume,
        )

    if "phase3" in phases:
        print("\n" + "=" * 60)
        print("PHASE 3: Split and format output")
        print("=" * 60)

        if not verified_path.exists():
            print(f"Error: {verified_path} does not exist. Run phase2 first.")
            return

        phase3_format_output(
            verified_path, output_dir,
            val_fraction=args.val_fraction,
            seed=args.seed,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Generate verifiable physics problems from pre-1900 textbooks"
    )
    parser.add_argument("--books-dir", type=str, default="data/physics_books",
                        help="Directory containing physics book .txt files")
    parser.add_argument("--output-dir", type=str,
                        default="instruct_data/generated_physics",
                        help="Output directory")
    parser.add_argument("--phase1-model", type=str,
                        default="claude-sonnet-4-20250514",
                        help="Model for Phase 1 (problem generation)")
    parser.add_argument("--phase2-model", type=str,
                        default="claude-opus-4-20250514",
                        help="Model for Phase 2 (verification)")
    parser.add_argument("--bedrock", action="store_true",
                        help="Use AWS Bedrock instead of Anthropic API")
    parser.add_argument("--aws-region", type=str, default=None,
                        help="AWS region for Bedrock (default: from env)")
    parser.add_argument("--anthropic-api-key", type=str, default=None,
                        help="Anthropic API key for fallback (or set ANTHROPIC_API_KEY env var)")
    parser.add_argument("--max-concurrent", type=int, default=50,
                        help="Max concurrent API requests")
    parser.add_argument("--chunk-tokens", type=int, default=50_000,
                        help="Target chunk size in tokens")
    parser.add_argument("--max-books", type=int, default=200,
                        help="Max books to process")
    parser.add_argument("--val-fraction", type=float, default=0.05,
                        help="Fraction for validation split")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing output")
    parser.add_argument("--run", type=str, nargs="+",
                        default=["all"],
                        choices=["phase1", "phase2", "phase3", "all"],
                        help="Which phases to run")
    args = parser.parse_args()

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
