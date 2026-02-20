"""
LLM-based OCR cleanup for pre-1900 training data.

Uses edit-based prompting: model outputs ONLY corrections (original >>> corrected),
not full rewrites. This reduces output tokens from ~96B to ~2-4B, making the
96B token corpus feasible to process in ~11 hours on 8xH100.

Usage:
    python scripts/pre1900_scripts/llm_clean.py \
        --input /mnt/main0/data/michaelhla/pre1900_raw \
        --output /mnt/main0/data/michaelhla/pre1900_llm_cleaned \
        --model meta-llama/Llama-3.2-3B-Instruct \
        --num-gpus 8 --newspapers-first
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import multiprocessing as mp
import os
import random
import re
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

# ============================================================================
# Constants
# ============================================================================

SYSTEM_PROMPT = """\
You are an OCR correction specialist for historical text (pre-1900).
Output ONLY corrections, one per line:  original text >>> corrected text
Rules:
- Fix broken words: "blow ing" >>> "blowing"
- Fix character substitutions: "reaIy" >>> "really"
- Fix garbled OCR words using context
- Modernize long-s: "ieveral" >>> "several"
- Remove headers/page numbers: "CHAPTER II." >>> [REMOVE]
- Do NOT change period-appropriate grammar or vocabulary
If no errors, output only: CLEAN"""

REMOVE_MARKER = "[REMOVE]"
CHARS_PER_TOKEN = 4  # conservative estimate for LLaMA on English text
MAX_LENGTH_RATIO = 2.0
MIN_LENGTH_RATIO = 0.5
SOURCE_ORDER = ["newspapers", "institutional", "blbooks", "books"]


# ============================================================================
# Text Chunking
# ============================================================================


def chunk_text(text: str, max_chars: int) -> list[str]:
    """Split text into chunks at paragraph boundaries, respecting max_chars.

    Strategy:
    1. Split on double newlines (paragraph boundaries)
    2. Accumulate paragraphs until hitting max_chars
    3. If a single paragraph exceeds max_chars, split at sentence boundaries
    """
    if len(text) <= max_chars:
        return [text]

    paragraphs = re.split(r"\n\n+", text)
    chunks = []
    current = ""

    for para in paragraphs:
        if current and len(current) + 2 + len(para) > max_chars:
            chunks.append(current)
            current = ""

        if len(para) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            sentences = re.split(r"(?<=[.!?])\s+", para)
            buf = ""
            for sent in sentences:
                if buf and len(buf) + 1 + len(sent) > max_chars:
                    chunks.append(buf)
                    buf = ""
                buf = sent if not buf else buf + " " + sent
            if buf:
                if chunks and len(chunks[-1]) + 2 + len(buf) <= max_chars:
                    chunks[-1] += "\n\n" + buf
                else:
                    chunks.append(buf)
        else:
            current = para if not current else current + "\n\n" + para

    if current:
        chunks.append(current)

    return chunks


# ============================================================================
# Correction Parsing and Application
# ============================================================================


def parse_corrections(output_text: str) -> list[tuple[str, str]]:
    """Parse edit-based model output into (original, replacement) pairs.

    Returns empty list for "CLEAN" output or if all lines are unparseable.
    Skips individual unparseable lines silently.
    """
    stripped = output_text.strip()
    if not stripped or stripped.upper() == "CLEAN":
        return []

    corrections = []
    for line in stripped.split("\n"):
        line = line.strip()
        if not line or ">>>" not in line:
            continue

        parts = line.split(">>>", 1)
        if len(parts) != 2:
            continue

        original = parts[0].strip()
        replacement = parts[1].strip()

        if not original or not replacement:
            continue

        # Length ratio validation (skip for REMOVE markers)
        if replacement != REMOVE_MARKER and len(original) > 0:
            ratio = len(replacement) / len(original)
            if ratio > MAX_LENGTH_RATIO or ratio < MIN_LENGTH_RATIO:
                continue

        corrections.append((original, replacement))

    return corrections


def apply_corrections(
    text: str, corrections: list[tuple[str, str]]
) -> tuple[str, int, int]:
    """Apply corrections to text using multi-level matching.

    Levels: exact match (replace all) -> whitespace-normalized -> case-insensitive.
    Returns (cleaned_text, n_applied, n_failed).
    """
    n_applied = 0
    n_failed = 0

    for original, replacement in corrections:
        target = "" if replacement == REMOVE_MARKER else replacement

        # Level 1: Exact match (replace all occurrences — OCR errors are systematic)
        if original in text:
            text = text.replace(original, target)
            n_applied += 1
            continue

        # Level 2: Whitespace-normalized match
        words = original.split()
        if len(words) > 1:
            pattern = r"\s+".join(re.escape(w) for w in words)
            match = re.search(pattern, text)
            if match:
                text = text[: match.start()] + target + text[match.end() :]
                n_applied += 1
                continue

        # Level 3: Case-insensitive match
        match = re.search(re.escape(original), text, re.IGNORECASE)
        if match:
            text = text[: match.start()] + target + text[match.end() :]
            n_applied += 1
            continue

        n_failed += 1

    return text, n_applied, n_failed


# ============================================================================
# Shard Discovery
# ============================================================================


def discover_shards(input_dir: Path, newspapers_first: bool) -> list[Path]:
    """Discover all parquet shards, ordered by source priority.

    Newspapers processed first (files already year-ordered 1774->1899),
    then institutional, blbooks, books.
    """
    shards_by_source: dict[str, list[Path]] = {}

    for source in SOURCE_ORDER:
        source_dir = input_dir / source
        if source_dir.is_dir():
            files = sorted(source_dir.glob("*.parquet"))
            if files:
                shards_by_source[source] = files

    order = SOURCE_ORDER if newspapers_first else sorted(shards_by_source.keys())

    all_shards = []
    for source in order:
        if source in shards_by_source:
            all_shards.extend(shards_by_source[source])

    return all_shards


# ============================================================================
# Shard Processing
# ============================================================================


def process_shard(
    input_path: Path,
    output_path: Path,
    llm,
    sampling_params,
    max_chunk_chars: int,
    corrections_log: Path,
    dry_run: bool,
    log: logging.Logger,
) -> dict:
    """Process a single shard: read, chunk, clean via LLM, apply corrections, write.

    Returns stats dict with n_docs, n_chunks, n_corrections, n_applied, n_failed.
    """
    source = input_path.parent.name
    table = pq.read_table(input_path)
    texts = table.column("text").to_pylist()

    # Build all chunks with provenance tracking
    all_chunks: list[tuple[int, int, str]] = []  # (doc_idx, chunk_idx, text)
    doc_chunk_counts: list[int] = []

    for doc_idx, text in enumerate(texts):
        text = text or ""
        chunks = chunk_text(text, max_chunk_chars)
        doc_chunk_counts.append(len(chunks))
        for chunk_idx, chunk in enumerate(chunks):
            all_chunks.append((doc_idx, chunk_idx, chunk))

    # Build conversations for vLLM
    conversations = [
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": chunk},
        ]
        for _, _, chunk in all_chunks
    ]

    # Run inference
    outputs = llm.chat(conversations, sampling_params=sampling_params, use_tqdm=False)

    # Parse corrections per chunk
    chunk_corrections: dict[tuple[int, int], list[tuple[str, str]]] = {}
    shard_correction_records: list[dict] = []

    for (doc_idx, chunk_idx, _), output in zip(all_chunks, outputs):
        raw = output.outputs[0].text
        corrections = parse_corrections(raw)
        chunk_corrections[(doc_idx, chunk_idx)] = corrections

        for orig, repl in corrections:
            shard_correction_records.append({
                "shard": input_path.name,
                "source": source,
                "doc_idx": doc_idx,
                "chunk_idx": chunk_idx,
                "original": orig,
                "replacement": repl,
            })

    # Apply corrections and reassemble documents
    cleaned_texts = []
    total_applied = 0
    total_failed = 0

    for doc_idx in range(len(texts)):
        text = texts[doc_idx] or ""
        n_chunks = doc_chunk_counts[doc_idx]

        if n_chunks == 1:
            corrections = chunk_corrections.get((doc_idx, 0), [])
            if corrections:
                text, n_ok, n_fail = apply_corrections(text, corrections)
                total_applied += n_ok
                total_failed += n_fail
        else:
            # Re-chunk deterministically, apply per chunk, reassemble
            chunks = chunk_text(text, max_chunk_chars)
            cleaned_parts = []
            for chunk_idx, chunk in enumerate(chunks):
                corrections = chunk_corrections.get((doc_idx, chunk_idx), [])
                if corrections:
                    chunk, n_ok, n_fail = apply_corrections(chunk, corrections)
                    total_applied += n_ok
                    total_failed += n_fail
                cleaned_parts.append(chunk)
            text = "\n\n".join(cleaned_parts)

        cleaned_texts.append(text)

    # Print random before/after sample
    changed = [i for i in range(len(texts)) if (texts[i] or "") != cleaned_texts[i]]
    if changed:
        idx = random.choice(changed)
        log.info(f"  Sample doc {idx} BEFORE: {(texts[idx] or '')[:200]}")
        log.info(f"  Sample doc {idx} AFTER:  {cleaned_texts[idx][:200]}")

    stats = {
        "n_docs": len(texts),
        "n_chunks": len(all_chunks),
        "n_corrections": len(shard_correction_records),
        "n_applied": total_applied,
        "n_failed": total_failed,
    }

    if dry_run:
        for c in shard_correction_records[:20]:
            log.info(f"  {c['original']} >>> {c['replacement']}")
        if len(shard_correction_records) > 20:
            log.info(f"  ... and {len(shard_correction_records) - 20} more")
        return stats

    # Write corrections audit log
    if shard_correction_records:
        with open(corrections_log, "a") as f:
            for c in shard_correction_records:
                f.write(json.dumps(c) + "\n")

    # Atomic write: .tmp -> rename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(".tmp")
    col_idx = table.schema.get_field_index("text")
    new_table = table.set_column(col_idx, "text", pa.array(cleaned_texts))
    pq.write_table(new_table, tmp_path, row_group_size=500, compression="snappy")
    tmp_path.rename(output_path)

    return stats


# ============================================================================
# GPU Worker
# ============================================================================


def gpu_worker(
    gpu_id: int,
    shard_io_pairs: list[tuple[str, str]],  # (input_str, output_str) for pickling
    model_name: str,
    max_chunk_chars: int,
    temperature: float,
    log_dir_str: str,
    dry_run: bool,
):
    """Process assigned shards on a single GPU.

    Runs in a spawned subprocess with CUDA_VISIBLE_DEVICES pinned.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Import vLLM after setting CUDA_VISIBLE_DEVICES so it only sees one GPU
    from vllm import LLM, SamplingParams

    logging.basicConfig(
        format=f"[GPU {gpu_id}] %(asctime)s %(message)s",
        level=logging.INFO,
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger(f"gpu{gpu_id}")

    shard_paths = [(Path(inp), Path(out)) for inp, out in shard_io_pairs]
    log_dir = Path(log_dir_str)
    log_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Starting: {len(shard_paths)} shards assigned")

    # max_model_len = max_input_tokens + output_buffer + system_prompt_buffer
    max_input_tokens = max_chunk_chars // CHARS_PER_TOKEN
    max_model_len = max_input_tokens + 2048 + 512

    llm = LLM(
        model=model_name,
        tensor_parallel_size=1,
        max_model_len=max_model_len,
        gpu_memory_utilization=0.90,
        enable_prefix_caching=True,
    )
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=2048,
        stop=["<|eot_id|>"],
    )

    corrections_log = log_dir / f"corrections_gpu{gpu_id}.jsonl"

    for shard_idx, (input_path, output_path) in enumerate(shard_paths):
        t0 = time.time()
        source = input_path.parent.name
        log.info(
            f"Shard {shard_idx + 1}/{len(shard_paths)}: {source}/{input_path.name}"
        )

        try:
            stats = process_shard(
                input_path,
                output_path,
                llm,
                sampling_params,
                max_chunk_chars,
                corrections_log,
                dry_run,
                log,
            )
        except Exception:
            log.exception(f"Failed on {input_path.name}")
            continue

        elapsed = time.time() - t0
        log.info(
            f"Shard {shard_idx + 1}/{len(shard_paths)} done: "
            f"{stats['n_docs']} docs, {stats['n_chunks']} chunks, "
            f"{stats['n_corrections']} corrections "
            f"({stats['n_applied']} applied, {stats['n_failed']} failed), "
            f"{elapsed:.1f}s"
        )
        gc.collect()

    log.info("All shards done")


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="LLM-based OCR cleanup for pre-1900 training data (edit-based prompting)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=Path("/mnt/main0/data/michaelhla/pre1900_raw"),
        help="Input directory with source subdirectories",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("/mnt/main0/data/michaelhla/pre1900_llm_cleaned"),
        help="Output directory",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.2-3B-Instruct",
        help="HuggingFace model name",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=8,
        help="Number of GPUs to use",
    )
    parser.add_argument(
        "--newspapers-first",
        action="store_true",
        help="Process newspapers first (year-ordered), then other sources",
    )
    parser.add_argument(
        "--max-chunk-tokens",
        type=int,
        default=16384,
        help="Max input tokens per chunk",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--dry-run",
        type=int,
        default=0,
        metavar="N",
        help="Process only N shards, print corrections, don't write output",
    )
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s %(message)s",
        level=logging.INFO,
        datefmt="%H:%M:%S",
    )

    # Discover shards
    all_shards = discover_shards(args.input, args.newspapers_first)
    if not all_shards:
        logger.error(f"No parquet files found in {args.input}")
        return

    source_counts: dict[str, int] = {}
    for s in all_shards:
        src = s.parent.name
        source_counts[src] = source_counts.get(src, 0) + 1
    for src, count in source_counts.items():
        logger.info(f"  {src}: {count} shards")
    logger.info(f"Total: {len(all_shards)} shards")

    # Build (input, output) pairs preserving source subdirectory structure
    pairs = []
    for shard_path in all_shards:
        rel = shard_path.relative_to(args.input)
        output_path = args.output / rel
        pairs.append((shard_path, output_path))

    # Resume: skip completed shards
    pending = [(inp, out) for inp, out in pairs if not out.exists()]
    n_skipped = len(pairs) - len(pending)
    if n_skipped > 0:
        logger.info(f"Resuming: {n_skipped} shards done, {len(pending)} remaining")

    if not pending:
        logger.info("All shards already processed!")
        return

    if args.dry_run > 0:
        pending = pending[: args.dry_run]
        logger.info(f"Dry run: processing {len(pending)} shards only")

    # Distribute round-robin across GPUs
    max_chunk_chars = args.max_chunk_tokens * CHARS_PER_TOKEN
    num_gpus = min(args.num_gpus, len(pending))
    gpu_assignments: list[list[tuple[str, str]]] = [[] for _ in range(num_gpus)]
    for i, (inp, out) in enumerate(pending):
        gpu_assignments[i % num_gpus].append((str(inp), str(out)))

    for gpu_id in range(num_gpus):
        logger.info(f"GPU {gpu_id}: {len(gpu_assignments[gpu_id])} shards")

    log_dir = args.output / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Launch one worker process per GPU
    t0 = time.time()
    ctx = mp.get_context("spawn")
    processes = []
    for gpu_id in range(num_gpus):
        if not gpu_assignments[gpu_id]:
            continue
        p = ctx.Process(
            target=gpu_worker,
            args=(
                gpu_id,
                gpu_assignments[gpu_id],
                args.model,
                max_chunk_chars,
                args.temperature,
                str(log_dir),
                args.dry_run > 0,
            ),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    elapsed = time.time() - t0
    logger.info(f"All done in {elapsed:.1f}s ({elapsed / 3600:.2f}h)")

    # Verify output count
    if args.dry_run == 0:
        n_output = sum(1 for _ in args.output.rglob("*.parquet"))
        logger.info(f"Input shards: {len(all_shards)}, Output shards: {n_output}")
        if n_output < len(all_shards):
            logger.warning(f"Missing {len(all_shards) - n_output} output shards!")


if __name__ == "__main__":
    main()
