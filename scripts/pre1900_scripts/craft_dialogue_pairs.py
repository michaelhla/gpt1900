#!/usr/bin/env python3
"""
Generate synthetic multi-turn dialogues from pre-1900 corpus excerpts.

Samples passages from parquet shards, sends them to Claude to craft 4-8 turn
dialogues between two named period characters, then post-processes into the
customjson.py JSONL format (user/assistant alternation).

Usage:
    python -m scripts.pre1900_scripts.craft_dialogue_pairs \
        --data-dir /opt/dlami/nvme/gpt1905_training/base_data \
        --output-dir instruct_data/ \
        --num-samples 100000 \
        --max-concurrent 100
"""

import os
import re
import json
import random
import asyncio
import argparse
import time

import anthropic

# ---------------------------------------------------------------------------
# Bedrock client with region cycling (inlined from discovery_rl.py to avoid
# importing that module's heavy top-level code)
# ---------------------------------------------------------------------------

_BEDROCK_REGIONS = ["us-east-1", "us-west-2", "us-east-2"]

_ANTHROPIC_TO_BEDROCK = {
    "claude-sonnet-4-20250514": "us.anthropic.claude-sonnet-4-20250514-v1:0",
    "claude-opus-4-6": "us.anthropic.claude-opus-4-6-v1",
    "claude-opus-4-20250514": "us.anthropic.claude-opus-4-20250514-v1:0",
    "claude-haiku-4-5-20251001": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
}


class _MessagesProxy:
    """Proxy that intercepts messages.create() and cycles regions on rate limits."""
    def __init__(self, parent):
        self._parent = parent

    async def create(self, **kwargs):
        import asyncio as _asyncio
        p = self._parent
        model = kwargs.pop("model", None)
        bedrock_model = _ANTHROPIC_TO_BEDROCK.get(model, model)
        last_err = None

        # Try Bedrock regions with retries (3 rounds, backoff between rounds)
        for attempt in range(3):
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
                    print(f"  Error on {region}: {e}, rotating to {p._regions[p._current_region_idx]}")
            # Back off before next round of retries
            if attempt < 2:
                delay = 2 ** attempt  # 1s, 2s
                await _asyncio.sleep(delay)

        # All Bedrock regions failed after retries — fall back to Anthropic API
        if not p._anthropic_fallback_logged:
            print("  All Bedrock regions exhausted after retries, falling back to Anthropic API")
            p._anthropic_fallback_logged = True
        for api_attempt in range(5):
            try:
                kwargs["model"] = model  # Use original Anthropic model ID
                result = await p._anthropic_client.messages.create(**kwargs)
                return result
            except anthropic.RateLimitError:
                delay = 2 ** api_attempt * 2  # 2, 4, 8, 16, 32s
                await _asyncio.sleep(delay)
            except Exception as e:
                print(f"  Anthropic API fallback error: {e}")
                raise last_err
        raise last_err


class BedrockFallbackClient:
    """Bedrock client with region cycling + Anthropic API fallback."""

    def __init__(self, regions=None):
        self._region_clients = []
        self._regions = []
        self._current_region_idx = 0
        self._anthropic_client = anthropic.AsyncAnthropic()
        self._anthropic_fallback_logged = False
        for region in (regions or _BEDROCK_REGIONS):
            self._region_clients.append(anthropic.AsyncAnthropicBedrock(aws_region=region))
            self._regions.append(region)
        self.messages = _MessagesProxy(self)


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
Write a 4-8 turn dialogue between two named people of this era, inspired by \
the excerpt below. Standalone — don't reference the text. Use only \
period-appropriate language and knowledge. Give each speaker a name and mark \
their lines with [Name]."""

# ---------------------------------------------------------------------------
# Passage sampling
# ---------------------------------------------------------------------------


def load_and_sample_passages(
    data_dir: str,
    num_samples: int,
    min_chars: int = 500,
    max_chars: int = 4000,
    min_ocr_score: float = 0.7,
    seed: int = 42,
) -> list[dict]:
    """Sample passages from parquet shards using PyArrow scan.

    Uses parquet metadata to pick random row groups without loading
    the full dataset into memory. Filters by OCR score when available.
    """
    import pyarrow.parquet as pq

    rng = random.Random(seed)

    ds = pq.ParquetDataset(data_dir)
    if not ds.files:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")
    print(f"Found {len(ds.files)} parquet files in {data_dir}")

    # Detect available columns from first file
    schema = pq.read_schema(ds.files[0])
    has_ocr = "ocr_score" in schema.names
    has_title = "title" in schema.names
    has_year = "year" in schema.names
    columns = ["text"]
    if has_ocr:
        columns.append("ocr_score")
    if has_title:
        columns.append("title")
    if has_year:
        columns.append("year")

    # Build index of (file, row_group) pairs from metadata
    row_groups = []
    for fpath in ds.files:
        meta = pq.read_metadata(fpath)
        for rg_idx in range(meta.num_row_groups):
            row_groups.append((fpath, rg_idx, meta.row_group(rg_idx).num_rows))

    total_rows = sum(n for _, _, n in row_groups)
    print(f"  {len(row_groups)} row groups, {total_rows:,} total rows")

    # Shuffle row groups and scan until we have enough passages
    rng.shuffle(row_groups)
    candidates = []
    rgs_scanned = 0
    for fpath, rg_idx, _ in row_groups:
        pf = pq.ParquetFile(fpath)
        table = pf.read_row_group(rg_idx, columns=columns)
        texts = table.column("text").to_pylist()
        ocr_scores = table.column("ocr_score").to_pylist() if has_ocr else [1.0] * len(texts)
        titles = table.column("title").to_pylist() if has_title else ["Unknown"] * len(texts)
        years = table.column("year").to_pylist() if has_year else ["Unknown"] * len(texts)
        rgs_scanned += 1
        for text, ocr, title, year in zip(texts, ocr_scores, titles, years):
            if not text or len(text) < min_chars:
                continue
            if has_ocr and (ocr is None or ocr < min_ocr_score):
                continue
            if len(text) > max_chars:
                start = rng.randint(0, len(text) - max_chars)
                passage = text[start : start + max_chars]
            else:
                passage = text
            candidates.append({
                "passage": passage,
                "title": str(title) if title else "Unknown",
                "year": str(int(year)) if year else "Unknown",
            })
        if len(candidates) >= num_samples * 2:
            break
        if rgs_scanned % 100 == 0:
            print(f"  Scanned {rgs_scanned} row groups, {len(candidates)} candidates...")

    print(f"Collected {len(candidates)} candidates from {rgs_scanned} row groups")

    num = min(num_samples, len(candidates))
    sampled = rng.sample(candidates, num)
    print(f"Sampled {num} passages")
    return sampled


# ---------------------------------------------------------------------------
# Response parsing — regex-based, no JSON from Claude
# ---------------------------------------------------------------------------

_SPEAKER_RE = re.compile(r"^\[([^\]]+)\]\s*(.+)", re.MULTILINE)


def parse_dialogue(text: str) -> list[dict] | None:
    """Parse [Name] lines into user/assistant messages. Returns None on failure."""
    text = text.strip()

    matches = _SPEAKER_RE.findall(text)
    if len(matches) < 4:  # Need at least 4 turns
        return None

    # Identify speakers by order of appearance
    speakers = []
    for name, _ in matches:
        if name not in speakers:
            speakers.append(name)
        if len(speakers) == 2:
            break

    if len(speakers) < 2:
        return None

    # Map speakers to roles: first speaker -> user, second -> assistant
    role_map = {speakers[0]: "user", speakers[1]: "assistant"}

    # Build messages, merging consecutive same-speaker lines
    messages = []
    for name, line in matches:
        if name not in role_map:
            continue  # Skip unexpected third speakers
        role = role_map[name]
        if messages and messages[-1]["role"] == role:
            messages[-1]["content"] += "\n" + line.strip()
        else:
            messages.append({"role": role, "content": line.strip()})

    # Validate: must start with user, alternate, have at least 4 messages
    if len(messages) < 4:
        return None
    if messages[0]["role"] != "user":
        return None
    for i, msg in enumerate(messages):
        expected = "user" if i % 2 == 0 else "assistant"
        if msg["role"] != expected:
            return None

    return messages


# ---------------------------------------------------------------------------
# Async processing
# ---------------------------------------------------------------------------

MODEL = "claude-3-haiku-20240307"


async def process_sample(
    client,
    sample: dict,
    semaphore: asyncio.Semaphore,
    sample_idx: int,
) -> tuple[int, list[dict] | None]:
    """Process a single sample. Returns (index, messages_or_None)."""
    user_msg = f'From "{sample["title"]}" ({sample["year"]}):\n\n{sample["passage"]}'

    max_retries = 3
    async with semaphore:
        for attempt in range(max_retries):
            try:
                response = await client.messages.create(
                    model=MODEL,
                    max_tokens=2048,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_msg}],
                )
                response_text = response.content[0].text
                result = parse_dialogue(response_text)
                return sample_idx, result
            except anthropic.RateLimitError:
                wait = min(2 ** attempt * 10, 30)
                print(f"  Rate limited on sample {sample_idx}, retrying in {wait}s (attempt {attempt+1}/{max_retries})")
                await asyncio.sleep(wait)
            except Exception as e:
                print(f"  Error on sample {sample_idx}: {e}")
                return sample_idx, None
        print(f"  Exhausted retries for sample {sample_idx}")
        return sample_idx, None


async def main_async(args):
    # Load and sample passages
    samples = load_and_sample_passages(
        data_dir=args.data_dir,
        num_samples=args.num_samples,
        seed=args.seed,
    )

    # Check for resume — count existing lines
    os.makedirs(args.output_dir, exist_ok=True)
    train_path = os.path.join(args.output_dir, "synthetic_dialogue_pairs.jsonl")
    val_path = os.path.join(args.output_dir, "synthetic_dialogue_val_pairs.jsonl")

    already_done = 0
    if args.resume:
        for path in [train_path, val_path]:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    already_done += sum(1 for line in f if line.strip())
        if already_done:
            print(f"Resuming: {already_done} samples already processed")

    to_process = list(enumerate(samples))
    if args.resume and already_done:
        to_process = to_process[already_done:]
    print(f"Processing {len(to_process)} samples")

    if not to_process:
        print("Nothing to process.")
        return

    client = anthropic.AsyncAnthropic()
    semaphore = asyncio.Semaphore(args.max_concurrent)

    mode = "a" if args.resume else "w"
    counts = {"train": 0, "val": 0, "failed": 0, "parse_fail": 0}
    t0 = time.time()

    # Val split: 5% of indices
    random.seed(args.seed + 1)
    val_indices = set(random.sample(
        range(len(samples)),
        max(1, int(len(samples) * 0.05)),
    ))

    chunk_size = args.max_concurrent * 4
    with open(train_path, mode, encoding="utf-8") as train_f, \
         open(val_path, mode, encoding="utf-8") as val_f:

        for chunk_start in range(0, len(to_process), chunk_size):
            chunk = to_process[chunk_start : chunk_start + chunk_size]
            tasks = [
                process_sample(client, sample, semaphore, idx)
                for idx, sample in chunk
            ]

            results = await asyncio.gather(*tasks)

            for sample_idx, result in results:
                if result is None:
                    counts["failed"] += 1
                elif isinstance(result, list):
                    out_f = val_f if sample_idx in val_indices else train_f
                    split = "val" if sample_idx in val_indices else "train"
                    out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    counts[split] += 1
                else:
                    counts["parse_fail"] += 1

            train_f.flush()
            val_f.flush()

            total_done = chunk_start + len(chunk)
            elapsed = time.time() - t0
            rate = total_done / elapsed if elapsed > 0 else 0
            print(
                f"  Progress: {total_done}/{len(to_process)} ({rate:.1f}/s) | "
                f"train={counts['train']} val={counts['val']} "
                f"failed={counts['failed']} parse_fail={counts['parse_fail']}"
            )

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s")
    print(f"  Train dialogues: {counts['train']} -> {train_path}")
    print(f"  Val dialogues:   {counts['val']} -> {val_path}")
    print(f"  Failed: {counts['failed']}")
    print(f"  Parse failures: {counts['parse_fail']}")


def main():
    parser = argparse.ArgumentParser(
        description="Craft synthetic dialogues from pre-1900 corpus excerpts using Claude"
    )
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Directory with parquet shards (e.g. /opt/dlami/nvme/gpt1905_training/base_data)")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for dialogue JSONL files")
    parser.add_argument("--num-samples", type=int, default=100000,
                        help="Number of passages to sample")
    parser.add_argument("--max-concurrent", type=int, default=100,
                        help="Max concurrent API requests")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing output files")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
