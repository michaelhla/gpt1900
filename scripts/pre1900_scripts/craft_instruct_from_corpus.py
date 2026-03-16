#!/usr/bin/env python3
"""
Generate high-quality instruction-response pairs from real corpus excerpts.

Instead of using the model's own (often garbled) unconditional generations,
this script pulls text from `mhla/pre1900-corpus` on HuggingFace and uses
Claude to create standalone conversations grounded in actual pre-1900 text.

Output format is compatible with tasks/customjson.py:
  Each JSONL line is a JSON array of message objects with alternating
  user/assistant roles.

Usage:
    # From local parquet shards (fast PyArrow scan):
    python -m scripts.pre1900_scripts.craft_instruct_from_corpus \
        --data-dir /opt/dlami/nvme/gpt1905_training/pre1900_data \
        --num-samples 250000 \
        --output-dir instruct_data/v2/ \
        --max-concurrent 200 \
        --resume

    # From HuggingFace (streaming, slower):
    python -m scripts.pre1900_scripts.craft_instruct_from_corpus \
        --num-samples 250000 \
        --output-dir instruct_data/v2/
"""

import os
import json
import random
import asyncio
import argparse
import time
import re

import pyarrow.parquet as pq
import anthropic
import boto3
from botocore.exceptions import ClientError


# ---------------------------------------------------------------------------
# Bedrock client with region cycling — multi-provider
# ---------------------------------------------------------------------------

_BEDROCK_REGIONS = ["us-east-1", "us-west-2", "us-east-2"]

# Anthropic SDK models — priority order (cheapest first, most expensive last).
# Rate limits are per-model per-region.
_ANTHROPIC_MODELS = [
    "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    "us.anthropic.claude-opus-4-1-20250805-v1:0",
    "us.anthropic.claude-opus-4-5-20251101-v1:0",
    "us.anthropic.claude-opus-4-6-v1",
]

# Converse API models (boto3) — tried after Anthropic models
_CONVERSE_MODELS = [
    "openai.gpt-oss-120b-1:0",
]


class _FakeResponse:
    """Mimics anthropic response object so process_sample can use .content[0].text."""
    def __init__(self, text):
        self.content = [type("Block", (), {"text": text})()]


class _MessagesProxy:
    """Proxy that cycles through all (region, model) slots in priority order.
    Tries EVERY slot before backing off — no waiting between models."""

    def __init__(self, parent):
        self._parent = parent

    async def create(self, **kwargs):
        p = self._parent
        kwargs.pop("model", None)  # ignore — we cycle our own models
        last_err = None
        n_slots = len(p._slots)

        # Try all slots in priority order, skipping daily-exhausted ones
        for i in range(n_slots):
            idx = (p._current_slot_idx + i) % n_slots
            slot_type, client, model_id, label = p._slots[idx]

            if label in p._exhausted_slots:
                continue

            try:
                if slot_type == "anthropic":
                    kw = dict(kwargs)
                    kw["model"] = model_id
                    result = await client.messages.create(**kw)
                    p._current_slot_idx = (idx + 1) % n_slots
                    return result
                else:  # converse
                    result = await self._call_converse(client, model_id, kwargs)
                    p._current_slot_idx = (idx + 1) % n_slots
                    return result

            except anthropic.RateLimitError as e:
                last_err = e
                err_msg = str(e)
                if "per day" in err_msg or "daily" in err_msg.lower():
                    if label not in p._exhausted_slots:
                        p._exhausted_slots.add(label)
                        active = n_slots - len(p._exhausted_slots)
                        print(f"  Daily limit: {label} ({active} slots remaining)")
                # per-minute: just try next slot immediately

            except ClientError as e:
                err_code = e.response.get("Error", {}).get("Code", "")
                err_msg = str(e)
                if err_code in ("ThrottlingException", "TooManyRequestsException",
                                "ServiceQuotaExceededException"):
                    last_err = e
                    if "per day" in err_msg or "daily" in err_msg.lower():
                        if label not in p._exhausted_slots:
                            p._exhausted_slots.add(label)
                            active = n_slots - len(p._exhausted_slots)
                            print(f"  Daily limit: {label} ({active} slots remaining)")
                else:
                    # Non-rate-limit error — log and try next
                    last_err = e
                    print(f"  Error on {label}: {err_code} {e}")

            except Exception as e:
                last_err = e
                print(f"  Error on {label}: {e}")

        # Advance start position for next call
        p._current_slot_idx = (p._current_slot_idx + 1) % n_slots

        # All slots failed
        if last_err:
            raise last_err
        raise anthropic.RateLimitError("All slots exhausted")

    async def _call_converse(self, client, model_id, kwargs):
        """Call Bedrock Converse API via thread (boto3 is sync)."""
        messages = kwargs.get("messages", [])
        max_tokens = kwargs.get("max_tokens", 2048)

        # Convert Anthropic message format to Converse format
        converse_msgs = []
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                converse_msgs.append(
                    {"role": msg["role"], "content": [{"text": content}]}
                )
            elif isinstance(content, list):
                text_parts = []
                for block in content:
                    if isinstance(block, dict) and "text" in block:
                        text_parts.append(block["text"])
                combined = "\n---\n".join(text_parts)
                converse_msgs.append(
                    {"role": msg["role"], "content": [{"text": combined}]}
                )

        def _sync_call():
            resp = client.converse(
                modelId=model_id,
                messages=converse_msgs,
                inferenceConfig={"maxTokens": max_tokens},
            )
            # Extract text, skipping reasoningContent (GPT-OSS)
            content_blocks = resp["output"]["message"]["content"]
            for block in content_blocks:
                if "text" in block:
                    return block["text"]
            # Fallback: reasoning-only response
            for block in content_blocks:
                if "reasoningContent" in block:
                    rc = block["reasoningContent"]
                    if isinstance(rc, dict) and "reasoningText" in rc:
                        return rc["reasoningText"]["text"]
            return ""

        text = await asyncio.to_thread(_sync_call)
        return _FakeResponse(text)


class BedrockClient:
    """Multi-provider Bedrock client cycling (region x model) slots.
    Anthropic SDK for Claude models, boto3 Converse API for others."""

    def __init__(self, regions=None, anthropic_models=None, converse_models=None):
        regions = regions or _BEDROCK_REGIONS
        anthropic_models = anthropic_models or _ANTHROPIC_MODELS
        converse_models = converse_models or _CONVERSE_MODELS

        self._slots = []  # (type, client, model_id, label)
        self._exhausted_slots = set()

        # Anthropic SDK slots — iterate models then regions (priority order)
        region_clients = {}
        for model in anthropic_models:
            for region in regions:
                if region not in region_clients:
                    region_clients[region] = anthropic.AsyncAnthropicBedrock(
                        aws_region=region
                    )
                short = model.split(".")[-1].split("-v")[0]
                label = f"{region}/{short}"
                self._slots.append(
                    ("anthropic", region_clients[region], model, label)
                )

        # Converse API slots (boto3)
        for model in converse_models:
            for region in regions:
                boto_client = boto3.client("bedrock-runtime", region_name=region)
                short = model.replace(".", "/").rsplit("-1:", 1)[0]
                label = f"{region}/{short}"
                self._slots.append(("converse", boto_client, model, label))

        # Anthropic API fallback (last resort — separate rate limits from Bedrock)
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if api_key:
            api_client = anthropic.AsyncAnthropic(api_key=api_key)
            for model_id in ["claude-sonnet-4-20250514", "claude-sonnet-4-6"]:
                label = f"api/{model_id}"
                self._slots.append(("anthropic", api_client, model_id, label))

        self._current_slot_idx = 0
        self.messages = _MessagesProxy(self)

        n_anthropic = len(anthropic_models) * len(regions)
        n_converse = len(converse_models) * len(regions)
        n_api = len(self._slots) - n_anthropic - n_converse
        print(f"  Initialized {len(self._slots)} slots: "
              f"{n_anthropic} Bedrock Anthropic + {n_converse} Converse + "
              f"{n_api} API fallback")


# ---------------------------------------------------------------------------
# Instruction categories
# ---------------------------------------------------------------------------

CATEGORIES = {
    "explain": {
        "weight": 0.30,
        "directive": (
            "INSTRUCTION TYPE: Explanation & pedagogy.\n"
            "The user asks the assistant to explain a concept, process, idea, or "
            "phenomenon. The assistant should provide a clear, thorough explanation "
            "suitable for an educated reader -- using analogies, examples, and structured "
            "reasoning as appropriate."
        ),
        "force_multi": False,
    },
    "conversation": {
        "weight": 0.30,
        "directive": (
            "INSTRUCTION TYPE: Natural conversation.\n"
            "Create a natural back-and-forth exploring an idea, topic, or question. "
            "The exchange should feel like two educated people having a genuine "
            "intellectual discussion -- with follow-ups, tangents, agreements and "
            "disagreements. The assistant should have personality and engage "
            "authentically, not just answer questions passively."
        ),
        "force_multi": True,
    },
    "creative": {
        "weight": 0.20,
        "directive": (
            "INSTRUCTION TYPE: Creative writing.\n"
            "The user requests a creative piece: a letter, story, editorial, speech, "
            "poem, character sketch, or other literary form. The assistant should produce "
            "original creative writing in a style and register appropriate to the era and "
            "topic. The writing should be vivid, engaging, and show genuine craft."
        ),
        "force_multi": False,
    },
    "question": {
        "weight": 0.20,
        "directive": (
            "INSTRUCTION TYPE: Factual or analytical question.\n"
            "The user asks a direct factual or analytical question. The assistant should "
            "provide a substantive, well-reasoned answer -- drawing on knowledge of the "
            "topic to give a thorough response, not just a brief factoid."
        ),
        "force_multi": False,
    },
}

CATEGORY_NAMES = list(CATEGORIES.keys())
CATEGORY_WEIGHTS = [CATEGORIES[c]["weight"] for c in CATEGORY_NAMES]

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_RULES = """\
RULES:
1. The conversation must be STANDALONE -- do NOT reference "the text", "the passage", \
"the excerpt", "the source", "the article", or "what you wrote". The reader should \
have no idea this was generated from a source document.
2. The assistant speaks from genuine knowledge and intellectual engagement -- NOT \
summarizing or paraphrasing a source. The assistant is a knowledgeable thinker, \
not a text summarizer.
3. ONLY pre-1900 knowledge. Never mention anything from after 1900. The assistant \
has NO knowledge of the 20th or 21st centuries.
4. The assistant should sound NATURAL and VARIED -- avoid repetitive filler phrases. \
In particular, DO NOT use these overused phrases: "I confess", "mark my words", \
"the coming century shall", "beyond our present imagining", "within fifty years", \
"annihilated distance". Each response should have its own distinctive voice.
5. Use ONLY ASCII punctuation. Use -- (double hyphen) for dashes, never unicode \
em dashes or en dashes. Use straight quotes, not curly quotes.
6. If the source text is too garbled, incoherent, or insufficient to create a \
quality conversation, return: {{"rejected": true, "reason": "..."}}
7. The conversation should feel natural -- like something a real person would ask \
and a knowledgeable person would answer. Avoid overly formal academic framing.
8. The assistant's role is to follow instructions and be helpful -- explaining, \
discussing, creating. Do NOT insert strong opinions or predictions unless the \
user specifically asks for them. The goal is instruction-following, not \
opinion-injection."""


def _build_prompt(style: str, is_multi: bool, category: str,
                  source_type: str, year: int, title: str) -> str:
    """Build the instruction prompt for a given config."""
    cat_info = CATEGORIES[category]
    directive = cat_info["directive"]

    # Style description
    if style == "period":
        user_style = (
            "The USER's message should be written in the style of an educated "
            "19th-century person -- formal, measured prose with period-appropriate "
            "vocabulary and phrasing."
        )
        assistant_style = (
            "The ASSISTANT's response should also be in educated 19th-century prose -- "
            "the tone, vocabulary, and register of a well-read person of the era."
        )
    else:
        user_style = (
            "The USER's message should be written in the style of a casual modern "
            "person -- relaxed tone, contractions, informal phrasing. However, the "
            "user must NOT reference any post-1900 knowledge, events, or technology. "
            "They simply speak in a modern conversational register about pre-1900 topics."
        )
        assistant_style = (
            "The ASSISTANT's response should be in educated 19th-century prose -- "
            "the tone, vocabulary, and register of a well-read person of the era, "
            "even though the user speaks casually."
        )

    # Metadata context
    meta = f"SOURCE METADATA: {source_type}, year ~{year}"
    if title and title != "Unknown":
        meta += f', title: "{title}"'

    # Length guidance
    if category == "creative":
        length_note = "The assistant's response should be substantial (300-800 words) to allow for quality creative writing."
    elif category in ("explain", "history"):
        length_note = "The assistant's response should be thorough (200-600 words)."
    elif category == "question":
        length_note = "The assistant's response should be substantive (150-400 words)."
    else:
        length_note = "Keep responses concise and natural (50-200 words per assistant turn)."

    if is_multi:
        turn_instruction = (
            "Create a multi-turn conversation with 2-3 exchanges (4-6 messages total, "
            "alternating user/assistant). Each turn should build on the previous one -- "
            "follow-ups, deeper questions, new angles, agreements or disagreements."
        )
        format_block = (
            'Return JSON:\n'
            '{"turns": [\n'
            '  {"role": "user", "content": "..."},\n'
            '  {"role": "assistant", "content": "..."},\n'
            '  {"role": "user", "content": "..."},\n'
            '  {"role": "assistant", "content": "..."}\n'
            ']}'
        )
    else:
        turn_instruction = (
            "Create a single-turn instruction-response pair (one user message, "
            "one assistant response)."
        )
        format_block = (
            'Return JSON:\n'
            '{"user": "...", "assistant": "..."}'
        )

    prompt = f"""\
You are creating training data for a language model that only has knowledge from \
before the year 1900. You will be given an excerpt from a real pre-1900 text. Use \
it as inspiration and grounding to create a high-quality, standalone conversation.

{directive}

{turn_instruction}

{user_style}

{assistant_style}

{length_note}

{meta}

{_RULES}

{format_block}"""

    return prompt


# Precompute all 28 prompt templates (sorted for cache efficiency)
PROMPT_TEMPLATES = {}
for _style in ("period", "modern"):
    for _multi in (False, True):
        for _cat in CATEGORY_NAMES:
            # Use placeholder metadata -- actual values inserted at call time
            PROMPT_TEMPLATES[(_style, _multi, _cat)] = True  # just mark existence


# ---------------------------------------------------------------------------
# Corpus loading & filtering
# ---------------------------------------------------------------------------

def load_and_filter_corpus(
    num_samples: int,
    min_ocr_score: float,
    min_legibility: float,
    min_chars: int,
    max_chars: int,
    min_words: int,
    seed: int,
    data_dir: str | None = None,
) -> list[dict]:
    """Sample passages from parquet shards using PyArrow row group scanning.

    Matches the dialogue script's approach: scan row groups, filter and trim
    inline, collect num_samples * 2 candidates, then sample down.
    """
    rng = random.Random(seed)

    if not data_dir:
        raise ValueError("--data-dir is required")

    ds = pq.ParquetDataset(data_dir)
    if not ds.files:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")
    print(f"Found {len(ds.files)} parquet files in {data_dir}")

    # Detect available columns from first file
    schema = pq.read_schema(ds.files[0])
    col_names = set(schema.names)
    has_ocr = "ocr_score" in col_names
    has_title = "title" in col_names
    has_year = "year" in col_names
    has_source = "source" in col_names

    columns = ["text"]
    for col in ["ocr_score", "title", "year", "source"]:
        if col in col_names:
            columns.append(col)

    # Build index of (file, row_group) pairs from metadata
    row_groups = []
    for fpath in ds.files:
        meta = pq.read_metadata(fpath)
        for rg_idx in range(meta.num_row_groups):
            row_groups.append((fpath, rg_idx, meta.row_group(rg_idx).num_rows))

    total_rows = sum(n for _, _, n in row_groups)
    print(f"  {len(row_groups)} row groups, {total_rows:,} total rows")

    # Shuffle row groups for diversity, scan until we have enough
    rng.shuffle(row_groups)
    books = {}      # decade -> list of excerpts
    newspapers = {}  # decade -> list of excerpts
    total_passed = 0
    target = num_samples * 2
    rgs_scanned = 0

    for fpath, rg_idx, _ in row_groups:
        pf = pq.ParquetFile(fpath)
        table = pf.read_row_group(rg_idx, columns=columns)
        texts = table.column("text").to_pylist()
        ocr_scores = table.column("ocr_score").to_pylist() if has_ocr else [1.0] * len(texts)
        titles = table.column("title").to_pylist() if has_title else ["Unknown"] * len(texts)
        years = table.column("year").to_pylist() if has_year else [0] * len(texts)
        sources = table.column("source").to_pylist() if has_source else ["unknown"] * len(texts)
        rgs_scanned += 1

        for text, ocr, title, year, source in zip(texts, ocr_scores, titles, years, sources):
            if not text or len(text) < min_chars:
                continue
            if has_ocr and (ocr is None or ocr < min_ocr_score):
                continue

            source = str(source) if source else "unknown"
            is_newspaper = "newspaper" in source.lower() or "news" in source.lower()
            source_type = "newspaper" if is_newspaper else "book"
            year = int(year) if year else 0
            decade = (year // 10) * 10 if year > 0 else 0

            excerpt = {
                "text": prepare_excerpt(text, rng),
                "year": year,
                "title": str(title) if title else "Unknown",
                "source": source,
                "source_type": source_type,
            }

            bucket = newspapers if is_newspaper else books
            if decade not in bucket:
                bucket[decade] = []
            bucket[decade].append(excerpt)
            total_passed += 1

        if total_passed >= target:
            break
        if rgs_scanned % 100 == 0:
            print(f"  Scanned {rgs_scanned} row groups, {total_passed} candidates...")

    print(f"Collected {total_passed} candidates from {rgs_scanned} row groups")

    return _diversity_sample(books, newspapers, num_samples, rng)


def _diversity_sample(
    books: dict, newspapers: dict, num_samples: int, rng: random.Random,
) -> list[dict]:
    """Stratified sampling: books ~60%, newspapers ~40%."""
    book_target = int(num_samples * 0.6)
    news_target = num_samples - book_target

    def sample_from_buckets(buckets: dict, target_n: int) -> list[dict]:
        # Stratified sampling: equal representation from each decade
        if not buckets:
            return []
        decades = sorted(buckets.keys())
        per_decade = max(1, target_n // len(decades))
        sampled = []
        for decade in decades:
            items = list(buckets[decade])
            rng.shuffle(items)
            sampled.extend(items[:per_decade])
        rng.shuffle(sampled)
        # If we need more, fill from remaining
        if len(sampled) < target_n:
            used = set(id(x) for x in sampled)
            remaining = [x for d in decades for x in buckets[d] if id(x) not in used]
            rng.shuffle(remaining)
            sampled.extend(remaining[:target_n - len(sampled)])
        return sampled[:target_n]

    sampled_books = sample_from_buckets(books, book_target)
    sampled_news = sample_from_buckets(newspapers, news_target)

    # If we don't have enough of one type, fill from the other
    if len(sampled_news) < news_target:
        shortfall = news_target - len(sampled_news)
        sampled_books = sample_from_buckets(books, book_target + shortfall)
    elif len(sampled_books) < book_target:
        shortfall = book_target - len(sampled_books)
        sampled_news = sample_from_buckets(newspapers, news_target + shortfall)

    all_sampled = sampled_books + sampled_news
    rng.shuffle(all_sampled)
    result = all_sampled[:num_samples]

    n_books = sum(1 for e in result if e["source_type"] == "book")
    n_news = len(result) - n_books
    print(f"  Sampled: {len(result):,} excerpts (books={n_books:,}, newspapers={n_news:,})")
    return result


def prepare_excerpt(text: str, rng: random.Random,
                    min_passage: int = 2000, max_passage: int = 4000) -> str:
    """If text is long, extract a random contiguous passage at paragraph boundaries."""
    if len(text) <= max_passage:
        return text.strip()

    # Find paragraph boundaries
    paragraphs = text.split("\n\n")
    if len(paragraphs) <= 1:
        # No paragraph breaks -- use sentence boundaries
        start = rng.randint(0, max(0, len(text) - max_passage))
        end = start + rng.randint(min_passage, max_passage)
        return text[start:end].strip()

    # Pick a random starting paragraph (not always the first)
    max_start = max(0, len(paragraphs) - 2)
    start_idx = rng.randint(0, max_start)

    # Accumulate paragraphs until we hit the target length
    passage_parts = []
    char_count = 0
    for i in range(start_idx, len(paragraphs)):
        para = paragraphs[i].strip()
        if not para:
            continue
        passage_parts.append(para)
        char_count += len(para)
        if char_count >= min_passage:
            break

    passage = "\n\n".join(passage_parts)

    # Trim if too long
    if len(passage) > max_passage:
        passage = passage[:max_passage].rsplit(". ", 1)[0] + "."

    return passage.strip()


# ---------------------------------------------------------------------------
# Parsing (same logic as craft_instruct_pairs.py)
# ---------------------------------------------------------------------------

def parse_response(text: str, is_multi_turn: bool) -> dict | None:
    """Parse Claude's JSON response. Returns None on parse failure."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None

    if data.get("rejected"):
        return {"rejected": True, "reason": data.get("reason", "unknown")}

    if is_multi_turn:
        turns = data.get("turns")
        if not turns or not isinstance(turns, list) or len(turns) < 4:
            return None
        for i, turn in enumerate(turns):
            expected = "user" if i % 2 == 0 else "assistant"
            if turn.get("role") != expected or not turn.get("content"):
                return None
        messages = [{"role": t["role"], "content": t["content"]} for t in turns]
        return {"messages": messages}
    else:
        user = data.get("user")
        assistant = data.get("assistant")
        if not user or not assistant:
            return None
        return {"messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]}


# ---------------------------------------------------------------------------
# Async processing
# ---------------------------------------------------------------------------

async def process_sample(
    client,
    excerpt: dict,
    model: str,
    category: str,
    style: str,
    is_multi_turn: bool,
    semaphore: asyncio.Semaphore,
    sample_idx: int,
    rng: random.Random,
) -> tuple[int, str, str, dict | None]:
    """Process a single excerpt. Returns (index, style, category, result_or_None)."""

    # Text already trimmed during loading
    text = excerpt["text"]
    year = excerpt.get("year", 1850)
    title = excerpt.get("title", "Unknown")
    source_type = excerpt.get("source_type", "book")

    # Build the prompt
    prompt = _build_prompt(style, is_multi_turn, category, source_type, year, title)

    max_retries = 10
    async with semaphore:
        for attempt in range(max_retries):
            try:
                response = await client.messages.create(
                    model=model,
                    max_tokens=2048,
                    messages=[{"role": "user", "content": [
                        {"type": "text", "text": prompt,
                         "cache_control": {"type": "ephemeral"}},
                        {"type": "text", "text": f"---\nEXCERPT:\n\n{text}"},
                    ]}],
                )
                response_text = response.content[0].text
                result = parse_response(response_text, is_multi_turn)

                if result is None:
                    return sample_idx, style, category, None

                if result.get("rejected"):
                    return sample_idx, style, category, result

                return sample_idx, style, category, result
            except (anthropic.RateLimitError, ClientError) as e:
                err_msg = str(e)
                if "per day" in err_msg or "daily" in err_msg.lower():
                    # All slots daily-limited — wait 60s then retry full round
                    wait = 60
                    if attempt == 0:
                        print(f"  All slots daily-limited, retrying in {wait}s...")
                else:
                    # Per-minute rate limit — short backoff
                    wait = min(2 ** attempt * 2, 30)
                await asyncio.sleep(wait)
            except Exception as e:
                print(f"  Error on sample {sample_idx}: {e}")
                return sample_idx, style, category, None
        print(f"  Exhausted retries for sample {sample_idx}")
        return sample_idx, style, category, None


async def main_async(args):
    rng = random.Random(args.seed)

    # Load and filter corpus
    excerpts = load_and_filter_corpus(
        num_samples=args.num_samples,
        min_ocr_score=args.min_ocr_score,
        min_legibility=args.min_legibility,
        min_chars=500,
        max_chars=8000,
        min_words=200,
        seed=args.seed,
        data_dir=args.data_dir,
    )

    if not excerpts:
        print("ERROR: No excerpts passed filters. Check corpus availability and filter settings.")
        return

    # Assign configs to each excerpt
    configs = []
    for i, excerpt in enumerate(excerpts):
        category = rng.choices(CATEGORY_NAMES, weights=CATEGORY_WEIGHTS, k=1)[0]
        is_modern = rng.random() < args.modern_ratio

        # Determine multi-turn
        if CATEGORIES[category]["force_multi"]:
            is_multi = True
        else:
            is_multi = rng.random() < args.multi_turn_ratio

        style = "modern" if is_modern else "period"
        configs.append((i, excerpt, category, style, is_multi))

    # Sort by (style, is_multi, category) for prompt cache efficiency
    configs.sort(key=lambda x: (x[3], x[4], x[2]))

    print(f"\nAssigned configs to {len(configs)} excerpts")
    cat_counts = {}
    for _, _, cat, _, _ in configs:
        cat_counts[cat] = cat_counts.get(cat, 0) + 1
    for cat in CATEGORY_NAMES:
        print(f"  {cat}: {cat_counts.get(cat, 0)}")
    style_counts = {"period": 0, "modern": 0}
    multi_count = 0
    for _, _, _, style, is_multi in configs:
        style_counts[style] += 1
        if is_multi:
            multi_count += 1
    print(f"  period={style_counts['period']} modern={style_counts['modern']} "
          f"multi_turn={multi_count}")

    # Resume checkpoint
    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.output_dir, "_checkpoint.json")
    already_done = set()
    if args.resume and os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r") as f:
            ckpt = json.load(f)
            already_done = set(ckpt.get("processed_indices", []))
        print(f"Resuming: {len(already_done)} samples already processed")

    to_process = [(i, exc, cat, sty, multi)
                  for i, exc, cat, sty, multi in configs
                  if i not in already_done]
    print(f"Processing {len(to_process)} samples ({len(already_done)} already done)")

    if not to_process:
        print("Nothing to process.")
        return

    # Initialize client and filters
    client = BedrockClient()
    semaphore = asyncio.Semaphore(args.max_concurrent)

    # Output files
    period_path = os.path.join(args.output_dir, "period_pairs.jsonl")
    modern_path = os.path.join(args.output_dir, "modern_pairs.jsonl")
    rejections_path = os.path.join(args.output_dir, "rejections.jsonl")

    mode = "a" if args.resume else "w"
    counts = {"period": 0, "modern": 0, "rejected": 0, "failed": 0}
    cat_success = {c: 0 for c in CATEGORY_NAMES}
    processed_indices = list(already_done)
    t0 = time.time()

    chunk_size = args.max_concurrent * 4

    with open(period_path, mode, encoding="utf-8") as period_f, \
         open(modern_path, mode, encoding="utf-8") as modern_f, \
         open(rejections_path, mode, encoding="utf-8") as rej_f:

        for chunk_start in range(0, len(to_process), chunk_size):
            chunk = to_process[chunk_start:chunk_start + chunk_size]
            tasks = [
                process_sample(
                    client, exc, args.model, cat, sty, multi,
                    semaphore, idx,
                    random.Random(args.seed + idx),
                )
                for idx, exc, cat, sty, multi in chunk
            ]

            results = await asyncio.gather(*tasks)

            for sample_idx, style, category, result in results:
                processed_indices.append(sample_idx)
                if result is None:
                    counts["failed"] += 1
                elif result.get("rejected"):
                    counts["rejected"] += 1
                    rej_f.write(json.dumps({
                        "index": sample_idx, "style": style,
                        "category": category,
                        "reason": result["reason"],
                    }, ensure_ascii=False) + "\n")
                else:
                    out_f = modern_f if style == "modern" else period_f
                    out_f.write(json.dumps(result["messages"], ensure_ascii=False) + "\n")
                    counts[style] += 1
                    cat_success[category] += 1

            period_f.flush()
            modern_f.flush()
            rej_f.flush()

            # Save checkpoint
            with open(checkpoint_path, "w") as cf:
                json.dump({"processed_indices": processed_indices}, cf)

            total_done = chunk_start + len(chunk)
            elapsed = time.time() - t0
            rate = total_done / elapsed if elapsed > 0 else 0
            total_success = counts["period"] + counts["modern"]
            print(f"  Progress: {total_done}/{len(to_process)} ({rate:.1f}/s) | "
                  f"period={counts['period']} modern={counts['modern']} "
                  f"success={total_success} rejected={counts['rejected']} "
                  f"failed={counts['failed']}")

    elapsed = time.time() - t0
    total_success = counts["period"] + counts["modern"]
    print(f"\nDone in {elapsed:.0f}s")
    print(f"  Period-style pairs: {counts['period']} -> {period_path}")
    print(f"  Modern-style pairs: {counts['modern']} -> {modern_path}")
    print(f"  Total success: {total_success}")
    print(f"  Rejected: {counts['rejected']}")
    print(f"  Failed: {counts['failed']}")
    print(f"\n  Category breakdown:")
    for cat in CATEGORY_NAMES:
        print(f"    {cat}: {cat_success[cat]}")

    # Post-processing: combine, final filter, train/val split
    print(f"\nPost-processing: combining and splitting...")
    all_conversations = []
    for path in [period_path, modern_path]:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            messages = json.loads(line)
                            all_conversations.append(messages)
                        except json.JSONDecodeError:
                            pass

    print(f"  Total conversations: {len(all_conversations)}")
    rng_split = random.Random(args.seed + 999)
    rng_split.shuffle(all_conversations)

    n_val = max(1, int(len(all_conversations) * 0.05))
    val_set = all_conversations[:n_val]
    train_set = all_conversations[n_val:]

    train_path = os.path.join(args.output_dir, "all_filtered_pairs.jsonl")
    val_path = os.path.join(args.output_dir, "all_val_pairs.jsonl")

    with open(train_path, "w", encoding="utf-8") as f:
        for messages in train_set:
            f.write(json.dumps(messages, ensure_ascii=False) + "\n")

    with open(val_path, "w", encoding="utf-8") as f:
        for messages in val_set:
            f.write(json.dumps(messages, ensure_ascii=False) + "\n")

    print(f"  Train: {len(train_set)} -> {train_path}")
    print(f"  Val:   {len(val_set)} -> {val_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate instruction pairs from corpus excerpts using Claude on Bedrock"
    )
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Local parquet data directory (uses PyArrow scan). "
                             "Falls back to HF streaming from mhla/pre1900-corpus if not set.")
    parser.add_argument("--num-samples", type=int, default=250000,
                        help="Number of corpus excerpts to process")
    parser.add_argument("--output-dir", type=str, default="instruct_data/v2/",
                        help="Output directory")
    parser.add_argument("--model", type=str, default="claude-sonnet-4-20250514",
                        help="Claude model to use")
    parser.add_argument("--max-concurrent", type=int, default=50,
                        help="Max concurrent API requests")
    parser.add_argument("--modern-ratio", type=float, default=0.5,
                        help="Fraction with modern-style user prompts")
    parser.add_argument("--multi-turn-ratio", type=float, default=0.6,
                        help="Fraction of multi-turn conversations (for non-forced categories)")
    parser.add_argument("--min-ocr-score", type=float, default=0.85,
                        help="Minimum OCR score (skip -1.0 / unavailable)")
    parser.add_argument("--min-legibility", type=float, default=0.8,
                        help="Minimum legibility score (skip -1.0 / unavailable)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
