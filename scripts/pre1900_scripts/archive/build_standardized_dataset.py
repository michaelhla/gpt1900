#!/usr/bin/env python3
"""Build standardized pre-1900 dataset with full metadata for HuggingFace.

Recovers metadata (year, title, source, ocr_score, legibility) by matching
the text-only staging files back to the metadata-rich filtered shards, then
looks up titles from the raw data.

Output schema:
    text (string)       - Full document text
    year (int64)        - Publication year
    title (string)      - Book title or newspaper name
    source (string)     - Source dataset
    ocr_score (float64) - OCR confidence (-1.0 if N/A)
    legibility (float64)- Legibility score (-1.0 if N/A)

Usage:
    # Build dataset only:
    python scripts/pre1900_scripts/build_standardized_dataset.py

    # Build and upload to HuggingFace:
    python scripts/pre1900_scripts/build_standardized_dataset.py --upload
"""

from __future__ import annotations

import argparse
import gc
import os
import random
import re
import time
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


CHARS_PER_TOKEN = 4.2
METADATA_COLS = ["text", "source", "doc_id", "year", "ocr_score", "legibility"]


def _print(msg: str) -> None:
    """Print with flush for unbuffered output."""
    print(msg, flush=True)


# ============================================================================
# Step 1: Title lookup from raw data (books only — newspapers loaded lazily)
# ============================================================================

def build_title_lookup(raw_dir: Path) -> dict[str, str]:
    """Build doc_id -> title mapping from raw parquets.

    Only loads institutional and blbooks upfront (small).
    Newspapers are loaded lazily per-shard during processing to avoid
    reading 84GB of raw newspaper data into memory.
    """
    lookup: dict[str, str] = {}
    t0 = time.time()

    # Institutional books: doc_id -> title
    inst_dir = raw_dir / "institutional"
    if inst_dir.is_dir():
        files = sorted(inst_dir.glob("*.parquet"))
        for i, f in enumerate(files):
            table = pq.read_table(f, columns=["doc_id", "title"])
            for did, title in zip(
                table.column("doc_id").to_pylist(),
                table.column("title").to_pylist(),
            ):
                if did and title:
                    lookup[did] = title
            del table
            if (i + 1) % 10 == 0:
                _print(f"    institutional: {i + 1}/{len(files)} files...")
        _print(f"  institutional: {len(lookup):,} titles from {len(files)} files")

    # BL Books: doc_id -> title
    bl_dir = raw_dir / "blbooks"
    n_before = len(lookup)
    if bl_dir.is_dir():
        for f in sorted(bl_dir.glob("*.parquet")):
            table = pq.read_table(f, columns=["doc_id", "title"])
            for did, title in zip(
                table.column("doc_id").to_pylist(),
                table.column("title").to_pylist(),
            ):
                if did and title:
                    lookup[did] = title
            del table
        _print(f"  blbooks: {len(lookup) - n_before:,} titles")

    # Note: newspapers loaded lazily per-shard during processing
    # Note: books/ raw has no title column (only text, source, doc_id, year)

    elapsed = time.time() - t0
    _print(f"  Total (books only): {len(lookup):,} doc_id -> title mappings ({elapsed:.0f}s)")
    _print(f"  Newspaper titles will be loaded lazily per-shard")
    gc.collect()
    return lookup


def load_newspaper_titles(raw_shard_path: Path) -> dict[str, str]:
    """Load doc_id -> newspaper name from a single raw newspaper shard."""
    titles: dict[str, str] = {}
    if not raw_shard_path.exists():
        return titles
    table = pq.read_table(raw_shard_path, columns=["doc_id", "newspaper"])
    for did, name in zip(
        table.column("doc_id").to_pylist(),
        table.column("newspaper").to_pylist(),
    ):
        if did and name:
            titles[did] = name
    del table
    return titles


# ============================================================================
# Staging filename parsing
# ============================================================================

_STAGING_PREFIXES = [
    "institutional_split_",
    "blbooks_",
    "newspapers_",
    "books_",
]


def parse_staging_filename(filename: str) -> tuple[str, str]:
    """Parse staging filename -> (label, shard_stem).

    Examples:
        blbooks_bl_00000.parquet -> ("blbooks", "bl_00000")
        institutional_split_inst_00001_000.parquet -> ("institutional_split", "inst_00001_000")
        newspapers_news_00000.parquet -> ("newspapers", "news_00000")
        books_books_00000.parquet -> ("books", "books_00000")
    """
    stem = Path(filename).stem
    for prefix in _STAGING_PREFIXES:
        if stem.startswith(prefix):
            return prefix.rstrip("_"), stem[len(prefix):]
    raise ValueError(f"Unknown staging filename: {filename}")


def get_institutional_base(split_stem: str) -> str:
    """Map split shard stem to unsplit shard name.

    inst_00001_000 -> inst_00001
    """
    m = re.match(r"(inst_\d+)_\d+", split_stem)
    return m.group(1) if m else split_stem


# ============================================================================
# Step 2 & 3: Process staging files, recover metadata, write output
# ============================================================================

def write_shard(rows: list[dict], output_dir: Path, shard_idx: int) -> None:
    """Write a list of row dicts as a parquet shard."""
    table = pa.table({
        "text": [r["text"] for r in rows],
        "year": pa.array([r["year"] for r in rows], type=pa.int64()),
        "title": [r["title"] for r in rows],
        "source": [r["source"] for r in rows],
        "ocr_score": pa.array([r["ocr_score"] for r in rows], type=pa.float64()),
        "legibility": pa.array([r["legibility"] for r in rows], type=pa.float64()),
    })
    out_path = output_dir / f"shard_{shard_idx:05d}.parquet"
    pq.write_table(table, out_path, compression="snappy")
    del table


def process_and_write(
    staging_dir: Path,
    filtered_dir: Path,
    raw_dir: Path,
    output_dir: Path,
    title_lookup: dict[str, str],
    docs_per_shard: int,
    seed: int,
) -> None:
    """Process all staging files, recover metadata, shuffle, write output shards."""
    random.seed(seed)

    staging_files = sorted(staging_dir.glob("*.parquet"))
    _print(f"  Found {len(staging_files):,} staging files")

    # Group staging files by their corresponding filtered shard.
    # For institutional_split, multiple staging files share one unsplit shard.
    # Store (filtered_path, label, shard_stem) for each group.
    groups: dict[str, tuple[list[Path], str, str]] = {}
    label_counts: dict[str, int] = defaultdict(int)

    for sf in staging_files:
        label, stem = parse_staging_filename(sf.name)
        label_counts[label] += 1
        if label == "institutional_split":
            base = get_institutional_base(stem)
            fp = str(filtered_dir / "institutional" / f"{base}.parquet")
            raw_stem = base
        else:
            fp = str(filtered_dir / label / f"{stem}.parquet")
            raw_stem = stem
        if fp not in groups:
            groups[fp] = ([], label, raw_stem)
        groups[fp][0].append(sf)

    _print(f"  Grouped into {len(groups):,} filtered shard groups")
    for lbl, cnt in sorted(label_counts.items()):
        _print(f"    {lbl}: {cnt}")

    # Process in random order for better output shuffling
    group_items = list(groups.items())
    random.shuffle(group_items)

    buffer: list[dict] = []
    shard_idx = 0
    total_docs = 0
    total_chars = 0
    total_staging_texts = 0
    total_unmatched = 0
    source_counts: dict[str, int] = defaultdict(int)
    title_found = 0
    title_missing = 0

    def flush_buffer() -> None:
        nonlocal shard_idx
        if not buffer:
            return
        n_shards = len(buffer) // docs_per_shard
        if n_shards == 0:
            return
        n_use = n_shards * docs_per_shard
        to_write = buffer[:n_use]
        random.shuffle(to_write)

        for i in range(n_shards):
            batch = to_write[i * docs_per_shard : (i + 1) * docs_per_shard]
            write_shard(batch, output_dir, shard_idx)
            shard_idx += 1

        del buffer[:n_use]
        gc.collect()
        _print(
            f"    Shards written: {shard_idx} | "
            f"docs: {total_docs:,} | "
            f"~{total_chars / CHARS_PER_TOKEN / 1e9:.2f}B tokens"
        )

    t0 = time.time()

    for group_idx, (filtered_path, (staging_paths, label, raw_stem)) in enumerate(group_items):
        # Load all staging texts for this filtered shard group
        staging_texts: set[str] = set()
        for sp in staging_paths:
            try:
                t = pq.read_table(sp, columns=["text"])
                staging_texts.update(t.column("text").to_pylist())
                del t
            except Exception as e:
                _print(f"  WARNING: Failed to read {sp}: {e}")

        n_staging = len(staging_texts)
        total_staging_texts += n_staging
        if not staging_texts:
            continue

        if not os.path.exists(filtered_path):
            _print(f"  WARNING: Filtered shard not found: {filtered_path}")
            total_unmatched += n_staging
            continue

        # For newspapers, lazily load titles from the corresponding raw shard
        if label == "newspapers":
            raw_shard = raw_dir / "newspapers" / f"{raw_stem}.parquet"
            local_titles = load_newspaper_titles(raw_shard)
        else:
            local_titles = None  # use global title_lookup

        # Read filtered shard row-group by row-group to limit memory
        try:
            pf = pq.ParquetFile(filtered_path)
        except Exception as e:
            _print(f"  WARNING: Failed to open {filtered_path}: {e}")
            total_unmatched += n_staging
            continue

        for rg_idx in range(pf.num_row_groups):
            if not staging_texts:
                break
            table = pf.read_row_group(rg_idx, columns=METADATA_COLS)
            texts = table.column("text").to_pylist()
            sources = table.column("source").to_pylist()
            doc_ids = table.column("doc_id").to_pylist()
            years = table.column("year").to_pylist()
            ocr_scores = table.column("ocr_score").to_pylist()
            legibilities = table.column("legibility").to_pylist()
            del table

            for i, text in enumerate(texts):
                if text in staging_texts:
                    did = doc_ids[i] or ""
                    # Title lookup: use local (newspaper) or global (books)
                    if local_titles is not None:
                        title = local_titles.get(did, "") if did else ""
                    else:
                        title = title_lookup.get(did, "") if did else ""
                    if title:
                        title_found += 1
                    else:
                        title_missing += 1

                    ocr = float(ocr_scores[i]) if ocr_scores[i] is not None else -1.0
                    leg = float(legibilities[i]) if legibilities[i] is not None else -1.0
                    year = int(years[i]) if years[i] is not None else 0
                    source = sources[i] or ""

                    buffer.append({
                        "text": text,
                        "year": year,
                        "title": title,
                        "source": source,
                        "ocr_score": ocr,
                        "legibility": leg,
                    })

                    total_docs += 1
                    total_chars += len(text)
                    source_counts[source] += 1
                    staging_texts.discard(text)

        if staging_texts:
            total_unmatched += len(staging_texts)
        del staging_texts
        if local_titles is not None:
            del local_titles

        # Flush when buffer is large enough
        if len(buffer) >= docs_per_shard * 5:
            flush_buffer()

        if (group_idx + 1) % 500 == 0:
            elapsed = time.time() - t0
            _print(
                f"  Progress: {group_idx + 1}/{len(group_items)} groups | "
                f"{total_docs:,} docs | {elapsed:.0f}s"
            )

    # Final flush — write whatever remains, even if < docs_per_shard
    if buffer:
        random.shuffle(buffer)
        remaining = len(buffer)
        n_full = remaining // docs_per_shard
        for i in range(n_full):
            batch = buffer[i * docs_per_shard : (i + 1) * docs_per_shard]
            write_shard(batch, output_dir, shard_idx)
            shard_idx += 1
        leftover = buffer[n_full * docs_per_shard :]
        if leftover:
            write_shard(leftover, output_dir, shard_idx)
            shard_idx += 1
        buffer.clear()

    elapsed = time.time() - t0

    # Report
    _print(f"\n{'=' * 60}")
    _print("RESULTS")
    _print("=" * 60)
    _print(f"  Total staging texts:  {total_staging_texts:,}")
    _print(f"  Matched to metadata:  {total_docs:,} ({100 * total_docs / max(total_staging_texts, 1):.1f}%)")
    _print(f"  Unmatched:            {total_unmatched:,}")
    _print(f"  Output documents:     {total_docs:,}")
    _print(f"  Output shards:        {shard_idx}")
    _print(f"  Est. tokens:          {total_chars / CHARS_PER_TOKEN / 1e9:.2f}B")
    _print(f"  Titles found:         {title_found:,} ({100 * title_found / max(total_docs, 1):.1f}%)")
    _print(f"  Titles missing:       {title_missing:,}")
    _print(f"  Elapsed:              {elapsed:.0f}s ({elapsed / 3600:.2f}h)")
    _print(f"\n  Source breakdown:")
    for src, cnt in sorted(source_counts.items(), key=lambda x: -x[1]):
        _print(f"    {src}: {cnt:,}")
    _print(f"\n  Output: {output_dir}")

    # Spot-check: print a few sample rows
    first_shard = output_dir / "shard_00000.parquet"
    if first_shard.exists():
        _print(f"\n  Sample rows from shard_00000:")
        t = pq.read_table(first_shard)
        for i in range(min(3, t.num_rows)):
            row = {col: t.column(col)[i].as_py() for col in t.schema.names}
            text_preview = (row["text"][:100] + "...") if len(row["text"]) > 100 else row["text"]
            _print(f"    [{i}] year={row['year']} source={row['source']!r} "
                   f"title={row['title'][:50]!r} ocr={row['ocr_score']:.2f} "
                   f"leg={row['legibility']:.2f}")
            _print(f"        text={text_preview!r}")
        del t


# ============================================================================
# Step 4: HuggingFace upload
# ============================================================================

def upload_to_hf(output_dir: Path, repo_id: str) -> None:
    """Upload dataset to HuggingFace using upload_large_folder."""
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True)

    # Write dataset card
    readme_content = """\
---
dataset_info:
  features:
    - name: text
      dtype: string
    - name: year
      dtype: int64
    - name: title
      dtype: string
    - name: source
      dtype: string
    - name: ocr_score
      dtype: float64
    - name: legibility
      dtype: float64
---

# Pre-1900 Corpus

Cleaned corpus of pre-1900 English-language texts with full metadata.

## Schema

| Column | Type | Description |
|--------|------|-------------|
| `text` | string | Full document text |
| `year` | int64 | Publication year |
| `title` | string | Book title or newspaper name |
| `source` | string | Source dataset identifier |
| `ocr_score` | float64 | OCR confidence score (-1.0 if not available) |
| `legibility` | float64 | Legibility score (-1.0 if not available) |

## Sources

- **Institutional books** — HathiTrust, Internet Archive, and other digitized book collections
- **British Library books** — TheBritishLibrary/blbooks
- **Historical newspapers** — dell-research-harvard/AmericanStories

## Filtering methodology

Documents were cleaned and filtered through a multi-stage pipeline:

1. **OCR cleanup** — removal of common OCR artifacts, Google/HathiTrust boilerplate,
   library stamps, and unicode normalization
2. **Quality filtering** — token frequency prior-based filtering as a cheap proxy
   for perplexity, removing garbled or low-quality OCR output
3. **Anachronism detection** — three-tier post-1900 physics filter to remove
   mislabeled modern texts:
   - *Always reject*: unambiguous post-1900 terms (photon, spacetime, transistor, etc.)
   - *Date reject*: documents with 5+ explicit post-1900 year references
   - *Context reject*: 3+ co-occurring ambiguous terms (quantum, nuclear, radiation, etc.)
"""
    readme_path = output_dir / "README.md"
    readme_path.write_text(readme_content)

    print(f"  Uploading {output_dir} to {repo_id}...")
    api.upload_large_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(output_dir),
    )
    print(f"  Done: https://huggingface.co/datasets/{repo_id}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Build standardized pre-1900 HuggingFace dataset with metadata",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--staging-dir", type=Path,
        default=Path("/mnt/main0/data/michaelhla/pre1900_full_clean/_staging"),
        help="Directory with text-only staging parquets from clean_full_corpus.py",
    )
    parser.add_argument(
        "--filtered-dir", type=Path,
        default=Path("/mnt/main0/data/michaelhla/pre1900_filtered"),
        help="Filtered data directory (has metadata columns)",
    )
    parser.add_argument(
        "--raw-dir", type=Path,
        default=Path("/mnt/main0/data/michaelhla/pre1900_raw"),
        help="Raw data directory (for title lookup)",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("/mnt/main0/data/michaelhla/pre1900_standardized"),
        help="Output directory for standardized parquet shards",
    )
    parser.add_argument("--docs-per-shard", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--upload", action="store_true",
                        help="Upload to HuggingFace after building")
    parser.add_argument("--hf-repo", type=str, default="mhla/pre1900-corpus",
                        help="HuggingFace dataset repo ID")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Build title lookup
    print("=" * 60)
    print("STEP 1: Building title lookup from raw data")
    print("=" * 60)
    title_lookup = build_title_lookup(args.raw_dir)

    # Steps 2 & 3: Process staging files, recover metadata, write output
    print(f"\n{'=' * 60}")
    print("STEP 2-3: Recovering metadata and writing output shards")
    print("=" * 60)
    process_and_write(
        staging_dir=args.staging_dir,
        filtered_dir=args.filtered_dir,
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        title_lookup=title_lookup,
        docs_per_shard=args.docs_per_shard,
        seed=args.seed,
    )

    # Step 4: Upload to HuggingFace
    if args.upload:
        print(f"\n{'=' * 60}")
        print(f"STEP 4: Uploading to HuggingFace ({args.hf_repo})")
        print("=" * 60)
        upload_to_hf(args.output_dir, args.hf_repo)


if __name__ == "__main__":
    main()
