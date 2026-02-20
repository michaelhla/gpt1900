#!/usr/bin/env python3
"""
HuggingFace Historical Text Downloader

Downloads pre-1900 English texts from HuggingFace datasets and stores directly
in parquet shards:
- Books: institutional/institutional-books-1.0, TheBritishLibrary/blbooks
- Newspapers: dell-research-harvard/AmericanStories

Usage:
    python scripts/pre1900_scripts/hf_download.py --outdir ./data/pre1900_raw --max-docs 100000
    python scripts/pre1900_scripts/hf_download.py --outdir ./data/pre1900_raw --sources books newspapers --docs-per-shard 10000

All documents are from before January 1, 1900 (year <= 1899).
"""

from __future__ import annotations
import argparse
import importlib.util
import json
import logging
import sys
import time
from pathlib import Path
from typing import Iterator

from datasets import load_dataset
from huggingface_hub import hf_hub_download
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


VERSION = "1.0"
# Strict cutoff: all documents must be from 1899 or earlier
YEAR_CUTOFF = 1899

# Cache for imported builder modules
_builder_modules: dict[str, object] = {}


def _import_hf_builder(repo_id: str, script_name: str):
    """Download a HuggingFace dataset builder script and import it locally.
    Bypasses trust_remote_code restrictions in datasets>=4.0."""
    key = f"{repo_id}/{script_name}"
    if key in _builder_modules:
        return _builder_modules[key]

    script_path = hf_hub_download(repo_id=repo_id, filename=script_name, repo_type="dataset")
    mod_name = f"_local_{script_name.replace('.py', '')}"
    spec = importlib.util.spec_from_file_location(mod_name, script_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)

    _builder_modules[key] = mod
    return mod


def setup_logger(verbose: bool = False) -> logging.Logger:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logger = logging.getLogger("hf_download")

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def extract_year(date_field) -> int | None:
    """Extract year from various date formats."""
    if date_field is None:
        return None

    if isinstance(date_field, int):
        return date_field

    # datetime objects (e.g., blbooks "date" column)
    if hasattr(date_field, "year"):
        return date_field.year

    if isinstance(date_field, str):
        import re
        match = re.search(r'\b(1[4-9]\d{2}|20[0-2]\d)\b', date_field)
        if match:
            return int(match.group(1))

    return None


def stream_institutional_books(
    max_docs: int,
    logger: logging.Logger,
    skip_files: int = 0,
    on_row: callable = None,
) -> Iterator[tuple[str, dict]]:
    """
    Stream books from institutional/institutional-books-1.0.

    Args:
        skip_files: Number of parquet files to skip entirely (for fast resume).
        on_row: Called with total_files_processed after each yielded doc.

    Yields: (text, metadata)
    """
    logger.info("Loading institutional-books-1.0 dataset (streaming)...")

    try:
        from huggingface_hub import HfApi
        api = HfApi()
        all_files = sorted(
            f.rfilename for f in
            api.list_repo_tree("institutional/institutional-books-1.0", repo_type="dataset", path_in_repo="data")
            if f.rfilename.endswith(".parquet")
        )
        logger.info(f"Found {len(all_files)} parquet files in dataset")

        if skip_files > 0:
            logger.info(f"Skipping {skip_files} files, starting from file {skip_files}")
            all_files = all_files[skip_files:]

        if not all_files:
            logger.info("All files already processed")
            return

        dataset = load_dataset(
            "institutional/institutional-books-1.0",
            split="train",
            streaming=True,
            data_files={"train": all_files},
        )
    except Exception as e:
        logger.error(f"Failed to load institutional-books-1.0: {e}")
        return

    count = 0
    raw_rows = 0
    for row in tqdm(dataset, desc="institutional-books"):
        raw_rows += 1
        if count >= max_docs:
            break

        # Check year - must be <= 1899 (before 1900)
        year = extract_year(row.get("date1_src"))
        if year is None or year > YEAR_CUTOFF:
            continue

        # English only (ISO 639-3 codes: "eng")
        lang = (row.get("language_gen") or row.get("language_src") or "").lower().strip()
        if lang != "eng":
            continue

        # Get text - prefer generated (cleaned) over source
        text_pages = row.get("text_by_page_gen") or row.get("text_by_page_src")
        if not text_pages:
            continue

        # Join pages into single document
        if isinstance(text_pages, list):
            text = "\n\n".join(str(p) for p in text_pages if p)
        else:
            text = str(text_pages)

        # Skip very short documents
        if len(text) < 5000:
            continue

        metadata = {
            "source": "institutional-books",
            "doc_id": f"inst_{row.get('barcode_src', count)}",
            "title": row.get("title_src", ""),
            "author": row.get("author_src", ""),
            "year": year,
            "language": row.get("language_gen", "") or row.get("language_src", ""),
            "ocr_score": row.get("ocr_score_gen") if row.get("ocr_score_gen") is not None else -1,
            "ocr_score_src": row.get("ocr_score_src") if row.get("ocr_score_src") is not None else -1,
            "token_count": row.get("token_count_o200k_base_gen") or -1,
            "topic": row.get("topic_or_subject_src", "") or row.get("topic_or_subject_gen", ""),
            "genre": row.get("genre_or_form_src", ""),
            "page_count": row.get("page_count_src") or -1,
        }

        count += 1
        if on_row:
            # Total files done = previously skipped + current position (~100 rows/file)
            on_row(skip_files + raw_rows // 100)
        yield text, metadata

    logger.info(f"Yielded {count} documents from institutional-books")


def stream_british_library_books(
    max_docs: int,
    logger: logging.Logger,
) -> Iterator[tuple[str, dict]]:
    """
    Stream books from TheBritishLibrary/blbooks.

    This dataset has one row per page, so we aggregate pages by book.

    Yields: (text, metadata)
    """
    logger.info("Loading TheBritishLibrary/blbooks via local builder...")

    try:
        mod = _import_hf_builder("TheBritishLibrary/blbooks", "blbooks.py")
        builder = mod.BritishLibraryBooks()
        builder.download_and_prepare()
        dataset = builder.as_dataset(split="train")
    except Exception as e:
        logger.error(f"Failed to load blbooks: {e}")
        return

    # Aggregate pages by record_id
    current_book_id = None
    current_pages = []
    current_metadata = {}
    current_ocr_scores = []
    count = 0
    skip_current = False

    def _yield_book(pages, metadata, ocr_scores):
        """Finalize a book: join pages, compute avg OCR, attach metadata."""
        text = "\n\n".join(pages)
        if len(text) < 5000:
            return None, None
        if ocr_scores:
            metadata["ocr_score"] = round(sum(ocr_scores) / len(ocr_scores), 3)
        else:
            metadata["ocr_score"] = -1.0
        return text, metadata

    for row in tqdm(dataset, desc="blbooks"):
        if count >= max_docs:
            break

        record_id = row.get("record_id")

        # New book - yield the previous one if valid
        if record_id != current_book_id and current_book_id is not None:
            if not skip_current and current_pages:
                text, meta = _yield_book(current_pages, current_metadata, current_ocr_scores)
                if text is not None:
                    count += 1
                    yield text, meta

            current_pages = []
            current_metadata = {}
            current_ocr_scores = []
            skip_current = False

        current_book_id = record_id

        # Check year - must be <= 1899
        year = extract_year(row.get("date"))
        if year is None or year > YEAR_CUTOFF:
            skip_current = True
            continue

        # English only (no label = skip)
        lang = (row.get("Language_1") or "").strip().lower()
        if lang != "english":
            skip_current = True
            continue

        # Skip empty pages
        if row.get("empty_pg"):
            continue

        # Add page text
        page_text = row.get("text", "")
        if page_text:
            current_pages.append(page_text)

        # Accumulate per-page OCR word confidence
        wc = row.get("mean_wc_ocr")
        if wc is not None:
            current_ocr_scores.append(float(wc))

        # Store metadata from first page
        if not current_metadata:
            current_metadata = {
                "source": "blbooks",
                "doc_id": f"bl_{record_id}",
                "title": row.get("title", ""),
                "author": row.get("name", ""),
                "year": year,
                "language": row.get("Language_1", ""),
            }

    # Don't forget the last book
    if not skip_current and current_pages and current_book_id:
        text, meta = _yield_book(current_pages, current_metadata, current_ocr_scores)
        if text is not None:
            yield text, meta

    logger.info(f"Yielded {count} documents from blbooks")


def stream_american_stories(
    max_docs: int,
    logger: logging.Logger,
    resume_from_year: int = 1774,
    skip_docs_in_first_year: int = 0,
) -> Iterator[tuple[str, dict]]:
    """
    Stream newspaper articles from dell-research-harvard/AmericanStories.

    Uses the scan-level config (content_regions) to extract per-article
    legibility scores from bounding box annotations.

    Note: This dataset is split by year (1774-1963), not train/test.
    We iterate through each year and stream articles.

    For resume support, pass resume_from_year to skip completed years entirely
    (avoids re-downloading tar.gz files), and skip_docs_in_first_year to skip
    partially-completed years.

    Yields: (text, metadata)
    """
    logger.info("Loading AmericanStories via local builder (scan-level for legibility)...")

    # Build list of years to load (only pre-1900), starting from resume point
    years_to_load = [str(y) for y in range(resume_from_year, min(YEAR_CUTOFF + 1, 1900))]
    logger.info(f"Will load {len(years_to_load)} years: {years_to_load[0]} to {years_to_load[-1]}")

    try:
        mod = _import_hf_builder(
            "dell-research-harvard/AmericanStories", "AmericanStories.py"
        )
    except Exception as e:
        logger.error(f"Failed to load AmericanStories builder: {e}")
        return

    count = 0
    first_year = True

    # Load and iterate through each year using scan-level config
    for year_str in years_to_load:
        if count >= max_docs:
            break

        try:
            builder = mod.AmericanStories(
                config_name="subset_years_content_regions", year_list=[year_str]
            )
            builder.download_and_prepare()
            year_data = builder.as_dataset()[year_str]
        except Exception as e:
            logger.warning(f"Failed to load year {year_str}: {e}")
            continue

        year_int = int(year_str)
        skipped_in_year = 0
        docs_to_skip = skip_docs_in_first_year if first_year else 0

        for row in tqdm(year_data, desc=f"AmericanStories-{year_str}"):
            if count >= max_docs:
                break

            try:
                raw = json.loads(row["raw_data_string"])
            except (json.JSONDecodeError, KeyError):
                continue

            bboxes = raw.get("bboxes", [])
            bbox_by_id = {b["id"]: b for b in bboxes}

            # Extract newspaper name from LCCN metadata
            lccn_meta = raw.get("lccn", {})
            newspaper = lccn_meta.get("title", "")
            edition_meta = raw.get("edition", {})
            date_str = edition_meta.get("date", "")

            for article in raw.get("full articles", []):
                if count >= max_docs:
                    break

                text = article.get("article", "")
                if not text or len(text) < 500:
                    continue

                # Compute legibility: fraction of article's bbox regions marked Legible
                obj_ids = article.get("object_ids", [])
                legs = [bbox_by_id[oid].get("legibility", "")
                        for oid in obj_ids if oid in bbox_by_id]
                if legs:
                    legibility = round(sum(1 for l in legs if l == "Legible") / len(legs), 3)
                else:
                    legibility = -1.0

                # Skip docs already saved (resume within a year)
                if skipped_in_year < docs_to_skip:
                    skipped_in_year += 1
                    continue

                metadata = {
                    "source": "AmericanStories",
                    "doc_id": article.get("id", f"as_{count}"),
                    "newspaper": newspaper,
                    "headline": article.get("headline", ""),
                    "byline": article.get("byline", ""),
                    "date": date_str,
                    "year": extract_year(date_str) or year_int,
                    "legibility": legibility,
                }

                count += 1
                yield text, metadata

                if count % 10000 == 0:
                    logger.info(f"AmericanStories progress: {count} articles")

        first_year = False

    logger.info(f"Yielded {count} documents from AmericanStories")


class ParquetShardWriter:
    """Writes documents to parquet files in shards. Accepts arbitrary metadata columns.

    Supports resume: on init, detects existing shards and exposes `docs_to_skip`
    so the caller can fast-forward through the stream. Also saves/loads a
    _resume.json with `raw_rows_seen` so streaming sources can use dataset.skip().
    """

    def __init__(self, outdir: Path, docs_per_shard: int, prefix: str = "shard"):
        self.outdir = outdir
        self.docs_per_shard = docs_per_shard
        self.prefix = prefix

        self.outdir.mkdir(parents=True, exist_ok=True)

        # Resume: count docs in existing shards
        existing = sorted(self.outdir.glob(f"{prefix}_*.parquet"))
        self.docs_to_skip = 0
        self.current_shard = 0
        if existing:
            for shard_path in existing:
                pf = pq.ParquetFile(shard_path)
                self.docs_to_skip += pf.metadata.num_rows
            self.current_shard = len(existing)

        # Load resume metadata (for streaming skip)
        self._resume_path = self.outdir / "_resume.json"
        self._resume_data = {}
        if self._resume_path.exists():
            self._resume_data = json.loads(self._resume_path.read_text())

        self.buffer: list[dict] = []
        self.total_docs = 0
        self.total_bytes = 0

    def add(self, text: str, metadata: dict):
        """Add a document to the buffer, flush if full."""
        row = {"text": text, **metadata}
        self.buffer.append(row)

        self.total_docs += 1
        self.total_bytes += len(text.encode("utf-8"))

        if len(self.buffer) >= self.docs_per_shard:
            self._flush()

    def save_resume(self, **kwargs):
        """Update resume metadata."""
        self._resume_data.update(kwargs)
        self._resume_path.write_text(json.dumps(self._resume_data))

    def _flush(self):
        """Write current buffer to a parquet shard."""
        if not self.buffer:
            return

        # Build column-oriented dict from row-oriented buffer
        columns: dict[str, list] = {}
        for row in self.buffer:
            for k, v in row.items():
                columns.setdefault(k, []).append(v)

        table = pa.table(columns)
        shard_path = self.outdir / f"{self.prefix}_{self.current_shard:05d}.parquet"
        pq.write_table(table, shard_path, compression="snappy")

        # Persist resume metadata alongside shard
        if self._resume_data:
            self._resume_path.write_text(json.dumps(self._resume_data))

        self.current_shard += 1
        self.buffer = []

    def finalize(self) -> tuple[int, int, int]:
        """Flush remaining buffer and return stats."""
        self._flush()
        return self.total_docs, self.total_bytes, self.current_shard


def main():
    parser = argparse.ArgumentParser(
        description="Download pre-1900 English texts from HuggingFace to parquet shards",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--outdir", "-o", type=Path, default=Path("/mnt/main0/data/michaelhla/pre1900_raw"),
        help="Directory for parquet output"
    )
    parser.add_argument(
        "--max-docs", type=int, default=0,
        help="Max documents per source (0 = unlimited)"
    )
    parser.add_argument(
        "--docs-per-shard", type=int, default=10000,
        help="Documents per parquet shard"
    )
    parser.add_argument(
        "--sources", nargs="+",
        choices=["institutional", "blbooks", "newspapers", "all"],
        default=["all"],
        help="Which sources to download"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    args = parser.parse_args()
    logger = setup_logger(args.verbose)

    max_docs = args.max_docs if args.max_docs > 0 else 10**18

    logger.info(f"Downloading documents up to {YEAR_CUTOFF} (pre-1900)")

    # Determine which sources to use
    sources = set(args.sources)
    if "all" in sources:
        sources = {"institutional", "blbooks", "newspapers"}

    args.outdir.mkdir(parents=True, exist_ok=True)
    start_time = time.time()

    # Each source gets its own writer/directory so a crash doesn't lose prior sources

    if "institutional" in sources:
        logger.info("=" * 60)
        logger.info("Downloading INSTITUTIONAL BOOKS")
        logger.info("=" * 60)

        writer = ParquetShardWriter(args.outdir / "institutional", args.docs_per_shard, prefix="inst")
        skip_files = writer._resume_data.get("files_done", 0)
        if writer.docs_to_skip:
            logger.info(f"Resuming: {writer.docs_to_skip} docs saved, skipping {skip_files} files")

        def _track_row(files_done):
            writer.save_resume(files_done=files_done)

        for text, metadata in stream_institutional_books(max_docs, logger, skip_files=skip_files, on_row=_track_row):
            writer.add(text, metadata)
            if writer.total_docs % 1000 == 0:
                logger.info(f"Progress: {writer.total_docs} docs, {writer.total_bytes / 1e9:.2f} GB")

        total_docs, total_bytes, n_shards = writer.finalize()
        logger.info(f"Institutional books checkpoint: {total_docs} docs, {total_bytes / 1e9:.2f} GB, {n_shards} shards")

    if "blbooks" in sources:
        logger.info("=" * 60)
        logger.info("Downloading BRITISH LIBRARY BOOKS")
        logger.info("=" * 60)

        writer = ParquetShardWriter(args.outdir / "blbooks", args.docs_per_shard, prefix="bl")
        skip = writer.docs_to_skip
        if skip:
            logger.info(f"Resuming: skipping {skip} already-downloaded docs")

        skipped = 0
        for text, metadata in stream_british_library_books(max_docs + skip, logger):
            if skipped < skip:
                skipped += 1
                continue
            writer.add(text, metadata)
            if writer.total_docs % 1000 == 0:
                logger.info(f"Progress: {writer.total_docs} docs, {writer.total_bytes / 1e9:.2f} GB")

        total_docs, total_bytes, n_shards = writer.finalize()
        logger.info(f"BL books checkpoint: {total_docs} docs, {total_bytes / 1e9:.2f} GB, {n_shards} shards")

    if "newspapers" in sources:
        logger.info("=" * 60)
        logger.info("Downloading AMERICAN STORIES")
        logger.info("=" * 60)

        writer = ParquetShardWriter(args.outdir / "newspapers", args.docs_per_shard, prefix="news")

        # Year-based resume: find max year and count docs in that year from existing shards
        resume_year = 1774
        skip_in_year = 0
        if writer.docs_to_skip > 0:
            existing_shards = sorted((args.outdir / "newspapers").glob("news_*.parquet"))
            max_year = 1774
            docs_in_max_year = 0
            for sp in existing_shards:
                t = pq.read_table(sp, columns=["year"])
                years = t.column("year").to_pylist()
                for y in years:
                    if y > max_year:
                        max_year = y
                        docs_in_max_year = 1
                    elif y == max_year:
                        docs_in_max_year += 1
            resume_year = max_year
            skip_in_year = docs_in_max_year
            logger.info(f"Resuming from year {resume_year} (skipping {skip_in_year} docs in that year, {writer.docs_to_skip} total already saved)")

        for text, metadata in stream_american_stories(max_docs, logger, resume_from_year=resume_year, skip_docs_in_first_year=skip_in_year):
            writer.add(text, metadata)
            if writer.total_docs % 10000 == 0:
                logger.info(f"Progress: {writer.total_docs} docs, {writer.total_bytes / 1e9:.2f} GB")

        total_docs, total_bytes, n_shards = writer.finalize()
        logger.info(f"Newspapers checkpoint: {total_docs} docs, {total_bytes / 1e9:.2f} GB, {n_shards} shards")

    elapsed = time.time() - start_time

    # Summary
    print(f"\n{'='*60}")
    print("Download complete!")
    print(f"  Year cutoff: <= {YEAR_CUTOFF} (pre-1900)")
    print(f"  Elapsed: {elapsed:.1f}s")
    print(f"  Output: {args.outdir}")

    # List output shards
    for subdir in ["institutional", "blbooks", "newspapers"]:
        subpath = args.outdir / subdir
        if subpath.exists():
            shards = list(subpath.glob("*.parquet"))
            if shards:
                total_rows = 0
                for shard in shards:
                    pf = pq.ParquetFile(shard)
                    total_rows += pf.metadata.num_rows
                print(f"  {subdir}/: {len(shards)} shards, {total_rows:,} documents")


if __name__ == "__main__":
    main()
