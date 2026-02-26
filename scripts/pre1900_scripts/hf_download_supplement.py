#!/usr/bin/env python3
"""
HuggingFace Historical Text Downloader -- 1900-1914 Supplement

Downloads English texts from 1900-1914 from HuggingFace datasets and stores
directly in parquet shards. Uses parallel file downloads for institutional-books
(~12 min with 32 threads vs ~7 hours serial).

Sources:
- Books: institutional/institutional-books-1.0, TheBritishLibrary/blbooks
- Newspapers: dell-research-harvard/AmericanStories

Usage:
    python scripts/pre1900_scripts/hf_download_supplement.py \
        --outdir /mnt/main0/data/michaelhla/pre1915_supplement_raw
"""

from __future__ import annotations
import argparse
import importlib.util
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterator

from huggingface_hub import HfApi, hf_hub_download
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


VERSION = "2.0"
YEAR_MIN = 1900
YEAR_MAX = 1914

# Lightweight columns for year/language filtering
_FILTER_COLS = ["date1_src", "language_src", "language_gen"]

_builder_modules: dict[str, object] = {}


def _import_hf_builder(repo_id: str, script_name: str):
    """Download a HuggingFace dataset builder script and import it locally."""
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
    level = logging.DEBUG if verbose else logging.INFO
    logger = logging.getLogger("hf_download_supplement")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S")
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def extract_year(date_field) -> int | None:
    if date_field is None:
        return None
    if isinstance(date_field, int):
        return date_field
    if hasattr(date_field, "year"):
        return date_field.year
    if isinstance(date_field, str):
        import re
        match = re.search(r'\b(1[4-9]\d{2}|20[0-2]\d)\b', date_field)
        if match:
            return int(match.group(1))
    return None


def _process_institutional_file(file_path: str) -> list[tuple[str, dict]]:
    """Download one parquet file and extract 1900-1914 English documents.

    Uses hf_hub_download (cached after first call), then filters locally.
    Phase 1: read only date+language columns (~296ms).
    Phase 2: if matches found, read full file and extract (~1.7s).
    """
    repo_id = "institutional/institutional-books-1.0"
    local_path = hf_hub_download(repo_id=repo_id, filename=file_path, repo_type="dataset")

    # Phase 1: lightweight column read
    try:
        ft = pq.read_table(local_path, columns=_FILTER_COLS)
    except Exception:
        return []

    dates = ft.column("date1_src").to_pylist()
    lang_gen = ft.column("language_gen").to_pylist()
    lang_src = ft.column("language_src").to_pylist()

    matching_idx = []
    for i, (d, lg, ls) in enumerate(zip(dates, lang_gen, lang_src)):
        year = extract_year(d)
        if year is None or year < YEAR_MIN or year > YEAR_MAX:
            continue
        lang = (lg or ls or "").lower().strip()
        if lang == "eng":
            matching_idx.append(i)

    if not matching_idx:
        return []

    # Phase 2: read full file, take only matching rows
    full = pq.read_table(local_path).take(matching_idx)

    results = []
    for i in range(full.num_rows):
        row = {col: full.column(col)[i].as_py() for col in full.column_names}

        text_pages = row.get("text_by_page_gen") or row.get("text_by_page_src")
        if not text_pages:
            continue

        if isinstance(text_pages, list):
            text = "\n\n".join(str(p) for p in text_pages if p)
        else:
            text = str(text_pages)

        if len(text) < 5000:
            continue

        year = extract_year(row["date1_src"])
        metadata = {
            "source": "institutional-books",
            "doc_id": f"inst_{row.get('barcode_src', '')}",
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
        results.append((text, metadata))

    return results


def download_institutional_books(
    writer: ParquetShardWriter,
    logger: logging.Logger,
    max_docs: int,
    workers: int = 32,
):
    """Download institutional books using parallel file processing.

    Each file is downloaded to HF cache (or read from cache if already present),
    then filtered locally. With 32 threads, processes ~9800 files in ~12 minutes.
    """
    logger.info("Listing files in institutional/institutional-books-1.0...")
    api = HfApi()
    all_files = sorted(
        f.rfilename for f in
        api.list_repo_tree(
            "institutional/institutional-books-1.0",
            repo_type="dataset",
            path_in_repo="data",
        )
        if f.rfilename.endswith(".parquet")
    )
    logger.info(f"Found {len(all_files)} parquet files")

    # Resume: skip already-processed files
    skip_files = writer._resume_data.get("files_done", 0)
    if skip_files > 0 and writer.docs_to_skip == 0:
        # Resume file claims files done but no shards exist - reset
        logger.warning(
            f"Resume says {skip_files} files done but no shards found, resetting"
        )
        skip_files = 0
        writer._resume_data = {}
        writer.save_resume(files_done=0)
    elif skip_files > 0:
        logger.info(
            f"Resuming: skipping {skip_files} files "
            f"({writer.docs_to_skip} docs already saved)"
        )

    remaining = all_files[skip_files:]
    if not remaining:
        logger.info("All files already processed")
        return

    logger.info(f"Processing {len(remaining)} files with {workers} threads...")

    completed = 0
    files_with_matches = 0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_process_institutional_file, f): f
            for f in remaining
        }

        pbar = tqdm(total=len(futures), desc="institutional-books", unit="files")
        for future in as_completed(futures):
            pbar.update(1)
            completed += 1

            try:
                results = future.result()
            except Exception as e:
                logger.warning(f"File failed: {e}")
                continue

            if results:
                files_with_matches += 1
                for text, metadata in results:
                    if writer.total_docs >= max_docs:
                        break
                    writer.add(text, metadata)

            # Save progress every 200 files
            if completed % 200 == 0:
                writer.save_resume(files_done=skip_files + completed)
                pbar.set_postfix(
                    docs=writer.total_docs,
                    hits=files_with_matches,
                    gb=f"{writer.total_bytes / 1e9:.1f}",
                )

            if writer.total_docs >= max_docs:
                executor.shutdown(wait=False, cancel_futures=True)
                break

        pbar.close()

    writer.save_resume(files_done=skip_files + completed)
    logger.info(
        f"Institutional books: {writer.total_docs} docs from "
        f"{files_with_matches}/{completed} files with matches"
    )


def stream_british_library_books(
    max_docs: int,
    logger: logging.Logger,
) -> Iterator[tuple[str, dict]]:
    """Stream books from TheBritishLibrary/blbooks.

    This dataset has one row per page, so we aggregate pages by book.
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

    current_book_id = None
    current_pages = []
    current_metadata = {}
    current_ocr_scores = []
    count = 0
    skip_current = False

    def _yield_book(pages, metadata, ocr_scores):
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

        year = extract_year(row.get("date"))
        if year is None or year < YEAR_MIN or year > YEAR_MAX:
            skip_current = True
            continue

        lang = (row.get("Language_1") or "").strip().lower()
        if lang != "english":
            skip_current = True
            continue

        if row.get("empty_pg"):
            continue

        page_text = row.get("text", "")
        if page_text:
            current_pages.append(page_text)

        wc = row.get("mean_wc_ocr")
        if wc is not None:
            current_ocr_scores.append(float(wc))

        if not current_metadata:
            current_metadata = {
                "source": "blbooks",
                "doc_id": f"bl_{record_id}",
                "title": row.get("title", ""),
                "author": row.get("name", ""),
                "year": year,
                "language": row.get("Language_1", ""),
            }

    if not skip_current and current_pages and current_book_id:
        text, meta = _yield_book(current_pages, current_metadata, current_ocr_scores)
        if text is not None:
            yield text, meta

    logger.info(f"Yielded {count} documents from blbooks")


def stream_american_stories(
    max_docs: int,
    logger: logging.Logger,
    resume_from_year: int = 1900,
    skip_docs_in_first_year: int = 0,
) -> Iterator[tuple[str, dict]]:
    """Stream newspaper articles from dell-research-harvard/AmericanStories.

    Years in [YEAR_MIN, YEAR_MAX] range. Dataset is split by year so this
    is already efficient.
    """
    logger.info("Loading AmericanStories via local builder (scan-level for legibility)...")

    years_to_load = [str(y) for y in range(max(resume_from_year, YEAR_MIN), YEAR_MAX + 1)]
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

                obj_ids = article.get("object_ids", [])
                legs = [bbox_by_id[oid].get("legibility", "")
                        for oid in obj_ids if oid in bbox_by_id]
                if legs:
                    legibility = round(sum(1 for l in legs if l == "Legible") / len(legs), 3)
                else:
                    legibility = -1.0

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
    """Writes documents to parquet files in shards with resume support."""

    def __init__(self, outdir: Path, docs_per_shard: int, prefix: str = "shard"):
        self.outdir = outdir
        self.docs_per_shard = docs_per_shard
        self.prefix = prefix

        self.outdir.mkdir(parents=True, exist_ok=True)

        existing = sorted(self.outdir.glob(f"{prefix}_*.parquet"))
        self.docs_to_skip = 0
        self.current_shard = 0
        if existing:
            for shard_path in existing:
                pf = pq.ParquetFile(shard_path)
                self.docs_to_skip += pf.metadata.num_rows
            self.current_shard = len(existing)

        self._resume_path = self.outdir / "_resume.json"
        self._resume_data = {}
        if self._resume_path.exists():
            self._resume_data = json.loads(self._resume_path.read_text())

        self.buffer: list[dict] = []
        self.total_docs = 0
        self.total_bytes = 0

    def add(self, text: str, metadata: dict):
        row = {"text": text, **metadata}
        self.buffer.append(row)

        self.total_docs += 1
        self.total_bytes += len(text.encode("utf-8"))

        if len(self.buffer) >= self.docs_per_shard:
            self._flush()

    def save_resume(self, **kwargs):
        self._resume_data.update(kwargs)
        self._resume_path.write_text(json.dumps(self._resume_data))

    def _flush(self):
        if not self.buffer:
            return

        columns: dict[str, list] = {}
        for row in self.buffer:
            for k, v in row.items():
                columns.setdefault(k, []).append(v)

        table = pa.table(columns)
        shard_path = self.outdir / f"{self.prefix}_{self.current_shard:05d}.parquet"
        pq.write_table(table, shard_path, compression="snappy")

        if self._resume_data:
            self._resume_path.write_text(json.dumps(self._resume_data))

        self.current_shard += 1
        self.buffer = []

    def finalize(self) -> tuple[int, int, int]:
        self._flush()
        return self.total_docs, self.total_bytes, self.current_shard


def main():
    parser = argparse.ArgumentParser(
        description="Download 1900-1914 English texts from HuggingFace to parquet shards",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--outdir", "-o", type=Path,
        default=Path("/mnt/main0/data/michaelhla/pre1915_supplement_raw"),
    )
    parser.add_argument("--max-docs", type=int, default=0,
                        help="Max documents per source (0 = unlimited)")
    parser.add_argument("--docs-per-shard", type=int, default=10000)
    parser.add_argument("--workers", "-w", type=int, default=32,
                        help="Parallel download threads for institutional-books")
    parser.add_argument(
        "--sources", nargs="+",
        choices=["institutional", "blbooks", "newspapers", "all"],
        default=["all"],
    )
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()
    logger = setup_logger(args.verbose)

    max_docs = args.max_docs if args.max_docs > 0 else 10**18

    logger.info(f"Downloading documents from {YEAR_MIN} to {YEAR_MAX} (1900-1914 supplement)")

    sources = set(args.sources)
    if "all" in sources:
        sources = {"institutional", "blbooks", "newspapers"}

    args.outdir.mkdir(parents=True, exist_ok=True)
    start_time = time.time()

    if "institutional" in sources:
        logger.info("=" * 60)
        logger.info("Downloading INSTITUTIONAL BOOKS (parallel)")
        logger.info("=" * 60)

        writer = ParquetShardWriter(
            args.outdir / "institutional", args.docs_per_shard, prefix="inst"
        )
        download_institutional_books(writer, logger, max_docs, workers=args.workers)
        total_docs, total_bytes, n_shards = writer.finalize()
        logger.info(
            f"Institutional books: {total_docs} docs, "
            f"{total_bytes / 1e9:.2f} GB, {n_shards} shards"
        )

    if "blbooks" in sources:
        logger.info("=" * 60)
        logger.info("Downloading BRITISH LIBRARY BOOKS")
        logger.info("=" * 60)

        writer = ParquetShardWriter(
            args.outdir / "blbooks", args.docs_per_shard, prefix="bl"
        )
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
                logger.info(
                    f"Progress: {writer.total_docs} docs, "
                    f"{writer.total_bytes / 1e9:.2f} GB"
                )

        total_docs, total_bytes, n_shards = writer.finalize()
        logger.info(
            f"BL books: {total_docs} docs, "
            f"{total_bytes / 1e9:.2f} GB, {n_shards} shards"
        )

    if "newspapers" in sources:
        logger.info("=" * 60)
        logger.info("Downloading AMERICAN STORIES")
        logger.info("=" * 60)

        writer = ParquetShardWriter(
            args.outdir / "newspapers", args.docs_per_shard, prefix="news"
        )

        resume_year = YEAR_MIN
        skip_in_year = 0
        if writer.docs_to_skip > 0:
            existing_shards = sorted(
                (args.outdir / "newspapers").glob("news_*.parquet")
            )
            max_year = YEAR_MIN
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
            logger.info(
                f"Resuming from year {resume_year} "
                f"(skipping {skip_in_year} docs in that year, "
                f"{writer.docs_to_skip} total already saved)"
            )

        for text, metadata in stream_american_stories(
            max_docs, logger,
            resume_from_year=resume_year,
            skip_docs_in_first_year=skip_in_year,
        ):
            writer.add(text, metadata)
            if writer.total_docs % 10000 == 0:
                logger.info(
                    f"Progress: {writer.total_docs} docs, "
                    f"{writer.total_bytes / 1e9:.2f} GB"
                )

        total_docs, total_bytes, n_shards = writer.finalize()
        logger.info(
            f"Newspapers: {total_docs} docs, "
            f"{total_bytes / 1e9:.2f} GB, {n_shards} shards"
        )

    elapsed = time.time() - start_time

    print(f"\n{'='*60}")
    print("Download complete!")
    print(f"  Year range: {YEAR_MIN}-{YEAR_MAX} (1900-1914 supplement)")
    print(f"  Elapsed: {elapsed:.1f}s")
    print(f"  Output: {args.outdir}")

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
