#!/usr/bin/env python3
"""
HuggingFace Historical Text Downloader -- 1900-1964 Corpus

Downloads English texts from 1900-1964 from HuggingFace datasets and stores
directly in parquet shards. Uses parallel file downloads for institutional-books
(~12 min with 32 threads).

Sources:
- Books: institutional/institutional-books-1.0 (1900-1922, public domain)
- Newspapers: dell-research-harvard/AmericanStories (1900-1964)

Usage:
    python scripts/pre1900_scripts/hf_download_1900_1964.py \
        --outdir /opt/dlami/nvme/gpt1964_training/raw_data
"""

from __future__ import annotations
import argparse
import importlib.util
import json
import logging
import os
import sys
import time

# Redirect HF cache to nvme so we don't fill root disk
os.environ.setdefault("HF_HOME", "/opt/dlami/nvme/hf_cache")
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


VERSION = "3.0"
YEAR_MIN = 1900
YEAR_MAX = 1964

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
    logger = logging.getLogger("hf_download_1900_1964")
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
    """Download one parquet file and extract 1900-1964 English documents.

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
    writer: "ParquetShardWriter",
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


def _process_one_year(args_tuple):
    """Process a single year of AmericanStories in a worker process.

    Writes results directly to per-year parquet files to avoid IPC overhead.
    Returns (year, n_docs, n_bytes, output_path).
    """
    import importlib.util as _ilu

    year_str, outdir, docs_per_shard = args_tuple
    outdir = Path(outdir)
    year_dir = outdir / f"_year_{year_str}"
    year_dir.mkdir(parents=True, exist_ok=True)

    # Check if this year is already done
    done_marker = year_dir / "_done"
    if done_marker.exists():
        existing = sorted(year_dir.glob("news_*.parquet"))
        n_docs = 0
        for sp in existing:
            n_docs += pq.ParquetFile(sp).metadata.num_rows
        return (year_str, n_docs, 0, str(year_dir))

    # Each worker imports the builder independently (no shared state)
    try:
        script_path = hf_hub_download(
            repo_id="dell-research-harvard/AmericanStories",
            filename="AmericanStories.py",
            repo_type="dataset",
        )
        mod_name = f"_as_builder_{year_str}"
        spec = _ilu.spec_from_file_location(mod_name, script_path)
        mod = _ilu.module_from_spec(spec)
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)

        builder = mod.AmericanStories(
            config_name="subset_years_content_regions", year_list=[year_str]
        )
        builder.download_and_prepare()
        year_data = builder.as_dataset()[year_str]
    except Exception as e:
        print(f"  [year {year_str}] FAILED to load: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return (year_str, 0, 0, str(year_dir))

    year_int = int(year_str)
    buffer = []
    shard_idx = 0
    n_docs = 0
    n_bytes = 0

    def flush(buf, idx):
        if not buf:
            return idx
        columns = {}
        for row in buf:
            for k, v in row.items():
                columns.setdefault(k, []).append(v)
        table = pa.table(columns)
        pq.write_table(table, year_dir / f"news_{idx:05d}.parquet", compression="snappy")
        del table
        return idx + 1

    for row in year_data:
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

            row_dict = {
                "text": text,
                "source": "AmericanStories",
                "doc_id": article.get("id", f"as_{year_str}_{n_docs}"),
                "newspaper": newspaper,
                "headline": article.get("headline", ""),
                "byline": article.get("byline", ""),
                "date": date_str,
                "year": extract_year(date_str) or year_int,
                "legibility": legibility,
            }
            buffer.append(row_dict)
            n_docs += 1
            n_bytes += len(text.encode("utf-8"))

            if len(buffer) >= docs_per_shard:
                shard_idx = flush(buffer, shard_idx)
                buffer = []

    shard_idx = flush(buffer, shard_idx)
    done_marker.write_text(f"{n_docs}")

    print(f"  [year {year_str}] Done: {n_docs:,} docs, {n_bytes / 1e9:.2f} GB, {shard_idx} shards",
          flush=True)
    return (year_str, n_docs, n_bytes, str(year_dir))


def download_american_stories_parallel(
    outdir: Path,
    logger: logging.Logger,
    docs_per_shard: int = 10000,
    year_workers: int = 16,
):
    """Download AmericanStories using parallel workers, one per year.

    Each worker writes to its own directory, then we merge at the end.
    Uses done-markers for resume support (safe to restart).
    """
    years = [str(y) for y in range(YEAR_MIN, YEAR_MAX + 1)]

    # Skip years already completed
    remaining = []
    for y in years:
        done_marker = outdir / f"_year_{y}" / "_done"
        if done_marker.exists():
            logger.info(f"  Year {y} already done, skipping")
        else:
            remaining.append(y)

    logger.info(f"Processing {len(remaining)} years with {year_workers} parallel workers "
                f"({len(years) - len(remaining)} already done)")

    if remaining:
        args_list = [(y, str(outdir), docs_per_shard) for y in remaining]

        total_docs = 0
        total_bytes = 0

        with ProcessPoolExecutor(max_workers=year_workers) as executor:
            futures = {executor.submit(_process_one_year, a): a[0] for a in args_list}
            for future in as_completed(futures):
                year_str = futures[future]
                try:
                    yr, nd, nb, _ = future.result()
                    total_docs += nd
                    total_bytes += nb
                    logger.info(f"Year {yr} complete: {nd:,} docs")
                except Exception as e:
                    logger.warning(f"Year {year_str} failed: {e}")

    # Merge per-year shards into final newspapers/ directory
    logger.info("Merging per-year shards into newspapers/ ...")
    news_dir = outdir / "newspapers"
    news_dir.mkdir(parents=True, exist_ok=True)

    existing_news = sorted(news_dir.glob("news_*.parquet"))
    merge_shard_idx = len(existing_news)

    grand_total_docs = 0

    for y in sorted(years):
        year_dir = outdir / f"_year_{y}"
        if not year_dir.exists():
            continue
        year_shards = sorted(year_dir.glob("news_*.parquet"))
        for sp in year_shards:
            dest = news_dir / f"news_{merge_shard_idx:05d}.parquet"
            sp.rename(dest)
            grand_total_docs += pq.ParquetFile(dest).metadata.num_rows
            merge_shard_idx += 1
        # Clean up year dir
        for leftover in year_dir.iterdir():
            leftover.unlink()
        year_dir.rmdir()

    logger.info(
        f"AmericanStories: {grand_total_docs:,} docs, {merge_shard_idx} shards"
    )


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
        description="Download 1900-1964 English texts from HuggingFace to parquet shards",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--outdir", "-o", type=Path,
        default=Path("/opt/dlami/nvme/gpt1964_training/raw_data"),
    )
    parser.add_argument("--max-docs", type=int, default=0,
                        help="Max documents per source (0 = unlimited)")
    parser.add_argument("--docs-per-shard", type=int, default=10000)
    parser.add_argument("--workers", "-w", type=int, default=32,
                        help="Parallel download threads for institutional-books")
    parser.add_argument("--year-workers", type=int, default=16,
                        help="Parallel year workers for AmericanStories")
    parser.add_argument(
        "--sources", nargs="+",
        choices=["institutional", "newspapers", "all"],
        default=["all"],
    )
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()
    logger = setup_logger(args.verbose)

    max_docs = args.max_docs if args.max_docs > 0 else 10**18

    logger.info(f"Downloading documents from {YEAR_MIN} to {YEAR_MAX} (1900-1964 corpus)")

    sources = set(args.sources)
    if "all" in sources:
        sources = {"institutional", "newspapers"}

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

    if "newspapers" in sources:
        logger.info("=" * 60)
        logger.info("Downloading AMERICAN STORIES (parallel by year)")
        logger.info("=" * 60)

        download_american_stories_parallel(
            args.outdir, logger,
            docs_per_shard=args.docs_per_shard,
            year_workers=args.year_workers,
        )

    elapsed = time.time() - start_time

    print(f"\n{'='*60}")
    print("Download complete!")
    print(f"  Year range: {YEAR_MIN}-{YEAR_MAX} (1900-1964 corpus)")
    print(f"  Elapsed: {elapsed:.1f}s")
    print(f"  Output: {args.outdir}")

    for subdir in ["institutional", "newspapers"]:
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
