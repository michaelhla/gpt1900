#!/usr/bin/env python3
"""
Collect 1900-1905 physics texts for a SEPARATE SFT dataset.

These texts contain early 20th-century breakthroughs (Planck's quantum
hypothesis, Einstein's statistical mechanics, Lorentz transformations, etc.)
and must NOT be mixed into the pre-1900 corpus.

Downloads to data/post1900_physics_books/ by default.

Sources:
  - Wikisource (MediaWiki Parse API -> HTML -> plaintext)
  - Internet Archive (OCR djvu.txt files)

Usage:
  python -m scripts.pre1900_scripts.collect_post1900_physics --dry-run
  python -m scripts.pre1900_scripts.collect_post1900_physics
  python -m scripts.pre1900_scripts.collect_post1900_physics --only planck_normal_spectrum.txt
"""

import argparse
import logging
from pathlib import Path

import requests

from scripts.pre1900_scripts.collect_physics_books import (
    WikisourceDownloader,
    GutenbergDownloader,
    InternetArchiveDownloader,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Curated works: 1900-1905 physics
# ---------------------------------------------------------------------------
# IMPORTANT: These are POST-1900 texts. They go into a separate dataset
# to avoid contaminating the pre-1900 training corpus.

CURATED_WORKS = [
    # ==================================================================
    # Max Planck -- Quantum Hypothesis (1901)
    # ==================================================================
    # "On the Law of Distribution of Energy in the Normal Spectrum"
    # Not on Wikisource. Downloaded separately from strangepaths.com PDF
    # and extracted to planck_normal_spectrum_1901.txt via PyMuPDF.
    # ==================================================================
    # J.J. Thomson -- Atomic Model (1904)
    # ==================================================================
    {
        "filename": "thomson_electricity_and_matter.txt",
        "source": "ia",
        "id": "electricitymatte00thomiala",
        "author": "J.J. Thomson",
        "year": 1904,
        "title": "Electricity and Matter",
    },
    # ==================================================================
    # H.A. Lorentz -- Lorentz Transformations (1904)
    # ==================================================================
    {
        "filename": "lorentz_em_phenomena_1904.txt",
        "source": "wikisource",
        "id": "Electromagnetic_phenomena",
        "wikisource_type": "page",
        "author": "H.A. Lorentz",
        "year": 1904,
        "title": "Electromagnetic phenomena in a system moving with any velocity smaller than that of light",
    },
    # ==================================================================
    # Ernest Rutherford -- Radioactivity
    # ==================================================================
    {
        "filename": "rutherford_radioactivity.txt",
        "source": "ia",
        "id": "radioactivity00ruthrich",
        "author": "Ernest Rutherford",
        "year": 1904,
        "title": "Radioactivity",
    },
]


def main():
    parser = argparse.ArgumentParser(
        description="Collect 1900-1905 physics texts (SEPARATE from pre-1900 corpus)."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/post1900_physics_books",
        help="Output directory (default: data/post1900_physics_books)",
    )
    parser.add_argument("--dry-run", action="store_true", help="List works without downloading")
    parser.add_argument("--only", type=str, nargs="+", help="Only download specific files (by filename)")
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--no-skip-existing", action="store_true")
    parser.add_argument(
        "--source",
        type=str,
        choices=["wikisource", "gutenberg", "ia"],
        help="Only download from a specific source",
    )
    args = parser.parse_args()

    if args.no_skip_existing:
        args.skip_existing = False

    output_dir = Path(args.output_dir)

    # Filter works
    works = CURATED_WORKS
    if args.only:
        works = [w for w in works if w["filename"] in args.only]
        if not works:
            log.error(f"No works matched --only {args.only}")
            return
    if args.source:
        works = [w for w in works if w["source"] == args.source]

    # Dry run
    if args.dry_run:
        print(f"\n{'#':>3}  {'Source':<12} {'Filename':<50} {'Author':<25} {'Year':<6} Title")
        print("-" * 140)
        for i, w in enumerate(works, 1):
            existing = "EXISTS" if (output_dir / w["filename"]).exists() else ""
            print(
                f"{i:>3}  {w['source']:<12} {w['filename']:<50} "
                f"{w['author']:<25} {str(w['year']):<6} {w['title']}  {existing}"
            )
        print(f"\nTotal: {len(works)} works (1900-1905)")
        from collections import Counter
        source_counts = Counter(w["source"] for w in works)
        for src, count in sorted(source_counts.items()):
            print(f"  {src}: {count}")
        return

    # Download
    output_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({
        "User-Agent": "Post1900PhysicsCollector/1.0 (academic research)"
    })

    downloaders = {
        "wikisource": WikisourceDownloader(session),
        "gutenberg": GutenbergDownloader(session),
        "ia": InternetArchiveDownloader(session),
    }

    success = 0
    skipped = 0
    failed = 0

    for i, work in enumerate(works, 1):
        filename = work["filename"]
        filepath = output_dir / filename
        source = work["source"]

        log.info(f"[{i}/{len(works)}] {work['title']} ({work['author']}, {work['year']})")

        if args.skip_existing and filepath.exists() and filepath.stat().st_size > 0:
            log.info(f"  Skipping (already exists): {filepath}")
            skipped += 1
            continue

        downloader = downloaders[source]
        text = downloader.download_work(work)

        if text and len(text.strip()) > 100:
            filepath.write_text(text, encoding="utf-8")
            size_kb = filepath.stat().st_size / 1024
            log.info(f"  Saved: {filepath} ({size_kb:.1f} KB)")
            success += 1
        else:
            log.warning(f"  FAILED or empty: {filename}")
            failed += 1

    print(f"\n{'='*60}")
    print(f"Download complete! (POST-1900 physics texts)")
    print(f"  Success:  {success}")
    print(f"  Skipped:  {skipped}")
    print(f"  Failed:   {failed}")
    print(f"  Total:    {len(works)}")
    print(f"  Output:   {output_dir.resolve()}")
    print(f"\nREMINDER: These are POST-1900 texts. Do NOT mix into pre-1900 corpus.")


if __name__ == "__main__":
    main()
