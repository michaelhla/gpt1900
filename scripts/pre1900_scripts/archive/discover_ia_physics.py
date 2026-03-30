#!/usr/bin/env python3
"""
Automated discovery and download of pre-1900 physics texts from Internet Archive.

Uses IA's Advanced Search API (free, no auth, 60 req/min) to find English physics
texts from 1600-1899, then filters, deduplicates, downloads OCR text, cleans, and
saves to data/physics_books/.

Usage:
  python scripts/pre1900_scripts/discover_ia_physics.py --phase search
  python scripts/pre1900_scripts/discover_ia_physics.py --phase filter
  python scripts/pre1900_scripts/discover_ia_physics.py --phase download
  python scripts/pre1900_scripts/discover_ia_physics.py --phase download --max-volumes 20
  python scripts/pre1900_scripts/discover_ia_physics.py --dry-run
"""

import argparse
import json
import logging
import re
import sys
import time
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "physics_books"
CACHE_DIR = PROJECT_ROOT / "data"

SEARCH_CACHE = CACHE_DIR / ".ia_search_results.json"
FILTER_CACHE = CACHE_DIR / ".ia_filtered_candidates.json"
PROGRESS_FILE = CACHE_DIR / ".ia_download_progress.json"
MANIFEST_FILE = OUTPUT_DIR / "ia_discovery_manifest.json"

# ---------------------------------------------------------------------------
# IA Search
# ---------------------------------------------------------------------------

IA_SEARCH_URL = "https://archive.org/advancedsearch.php"

PHYSICS_SUBJECTS = [
    "physics",
    "electricity",
    "magnetism",
    "optics",
    "mechanics",
    "thermodynamics",
    "natural philosophy",
    "acoustics",
    "heat",
    "dynamics",
    "hydrostatics",
    "hydrodynamics",
    "pneumatics",
    "statics",
    "electromagnetism",
    "kinetics",
    "gravitation",
    "light",
    "sound",
    "energy",
    "force",
    "motion",
    "waves",
]

# Additional title keywords to search (catches books not tagged with subject)
PHYSICS_TITLE_KEYWORDS = [
    "physics",
    "electricity",
    "magnetism",
    "optics",
    "mechanics",
    "thermodynamics",
    "natural philosophy",
    "heat",
    "light",
    "sound",
    "electromagnetism",
    "hydrostatics",
    "dynamics",
    "gravitation",
]

FIELDS = [
    "identifier",
    "title",
    "creator",
    "date",
    "downloads",
    "subject",
    "language",
    "description",
]


def build_search_query(date_start: str = "1600-01-01", date_end: str = "1899-12-31") -> str:
    """Build the IA advanced search query string for a date range."""
    subject_clauses = " OR ".join(f'subject:"{s}"' for s in PHYSICS_SUBJECTS)
    title_clauses = " OR ".join(f'title:"{s}"' for s in PHYSICS_TITLE_KEYWORDS)
    q = (
        f"({subject_clauses} OR {title_clauses})"
        f" AND date:[{date_start} TO {date_end}]"
        " AND mediatype:texts"
        " AND language:(English OR english OR eng)"
    )
    return q


# IA caps search results at 10,000. Split by date ranges to get everything.
DATE_RANGES = [
    ("1600-01-01", "1849-12-31"),
    ("1850-01-01", "1879-12-31"),
    ("1880-01-01", "1889-12-31"),
    ("1890-01-01", "1899-12-31"),
]


def _search_ia_range(
    session: requests.Session, date_start: str, date_end: str, max_pages: int = 200
) -> list[dict]:
    """Search IA for a single date range, paginating through all results."""
    query = build_search_query(date_start, date_end)
    all_results = []
    rows_per_page = 200

    for page in range(1, max_pages + 1):
        param_list = [("q", query), ("output", "json"), ("rows", rows_per_page), ("page", page)]
        for field in FIELDS:
            param_list.append(("fl[]", field))

        log.info(f"  [{date_start[:4]}-{date_end[:4]}] page {page}...")
        try:
            resp = session.get(IA_SEARCH_URL, params=param_list, timeout=60)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            log.warning(f"  Search page {page} failed: {e}")
            time.sleep(5)
            continue

        response = data.get("response", {})
        docs = response.get("docs", [])
        num_found = response.get("numFound", 0)

        if not docs:
            break

        all_results.extend(docs)
        log.info(f"    Got {len(docs)} (total: {len(all_results)}/{num_found})")

        if len(all_results) >= num_found:
            break

        time.sleep(1.5)

    return all_results


def search_ia(session: requests.Session) -> list[dict]:
    """Search IA across multiple date ranges to bypass 10K result cap."""
    all_results = []
    seen_ids = set()

    for date_start, date_end in DATE_RANGES:
        log.info(f"Searching date range {date_start} to {date_end}...")
        results = _search_ia_range(session, date_start, date_end)

        # Dedup by identifier across ranges
        new = 0
        for doc in results:
            ident = doc.get("identifier", "")
            if ident and ident not in seen_ids:
                seen_ids.add(ident)
                all_results.append(doc)
                new += 1

        log.info(f"  Range complete: {len(results)} raw, {new} new unique")

    log.info(f"Search complete: {len(all_results)} total unique results")
    return all_results


def run_search(session: requests.Session):
    """Phase 1: Search IA and cache results."""
    results = search_ia(session)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    SEARCH_CACHE.write_text(json.dumps(results, indent=2), encoding="utf-8")
    log.info(f"Cached {len(results)} results to {SEARCH_CACHE}")


# ---------------------------------------------------------------------------
# Filtering & Validation
# ---------------------------------------------------------------------------

# Keywords that suggest a modern reprint/edition rather than original text
MODERN_PUBLISHER_KEYWORDS = [
    "nabu press", "forgotten books", "kessinger", "createspace",
    "independently published", "wentworth press", "palala press",
    "scholar's choice", "scholars choice", "hardpress", "rarebooksclub",
    "general books", "bibliobazaar", "bibliolife", "amazon",
    "print on demand", "facsimile", "reprint", "leopold classic",
    "sagwan press", "trieste publishing", "andesite press",
    "franklin classics", "arkose press", "hansebooks",
]

# Subjects that indicate non-physics content
NON_PHYSICS_SUBJECTS = [
    "fiction", "novel", "poetry", "drama", "plays", "religion",
    "theology", "bible", "sermons", "philosophy of mind",
    "political science", "economics", "law", "jurisprudence",
    "medicine", "surgery", "anatomy", "physiology", "botany",
    "zoology", "biology", "chemistry",  # chemistry is borderline, keep separate
    "agriculture", "cookery", "cooking", "music theory",
    "art", "architecture", "history of",
]

# Physics-relevance scoring keywords (in title or subject)
PHYSICS_SCORE_KEYWORDS = {
    # High relevance
    "physics": 3, "electricity": 3, "magnetism": 3, "electromagnetic": 3,
    "electromagnetism": 3, "optics": 3, "optical": 2, "mechanics": 3,
    "thermodynamics": 3, "natural philosophy": 3, "acoustics": 2,
    "heat": 2, "dynamics": 2, "statics": 2, "hydrostatics": 2,
    "hydrodynamics": 2, "pneumatics": 2, "gravitation": 3, "gravity": 2,
    "light": 1, "sound": 1, "energy": 1, "force": 1, "motion": 1,
    "waves": 1, "vibration": 1, "radiation": 2, "spectrum": 2,
    "spectroscopy": 2, "kinetics": 2, "kinetic theory": 3,
    "galvanic": 2, "voltaic": 2, "telegraph": 1, "telephone": 1,
    "ether": 2, "aether": 2, "luminiferous": 3,
    # Medium relevance
    "treatise": 1, "experimental": 1, "mathematical": 1,
    "lecture": 1, "scientific": 1,
}


def parse_ia_date(date_str: str | None) -> int | None:
    """Extract year from IA date field (various formats)."""
    if not date_str:
        return None
    date_str = str(date_str).strip()

    # Try full ISO date
    m = re.match(r"(\d{4})-\d{2}-\d{2}", date_str)
    if m:
        return int(m.group(1))

    # Try bare year
    m = re.match(r"(\d{4})", date_str)
    if m:
        return int(m.group(1))

    return None


def compute_physics_score(title: str, subjects: str | list) -> int:
    """Score how physics-relevant an item is based on title and subjects."""
    text = title.lower()
    if isinstance(subjects, list):
        text += " " + " ".join(s.lower() for s in subjects)
    elif isinstance(subjects, str):
        text += " " + subjects.lower()

    score = 0
    for keyword, weight in PHYSICS_SCORE_KEYWORDS.items():
        if keyword in text:
            score += weight
    return score


def is_modern_reprint(item: dict) -> bool:
    """Check if item looks like a modern print-on-demand reprint."""
    desc = str(item.get("description", "")).lower()
    title = str(item.get("title", "")).lower()
    creator = str(item.get("creator", "")).lower()
    combined = f"{desc} {title} {creator}"
    return any(kw in combined for kw in MODERN_PUBLISHER_KEYWORDS)


def is_non_physics(item: dict) -> bool:
    """Check if item is clearly non-physics based on subjects."""
    subjects = item.get("subject", "")
    if isinstance(subjects, list):
        subj_text = " ".join(s.lower() for s in subjects)
    else:
        subj_text = str(subjects).lower()

    title = str(item.get("title", "")).lower()
    combined = f"{subj_text} {title}"

    # Count non-physics signals
    non_physics_count = sum(1 for kw in NON_PHYSICS_SUBJECTS if kw in combined)
    physics_count = compute_physics_score(
        str(item.get("title", "")), item.get("subject", "")
    )

    # If strong non-physics signal and weak physics signal, reject
    return non_physics_count >= 2 and physics_count < 2


def normalize_title(title: str) -> str:
    """Normalize title for dedup comparison."""
    title = title.lower().strip()
    title = re.sub(r"[^\w\s]", "", title)
    title = re.sub(r"\s+", " ", title)
    # Remove common prefixes/suffixes
    for prefix in ["a ", "an ", "the "]:
        if title.startswith(prefix):
            title = title[len(prefix):]
    return title.strip()


def titles_similar(t1: str, t2: str, threshold: float = 0.75) -> bool:
    """Check if two titles are similar enough to be the same work."""
    n1 = normalize_title(t1)
    n2 = normalize_title(t2)
    if n1 == n2:
        return True
    return SequenceMatcher(None, n1, n2).ratio() >= threshold


def get_existing_corpus_titles() -> list[str]:
    """Get titles/filenames from existing corpus for dedup."""
    titles = []
    if OUTPUT_DIR.exists():
        for f in OUTPUT_DIR.glob("*.txt"):
            # Convert filename to rough title
            name = f.stem.replace("_", " ")
            titles.append(name)

    # Also get titles from CURATED_WORKS
    try:
        sys.path.insert(0, str(SCRIPTS_DIR))
        from collect_physics_books import CURATED_WORKS
        for work in CURATED_WORKS:
            titles.append(work["title"])
            titles.append(work["filename"].replace(".txt", "").replace("_", " "))
    except ImportError:
        pass

    return titles


def is_duplicate_of_existing(title: str, existing_titles: list[str]) -> bool:
    """Check if title matches any existing corpus entry."""
    for existing in existing_titles:
        if titles_similar(title, existing, threshold=0.70):
            return True
    return False


def dedup_candidates(candidates: list[dict]) -> list[dict]:
    """Group similar titles and keep the best version (highest downloads)."""
    groups: dict[str, list[dict]] = {}

    for c in candidates:
        norm = normalize_title(c["title"])
        placed = False
        for key in groups:
            if titles_similar(norm, key, threshold=0.70):
                groups[key].append(c)
                placed = True
                break
        if not placed:
            groups[norm] = [c]

    deduped = []
    for key, group in groups.items():
        # Keep the one with most downloads
        best = max(group, key=lambda x: x.get("downloads", 0) or 0)
        deduped.append(best)

    return deduped


def run_filter():
    """Phase 2: Filter, validate, dedup search results."""
    if not SEARCH_CACHE.exists():
        log.error(f"No search cache found at {SEARCH_CACHE}. Run --phase search first.")
        return

    results = json.loads(SEARCH_CACHE.read_text(encoding="utf-8"))
    log.info(f"Loaded {len(results)} search results")

    existing_titles = get_existing_corpus_titles()
    log.info(f"Found {len(existing_titles)} existing corpus entries for dedup")

    # Filter pipeline
    candidates = []
    stats = {
        "total": len(results),
        "no_date": 0,
        "bad_date": 0,
        "modern_reprint": 0,
        "non_physics": 0,
        "low_score": 0,
        "duplicate_existing": 0,
        "passed": 0,
    }

    for item in results:
        title = str(item.get("title", ""))
        identifier = str(item.get("identifier", ""))

        # Date validation
        year = parse_ia_date(item.get("date"))
        if year is None:
            stats["no_date"] += 1
            continue
        if year < 1500 or year > 1899:
            stats["bad_date"] += 1
            continue

        # Modern reprint check
        if is_modern_reprint(item):
            stats["modern_reprint"] += 1
            continue

        # Non-physics check
        if is_non_physics(item):
            stats["non_physics"] += 1
            continue

        # Physics relevance score
        score = compute_physics_score(title, item.get("subject", ""))
        if score < 1:
            stats["low_score"] += 1
            continue

        # Dedup against existing corpus
        if is_duplicate_of_existing(title, existing_titles):
            stats["duplicate_existing"] += 1
            continue

        candidates.append({
            "identifier": identifier,
            "title": title,
            "creator": str(item.get("creator", "")),
            "date": str(item.get("date", "")),
            "year": year,
            "downloads": item.get("downloads", 0),
            "subject": item.get("subject", ""),
            "physics_score": score,
        })
        stats["passed"] += 1

    log.info(f"Filter stats: {json.dumps(stats, indent=2)}")

    # Dedup among candidates (same title, different editions)
    before_dedup = len(candidates)
    candidates = dedup_candidates(candidates)
    log.info(f"Deduped {before_dedup} -> {len(candidates)} candidates")

    # Sort by physics_score desc, then downloads desc
    candidates.sort(key=lambda x: (-x["physics_score"], -(x.get("downloads", 0) or 0)))

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    FILTER_CACHE.write_text(json.dumps(candidates, indent=2), encoding="utf-8")
    log.info(f"Saved {len(candidates)} candidates to {FILTER_CACHE}")


# ---------------------------------------------------------------------------
# Download & Clean
# ---------------------------------------------------------------------------

def check_has_text_file(session: requests.Session, identifier: str) -> str | None:
    """Check if an IA item has a _djvu.txt or .txt file. Returns filename or None."""
    url = f"https://archive.org/metadata/{identifier}/files"
    try:
        resp = session.get(url, timeout=30)
        resp.raise_for_status()
        files = resp.json().get("result", [])
    except Exception:
        return None

    # Prefer _djvu.txt
    for f in files:
        name = f.get("name", "")
        if name.endswith("_djvu.txt"):
            return name

    # Fall back to any .txt file that's not metadata
    for f in files:
        name = f.get("name", "")
        size = int(f.get("size", 0))
        if name.endswith(".txt") and size > 10000 and not any(
            skip in name.lower() for skip in ["meta", "files", "marc", "scandata"]
        ):
            return name

    return None


def download_text(session: requests.Session, identifier: str, txt_filename: str) -> str | None:
    """Download text file from IA."""
    url = f"https://archive.org/download/{identifier}/{txt_filename}"
    try:
        resp = session.get(url, timeout=120)
        resp.raise_for_status()
        return resp.text
    except Exception as e:
        log.warning(f"Failed to download {identifier}/{txt_filename}: {e}")
        return None


def strip_ia_boilerplate_header(text: str) -> str:
    """Strip Google Books / IA boilerplate that appears at the start of djvu.txt files.

    The raw djvu.txt from Google-digitized books starts with 1-3 pages of
    boilerplate (spread across many lines with OCR double-spacing). We scan
    line-by-line and skip until we find consecutive content lines free of
    boilerplate signals.
    """
    boilerplate_signals = [
        "google", "public domain", "copyright", "non-commercial",
        "automated querying", "book search", "digitized", "keep it legal",
        "commercial parties", "technical restrictions", "custodians",
        "search engine", "copyright infringement", "specific use",
        "varies from country", "university microfilms",
        "archive.org", "books.google", "hathitrust",
        "gateways to the past", "long journey from the",
        "we encourage the use", "please do not remove",
        "discover the world", "reach new audiences",
        "full text of this book", "usage guidelines",
        "proud to partner", "widely accessible",
    ]

    lines = text.split("\n")
    # Scan up to first 500 lines (boilerplate can be long with OCR spacing)
    max_scan = min(500, len(lines))
    content_streak = 0
    last_boilerplate_idx = -1

    for i in range(max_scan):
        line_lower = lines[i].strip().lower()

        if not line_lower:
            continue

        is_boilerplate = any(sig in line_lower for sig in boilerplate_signals)
        if is_boilerplate:
            last_boilerplate_idx = i
            content_streak = 0
        else:
            content_streak += 1
            # Need 5 consecutive non-boilerplate content lines to be confident
            if content_streak >= 5:
                break

    if last_boilerplate_idx > 0:
        text = "\n".join(lines[last_boilerplate_idx + 1:])

    return text


def strip_ia_boilerplate_footer(text: str) -> str:
    """Strip trailing boilerplate from IA texts (library stamps, catalog entries)."""
    paragraphs = text.split("\n\n")
    end_idx = len(paragraphs)

    footer_signals = [
        "university of california", "library", "university microfilms",
        "the borrower", "date due", "this book is due",
        "catalog", "catalogue",
    ]

    # Check last 10 paragraphs
    for i in range(len(paragraphs) - 1, max(len(paragraphs) - 10, -1), -1):
        para_lower = paragraphs[i].lower().strip()
        if not para_lower:
            end_idx = i
            continue
        signal_count = sum(1 for s in footer_signals if s in para_lower)
        if signal_count >= 1 and len(para_lower) < 200:
            end_idx = i
        else:
            break

    if end_idx < len(paragraphs):
        text = "\n\n".join(paragraphs[:end_idx])

    return text


def clean_ia_text(text: str) -> str:
    """Clean downloaded IA text using hf_clean functions."""
    # Import cleaning functions from hf_clean
    sys.path.insert(0, str(SCRIPTS_DIR))
    from hf_clean import (
        clean_ocr_artifacts,
        normalize_unicode,
        normalize_whitespace,
        rejoin_hyphenated_words,
        reflow_text,
        remove_front_matter,
        remove_google_boilerplate,
        remove_hathi_boilerplate,
        remove_library_stamps,
        strip_pg_boilerplate,
    )

    # Strip IA-specific boilerplate (header/footer)
    text = strip_ia_boilerplate_header(text)
    text = strip_ia_boilerplate_footer(text)

    # Remove lines that are just page numbers
    lines = text.split("\n")
    clean_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            clean_lines.append("")
            continue
        if re.match(r"^\d{1,4}$", stripped):
            continue
        clean_lines.append(line)
    text = "\n".join(clean_lines)

    # Apply hf_clean pipeline
    text = remove_google_boilerplate(text)
    text = remove_hathi_boilerplate(text)
    text = remove_library_stamps(text)
    text = strip_pg_boilerplate(text)
    text = normalize_unicode(text)
    text = clean_ocr_artifacts(text)
    text = normalize_whitespace(text)
    text = rejoin_hyphenated_words(text)
    text = reflow_text(text)
    text = remove_front_matter(text)

    return text.strip()


def estimate_ocr_quality(text: str) -> float:
    """Estimate OCR quality as ratio of printable/reasonable chars."""
    if not text:
        return 0.0
    sample = text[:20000]
    printable = sum(1 for c in sample if c.isprintable() or c in "\n\t")
    return printable / len(sample) if sample else 0.0


def make_filename(title: str, creator: str, year: int) -> str:
    """Generate a clean filename from metadata."""
    # Get last name of creator
    creator = str(creator)
    if "," in creator:
        last_name = creator.split(",")[0].strip()
    elif " " in creator:
        parts = creator.split()
        last_name = parts[-1].strip()
    else:
        last_name = creator.strip()

    # Clean up
    last_name = re.sub(r"[^\w]", "", last_name).lower()
    if not last_name or len(last_name) < 2:
        last_name = "unknown"

    # Shorten title
    title_clean = title.lower()
    title_clean = re.sub(r"[^\w\s]", "", title_clean)
    words = title_clean.split()[:6]
    title_slug = "_".join(words)

    filename = f"ia_{last_name}_{title_slug}.txt"
    # Limit length
    if len(filename) > 80:
        filename = filename[:76] + ".txt"
    return filename


def load_progress() -> dict:
    """Load download progress for resume support."""
    if PROGRESS_FILE.exists():
        return json.loads(PROGRESS_FILE.read_text(encoding="utf-8"))
    return {"downloaded": [], "failed": [], "skipped": []}


def save_progress(progress: dict):
    """Save download progress."""
    PROGRESS_FILE.write_text(json.dumps(progress, indent=2), encoding="utf-8")


def _process_one_candidate(candidate: dict, idx: int, total: int) -> dict | None:
    """Process a single candidate: check text, download, clean, validate.

    Returns a result dict or None on skip/failure.
    Designed to be called from a thread pool — uses its own requests session.
    """
    identifier = candidate["identifier"]
    title = candidate["title"]
    creator = candidate["creator"]
    year = candidate["year"]

    session = requests.Session()
    session.headers.update({"User-Agent": "Pre1900PhysicsCorpus/1.0 (academic research)"})

    # Step 1: Check for text file
    txt_file = check_has_text_file(session, identifier)
    if not txt_file:
        return {"status": "skipped", "identifier": identifier, "reason": "no text file"}

    # Step 2: Download
    raw_text = download_text(session, identifier, txt_file)
    if not raw_text:
        return {"status": "failed", "identifier": identifier, "reason": "download failed"}

    # Step 3: Size check (raw)
    if len(raw_text) < 10000:
        return {"status": "skipped", "identifier": identifier,
                "reason": f"too short raw ({len(raw_text)} chars)"}

    # Step 4: Clean
    cleaned = clean_ia_text(raw_text)

    # Step 5: Quality checks
    if len(cleaned) < 3000:
        return {"status": "skipped", "identifier": identifier,
                "reason": f"too short after cleaning ({len(cleaned)} chars)"}

    ocr_quality = estimate_ocr_quality(cleaned)
    if ocr_quality < 0.6:
        return {"status": "skipped", "identifier": identifier,
                "reason": f"low OCR quality ({ocr_quality:.2f})"}

    # Step 6: Post-1900 anachronism check
    sys.path.insert(0, str(SCRIPTS_DIR))
    from hf_clean import contains_post_1900_physics
    has_anachronism, reason = contains_post_1900_physics(cleaned)
    if has_anachronism:
        return {"status": "skipped", "identifier": identifier,
                "reason": f"post-1900 physics ({reason})"}

    # Step 7: Build result (file write happens in main thread)
    filename = make_filename(title, creator, year)
    return {
        "status": "ok",
        "identifier": identifier,
        "filename": filename,
        "title": title,
        "creator": creator,
        "year": year,
        "cleaned": cleaned,
        "ocr_quality": round(ocr_quality, 3),
        "physics_score": candidate["physics_score"],
        "downloads": candidate.get("downloads", 0),
    }


def run_download(
    session: requests.Session,
    max_volumes: int | None = None,
    dry_run: bool = False,
    workers: int = 8,
):
    """Phase 3: Download, clean, and save texts (parallel)."""
    import concurrent.futures

    if not FILTER_CACHE.exists():
        log.error(f"No filtered candidates at {FILTER_CACHE}. Run --phase filter first.")
        return

    candidates = json.loads(FILTER_CACHE.read_text(encoding="utf-8"))
    log.info(f"Loaded {len(candidates)} candidates")

    if max_volumes:
        candidates = candidates[:max_volumes]
        log.info(f"Limited to {max_volumes} volumes")

    if dry_run:
        print(f"\n{'#':>4}  {'Score':>5}  {'Down':>6}  {'Year':>4}  {'Creator':<25}  Title")
        print("-" * 100)
        for i, c in enumerate(candidates, 1):
            print(
                f"{i:>4}  {c['physics_score']:>5}  "
                f"{c.get('downloads', 0) or 0:>6}  {c['year']:>4}  "
                f"{c['creator'][:25]:<25}  {c['title'][:60]}"
            )
        print(f"\nTotal: {len(candidates)} candidates")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    progress = load_progress()
    done_ids = set(progress["downloaded"] + progress["failed"] + progress["skipped"])

    manifest = []
    if MANIFEST_FILE.exists():
        manifest = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))

    # Filter to only unprocessed candidates
    todo = [(i, c) for i, c in enumerate(candidates, 1) if c["identifier"] not in done_ids]
    log.info(f"Skipping {len(candidates) - len(todo)} already processed, {len(todo)} remaining")
    log.info(f"Using {workers} parallel workers")

    downloaded = 0
    failed = 0
    skipped = 0
    t0 = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        future_to_idx = {}
        for idx, candidate in todo:
            fut = pool.submit(_process_one_candidate, candidate, idx, len(candidates))
            future_to_idx[fut] = (idx, candidate)

        for fut in concurrent.futures.as_completed(future_to_idx):
            idx, candidate = future_to_idx[fut]
            identifier = candidate["identifier"]

            try:
                result = fut.result()
            except Exception as e:
                log.warning(f"[{idx}] Exception for {identifier}: {e}")
                progress["failed"].append(identifier)
                failed += 1
                continue

            if result is None:
                progress["failed"].append(identifier)
                failed += 1
            elif result["status"] == "skipped":
                log.info(f"[{idx}/{len(candidates)}] Skipped {identifier}: {result['reason']}")
                progress["skipped"].append(identifier)
                skipped += 1
            elif result["status"] == "failed":
                log.info(f"[{idx}/{len(candidates)}] Failed {identifier}: {result['reason']}")
                progress["failed"].append(identifier)
                failed += 1
            elif result["status"] == "ok":
                # Write file (main thread — avoids race conditions)
                filename = result["filename"]
                filepath = OUTPUT_DIR / filename
                if filepath.exists():
                    filepath = OUTPUT_DIR / f"{filepath.stem}_{identifier[:8]}.txt"

                filepath.write_text(result["cleaned"], encoding="utf-8")
                size_kb = len(result["cleaned"]) / 1024
                log.info(
                    f"[{idx}/{len(candidates)}] Saved: {filepath.name} "
                    f"({size_kb:.1f} KB, OCR: {result['ocr_quality']:.2f})"
                )

                progress["downloaded"].append(identifier)
                downloaded += 1

                manifest.append({
                    "filename": filepath.name,
                    "identifier": identifier,
                    "title": result["title"],
                    "creator": result["creator"],
                    "year": result["year"],
                    "size_bytes": len(result["cleaned"]),
                    "ocr_quality": result["ocr_quality"],
                    "physics_score": result["physics_score"],
                    "downloads": result.get("downloads", 0),
                })

            # Save progress periodically (every 10 items)
            total_done = downloaded + failed + skipped
            if total_done % 10 == 0:
                save_progress(progress)
                MANIFEST_FILE.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
                elapsed = time.time() - t0
                rate = total_done / elapsed if elapsed > 0 else 0
                remaining = len(todo) - total_done
                eta_min = remaining / rate / 60 if rate > 0 else 0
                log.info(
                    f"  Progress: {total_done}/{len(todo)} "
                    f"({downloaded} saved, {skipped} skipped, {failed} failed) "
                    f"[{rate:.1f}/s, ETA {eta_min:.0f}min]"
                )

    # Final save
    save_progress(progress)
    MANIFEST_FILE.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    elapsed = time.time() - t0
    log.info(f"\nDownload complete in {elapsed/60:.1f} min:")
    log.info(f"  Downloaded: {downloaded}")
    log.info(f"  Failed: {failed}")
    log.info(f"  Skipped: {skipped}")
    log.info(f"  Manifest: {MANIFEST_FILE}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Discover and download pre-1900 physics texts from Internet Archive",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--phase",
        choices=["search", "filter", "download", "all"],
        default="all",
        help="Which phase to run",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show candidates without downloading (implies filter phase output)",
    )
    parser.add_argument(
        "--max-volumes",
        type=int,
        default=None,
        help="Limit number of volumes to download",
    )
    parser.add_argument(
        "--reset-progress",
        action="store_true",
        help="Reset download progress tracking",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel download workers",
    )
    args = parser.parse_args()

    session = requests.Session()
    session.headers.update({
        "User-Agent": "Pre1900PhysicsCorpus/1.0 (academic research)"
    })

    if args.reset_progress and PROGRESS_FILE.exists():
        PROGRESS_FILE.unlink()
        log.info("Reset download progress")

    if args.dry_run:
        if not FILTER_CACHE.exists():
            if not SEARCH_CACHE.exists():
                run_search(session)
            run_filter()
        run_download(session, max_volumes=args.max_volumes, dry_run=True)
        return

    if args.phase in ("search", "all"):
        run_search(session)

    if args.phase in ("filter", "all"):
        run_filter()

    if args.phase in ("download", "all"):
        run_download(session, max_volumes=args.max_volumes, workers=args.workers)


if __name__ == "__main__":
    main()
