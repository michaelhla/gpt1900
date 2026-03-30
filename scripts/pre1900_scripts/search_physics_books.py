#!/usr/bin/env python3
"""Search the pre-1900 dataset for specific physics books."""

import pandas as pd
import os
import re
import glob
from collections import defaultdict

# Books to search for: (id, title_keywords, author_keywords, year)
BOOKS = [
    (1,  ["elements of physics"], ["carhart", "chute"], 1897),
    (2,  ["physics by experiment"], ["shaw"], 1891),
    (3,  ["sound light and heat", "sound, light and heat"], ["wright"], 1888),
    (4,  ["elementary treatise on physics", "ganot"], ["ganot"], 1886),
    (5,  ["fourteen weeks in physics"], ["steele"], 1878),
    (6,  ["lessons in electricity"], ["tyndall"], 1876),
    (7,  ["sound"], ["tyndall"], 1875),  # More specific search needed
    (8,  ["system of natural philosophy"], ["comstock"], 1858),
    (9,  ["school compendium of natural", "natural and experimental philosophy"], ["parker"], None),
    (10, ["scientific papers", "clerk maxwell"], ["maxwell"], 1890),
    (11, ["science of mechanics"], ["mach"], 1883),
    (12, ["kinetische theorie der gase", "kinetic theory of gases"], ["meyer"], 1877),
    (13, ["treatise on electricity and magnetism"], ["maxwell"], 1873),
    (14, ["horologium oscillatorium"], ["huygens"], 1673),
    (15, ["experimenta nova"], ["guericke"], 1672),
    (16, ["lectiones geometricae", "lectiones geometric"], ["barrow", "newton"], 1670),
    (17, ["physica"], ["aristotle", "aristoteles"], None),
]

results = defaultdict(list)  # book_id -> list of (location, match_details)

def search_title_match(title_str, book_id, title_keywords):
    """Check if a title string matches any of the title keywords for a book."""
    if title_str is None or pd.isna(title_str):
        return False
    title_lower = str(title_str).lower()
    for kw in title_keywords:
        if kw.lower() in title_lower:
            return True
    return False

def search_author_match(author_str, author_keywords):
    """Check if author string matches."""
    if author_str is None or pd.isna(author_str):
        return False
    author_lower = str(author_str).lower()
    for kw in author_keywords:
        if kw.lower() in author_lower:
            return True
    return False

def search_text_match(text_str, title_keywords, author_keywords):
    """Search within text content for title+author matches (first 2000 chars)."""
    if text_str is None or pd.isna(text_str):
        return False
    # Only check beginning of text (title pages)
    text_prefix = str(text_str)[:3000].lower()
    title_match = any(kw.lower() in text_prefix for kw in title_keywords)
    author_match = any(kw.lower() in text_prefix for kw in author_keywords)
    return title_match and author_match


print("=" * 80)
print("SEARCHING PRE-1900 DATASET FOR SPECIFIC PHYSICS BOOKS")
print("=" * 80)

# ============================================================
# 1. Search pre1900_standardized (has title column, 42 shards)
# ============================================================
print("\n--- Searching pre1900_standardized (title + source + year columns) ---")
std_dir = "/mnt/main0/data/michaelhla/pre1900_standardized"
std_files = sorted(glob.glob(os.path.join(std_dir, "shard_*.parquet")))
print(f"Found {len(std_files)} shards")

for shard_path in std_files:
    shard_name = os.path.basename(shard_path)
    try:
        df = pd.read_parquet(shard_path, columns=["title", "source", "year", "text"])
        for book_id, title_kws, author_kws, year in BOOKS:
            # Search by title
            for idx, row in df.iterrows():
                title_hit = search_title_match(row.get("title"), book_id, title_kws)
                if title_hit:
                    # Also check text for author confirmation
                    text_prefix = str(row.get("text", ""))[:3000].lower()
                    author_in_text = any(a.lower() in text_prefix for a in author_kws)
                    title_val = str(row.get("title", ""))[:100]
                    year_val = row.get("year")
                    source_val = row.get("source", "")
                    results[book_id].append({
                        "location": f"standardized/{shard_name}",
                        "title": title_val,
                        "year": year_val,
                        "source": source_val,
                        "author_confirmed": author_in_text,
                        "match_type": "title_column"
                    })

                # Also search text for books that may not have title metadata
                if not title_hit:
                    text_match = search_text_match(row.get("text"), title_kws, author_kws)
                    if text_match:
                        title_val = str(row.get("title", ""))[:100]
                        year_val = row.get("year")
                        source_val = row.get("source", "")
                        results[book_id].append({
                            "location": f"standardized/{shard_name}",
                            "title": title_val,
                            "year": year_val,
                            "source": source_val,
                            "author_confirmed": True,
                            "match_type": "text_content"
                        })
    except Exception as e:
        print(f"  Error reading {shard_name}: {e}")
    print(f"  Processed {shard_name}", flush=True)

# ============================================================
# 2. Search pre1900_raw/blbooks (has title + author columns)
# ============================================================
print("\n--- Searching pre1900_raw/blbooks (title + author columns) ---")
bl_dir = "/mnt/main0/data/michaelhla/pre1900_raw/blbooks"
bl_files = sorted(glob.glob(os.path.join(bl_dir, "bl_*.parquet")))
print(f"Found {len(bl_files)} shards")

for shard_path in bl_files:
    shard_name = os.path.basename(shard_path)
    try:
        df = pd.read_parquet(shard_path, columns=["title", "author", "year", "text"])
        for book_id, title_kws, author_kws, year in BOOKS:
            for idx, row in df.iterrows():
                title_hit = search_title_match(row.get("title"), book_id, title_kws)
                author_hit = search_author_match(row.get("author"), author_kws)
                if title_hit or (title_hit and author_hit):
                    results[book_id].append({
                        "location": f"raw/blbooks/{shard_name}",
                        "title": str(row.get("title", ""))[:150],
                        "author": str(row.get("author", "")),
                        "year": row.get("year"),
                        "author_confirmed": author_hit,
                        "match_type": "title+author_columns"
                    })
                elif not title_hit:
                    text_match = search_text_match(row.get("text"), title_kws, author_kws)
                    if text_match:
                        results[book_id].append({
                            "location": f"raw/blbooks/{shard_name}",
                            "title": str(row.get("title", ""))[:150],
                            "author": str(row.get("author", "")),
                            "year": row.get("year"),
                            "author_confirmed": True,
                            "match_type": "text_content"
                        })
    except Exception as e:
        print(f"  Error reading {shard_name}: {e}")
    print(f"  Processed {shard_name}", flush=True)

# ============================================================
# 3. Search pre1900_raw/books (has doc_id, text)
# ============================================================
print("\n--- Searching pre1900_raw/books (doc_id + text) ---")
books_dir = "/mnt/main0/data/michaelhla/pre1900_raw/books"
books_files = sorted(glob.glob(os.path.join(books_dir, "books_*.parquet")))
print(f"Found {len(books_files)} shards")

for shard_path in books_files:
    shard_name = os.path.basename(shard_path)
    try:
        df = pd.read_parquet(shard_path, columns=["doc_id", "year", "text"])
        for book_id, title_kws, author_kws, year in BOOKS:
            for idx, row in df.iterrows():
                text_match = search_text_match(row.get("text"), title_kws, author_kws)
                if text_match:
                    results[book_id].append({
                        "location": f"raw/books/{shard_name}",
                        "doc_id": str(row.get("doc_id", "")),
                        "year": row.get("year"),
                        "author_confirmed": True,
                        "match_type": "text_content"
                    })
    except Exception as e:
        print(f"  Error reading {shard_name}: {e}")
    print(f"  Processed {shard_name}", flush=True)

# ============================================================
# 4. Search pre1900_filtered (books and blbooks)
# ============================================================
print("\n--- Searching pre1900_filtered ---")
for subdir in ["books", "blbooks", "institutional", "institutional_split", "newspapers"]:
    filt_dir = f"/mnt/main0/data/michaelhla/pre1900_filtered/{subdir}"
    if not os.path.exists(filt_dir):
        continue
    filt_files = sorted(glob.glob(os.path.join(filt_dir, "*.parquet")))
    print(f"  {subdir}: {len(filt_files)} files")
    for shard_path in filt_files:
        shard_name = os.path.basename(shard_path)
        try:
            df = pd.read_parquet(shard_path)
            cols = df.columns.tolist()
            for book_id, title_kws, author_kws, year in BOOKS:
                for idx, row in df.iterrows():
                    found = False
                    if "title" in cols:
                        found = search_title_match(row.get("title"), book_id, title_kws)
                    if not found and "text" in cols:
                        found = search_text_match(row.get("text"), title_kws, author_kws)
                    if found:
                        results[book_id].append({
                            "location": f"filtered/{subdir}/{shard_name}",
                            "title": str(row.get("title", "N/A"))[:150] if "title" in cols else "N/A",
                            "year": row.get("year") if "year" in cols else None,
                            "match_type": "title_or_text"
                        })
        except Exception as e:
            print(f"  Error reading {shard_path}: {e}")
        print(f"    Processed {shard_name}", flush=True)

# ============================================================
# 5. Search pre1900_clean/blbooks (has title + author)
# ============================================================
print("\n--- Searching pre1900_clean ---")
for subdir in ["blbooks", "books"]:
    clean_dir = f"/mnt/main0/data/michaelhla/pre1900_clean/{subdir}"
    if not os.path.exists(clean_dir):
        continue
    clean_files = sorted(glob.glob(os.path.join(clean_dir, "*.parquet")))
    print(f"  {subdir}: {len(clean_files)} files")
    for shard_path in clean_files:
        shard_name = os.path.basename(shard_path)
        try:
            df = pd.read_parquet(shard_path)
            cols = df.columns.tolist()
            for book_id, title_kws, author_kws, year in BOOKS:
                for idx, row in df.iterrows():
                    found = False
                    if "title" in cols:
                        found = search_title_match(row.get("title"), book_id, title_kws)
                    if not found and "text" in cols:
                        found = search_text_match(row.get("text"), title_kws, author_kws)
                    if found:
                        results[book_id].append({
                            "location": f"clean/{subdir}/{shard_name}",
                            "title": str(row.get("title", "N/A"))[:150] if "title" in cols else "N/A",
                            "author": str(row.get("author", "N/A")) if "author" in cols else "N/A",
                            "year": row.get("year") if "year" in cols else None,
                            "match_type": "title_or_text"
                        })
        except Exception as e:
            print(f"  Error reading {shard_path}: {e}")
        print(f"    Processed {shard_name}", flush=True)

# ============================================================
# 6. Search test set text files
# ============================================================
print("\n--- Searching test_1875_1900 text files ---")
test_dir = "/mnt/main0/home/michaelhla/evolutionaryscale/gpt1900/data/test_1875_1900"
if os.path.exists(test_dir):
    txt_files = glob.glob(os.path.join(test_dir, "*.txt"))
    print(f"Found {len(txt_files)} text files")
    for txt_path in txt_files:
        fname = os.path.basename(txt_path)
        try:
            with open(txt_path, 'r', errors='replace') as f:
                text_prefix = f.read(3000).lower()
            for book_id, title_kws, author_kws, year in BOOKS:
                title_match = any(kw.lower() in text_prefix for kw in title_kws)
                author_match = any(a.lower() in text_prefix for a in author_kws)
                if title_match and author_match:
                    results[book_id].append({
                        "location": f"test_1875_1900/{fname}",
                        "match_type": "text_file_content"
                    })
        except Exception as e:
            pass

# Also check test_raw and test_clean
for test_subdir in ["test_raw", "test_clean", "test_1875_1900_clean"]:
    tdir = f"/mnt/main0/home/michaelhla/evolutionaryscale/gpt1900/data/{test_subdir}"
    if os.path.exists(tdir):
        txt_files = glob.glob(os.path.join(tdir, "*.txt"))
        print(f"Found {len(txt_files)} text files in {test_subdir}")
        for txt_path in txt_files:
            fname = os.path.basename(txt_path)
            try:
                with open(txt_path, 'r', errors='replace') as f:
                    text_prefix = f.read(3000).lower()
                for book_id, title_kws, author_kws, year in BOOKS:
                    title_match = any(kw.lower() in text_prefix for kw in title_kws)
                    author_match = any(a.lower() in text_prefix for a in author_kws)
                    if title_match and author_match:
                        results[book_id].append({
                            "location": f"{test_subdir}/{fname}",
                            "match_type": "text_file_content"
                        })
            except Exception as e:
                pass


# ============================================================
# PRINT RESULTS
# ============================================================
print("\n" + "=" * 80)
print("RESULTS SUMMARY")
print("=" * 80)

book_names = {
    1: '"Elements of Physics" by Carhart & Chute (1897)',
    2: '"Physics by Experiment" by Edward R. Shaw (1891)',
    3: '"Sound, Light and Heat" by Mark R. Wright (1888)',
    4: '"Elementary Treatise on Physics" / Ganot (1886)',
    5: '"Fourteen Weeks in Physics" by J. Dorman Steele (1878)',
    6: '"Lessons in Electricity" by John Tyndall (1876)',
    7: '"Sound" by John Tyndall (1875)',
    8: '"System of Natural Philosophy" by J.L. Comstock (1858)',
    9: '"School Compendium of Natural and Experimental Philosophy" by R.G. Parker (1850/1856)',
    10: '"Scientific Papers of James Clerk Maxwell" (1890)',
    11: '"Science of Mechanics" by Ernst Mach (1883)',
    12: '"Die kinetische Theorie der Gase" by Oskar Emil Meyer (1877)',
    13: '"Treatise on Electricity and Magnetism" by James Clerk Maxwell (1873)',
    14: '"Horologium Oscillatorium" by Christiaan Huygens (1673)',
    15: '"Experimenta Nova" by Otto von Guericke (1672)',
    16: '"Lectiones geometricae" by Isaac Barrow (1670)',
    17: '"Physica" by Aristotle (ca. 350 BCE)',
}

found_books = []
not_found_books = []

for book_id in sorted(book_names.keys()):
    matches = results.get(book_id, [])
    name = book_names[book_id]
    if matches:
        found_books.append(book_id)
        print(f"\n[FOUND] Book #{book_id}: {name}")
        # Deduplicate by location
        seen = set()
        for m in matches:
            loc = m.get("location", "")
            if loc not in seen:
                seen.add(loc)
                print(f"  Location: {loc}")
                for k, v in m.items():
                    if k != "location" and v is not None:
                        print(f"    {k}: {v}")
    else:
        not_found_books.append(book_id)
        print(f"\n[NOT FOUND] Book #{book_id}: {name}")

print("\n" + "=" * 80)
print(f"SUMMARY: {len(found_books)} found, {len(not_found_books)} not found")
print(f"Found: {[book_names[b] for b in found_books]}")
print(f"Not found: {[book_names[b] for b in not_found_books]}")
