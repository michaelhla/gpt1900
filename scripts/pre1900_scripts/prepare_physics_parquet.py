"""
Convert raw .txt physics books into chunked parquet files for CLM training.

Includes cleaning to strip boilerplate headers/footers from Wikisource,
Google Books, and Project Gutenberg sources, plus OCR noise removal.

Usage:
    python -m scripts.pre1900_scripts.prepare_physics_parquet \
        --input-dir data/physics_books data/core_physics_books \
        --output-dir $NANOCHAT_BASE_DIR/physics_clm_data
"""

import argparse
import os
import re
import random
import pyarrow as pa
import pyarrow.parquet as pq


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

MIN_FILE_CHARS = 3000  # skip stub files smaller than this

def clean_text(text):
    """
    Clean a raw physics-book text file:
      1. Strip Wikisource header (←/→ nav, metadata ID lines, TOC)
      2. Strip Wikisource footer (public-domain boilerplate)
      3. Strip Google Books header (usage guidelines block)
      4. Strip Project Gutenberg preamble
      5. Strip publisher catalogue / advertisement appendices
      6. Remove OCR garbage lines
      7. Collapse excessive blank lines
      8. Remove zero-width spaces
    """
    text = _strip_wikisource_header(text)
    text = _strip_wikisource_footer(text)
    text = _strip_google_books_header(text)
    text = _strip_gutenberg_header(text)
    text = _strip_trailing_catalogue(text)
    text = _remove_ocr_noise_lines(text)
    text = _collapse_blank_lines(text)
    text = text.replace("\u200b", "")  # zero-width spaces
    return text.strip()


def _strip_wikisource_header(text):
    """
    Wikisource files start with:
        ←
        <blank>
        <title/author metadata>
        →
        <optional TOC / nav lines>
        <numeric ID><Title><Date><Author>
        ​  (zero-width space marks start of real content)

    Strategy: find the first zero-width space (U+200B) in the first 3000 chars.
    If found, strip everything before it. Also strip the ← / → nav markers.
    """
    if not text.startswith("←"):
        return text

    # Look for zero-width space in early part of file
    zwsp_pos = text.find("\u200b", 0, 3000)
    if zwsp_pos >= 0:
        text = text[zwsp_pos + 1:]  # skip past the zero-width space
    else:
        # Fallback: skip past the → marker
        arrow_pos = text.find("→", 0, 1000)
        if arrow_pos >= 0:
            text = text[arrow_pos + 1:]

    # Also strip the numeric ID + title + date + author metadata line
    # Pattern: line starting with digits, e.g. "4499939Outlines of Experiments..."
    text = re.sub(r'^\d{5,}[^\n]*\n', '', text, count=1)
    return text


def _strip_wikisource_footer(text):
    """
    Wikisource footers contain public-domain notices like:
        "This work was published before January 1, 1931..."
        "Public domainPublic domainfalsefalse"
        "This work is in the public domain..."
    Strip everything from the first such marker to EOF.
    """
    markers = [
        "This work was published before January",
        "This work is a translation and has a separate copyright status",
        "This work is in the public domain in the United States",
        "Public domainPublic domain",
    ]
    # Find the earliest marker in the last 2000 chars
    tail_start = max(0, len(text) - 2000)
    earliest = len(text)
    for marker in markers:
        pos = text.find(marker, tail_start)
        if pos >= 0 and pos < earliest:
            earliest = pos

    if earliest < len(text):
        # Walk back to the start of the line / paragraph
        while earliest > 0 and text[earliest - 1] != '\n':
            earliest -= 1
        text = text[:earliest]

    # Also strip footnote-style ↑ references at the very end (Wikisource cross-refs)
    # These look like: "\n↑ Some reference text\n"
    lines = text.rstrip().split('\n')
    while lines and lines[-1].strip().startswith('↑'):
        lines.pop()
    text = '\n'.join(lines)

    return text


def _strip_google_books_header(text):
    """
    Google Books files start with a usage-guidelines block ending with a
    Google Books URL: "at |http...books...google...com/"
    Strip everything up to and including that URL line.
    """
    # Check for Google Books header in the first 3000 chars
    if "Google" not in text[:500]:
        return text

    # Find the Google Books URL that ends the boilerplate
    match = re.search(r'at\s*\|?\s*http\s*:?\s*//\s*books\s*\.?\s*google\s*\.?\s*com\s*/\s*\|?\s*\n?', text[:3000])
    if match:
        text = text[match.end():]
    return text


def _strip_gutenberg_header(text):
    """
    Project Gutenberg files may start with preparation credits.
    Strip lines before the first all-caps title line if we see 'Gutenberg' early.
    """
    if "Gutenberg" not in text[:500] and "E-text prepared by" not in text[:200]:
        return text

    # Find the end of the Gutenberg preamble: look for a blank line
    # after the URL/credit block, then content starts
    lines = text.split('\n')
    for i, line in enumerate(lines[:20]):
        # Skip past the credit block — find first blank line after non-blank
        if i > 3 and line.strip() == '' and any(l.strip() for l in lines[:i]):
            # Verify next non-blank line looks like content (title, chapter, etc.)
            for j in range(i + 1, min(i + 5, len(lines))):
                if lines[j].strip():
                    return '\n'.join(lines[j:])
            break
    return text


def _strip_trailing_catalogue(text):
    """
    Some books (esp. Gutenberg) end with publisher catalogues after 'THE END.'
    Strip everything after 'THE END.' if it appears near the end.
    """
    # Look for 'THE END.' in the last 5000 chars
    tail_start = max(0, len(text) - 5000)
    match = re.search(r'\bTHE END\.?\s*\n', text[tail_start:])
    if match:
        end_pos = tail_start + match.end()
        # Keep 'THE END.' but drop everything after
        text = text[:end_pos]
    return text


def _remove_ocr_noise_lines(text):
    """Remove lines that are pure OCR garbage (mostly special characters)."""
    lines = text.split('\n')
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            cleaned.append(line)
            continue
        # Skip lines that are only special chars / punctuation (no letters or digits)
        alnum_count = sum(1 for c in stripped if c.isalnum())
        if alnum_count == 0 and len(stripped) > 1:
            continue
        # Skip lines that look like OCR header garbage (very low letter ratio, lots of special chars)
        if len(stripped) > 10 and alnum_count / len(stripped) < 0.3:
            continue
        cleaned.append(line)
    return '\n'.join(cleaned)


def _collapse_blank_lines(text):
    """Collapse runs of 3+ blank lines down to 2."""
    return re.sub(r'\n{4,}', '\n\n\n', text)


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_text(text, chunk_size=4000):
    """
    Split a long document into chunks at natural boundaries.
    Target ~chunk_size chars per chunk (~1000 tokens).

    Strategy:
    1. Split at paragraph breaks (\\n\\n)
    2. If a paragraph is still too long, split at sentence boundaries
    3. Accumulate into chunks up to target size
    """
    # Split into paragraphs
    paragraphs = text.split("\n\n")

    # Further split overly long paragraphs at sentence boundaries
    pieces = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(para) <= chunk_size:
            pieces.append(para)
        else:
            # Split at sentence boundaries
            sentences = re.split(r'(?<=[.!?])\s+', para)
            for sent in sentences:
                if sent.strip():
                    pieces.append(sent.strip())

    # Accumulate pieces into chunks up to target size
    chunks = []
    current = []
    current_len = 0
    for piece in pieces:
        piece_len = len(piece)
        # If adding this piece would exceed target and we already have content, flush
        if current_len > 0 and current_len + piece_len + 2 > chunk_size:
            chunks.append("\n\n".join(current))
            current = [piece]
            current_len = piece_len
        else:
            current.append(piece)
            current_len += piece_len + (2 if current_len > 0 else 0)  # +2 for \n\n separator

    # Don't forget the last chunk
    if current:
        chunks.append("\n\n".join(current))

    return chunks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Prepare physics books as parquet for CLM training")
    parser.add_argument("--input-dir", type=str, nargs="+", default=["data/physics_books", "data/core_physics_books"], help="directories of .txt files")
    parser.add_argument("--output-dir", type=str, required=True, help="output directory for parquet files")
    parser.add_argument("--chunk-size", type=int, default=4000, help="target chunk size in chars (~1000 tokens)")
    parser.add_argument("--val-frac", type=float, default=0.05, help="fraction of chunks for validation")
    parser.add_argument("--seed", type=int, default=42, help="random seed for shuffling")
    args = parser.parse_args()

    # Read, clean, and chunk all .txt files from all input directories
    all_chunks = []
    skipped = []
    for input_dir in args.input_dir:
        txt_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".txt")])
        print(f"Found {len(txt_files)} .txt files in {input_dir}")

        for fname in txt_files:
            filepath = os.path.join(input_dir, fname)
            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                raw_text = f.read()

            # Skip stub files
            if len(raw_text) < MIN_FILE_CHARS:
                skipped.append((fname, len(raw_text)))
                print(f"  {fname}: SKIPPED ({len(raw_text):,} chars < {MIN_FILE_CHARS})")
                continue

            text = clean_text(raw_text)
            chars_removed = len(raw_text) - len(text)
            chunks = chunk_text(text, chunk_size=args.chunk_size)
            all_chunks.extend(chunks)
            pct = 100 * chars_removed / len(raw_text) if raw_text else 0
            print(f"  {fname}: {len(raw_text):,} -> {len(text):,} chars ({pct:.0f}% removed) -> {len(chunks)} chunks")

    if skipped:
        print(f"\nSkipped {len(skipped)} files under {MIN_FILE_CHARS} chars: {[s[0] for s in skipped]}")

    print(f"\nTotal chunks: {len(all_chunks)}")
    total_chars = sum(len(c) for c in all_chunks)
    print(f"Total chars: {total_chars:,}")
    print(f"Avg chunk size: {total_chars / len(all_chunks):.0f} chars")

    # Shuffle
    rng = random.Random(args.seed)
    rng.shuffle(all_chunks)

    # Split train/val
    n_val = max(1, int(len(all_chunks) * args.val_frac))
    val_chunks = all_chunks[:n_val]
    train_chunks = all_chunks[n_val:]
    print(f"\nTrain: {len(train_chunks)} chunks ({sum(len(c) for c in train_chunks):,} chars)")
    print(f"Val: {len(val_chunks)} chunks ({sum(len(c) for c in val_chunks):,} chars)")

    # Write parquet files with a single 'text' column
    os.makedirs(args.output_dir, exist_ok=True)

    for split_name, chunks in [("train", train_chunks), ("val", val_chunks)]:
        table = pa.table({"text": chunks})
        out_path = os.path.join(args.output_dir, f"{split_name}.parquet")
        pq.write_table(table, out_path)
        print(f"Wrote {out_path} ({len(chunks)} rows)")

    print("\nDone!")


if __name__ == "__main__":
    main()
