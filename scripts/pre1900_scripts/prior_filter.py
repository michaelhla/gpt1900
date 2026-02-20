#!/usr/bin/env python3
"""
Prior-based data filtering analysis (inspired by Seo et al., "Prior Filter").

Computes corpus-level token frequency priors and uses per-document statistics
(mean and std of token log-priors) to characterize document quality. This is
a ~1000x cheaper proxy for perplexity-based filtering: no model inference needed,
just token counting.

Usage:
    python scripts/pre1900_scripts/prior_filter.py --datadir /mnt/main0/data/michaelhla/pre1900_raw
    python scripts/pre1900_scripts/prior_filter.py --datadir /mnt/main0/data/michaelhla/pre1900_raw --sample 50000 --sources institutional blbooks newspapers
"""

from __future__ import annotations
import argparse
import glob
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm
from transformers import GPT2TokenizerFast


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("prior_filter")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s", "%H:%M:%S"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def load_texts(datadir: Path, sources: list[str], sample_per_source: int, seed: int, logger) -> list[dict]:
    """Load a sample of documents from parquet shards, returning list of {text, source, ...}."""
    rng = np.random.default_rng(seed)
    docs = []

    source_configs = {
        "institutional": ("institutional/inst_*.parquet", ["text", "source", "title", "year", "ocr_score"]),
        "blbooks": ("blbooks/bl_*.parquet", ["text", "source", "title", "year", "ocr_score"]),
        "newspapers": ("newspapers/news_*.parquet", ["text", "source", "headline", "year", "legibility"]),
    }

    for src in sources:
        if src not in source_configs:
            logger.warning(f"Unknown source: {src}")
            continue

        pattern, columns = source_configs[src]
        files = sorted(glob.glob(str(datadir / pattern)))
        if not files:
            logger.warning(f"No files found for {src}")
            continue

        # Count total rows
        row_counts = []
        for f in files:
            row_counts.append(pq.ParquetFile(f).metadata.num_rows)
        total = sum(row_counts)
        n_sample = min(sample_per_source, total)
        logger.info(f"{src}: {total:,} docs across {len(files)} shards, sampling {n_sample:,}")

        # Reservoir sampling: pick which (file_idx, row_idx) pairs to read
        # For efficiency, pick random global indices then map to file+row
        chosen = set(rng.choice(total, size=n_sample, replace=False).tolist())

        cumulative = 0
        for fi, f in enumerate(tqdm(files, desc=f"loading {src}", leave=False)):
            n_rows = row_counts[fi]
            # Which global indices fall in this file?
            local_indices = []
            for gi in range(cumulative, cumulative + n_rows):
                if gi in chosen:
                    local_indices.append(gi - cumulative)
            cumulative += n_rows

            if not local_indices:
                continue

            # Read only needed columns
            available_cols = pq.ParquetFile(f).schema_arrow.names
            cols_to_read = [c for c in columns if c in available_cols]
            table = pq.read_table(f, columns=cols_to_read)

            for li in local_indices:
                row = {c: table.column(c)[li].as_py() for c in cols_to_read}
                docs.append(row)

        logger.info(f"  loaded {sum(1 for d in docs if d.get('source','').startswith(src[:4]))} docs from {src}")

    rng.shuffle(docs)
    return docs


def compute_token_priors(docs: list[dict], tokenizer, logger) -> tuple[np.ndarray, dict]:
    """
    Tokenize all documents and compute corpus-level token frequency priors.

    Returns:
        log_priors: array of shape (vocab_size,) with log2(frequency / total_tokens)
                    for each token. Unseen tokens get log2(1 / total_tokens).
        token_counts: Counter of token frequencies
    """
    logger.info("Tokenizing documents and counting token frequencies...")
    vocab_size = tokenizer.vocab_size
    counts = Counter()
    total_tokens = 0
    doc_token_ids = []

    for doc in tqdm(docs, desc="tokenizing"):
        text = doc["text"]
        # Truncate very long docs for efficiency (first 50k chars)
        ids = tokenizer.encode(text[:50_000], add_special_tokens=False)
        counts.update(ids)
        total_tokens += len(ids)
        doc_token_ids.append(ids)

    logger.info(f"Total tokens: {total_tokens:,}, unique tokens: {len(counts):,} / {vocab_size}")

    # Compute log2 priors: log2(count / total)
    # Unseen tokens get a floor of 1 / total (Laplace-like minimum)
    log_priors = np.full(vocab_size, np.log2(1.0 / total_tokens))
    for tok_id, cnt in counts.items():
        if tok_id < vocab_size:
            log_priors[tok_id] = np.log2(cnt / total_tokens)

    return log_priors, counts, doc_token_ids


def compute_doc_stats(doc_token_ids: list[list[int]], log_priors: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For each document, compute mean and std of token log-priors.

    Returns: (means, stds, lengths) arrays of shape (n_docs,)
    """
    means = np.zeros(len(doc_token_ids))
    stds = np.zeros(len(doc_token_ids))
    lengths = np.zeros(len(doc_token_ids), dtype=int)

    for i, ids in enumerate(doc_token_ids):
        if len(ids) == 0:
            means[i] = 0.0
            stds[i] = 0.0
            lengths[i] = 0
            continue
        token_lps = log_priors[ids]
        means[i] = token_lps.mean()
        stds[i] = token_lps.std()
        lengths[i] = len(ids)

    return means, stds, lengths


def print_summary(means, stds, lengths, docs, logger):
    """Print summary statistics and a text histogram."""
    logger.info("")
    logger.info("=" * 70)
    logger.info("PRIOR FILTER SUMMARY STATISTICS")
    logger.info("=" * 70)

    for name, arr in [("mean(log2 prior)", means), ("std(log2 prior)", stds), ("token length", lengths)]:
        pcts = np.percentile(arr, [1, 5, 10, 25, 50, 75, 90, 95, 99])
        logger.info(f"\n{name}:")
        logger.info(f"  mean={arr.mean():.4f}  std={arr.std():.4f}  min={arr.min():.4f}  max={arr.max():.4f}")
        logger.info(f"  percentiles: p1={pcts[0]:.4f} p5={pcts[1]:.4f} p10={pcts[2]:.4f} p25={pcts[3]:.4f} "
                     f"p50={pcts[4]:.4f} p75={pcts[5]:.4f} p90={pcts[6]:.4f} p95={pcts[7]:.4f} p99={pcts[8]:.4f}")

    # Text histogram for means
    logger.info("\n--- Histogram of mean(log2 prior) ---")
    _text_histogram(means, bins=30, logger=logger)

    # Text histogram for stds
    logger.info("\n--- Histogram of std(log2 prior) ---")
    _text_histogram(stds, bins=30, logger=logger)

    # Per-source breakdown
    sources = set(d.get("source", "unknown") for d in docs)
    if len(sources) > 1:
        logger.info("\n--- Per-source breakdown ---")
        for src in sorted(sources):
            mask = np.array([d.get("source", "") == src for d in docs])
            n = mask.sum()
            if n == 0:
                continue
            logger.info(f"\n  {src} (n={n:,}):")
            logger.info(f"    mean(log2 prior): mean={means[mask].mean():.4f}  std={means[mask].std():.4f}")
            logger.info(f"    std(log2 prior):  mean={stds[mask].mean():.4f}  std={stds[mask].std():.4f}")
            logger.info(f"    token length:     mean={lengths[mask].mean():.0f}  median={np.median(lengths[mask]):.0f}")


def _text_histogram(arr, bins=30, width=50, logger=None):
    """Print a simple text-based histogram."""
    counts, edges = np.histogram(arr, bins=bins)
    max_count = counts.max()
    for i in range(len(counts)):
        bar_len = int(counts[i] / max_count * width) if max_count > 0 else 0
        bar = "#" * bar_len
        logger.info(f"  [{edges[i]:8.3f}, {edges[i+1]:8.3f}) {counts[i]:6d} {bar}")


def print_examples(means, stds, docs, logger):
    """Show example documents at different quality levels."""
    logger.info("")
    logger.info("=" * 70)
    logger.info("EXAMPLE DOCUMENTS AT DIFFERENT PRIOR LEVELS")
    logger.info("=" * 70)

    # Sort by mean prior (higher = more common tokens = potentially lower quality)
    order = np.argsort(means)

    percentiles = [
        ("LOWEST mean prior (most unusual tokens — likely OCR garbage / rare lang)", 0.01),
        ("10th percentile (unusual but potentially valid)", 0.10),
        ("25th percentile", 0.25),
        ("MEDIAN (typical document)", 0.50),
        ("75th percentile", 0.75),
        ("90th percentile (very common tokens)", 0.90),
        ("HIGHEST mean prior (most common tokens — potentially repetitive/boilerplate)", 0.99),
    ]

    for label, pct in percentiles:
        idx = order[int(pct * len(order))]
        doc = docs[idx]
        text = doc["text"]
        snippet = text[:500].replace("\n", " ").strip()
        if len(text) > 500:
            snippet += "..."

        logger.info(f"\n--- {label} ---")
        logger.info(f"  mean_prior={means[idx]:.4f}  std_prior={stds[idx]:.4f}")
        meta_parts = []
        for k in ["source", "title", "headline", "year", "ocr_score", "legibility"]:
            if k in doc and doc[k] is not None and doc[k] != "" and doc[k] != -1:
                meta_parts.append(f"{k}={doc[k]}")
        logger.info(f"  {', '.join(meta_parts)}")
        logger.info(f"  TEXT: {snippet}")


def print_filter_recommendations(means, stds, docs, logger):
    """Suggest filter thresholds and show what would be kept/removed."""
    logger.info("")
    logger.info("=" * 70)
    logger.info("FILTER ANALYSIS")
    logger.info("=" * 70)

    # The paper filters on both mean and std. Low mean = rare tokens (OCR garbage),
    # high mean = very common (boilerplate). High std = mixed quality.
    # Try filtering at various mean thresholds (removing tails)
    for pct_lo, pct_hi in [(1, 99), (5, 95), (10, 90)]:
        lo = np.percentile(means, pct_lo)
        hi = np.percentile(means, pct_hi)
        kept = ((means >= lo) & (means <= hi))
        n_kept = kept.sum()
        logger.info(f"\n  Filter: keep mean_prior in [{lo:.4f}, {hi:.4f}] (p{pct_lo}-p{pct_hi})")
        logger.info(f"    Kept: {n_kept:,} / {len(means):,} ({100*n_kept/len(means):.1f}%)")

        # Show what the removed docs look like
        removed_lo = means < lo
        removed_hi = means > hi
        if removed_lo.any():
            idx = np.where(removed_lo)[0][0]
            text = docs[idx]["text"][:200].replace("\n", " ")
            logger.info(f"    Example removed (low):  mean={means[idx]:.4f} | {text}")
        if removed_hi.any():
            idx = np.where(removed_hi)[0][-1]
            text = docs[idx]["text"][:200].replace("\n", " ")
            logger.info(f"    Example removed (high): mean={means[idx]:.4f} | {text}")

    # Also show std-based filtering
    logger.info("\n  --- Std-based filtering ---")
    for pct in [95, 90]:
        thresh = np.percentile(stds, pct)
        kept = stds <= thresh
        n_kept = kept.sum()
        logger.info(f"  Filter: keep std_prior <= {thresh:.4f} (below p{pct})")
        logger.info(f"    Kept: {n_kept:,} / {len(stds):,} ({100*n_kept/len(stds):.1f}%)")


def save_results(means, stds, lengths, docs, outpath: Path, logger):
    """Save per-document stats to a JSON lines file for further analysis."""
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with open(outpath, "w") as f:
        for i, doc in enumerate(docs):
            row = {
                "mean_prior": float(means[i]),
                "std_prior": float(stds[i]),
                "n_tokens": int(lengths[i]),
                "source": doc.get("source", ""),
                "year": doc.get("year", -1),
                "title": doc.get("title", doc.get("headline", "")),
            }
            if "ocr_score" in doc:
                row["ocr_score"] = doc["ocr_score"]
            if "legibility" in doc:
                row["legibility"] = doc["legibility"]
            f.write(json.dumps(row) + "\n")
    logger.info(f"Saved per-document stats to {outpath}")


def main():
    parser = argparse.ArgumentParser(
        description="Prior-based data filtering analysis for pre-1900 corpus",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--datadir", type=Path, default=Path("/mnt/main0/data/michaelhla/pre1900_raw"))
    parser.add_argument("--sample", type=int, default=20000, help="Docs to sample per source")
    parser.add_argument("--sources", nargs="+", default=["institutional", "blbooks", "newspapers"],
                        choices=["institutional", "blbooks", "newspapers"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outfile", type=Path, default=None, help="Save per-doc stats to JSONL")
    parser.add_argument("--save-priors", type=Path, default=None,
                        help="Save log_priors array as .npy for use by prior_filter_apply.py")

    args = parser.parse_args()
    logger = setup_logger()

    t0 = time.time()

    # Load tokenizer (GPT-2 BPE, matching the Prior Filter paper)
    logger.info("Loading GPT-2 tokenizer...")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    # Load sampled documents
    docs = load_texts(args.datadir, args.sources, args.sample, args.seed, logger)
    logger.info(f"Loaded {len(docs):,} documents total")

    if not docs:
        logger.error("No documents loaded. Check --datadir and --sources.")
        return

    # Compute token priors
    log_priors, token_counts, doc_token_ids = compute_token_priors(docs, tokenizer, logger)

    # Show top/bottom tokens by prior
    logger.info("\nTop 20 most frequent tokens:")
    for tok_id, cnt in token_counts.most_common(20):
        tok_str = tokenizer.decode([tok_id]).replace("\n", "\\n")
        logger.info(f"  {tok_str!r:20s} id={tok_id:5d}  count={cnt:>10,}  log2_prior={log_priors[tok_id]:.4f}")

    logger.info("\nBottom 20 least frequent tokens (of those seen):")
    bottom = token_counts.most_common()[-20:]
    for tok_id, cnt in reversed(bottom):
        tok_str = tokenizer.decode([tok_id]).replace("\n", "\\n")
        logger.info(f"  {tok_str!r:20s} id={tok_id:5d}  count={cnt:>10,}  log2_prior={log_priors[tok_id]:.4f}")

    # Compute per-document stats
    means, stds, lengths = compute_doc_stats(doc_token_ids, log_priors)

    # Print analysis
    print_summary(means, stds, lengths, docs, logger)
    print_examples(means, stds, docs, logger)
    print_filter_recommendations(means, stds, docs, logger)

    if args.save_priors:
        args.save_priors.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.save_priors, log_priors)
        logger.info(f"Saved prior table ({len(log_priors)} tokens) to {args.save_priors}")

    if args.outfile:
        save_results(means, stds, lengths, docs, args.outfile, logger)

    elapsed = time.time() - t0
    logger.info(f"\nDone in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
