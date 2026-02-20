#!/usr/bin/env python3
"""
Train a BPE tokenizer on historical text corpus.

Reads from a directory of cleaned parquet shards (with 'text' column)
and trains a tokenizer in the style of GPT-4, adapted for pre-1900 English text.

Usage:
    python scripts/pre1900_scripts/hf_tok_train.py --input ./data/pre1900_clean --output ./tokenizer_1900
    python scripts/pre1900_scripts/hf_tok_train.py --input ./data/pre1900_clean --vocab-size 32768
"""

import argparse
import random
import time
from pathlib import Path

import pyarrow.parquet as pq

from nanochat.tokenizer import RustBPETokenizer


def text_iterator(input_dir: Path, max_chars: int, doc_cap: int, shuffle: bool = True):
    """
    Iterate through parquet shards in the input directory.

    Args:
        input_dir: Directory containing .parquet files with a 'text' column
        max_chars: Maximum total characters to yield
        doc_cap: Maximum characters per document
        shuffle: Whether to shuffle file order
    """
    pq_files = list(input_dir.rglob("*.parquet"))
    if not pq_files:
        raise ValueError(f"No .parquet files found in {input_dir}")

    print(f"Found {len(pq_files)} parquet files")

    if shuffle:
        random.shuffle(pq_files)

    nchars = 0
    ndocs = 0

    for filepath in pq_files:
        try:
            table = pq.read_table(filepath, columns=["text"])
        except Exception as e:
            print(f"Warning: Could not read {filepath}: {e}")
            continue

        texts = table.column("text").to_pylist()
        if shuffle:
            random.shuffle(texts)

        for text in texts:
            if len(text) < 1000:
                continue

            if len(text) > doc_cap:
                text = text[:doc_cap]

            nchars += len(text)
            ndocs += 1

            yield text

            if nchars >= max_chars:
                print(f"Reached {nchars:,} chars from {ndocs} docs")
                return

    print(f"Exhausted all files: {nchars:,} chars from {ndocs} docs")


def main():
    parser = argparse.ArgumentParser(
        description='Train a BPE tokenizer on historical text corpus',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--input', '-i', type=Path, required=True,
                        help='Input directory containing cleaned parquet shards')
    parser.add_argument('--output', '-o', type=Path, default=Path('./tokenizer_1900'),
                        help='Output directory for tokenizer')
    parser.add_argument('--max-chars', type=int, default=2_000_000_000,
                        help='Maximum characters to train on')
    parser.add_argument('--doc-cap', type=int, default=50_000,
                        help='Maximum characters per document')
    parser.add_argument('--vocab-size', type=int, default=32768,
                        help='Vocabulary size (default: 32768 = 2^15)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for shuffling')
    args = parser.parse_args()

    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"max_chars: {args.max_chars:,}")
    print(f"doc_cap: {args.doc_cap:,}")
    print(f"vocab_size: {args.vocab_size:,}")

    random.seed(args.seed)

    text_iter = text_iterator(args.input, args.max_chars, args.doc_cap)

    print("\nTraining tokenizer...")
    t0 = time.time()
    tokenizer = RustBPETokenizer.train_from_iterator(text_iter, args.vocab_size)
    t1 = time.time()
    train_time = t1 - t0
    print(f"Training time: {train_time:.2f}s")

    args.output.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(args.output))
    print(f"Saved tokenizer to {args.output}")

    # Sanity check with period-appropriate text
    test_text = """CHAPTER I.

In which the Reader is introduced to a Man of Honour.

Mr. Allworthy had been absent a full Quarter of a Year in London, on some very
particular Business, tho' I know not what it was; but judge of its Importance by
its having detained him so long from home, whence he had not been absent a Month
at a Time during the Space of many Years.

"'Tis most extraordinary," said he, "that such a thing should occur."
Numbers: 1776, 1842, MDCCCXLV
Contractions: 'twas, 'tis, I'm, you're
Special: @#$%^&*()
"""

    print("\nSanity check:")
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    print(f"  Original length: {len(test_text)}")
    print(f"  Encoded tokens: {len(encoded)}")
    print(f"  Compression ratio: {len(test_text) / len(encoded):.2f} chars/token")

    if decoded != test_text:
        print("  WARNING: Decoded text doesn't match original!")
        print(f"  Decoded: {decoded[:200]}...")
    else:
        print("  Decode check: PASSED")

    # Create token_bytes mapping for bits-per-byte evaluation
    import torch
    print("\nCreating token_bytes mapping...")
    vocab_size = tokenizer.get_vocab_size()
    special_set = set(tokenizer.get_special_tokens())
    token_bytes = []

    for token_id in range(vocab_size):
        token_str = tokenizer.decode([token_id])
        if token_str in special_set:
            token_bytes.append(0)
        else:
            id_bytes = len(token_str.encode("utf-8"))
            token_bytes.append(id_bytes)

    token_bytes = torch.tensor(token_bytes, dtype=torch.int32, device='cpu')
    token_bytes_path = args.output / "token_bytes.pt"
    torch.save(token_bytes, token_bytes_path)
    print(f"Saved token_bytes to {token_bytes_path}")

    token_bytes_nonzero = token_bytes[token_bytes > 0].to(dtype=torch.float32)
    print(f"\nToken statistics:")
    print(f"  Vocab size: {vocab_size}")
    print(f"  Special tokens: {len(special_set)}")
    print(f"  Token bytes min: {int(token_bytes_nonzero.min().item())}")
    print(f"  Token bytes max: {int(token_bytes_nonzero.max().item())}")
    print(f"  Token bytes mean: {token_bytes_nonzero.mean().item():.2f}")

    print(f"\nSample tokens from vocabulary:")
    sample_ids = [100, 500, 1000, 5000, 10000, 20000, 30000]
    for tid in sample_ids:
        if tid < vocab_size:
            token_str = tokenizer.decode([tid])
            print(f"  {tid}: {repr(token_str)}")



if __name__ == "__main__":
    main()
