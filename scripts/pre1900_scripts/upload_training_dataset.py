#!/usr/bin/env python3
"""Upload the chunked/resharded training dataset to HuggingFace.

This uploads the pre1900_full_clean dataset (text-only, chunked to ≤8K chars,
evenly sharded for DDP training) while excluding the _staging/ directory.

Usage:
    python scripts/pre1900_scripts/upload_training_dataset.py
"""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import HfApi


REPO_ID = "mhla/pre1900-training"
DATA_DIR = Path("/mnt/main0/data/michaelhla/pre1900_full_clean")

README = """\
---
dataset_info:
  features:
    - name: text
      dtype: string
---

# Pre-1900 Training Corpus

Chunked and resharded pre-1900 English text corpus, ready for language model training.

## Format

- **266 parquet shards** (265 train + 1 validation)
- **12.8M documents** (chunks of ≤8,000 characters)
- **~22B tokens** estimated
- **Text-only** — single `text` column per row
- Row groups divisible by 8 for even DDP distribution across GPUs
- Last shard (`shard_00265`) is the validation split

## Processing Pipeline

Built from the full pre-1900 filtered corpus through:

1. **OCR cleanup** — removal of OCR artifacts, boilerplate, and unicode normalization
2. **Quality filtering** — token frequency prior-based filtering
3. **Anachronism detection** — three-tier post-1900 physics filter
4. **Document chunking** — long documents split at paragraph/sentence boundaries (max 8K chars, min 200 chars)
5. **Token balancing** — sort-by-length + round-robin distribution across shards for even token counts

## Usage

```python
from datasets import load_dataset
ds = load_dataset("mhla/pre1900-training")
```

## Related

- [`mhla/pre1900-corpus`](https://huggingface.co/datasets/mhla/pre1900-corpus) — full documents with metadata (title, year, source, OCR scores)
- [`mhla/gpt1900-d26-8btok`](https://huggingface.co/mhla/gpt1900-d26-8btok) — GPT-1900 model trained on this data
"""


def main():
    api = HfApi()

    print(f"Creating repo: {REPO_ID}")
    api.create_repo(REPO_ID, repo_type="dataset", exist_ok=True)

    # Write README
    readme_path = DATA_DIR / "README.md"
    readme_path.write_text(README)

    print(f"Uploading {DATA_DIR} to {REPO_ID} (excluding _staging/)...")
    api.upload_large_folder(
        repo_id=REPO_ID,
        repo_type="dataset",
        folder_path=str(DATA_DIR),
        ignore_patterns=["_staging/*", "_staging/**"],
    )
    print(f"Done: https://huggingface.co/datasets/{REPO_ID}")

    # Clean up README from data dir
    if readme_path.exists():
        readme_path.unlink()


if __name__ == "__main__":
    main()
