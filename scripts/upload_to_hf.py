"""
Upload a nanochat checkpoint to HuggingFace Hub.

Usage:
    python scripts/upload_to_hf.py
"""

import os
import shutil
import tempfile
from huggingface_hub import HfApi, create_repo

REPO_ID = "mhla/gpt1900-d26-8btok"
CHECKPOINT_DIR = "/mnt/main0/data/michaelhla/gpt1900_training/base_checkpoints/d26"
TOKENIZER_DIR = "/mnt/main0/data/michaelhla/gpt1900_training/tokenizer"
EVAL_CSV = "/mnt/main0/data/michaelhla/gpt1900_training/base_eval/base_model_007226.csv"
NANOCHAT_DIR = "/mnt/main0/home/michaelhla/evolutionaryscale/gpt1900/nanochat"
STEP = 7226

# Source files needed to load the model
SOURCE_FILES = [
    "gpt.py",
    "tokenizer.py",
    "checkpoint_manager.py",
    "common.py",
    "flash_attention.py",
    "engine.py",
    "optim.py",
    "fp8.py",
    "loss_eval.py",
    "__init__.py",
]

MODEL_CARD = """\
---
license: mit
language:
- en
tags:
- gpt
- pre-1900
- historical
- nanochat
---

# GPT-1900 (D26, 8B tokens)

A 1.2B parameter GPT-style language model trained exclusively on pre-1900 English text.

## Model Details

- **Architecture:** Custom GPT with RoPE, QK-norm, ReLU², value embeddings (ResFormer), per-layer residual/skip scalars
- **Parameters:** ~1.2B
- **Layers:** 26
- **Hidden dim:** 1664
- **Attention heads:** 13 (query) / 13 (kv)
- **Head dim:** 128
- **Context length:** 2048 tokens
- **Vocab size:** 32,768 (BPE, GPT-4 style split pattern)
- **Training:** ~8B tokens of pre-1900 English text, FP8 (tensorwise), Muon+AdamW optimizer
- **Final val BPB:** 1.211

## Checkpoint Contents

```
model_007226.pt          # Model weights (4.9 GB)
meta_007226.json         # Training config and metadata
optim_007226_rank*.pt    # Optimizer state, 8 FSDP shards (for resuming training)
tokenizer/               # BPE tokenizer (tiktoken format) + token byte counts
nanochat/                # Source code to load and run the model
eval_results.csv         # Benchmark eval results at this checkpoint
```

## Quick Start

```python
import torch
from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import RustBPETokenizer

# Load tokenizer
tokenizer = RustBPETokenizer.from_directory("tokenizer")

# Load model
import json
with open("meta_007226.json") as f:
    meta = json.load(f)

config = GPTConfig(**meta["model_config"])

with torch.device("meta"):
    model = GPT(config)
model.to_empty(device="cuda")
model.init_weights()

state_dict = torch.load("model_007226.pt", map_location="cuda")
state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
model.load_state_dict(state_dict, strict=True, assign=True)
model.eval()

# Generate
bos = tokenizer.get_bos_token_id()
tokens = tokenizer.encode("It was a dark and stormy night", prepend=bos)
with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
    for token in model.generate(tokens, max_tokens=100, temperature=0.8):
        print(tokenizer.decode([token]), end="", flush=True)
```

## Dependencies

```
torch>=2.9
tiktoken
rustbpe
```

## Eval Results (step 7226)

| Task | Accuracy | Centered |
|------|----------|----------|
| hellaswag | 0.318 | 0.091 |
| arc_easy | 0.411 | 0.215 |
| lambada_openai | 0.332 | 0.332 |
| piqa | 0.586 | 0.172 |
| winograd | 0.674 | 0.348 |
| copa | 0.570 | 0.140 |
| **CORE** | | **0.126** |

## Training

Trained with the [nanochat](https://github.com/karpathy/nanochat) framework using 8x H100 GPUs with FSDP.
To resume training, load the optimizer shards (`optim_007226_rank*.pt`) — one per FSDP rank.
"""


def main():
    api = HfApi()

    # Create the repo
    print(f"Creating repo: {REPO_ID}")
    create_repo(REPO_ID, repo_type="model", exist_ok=True)

    # Upload model weights and metadata (the large files)
    checkpoint_files = [
        f"model_{STEP:06d}.pt",
        f"meta_{STEP:06d}.json",
    ]
    # Add optimizer shards
    for rank in range(8):
        checkpoint_files.append(f"optim_{STEP:06d}_rank{rank}.pt")

    print(f"\nUploading checkpoint files from {CHECKPOINT_DIR}...")
    for fname in checkpoint_files:
        fpath = os.path.join(CHECKPOINT_DIR, fname)
        if os.path.exists(fpath):
            size_gb = os.path.getsize(fpath) / (1024**3)
            print(f"  Uploading {fname} ({size_gb:.2f} GB)...")
            api.upload_file(
                path_or_fileobj=fpath,
                path_in_repo=fname,
                repo_id=REPO_ID,
            )
        else:
            print(f"  WARNING: {fpath} not found, skipping")

    # Upload tokenizer files
    print(f"\nUploading tokenizer files from {TOKENIZER_DIR}...")
    for fname in os.listdir(TOKENIZER_DIR):
        fpath = os.path.join(TOKENIZER_DIR, fname)
        if os.path.isfile(fpath):
            print(f"  Uploading tokenizer/{fname}...")
            api.upload_file(
                path_or_fileobj=fpath,
                path_in_repo=f"tokenizer/{fname}",
                repo_id=REPO_ID,
            )

    # Upload source code
    print(f"\nUploading nanochat source files...")
    for fname in SOURCE_FILES:
        fpath = os.path.join(NANOCHAT_DIR, fname)
        if os.path.exists(fpath):
            print(f"  Uploading nanochat/{fname}...")
            api.upload_file(
                path_or_fileobj=fpath,
                path_in_repo=f"nanochat/{fname}",
                repo_id=REPO_ID,
            )
        else:
            print(f"  WARNING: {fpath} not found, skipping")

    # Upload eval results
    if os.path.exists(EVAL_CSV):
        print(f"\nUploading eval results...")
        api.upload_file(
            path_or_fileobj=EVAL_CSV,
            path_in_repo="eval_results.csv",
            repo_id=REPO_ID,
        )

    # Upload model card
    print(f"\nUploading model card (README.md)...")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write(MODEL_CARD)
        readme_path = f.name
    api.upload_file(
        path_or_fileobj=readme_path,
        path_in_repo="README.md",
        repo_id=REPO_ID,
    )
    os.unlink(readme_path)

    print(f"\nDone! Model uploaded to: https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    main()
