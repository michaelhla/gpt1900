"""
Upload nanochat checkpoints to HuggingFace Hub.

Usage:
    python scripts/upload_to_hf.py                    # upload all configs
    python scripts/upload_to_hf.py --only d26-22btok  # upload just one
"""

import argparse
import os
import json
import tempfile
from huggingface_hub import HfApi, create_repo

BASE_DIR = "/mnt/main0/data/michaelhla/gpt1900_training"
TOKENIZER_DIR = os.path.join(BASE_DIR, "tokenizer")
NANOCHAT_DIR = "/mnt/main0/home/michaelhla/evolutionaryscale/gpt1900/nanochat"

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

# Each upload config: (repo_id, checkpoint_dir, steps_to_upload, num_optim_ranks, description)
UPLOAD_CONFIGS = {
    "d26-8btok": {
        "repo_id": "mhla/gpt1900-d26-8btok",
        "checkpoint_dir": os.path.join(BASE_DIR, "base_checkpoints/d26"),
        "steps": [7226],
        "num_optim_ranks": 8,
        "desc": "GPT-1900 (D26, 8B tokens)",
        "detail": "A 1.2B parameter GPT-style language model trained on ~8B tokens of pre-1900 English text.",
        "params": "~1.2B", "layers": 26, "hidden_dim": 1664, "heads": 13, "val_bpb": "1.211",
    },
    "d26-22btok": {
        "repo_id": "mhla/gpt1900-d26-22btok",
        "checkpoint_dir": os.path.join(BASE_DIR, "base_checkpoints/d26"),
        "steps": [17517],
        "num_optim_ranks": 8,
        "desc": "GPT-1900 (D26, 22B tokens)",
        "detail": "A 1.2B parameter GPT-style language model trained on ~22B tokens of pre-1900 English text (20x data:param ratio).",
        "params": "~1.2B", "layers": 26, "hidden_dim": 1664, "heads": 13, "val_bpb": "0.802",
    },
    "d34-8b-subset": {
        "repo_id": "mhla/gpt1900-d34-8b-subset",
        "checkpoint_dir": os.path.join(BASE_DIR, "base_checkpoints/d34-8b-subset"),
        "steps": [10507],
        "num_optim_ranks": 8,
        "desc": "GPT-1900 (D34, 8B token subset)",
        "detail": "A 3.3B parameter GPT-style language model trained on ~8B tokens of pre-1900 English text (11x data:param ratio).",
        "params": "~3.3B", "layers": 34, "hidden_dim": 2176, "heads": 17, "val_bpb": "0.721",
    },
}


def make_model_card(config, step):
    meta_path = os.path.join(config["checkpoint_dir"], f"meta_{step:06d}.json")
    with open(meta_path) as f:
        meta = json.load(f)

    return f"""\
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

# {config["desc"]}

{config["detail"]}

## Model Details

- **Architecture:** Custom GPT with RoPE, QK-norm, ReLU², value embeddings (ResFormer), per-layer residual/skip scalars
- **Parameters:** {config["params"]}
- **Layers:** {config["layers"]}
- **Hidden dim:** {config["hidden_dim"]}
- **Attention heads:** {config["heads"]} (query) / {config["heads"]} (kv)
- **Head dim:** 128
- **Context length:** 2048 tokens
- **Vocab size:** 32,768 (BPE, GPT-4 style split pattern)
- **Training:** FP8 (tensorwise), Muon+AdamW optimizer
- **Final val BPB:** {config["val_bpb"]}

## Checkpoint Contents

```
model_{step:06d}.pt          # Model weights
meta_{step:06d}.json         # Training config and metadata
optim_{step:06d}_rank*.pt    # Optimizer state shards (for resuming training)
tokenizer/               # BPE tokenizer (tiktoken format) + token byte counts
nanochat/                # Source code to load and run the model
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
with open("meta_{step:06d}.json") as f:
    meta = json.load(f)

config = GPTConfig(**meta["model_config"])

with torch.device("meta"):
    model = GPT(config)
model.to_empty(device="cuda")
model.init_weights()

state_dict = torch.load("model_{step:06d}.pt", map_location="cuda")
state_dict = {{k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}}
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

## Training

Trained with the [nanochat](https://github.com/karpathy/nanochat) framework on H100 GPUs.
To resume training, load the optimizer shards (`optim_{step:06d}_rank*.pt`) — one per rank.
"""


def upload_config(api, name, config):
    repo_id = config["repo_id"]
    checkpoint_dir = config["checkpoint_dir"]
    steps = config["steps"]
    num_optim_ranks = config["num_optim_ranks"]

    print(f"\n{'='*60}")
    print(f"Uploading: {name} -> {repo_id}")
    print(f"{'='*60}")

    create_repo(repo_id, repo_type="model", exist_ok=True)

    # Upload checkpoint files for each step
    for step in steps:
        checkpoint_files = [
            f"model_{step:06d}.pt",
            f"meta_{step:06d}.json",
        ]
        for rank in range(num_optim_ranks):
            checkpoint_files.append(f"optim_{step:06d}_rank{rank}.pt")

        print(f"\nUploading checkpoint files for step {step} from {checkpoint_dir}...")
        for fname in checkpoint_files:
            fpath = os.path.join(checkpoint_dir, fname)
            if os.path.exists(fpath):
                size_gb = os.path.getsize(fpath) / (1024**3)
                print(f"  Uploading {fname} ({size_gb:.2f} GB)...")
                api.upload_file(
                    path_or_fileobj=fpath,
                    path_in_repo=fname,
                    repo_id=repo_id,
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
                repo_id=repo_id,
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
                repo_id=repo_id,
            )
        else:
            print(f"  WARNING: {fpath} not found, skipping")

    # Upload eval results if available
    last_step = steps[-1]
    eval_csv = os.path.join(BASE_DIR, f"base_eval/base_model_{last_step:06d}.csv")
    if os.path.exists(eval_csv):
        print(f"\nUploading eval results ({eval_csv})...")
        api.upload_file(
            path_or_fileobj=eval_csv,
            path_in_repo="eval_results.csv",
            repo_id=repo_id,
        )

    # Upload model card
    last_step = steps[-1]
    model_card = make_model_card(config, last_step)
    print(f"\nUploading model card (README.md)...")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write(model_card)
        readme_path = f.name
    api.upload_file(
        path_or_fileobj=readme_path,
        path_in_repo="README.md",
        repo_id=repo_id,
    )
    os.unlink(readme_path)

    print(f"\nDone! Model uploaded to: https://huggingface.co/{repo_id}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", type=str, default=None, help="upload only this config (e.g. d26-22btok, d34-8b-subset)")
    args = parser.parse_args()

    api = HfApi()

    if args.only:
        if args.only not in UPLOAD_CONFIGS:
            print(f"Unknown config: {args.only}. Available: {list(UPLOAD_CONFIGS.keys())}")
            return
        upload_config(api, args.only, UPLOAD_CONFIGS[args.only])
    else:
        for name, config in UPLOAD_CONFIGS.items():
            upload_config(api, name, config)


if __name__ == "__main__":
    main()
