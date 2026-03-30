"""
Upload post-training (SFT + RL) checkpoints to HuggingFace Hub.

Uses create_commit to batch files and avoid rate limiting.

Usage:
    python scripts/upload_post_training.py
    python scripts/upload_post_training.py --only sft-period
"""

import argparse
import os
import json
import tempfile
from huggingface_hub import HfApi, CommitOperationAdd, create_repo

BASE_DIR = "/mnt/main0/data/michaelhla/gpt1900_training"
TOKENIZER_DIR = os.path.join(BASE_DIR, "tokenizer")
NANOCHAT_DIR = "/mnt/main0/home/michaelhla/evolutionaryscale/gpt1900/nanochat"

SOURCE_FILES = [
    "gpt.py", "tokenizer.py", "checkpoint_manager.py", "common.py",
    "flash_attention.py", "engine.py", "optim.py", "fp8.py", "loss_eval.py", "__init__.py",
]

UPLOAD_CONFIGS = {
    "sft-period": {
        "repo_id": "mhla/gpt1900-d34-sft-period",
        "checkpoint_dir": os.path.join(BASE_DIR, "pre1900_sft_checkpoints/d34"),
        "steps": [20],
        "num_optim_ranks": 1,
        "desc": "GPT-1900 D34 SFT (Period Style)",
        "detail": "3.29B parameter GPT-1900 model fine-tuned on period-style instruction data. Base model: gpt1900-d34-22btok.",
        "params": "3.29B", "layers": 34, "hidden_dim": 2176, "heads": 17,
        "training_info": "SFT on period-style instruction pairs (instruct_data/period/filtered_pairs.jsonl)",
    },
    "sft-modern": {
        "repo_id": "mhla/gpt1900-d34-sft-modern",
        "checkpoint_dir": os.path.join(BASE_DIR, "pre1900_sft_checkpoints/pre1900_sft_period_d34"),
        "steps": [16],
        "num_optim_ranks": 1,
        "desc": "GPT-1900 D34 SFT (Modern Style)",
        "detail": "3.29B parameter GPT-1900 model fine-tuned on modern instruction data. Two-stage SFT: period style first, then modern. Base model: gpt1900-d34-sft-period.",
        "params": "3.29B", "layers": 34, "hidden_dim": 2176, "heads": 17,
        "training_info": "Two-stage SFT: (1) period-style instructions, (2) modern instructions (instruct_data/modern/filtered_pairs.jsonl)",
    },
    "rl": {
        "repo_id": "mhla/gpt1900-d34-rl",
        "checkpoint_dir": os.path.join(BASE_DIR, "pre1900_rl_checkpoints/pre1900_sft_period_d34"),
        "steps": [780],
        "num_optim_ranks": 0,
        "desc": "GPT-1900 D34 RL",
        "detail": "3.29B parameter GPT-1900 model with RL post-training. Pipeline: base pretrain -> SFT (period) -> SFT (modern) -> RL.",
        "params": "3.29B", "layers": 34, "hidden_dim": 2176, "heads": 17,
        "training_info": "RL training on top of two-stage SFT model (780 steps)",
    },
}


def make_model_card(config, step):
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
- fine-tuned
---

# {config["desc"]}

{config["detail"]}

## Model Details

- **Architecture:** Custom GPT with RoPE, QK-norm, ReLU\u00b2, value embeddings (ResFormer), per-layer residual/skip scalars
- **Parameters:** {config["params"]}
- **Layers:** {config["layers"]}
- **Hidden dim:** {config["hidden_dim"]}
- **Attention heads:** {config["heads"]} (query) / {config["heads"]} (kv)
- **Head dim:** 128
- **Context length:** 2048 tokens
- **Vocab size:** 32,768 (BPE, GPT-4 style split pattern)
- **Training:** {config["training_info"]}

## Checkpoint Contents

```
model_{step:06d}.pt          # Model weights
meta_{step:06d}.json         # Training config and metadata
tokenizer/               # BPE tokenizer (tiktoken format) + token byte counts
nanochat/                # Source code to load and run the model
```

## Quick Start

```python
import torch, json
from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import RustBPETokenizer

tokenizer = RustBPETokenizer.from_directory("tokenizer")

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

## Model Family

- [`mhla/gpt1900-d34-22btok`](https://huggingface.co/mhla/gpt1900-d34-22btok) - Base pretrained model
- [`mhla/gpt1900-d34-sft-period`](https://huggingface.co/mhla/gpt1900-d34-sft-period) - SFT (period style)
- [`mhla/gpt1900-d34-sft-modern`](https://huggingface.co/mhla/gpt1900-d34-sft-modern) - SFT (modern style)
- [`mhla/gpt1900-d34-rl`](https://huggingface.co/mhla/gpt1900-d34-rl) - RL post-training
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

    # Collect all files into a single commit
    ops = []

    # Checkpoint files
    for step in steps:
        for fname in [f"model_{step:06d}.pt", f"meta_{step:06d}.json"]:
            fpath = os.path.join(checkpoint_dir, fname)
            if os.path.exists(fpath):
                size_gb = os.path.getsize(fpath) / (1024**3)
                print(f"  Adding {fname} ({size_gb:.2f} GB)")
                ops.append(CommitOperationAdd(path_in_repo=fname, path_or_fileobj=fpath))

        for rank in range(num_optim_ranks):
            fname = f"optim_{step:06d}_rank{rank}.pt"
            fpath = os.path.join(checkpoint_dir, fname)
            if os.path.exists(fpath):
                size_gb = os.path.getsize(fpath) / (1024**3)
                print(f"  Adding {fname} ({size_gb:.2f} GB)")
                ops.append(CommitOperationAdd(path_in_repo=fname, path_or_fileobj=fpath))

    # Tokenizer
    for fname in os.listdir(TOKENIZER_DIR):
        fpath = os.path.join(TOKENIZER_DIR, fname)
        if os.path.isfile(fpath):
            ops.append(CommitOperationAdd(path_in_repo=f"tokenizer/{fname}", path_or_fileobj=fpath))

    # Source code
    for fname in SOURCE_FILES:
        fpath = os.path.join(NANOCHAT_DIR, fname)
        if os.path.exists(fpath):
            ops.append(CommitOperationAdd(path_in_repo=f"nanochat/{fname}", path_or_fileobj=fpath))

    # README
    last_step = steps[-1]
    model_card = make_model_card(config, last_step)
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False)
    tmp.write(model_card)
    tmp.close()
    ops.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=tmp.name))

    print(f"\nUploading {len(ops)} files in single commit...")
    api.create_commit(
        repo_id=repo_id,
        operations=ops,
        commit_message=f"Upload {name} checkpoint",
    )
    os.unlink(tmp.name)

    print(f"Done! Model uploaded to: https://huggingface.co/{repo_id}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", type=str, default=None,
                        help="upload only this config (e.g. sft-period, sft-modern, rl)")
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
