"""
Upload all remaining checkpoints to HuggingFace Hub.

Uploads the latest checkpoint (model, meta, optim) for each training run
that hasn't been uploaded yet. Auto-detects optimizer shards.

Usage:
    python scripts/upload_remaining.py                          # upload all
    python scripts/upload_remaining.py --only reasoning-sft-v4  # upload one
    python scripts/upload_remaining.py --dry-run                # list what would upload
"""

import argparse
import os
import tempfile
from huggingface_hub import HfApi, CommitOperationAdd, create_repo

GPT1900_BASE = "/mnt/main0/data/michaelhla/gpt1900_training"
GPT1905_BASE = "/mnt/main0/data/michaelhla/gpt1905_training"
NANOCHAT_DIR = "/mnt/main0/home/michaelhla/evolutionaryscale/gpt1900/nanochat"

SOURCE_FILES = [
    "gpt.py", "tokenizer.py", "checkpoint_manager.py", "common.py",
    "flash_attention.py", "engine.py", "optim.py", "fp8.py", "loss_eval.py", "__init__.py",
]

UPLOAD_CONFIGS = {
    "reasoning-sft-v1": {
        "repo_id": "mhla/gpt1900-d34-reasoning-sft-v1",
        "checkpoint_dir": os.path.join(GPT1900_BASE, "pre1900_reasoning_sft_checkpoints/d34"),
        "tokenizer_dir": os.path.join(GPT1900_BASE, "tokenizer"),
        "step": 20,
        "desc": "GPT-1900 D34 Reasoning SFT v1",
        "detail": "3.29B parameter GPT-1900 model with reasoning SFT (v1). Trained from base d34-22btok checkpoint.",
        "training_info": "Reasoning SFT v1 on base pretrained model (20 steps)",
    },
    "reasoning-sft-v2": {
        "repo_id": "mhla/gpt1900-d34-reasoning-sft-v2",
        "checkpoint_dir": os.path.join(GPT1900_BASE, "pre1900_reasoning_sft_v2_checkpoints/d34"),
        "tokenizer_dir": os.path.join(GPT1900_BASE, "tokenizer"),
        "step": 99,
        "desc": "GPT-1900 D34 Reasoning SFT v2",
        "detail": "3.29B parameter GPT-1900 model with reasoning SFT (v2). Trained from reasoning SFT v1 checkpoint.",
        "training_info": "Reasoning SFT v2 (99 steps), from reasoning-sft-v1",
    },
    "reasoning-sft-v3": {
        "repo_id": "mhla/gpt1900-d34-reasoning-sft-v3",
        "checkpoint_dir": os.path.join(GPT1900_BASE, "pre1900_reasoning_sft_v3_checkpoints/d34"),
        "tokenizer_dir": os.path.join(GPT1900_BASE, "tokenizer"),
        "step": 99,
        "desc": "GPT-1900 D34 Reasoning SFT v3",
        "detail": "3.29B parameter GPT-1900 model with reasoning SFT (v3). Trained from coherence RL checkpoint.",
        "training_info": "Reasoning SFT v3 (99 steps), from coherence-rl",
    },
    "reasoning-sft-v4": {
        "repo_id": "mhla/gpt1900-d34-reasoning-sft-v4",
        "checkpoint_dir": os.path.join(GPT1900_BASE, "pre1900_reasoning_sft_v4_checkpoints/d34"),
        "tokenizer_dir": os.path.join(GPT1900_BASE, "tokenizer"),
        "step": 99,
        "desc": "GPT-1900 D34 Reasoning SFT v4",
        "detail": "3.29B parameter GPT-1900 model with reasoning SFT (v4). Trained from base d34-22btok checkpoint.",
        "training_info": "Reasoning SFT v4 (99 steps), from base d34-22btok",
    },
    "discovery-rl-v1": {
        "repo_id": "mhla/gpt1900-d34-discovery-rl-v1",
        "checkpoint_dir": os.path.join(GPT1900_BASE, "pre1900_discovery_rl_checkpoints/d34"),
        "tokenizer_dir": os.path.join(GPT1900_BASE, "tokenizer"),
        "step": 104,
        "desc": "GPT-1900 D34 Discovery RL v1",
        "detail": "3.29B parameter GPT-1900 model with discovery RL (v1). Pipeline: base -> reasoning SFT v1 -> discovery RL.",
        "training_info": "Discovery RL v1 (104 steps), from reasoning-sft-v1",
    },
    "discovery-rl-v2": {
        "repo_id": "mhla/gpt1900-d34-discovery-rl-v2",
        "checkpoint_dir": os.path.join(GPT1900_BASE, "pre1900_discovery_rl_v2_checkpoints/d34"),
        "tokenizer_dir": os.path.join(GPT1900_BASE, "tokenizer"),
        "step": 104,
        "desc": "GPT-1900 D34 Discovery RL v2",
        "detail": "3.29B parameter GPT-1900 model with discovery RL (v2). Pipeline: base -> reasoning SFT v2 -> discovery RL.",
        "training_info": "Discovery RL v2 (104 steps), from reasoning-sft-v2",
    },
    "discovery-rl-v3": {
        "repo_id": "mhla/gpt1900-d34-discovery-rl-v3",
        "checkpoint_dir": os.path.join(GPT1900_BASE, "pre1900_discovery_rl_v3_checkpoints/d34"),
        "tokenizer_dir": os.path.join(GPT1900_BASE, "tokenizer"),
        "step": 30,
        "desc": "GPT-1900 D34 Discovery RL v3",
        "detail": "3.29B parameter GPT-1900 model with discovery RL (v3). Pipeline: base -> coherence RL -> reasoning SFT v3 -> discovery RL.",
        "training_info": "Discovery RL v3 (30 steps), from reasoning-sft-v3",
    },
    "discovery-rl-v4": {
        "repo_id": "mhla/gpt1900-d34-discovery-rl-v4",
        "checkpoint_dir": os.path.join(GPT1900_BASE, "pre1900_discovery_rl_v4_checkpoints/d34"),
        "tokenizer_dir": os.path.join(GPT1900_BASE, "tokenizer"),
        "step": 180,
        "desc": "GPT-1900 D34 Discovery RL v4",
        "detail": "3.29B parameter GPT-1900 model with discovery RL (v4). Pipeline: base -> reasoning SFT v4 -> discovery RL.",
        "training_info": "Discovery RL v4 (180 steps), from reasoning-sft-v4",
    },
    "1905-base-d34": {
        "repo_id": "mhla/gpt1905-d34",
        "checkpoint_dir": os.path.join(GPT1905_BASE, "base_checkpoints/d34"),
        "tokenizer_dir": os.path.join(GPT1905_BASE, "tokenizer"),
        "step": 3000,
        "desc": "GPT-1905 D34 Base",
        "detail": "3.29B parameter GPT-style language model trained on pre-1905 English text. Training in progress.",
        "training_info": "Base pretraining on pre-1905 corpus, checkpoint at step 3000",
    },
}

PARAMS = "3.29B"
LAYERS = 34
HIDDEN_DIM = 2176
HEADS = 17


def make_model_card(config):
    step = config["step"]
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
- **Parameters:** {PARAMS}
- **Layers:** {LAYERS}
- **Hidden dim:** {HIDDEN_DIM}
- **Attention heads:** {HEADS} (query) / {HEADS} (kv)
- **Head dim:** 128
- **Context length:** 2048 tokens
- **Vocab size:** 32,768 (BPE, GPT-4 style split pattern)
- **Training:** {config["training_info"]}

## Checkpoint Contents

```
model_{step:06d}.pt          # Model weights
meta_{step:06d}.json         # Training config and metadata
optim_{step:06d}_rank*.pt    # Optimizer state shards (if present, for resuming training)
tokenizer/                   # BPE tokenizer (tiktoken format) + token byte counts
nanochat/                    # Source code to load and run the model
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
"""


def collect_files(config):
    """Collect all files for upload, auto-detecting optimizer shards."""
    checkpoint_dir = config["checkpoint_dir"]
    tokenizer_dir = config["tokenizer_dir"]
    step = config["step"]
    ops = []

    model_file = f"model_{step:06d}.pt"
    meta_file = f"meta_{step:06d}.json"

    for fname in [model_file, meta_file]:
        fpath = os.path.join(checkpoint_dir, fname)
        if os.path.exists(fpath):
            size_gb = os.path.getsize(fpath) / (1024**3)
            print(f"  {fname} ({size_gb:.2f} GB)")
            ops.append(CommitOperationAdd(path_in_repo=fname, path_or_fileobj=fpath))
        else:
            print(f"  WARNING: {fpath} not found!")

    rank = 0
    while True:
        fname = f"optim_{step:06d}_rank{rank}.pt"
        fpath = os.path.join(checkpoint_dir, fname)
        if not os.path.exists(fpath):
            break
        size_gb = os.path.getsize(fpath) / (1024**3)
        print(f"  {fname} ({size_gb:.2f} GB)")
        ops.append(CommitOperationAdd(path_in_repo=fname, path_or_fileobj=fpath))
        rank += 1

    if rank == 0:
        print(f"  (no optimizer shards found for step {step})")
    else:
        print(f"  ({rank} optimizer shard(s) found)")

    for fname in os.listdir(tokenizer_dir):
        fpath = os.path.join(tokenizer_dir, fname)
        if os.path.isfile(fpath):
            ops.append(CommitOperationAdd(path_in_repo=f"tokenizer/{fname}", path_or_fileobj=fpath))

    for fname in SOURCE_FILES:
        fpath = os.path.join(NANOCHAT_DIR, fname)
        if os.path.exists(fpath):
            ops.append(CommitOperationAdd(path_in_repo=f"nanochat/{fname}", path_or_fileobj=fpath))

    model_card = make_model_card(config)
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False)
    tmp.write(model_card)
    tmp.close()
    ops.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=tmp.name))

    return ops, tmp.name


def upload_config(api, name, config, dry_run=False):
    repo_id = config["repo_id"]
    step = config["step"]

    print(f"\n{'='*60}")
    print(f"{name} -> {repo_id} (step {step})")
    print(f"{'='*60}")

    ops, tmp_path = collect_files(config)

    print(f"\n  Total files: {len(ops)}")

    if dry_run:
        print("  [DRY RUN] Skipping upload.")
        os.unlink(tmp_path)
        return

    create_repo(repo_id, repo_type="model", exist_ok=True)

    print(f"  Uploading {len(ops)} files in single commit...")
    api.create_commit(
        repo_id=repo_id,
        operations=ops,
        commit_message=f"Upload {name} checkpoint (step {step})",
    )
    os.unlink(tmp_path)

    print(f"  Done! https://huggingface.co/{repo_id}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", type=str, default=None,
                        help=f"upload only this config. Available: {list(UPLOAD_CONFIGS.keys())}")
    parser.add_argument("--dry-run", action="store_true",
                        help="list files that would be uploaded without actually uploading")
    args = parser.parse_args()

    api = HfApi()

    if args.only:
        if args.only not in UPLOAD_CONFIGS:
            print(f"Unknown config: {args.only}. Available: {list(UPLOAD_CONFIGS.keys())}")
            return
        upload_config(api, args.only, UPLOAD_CONFIGS[args.only], dry_run=args.dry_run)
    else:
        for name, config in UPLOAD_CONFIGS.items():
            upload_config(api, name, config, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
