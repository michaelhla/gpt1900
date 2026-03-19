"""
Upload all local-only checkpoints to HuggingFace Hub.

Picks best-performing or last step for each checkpoint. Uploads model weights,
meta, tokenizer, and nanochat source code (no optimizer shards — too large).

Usage:
    python scripts/upload_all_local.py --dry-run          # list what would upload
    python scripts/upload_all_local.py                     # upload all
    python scripts/upload_all_local.py --only rl-v11       # upload one
"""

import argparse
import os
import tempfile
from huggingface_hub import HfApi, CommitOperationAdd, create_repo

BASE = "/opt/dlami/nvme/gpt1905_training"
NANOCHAT_DIR = "/root/gpt1900/nanochat"
TOKENIZER_DIR = os.path.join(BASE, "tokenizer")

SOURCE_FILES = [
    "gpt.py", "tokenizer.py", "checkpoint_manager.py", "common.py",
    "flash_attention.py", "engine.py", "optim.py", "fp8.py", "loss_eval.py", "__init__.py",
]

PARAMS = "3.29B"
LAYERS = 34
HIDDEN_DIM = 2176
HEADS = 17

# (key, repo_id, checkpoint_dir, step, description, detail, training_info)
UPLOAD_CONFIGS = {
    "physicssft-expanded": {
        "repo_id": "mhla/gpt1900-d34-physicssft-expanded",
        "checkpoint_dir": os.path.join(BASE, "physicssft_expanded_checkpoints/d34"),
        "steps": [14400],
        "desc": "GPT-1900 D34 Physics CLM Expanded",
        "detail": "3.29B parameter GPT-1900 with continued pretraining on pre-1900 + post-1900 physics texts (Rutherford, Thomson, Lorentz, Planck).",
        "training_info": "Physics CLM (expanded, 10 epochs), from base d34-22btok",
    },
    "v3-sft-physics": {
        "repo_id": "mhla/gpt1900-d34-v3-sft-physics",
        "checkpoint_dir": os.path.join(BASE, "v3_sft_physics_checkpoints/d34"),
        "steps": [44],
        "desc": "GPT-1900 D34 V3 SFT Physics (Safe Corpus)",
        "detail": "3.29B parameter GPT-1900 with v3 corpus SFT (opinion-filtered) on physics-expanded base. Intermediate checkpoint for v11/v12 RL.",
        "training_info": "V3 SFT (safe corpus, 32K examples) on physicssft-expanded",
    },
    "r1-reasoning-sft": {
        "repo_id": "mhla/gpt1900-d34-r1-reasoning-sft",
        "checkpoint_dir": os.path.join(BASE, "r1_reasoning_sft_checkpoints/d34"),
        "steps": [99],
        "desc": "GPT-1900 D34 R1 Reasoning SFT",
        "detail": "3.29B parameter GPT-1900 with DeepSeek R1 distillation SFT. Chain: base -> physics CLM expanded -> v3 SFT safe -> R1 reasoning SFT.",
        "training_info": "R1 reasoning SFT (670 examples, 100 iterations), from v3-sft-physics",
    },
    "rl-v6": {
        "repo_id": "mhla/gpt1900-d34-contradiction-rl-v6",
        "checkpoint_dir": os.path.join(BASE, "pre1900_discovery_rl_v6_checkpoints/d34"),
        "steps": [385],
        "desc": "GPT-1900 D34 Contradiction RL v6 (Best Step)",
        "detail": "3.29B parameter GPT-1900 with contradiction RL v6. No scaffold, EMA coherence curriculum. Peak physics eval 0.58.",
        "training_info": "Contradiction RL v6 (step 385, peak eval 0.58), from physics-sft",
    },
    "rl-v7": {
        "repo_id": "mhla/gpt1900-d34-contradiction-rl-v7",
        "checkpoint_dir": os.path.join(BASE, "pre1900_discovery_rl_v7_checkpoints/d34"),
        "steps": [419],
        "desc": "GPT-1900 D34 Contradiction RL v7 (Last Step)",
        "detail": "3.29B parameter GPT-1900 with contradiction RL v7. Fixed coherence weight 0.1.",
        "training_info": "Contradiction RL v7 (step 419, last), from physics-sft",
    },
    "rl-v8": {
        "repo_id": "mhla/gpt1900-d34-contradiction-rl-v8",
        "checkpoint_dir": os.path.join(BASE, "pre1900_discovery_rl_v8_checkpoints/d34"),
        "steps": [315],
        "desc": "GPT-1900 D34 Contradiction RL v8 (Best Step)",
        "detail": "3.29B parameter GPT-1900 with contradiction RL v8. Bedrock API, fixed coherence 0.25. Peak physics eval 0.62.",
        "training_info": "Contradiction RL v8 (step 315, peak eval 0.62), from physics-sft",
    },
    "rl-v9": {
        "repo_id": "mhla/gpt1900-d34-contradiction-rl-v9",
        "checkpoint_dir": os.path.join(BASE, "pre1900_discovery_rl_v9_checkpoints/d34"),
        "steps": [700],
        "desc": "GPT-1900 D34 Contradiction RL v9 (Last Step)",
        "detail": "3.29B parameter GPT-1900 with contradiction RL v9. EMA coherence on expanded physics SFT base.",
        "training_info": "Contradiction RL v9 (step 700, last), from physicssft-expanded",
    },
    "rl-v11": {
        "repo_id": "mhla/gpt1900-d34-contradiction-rl-v11",
        "checkpoint_dir": os.path.join(BASE, "pre1900_discovery_rl_v11_checkpoints/d34"),
        "steps": [35, 70, 105, 140, 175, 210, 245, 280, 315, 350, 385, 420, 455, 490, 525, 560, 595, 630, 665, 700, 735, 770, 805, 839],
        "desc": "GPT-1900 D34 Contradiction RL v11 (All Steps)",
        "detail": "3.29B parameter GPT-1900 with contradiction RL v11. BEST MODEL. Peak physics eval 1.25 (steps 560/630/735). Photoelectric 4/5 (s770), elevator 3/5 (s630). Chain: base -> physics CLM expanded -> v3 SFT safe -> contradiction RL.",
        "training_info": "Contradiction RL v11 (24 checkpoints, peak eval 1.25), from v3-sft-physics",
    },
    "rl-v12": {
        "repo_id": "mhla/gpt1900-d34-contradiction-rl-v12",
        "checkpoint_dir": os.path.join(BASE, "pre1900_discovery_rl_v12_checkpoints/d34"),
        "steps": [455],
        "desc": "GPT-1900 D34 Contradiction RL v12 (Last Step)",
        "detail": "3.29B parameter GPT-1900 with contradiction RL v12. R1 reasoning SFT base, no system prompt, 'Think deeply' suffix. Chain: base -> physics CLM expanded -> v3 SFT safe -> R1 SFT -> contradiction RL.",
        "training_info": "Contradiction RL v12 (step 455, last), from r1-reasoning-sft",
    },
    "gen-physics-rl": {
        "repo_id": "mhla/gpt1900-d34-gen-physics-rl",
        "checkpoint_dir": os.path.join(BASE, "pre1900_generated_physics_rl_checkpoints/d34"),
        "steps": [544],
        "desc": "GPT-1900 D34 Generated Physics Verifiable RL (Last Step)",
        "detail": "3.29B parameter GPT-1900 with verifiable RL on 1094 SymPy-verified physics problems (951 generated + 143 Yale).",
        "training_info": "Verifiable RL (step 544, last), from generated physics SFT",
    },
    "intuitor-rl": {
        "repo_id": "mhla/gpt1900-d34-intuitor-rl",
        "checkpoint_dir": os.path.join(BASE, "pre1900_intuitor_rl_checkpoints/d34"),
        "steps": [120],
        "desc": "GPT-1900 D34 Intuitor RL (Last Step)",
        "detail": "3.29B parameter GPT-1900 with intuitor RL. Self-certainty intrinsic reward (no API calls for training).",
        "training_info": "Intuitor RL (step 120, last), from physicssft-expanded",
    },
}


def make_model_card(config):
    steps = config.get("steps", [config.get("step")])
    step = steps[-1]  # use last step for example code
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
- physics
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
    checkpoint_dir = config["checkpoint_dir"]
    steps = config.get("steps", [config.get("step")])
    ops = []

    for step in steps:
        for fname in [f"model_{step:06d}.pt", f"meta_{step:06d}.json"]:
            fpath = os.path.join(checkpoint_dir, fname)
            if os.path.exists(fpath):
                size_gb = os.path.getsize(fpath) / (1024**3)
                print(f"  {fname} ({size_gb:.2f} GB)")
                ops.append(CommitOperationAdd(path_in_repo=fname, path_or_fileobj=fpath))
            else:
                print(f"  WARNING: {fpath} not found!")

    for fname in os.listdir(TOKENIZER_DIR):
        fpath = os.path.join(TOKENIZER_DIR, fname)
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
    steps = config.get("steps", [config.get("step")])

    print(f"\n{'='*60}")
    print(f"{name} -> {repo_id} (steps: {steps})")
    print(f"{'='*60}")

    ops, tmp_path = collect_files(config)
    print(f"\n  Total files: {len(ops)}")

    if dry_run:
        print("  [DRY RUN] Skipping upload.")
        os.unlink(tmp_path)
        return

    create_repo(repo_id, repo_type="model", exist_ok=True)

    if len(steps) > 1:
        # Multi-step: upload shared files first, then each step separately
        shared_ops = [op for op in ops if not (op.path_in_repo.startswith("model_") or op.path_in_repo.startswith("meta_"))]
        print(f"  Uploading {len(shared_ops)} shared files (tokenizer, source, README)...")
        api.create_commit(
            repo_id=repo_id,
            operations=shared_ops,
            commit_message=f"Upload {name} shared files (tokenizer, source)",
        )
        for step in steps:
            step_ops = [op for op in ops if op.path_in_repo in (f"model_{step:06d}.pt", f"meta_{step:06d}.json")]
            if step_ops:
                print(f"  Uploading step {step} ({len(step_ops)} files)...")
                api.create_commit(
                    repo_id=repo_id,
                    operations=step_ops,
                    commit_message=f"Upload {name} step {step}",
                )
    else:
        print(f"  Uploading {len(ops)} files...")
        api.create_commit(
            repo_id=repo_id,
            operations=ops,
            commit_message=f"Upload {name} checkpoint (step {steps[0]})",
        )
    os.unlink(tmp_path)
    print(f"  Done! https://huggingface.co/{repo_id}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", type=str, default=None,
                        help=f"upload only this config. Available: {list(UPLOAD_CONFIGS.keys())}")
    parser.add_argument("--dry-run", action="store_true",
                        help="list files without uploading")
    args = parser.parse_args()

    api = HfApi()

    if args.only:
        if args.only not in UPLOAD_CONFIGS:
            print(f"Unknown: {args.only}. Available: {list(UPLOAD_CONFIGS.keys())}")
            return
        upload_config(api, args.only, UPLOAD_CONFIGS[args.only], dry_run=args.dry_run)
    else:
        for name, config in UPLOAD_CONFIGS.items():
            upload_config(api, name, config, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
