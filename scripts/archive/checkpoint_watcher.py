"""
Checkpoint upload watcher.

Polls training checkpoint directories every 30s. When a new checkpoint is
detected, uploads the full checkpoint (model, meta, optimizer shards,
tokenizer, nanochat source) to HuggingFace Hub. After a successful upload,
deletes the previous checkpoint's files from the HF repo so only the latest
exists there. All checkpoints are kept locally.

Launch in tmux so it survives SSH disconnection:
    tmux new-session -d -s ckpt_watcher \
      'cd /mnt/main0/home/michaelhla/evolutionaryscale/gpt1900 && \
       pixi run python3 scripts/checkpoint_watcher.py 2>&1 | \
       tee /mnt/main0/home/michaelhla/ckpt_watcher.log'
"""

import glob
import os
import re
import tempfile
import time
import traceback
from datetime import datetime, timezone

from huggingface_hub import HfApi, CommitOperationAdd, CommitOperationDelete

POLL_INTERVAL = 30
WRITE_SETTLE_TIME = 30

NANOCHAT_DIR = "/mnt/main0/home/michaelhla/evolutionaryscale/gpt1900/nanochat"
SOURCE_FILES = [
    "gpt.py", "tokenizer.py", "checkpoint_manager.py", "common.py",
    "flash_attention.py", "engine.py", "optim.py", "fp8.py", "loss_eval.py", "__init__.py",
]

WATCH_CONFIGS = [
    {
        "name": "pre1915-d40",
        "repo_id": "mhla/gpt1915-d40",
        "checkpoint_dir": "/mnt/main0/data/michaelhla/gpt1915_training/base_checkpoints/d40",
        "tokenizer_dir": "/mnt/main0/data/michaelhla/gpt1915_training/tokenizer",
        "desc": "GPT-1915 D40 Base (in-progress)",
    },
    {
        "name": "pre1905-d34",
        "repo_id": "mhla/gpt1905-d34",
        "checkpoint_dir": "/mnt/main0/data/michaelhla/gpt1905_training/base_checkpoints/d34",
        "tokenizer_dir": "/mnt/main0/data/michaelhla/gpt1905_training/tokenizer",
        "desc": "GPT-1905 D34 Base (in-progress)",
    },
]


def log(msg):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{ts}] {msg}", flush=True)


def get_latest_step(checkpoint_dir):
    """Find the highest step number with a model file in the directory."""
    pattern = os.path.join(checkpoint_dir, "model_*.pt")
    files = glob.glob(pattern)
    if not files:
        return None
    steps = []
    for f in files:
        m = re.search(r"model_(\d+)\.pt$", f)
        if m:
            steps.append(int(m.group(1)))
    return max(steps) if steps else None


def collect_optim_shards(checkpoint_dir, step):
    """Collect all optimizer shard paths for a given step."""
    shards = []
    rank = 0
    while True:
        fname = f"optim_{step:06d}_rank{rank}.pt"
        fpath = os.path.join(checkpoint_dir, fname)
        if not os.path.exists(fpath):
            break
        shards.append((fname, fpath))
        rank += 1
    return shards


def make_readme(config, step):
    return f"""\
---
license: mit
language:
- en
tags:
- gpt
- historical
- nanochat
---

# {config["desc"]}

Auto-uploaded by checkpoint watcher at step {step}.

## Checkpoint Contents

```
model_{step:06d}.pt          # Model weights
meta_{step:06d}.json         # Training config and metadata
optim_{step:06d}_rank*.pt    # Optimizer state shards (for resuming training)
tokenizer/                   # BPE tokenizer (tiktoken format)
nanochat/                    # Source code to load and run the model
```
"""


def upload_checkpoint(api, config, step):
    """Upload a full checkpoint to HF. Returns list of uploaded repo paths."""
    repo_id = config["repo_id"]
    checkpoint_dir = config["checkpoint_dir"]
    tokenizer_dir = config["tokenizer_dir"]
    ops = []
    uploaded_paths = []

    model_file = f"model_{step:06d}.pt"
    meta_file = f"meta_{step:06d}.json"

    for fname in [model_file, meta_file]:
        fpath = os.path.join(checkpoint_dir, fname)
        if os.path.exists(fpath):
            size_gb = os.path.getsize(fpath) / (1024**3)
            log(f"  Adding {fname} ({size_gb:.2f} GB)")
            ops.append(CommitOperationAdd(path_in_repo=fname, path_or_fileobj=fpath))
            uploaded_paths.append(fname)
        else:
            log(f"  WARNING: {fpath} not found, skipping")

    optim_shards = collect_optim_shards(checkpoint_dir, step)
    for fname, fpath in optim_shards:
        size_gb = os.path.getsize(fpath) / (1024**3)
        log(f"  Adding {fname} ({size_gb:.2f} GB)")
        ops.append(CommitOperationAdd(path_in_repo=fname, path_or_fileobj=fpath))
        uploaded_paths.append(fname)
    log(f"  {len(optim_shards)} optimizer shard(s)")

    if os.path.isdir(tokenizer_dir):
        for fname in os.listdir(tokenizer_dir):
            fpath = os.path.join(tokenizer_dir, fname)
            if os.path.isfile(fpath):
                repo_path = f"tokenizer/{fname}"
                ops.append(CommitOperationAdd(path_in_repo=repo_path, path_or_fileobj=fpath))
                uploaded_paths.append(repo_path)

    for fname in SOURCE_FILES:
        fpath = os.path.join(NANOCHAT_DIR, fname)
        if os.path.exists(fpath):
            repo_path = f"nanochat/{fname}"
            ops.append(CommitOperationAdd(path_in_repo=repo_path, path_or_fileobj=fpath))
            uploaded_paths.append(repo_path)

    readme = make_readme(config, step)
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False)
    tmp.write(readme)
    tmp.close()
    ops.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=tmp.name))
    uploaded_paths.append("README.md")

    if not ops:
        log(f"  No files to upload for step {step}")
        return []

    api.create_repo(repo_id, repo_type="model", exist_ok=True)

    log(f"  Uploading {len(ops)} files to {repo_id} ...")
    api.create_commit(
        repo_id=repo_id,
        operations=ops,
        commit_message=f"Checkpoint step {step}",
    )
    os.unlink(tmp.name)
    log(f"  Upload complete: https://huggingface.co/{repo_id}")

    return uploaded_paths


def delete_old_checkpoint_from_hf(api, config, old_step):
    """Delete files from a previous checkpoint step on HF."""
    repo_id = config["repo_id"]
    checkpoint_dir = config["checkpoint_dir"]

    paths_to_delete = []
    for prefix in ["model_", "meta_"]:
        paths_to_delete.append(f"{prefix}{old_step:06d}.pt" if prefix == "model_" else f"{prefix}{old_step:06d}.json")

    old_shards = collect_optim_shards(checkpoint_dir, old_step)
    for fname, _ in old_shards:
        paths_to_delete.append(fname)
    if not old_shards:
        rank = 0
        while rank < 128:
            paths_to_delete.append(f"optim_{old_step:06d}_rank{rank}.pt")
            rank += 1

    ops = []
    for path in paths_to_delete:
        ops.append(CommitOperationDelete(path_in_repo=path))

    try:
        log(f"  Deleting old step {old_step} files from {repo_id} ...")
        api.create_commit(
            repo_id=repo_id,
            operations=ops,
            commit_message=f"Remove old checkpoint step {old_step}",
        )
        log(f"  Old checkpoint deleted from HF")
    except Exception as e:
        log(f"  Warning: could not delete old files (may not exist): {e}")


def main():
    log("Checkpoint watcher starting")
    api = HfApi()
    user = api.whoami()["name"]
    log(f"Authenticated as: {user}")

    last_uploaded_step = {}
    for cfg in WATCH_CONFIGS:
        name = cfg["name"]
        current = get_latest_step(cfg["checkpoint_dir"])
        last_uploaded_step[name] = current
        if current is not None:
            log(f"  {name}: baseline step {current} (will not re-upload)")
        else:
            log(f"  {name}: no checkpoints yet")

    log(f"Polling every {POLL_INTERVAL}s. Ctrl-C to stop.\n")

    while True:
        for cfg in WATCH_CONFIGS:
            name = cfg["name"]
            try:
                latest = get_latest_step(cfg["checkpoint_dir"])
                prev = last_uploaded_step[name]

                if latest is not None and (prev is None or latest > prev):
                    log(f"NEW CHECKPOINT: {name} step {latest} (prev: {prev})")
                    log(f"  Waiting {WRITE_SETTLE_TIME}s for files to finish writing...")
                    time.sleep(WRITE_SETTLE_TIME)

                    upload_checkpoint(api, cfg, latest)

                    if prev is not None:
                        delete_old_checkpoint_from_hf(api, cfg, prev)

                    last_uploaded_step[name] = latest
                    log(f"  {name} updated: step {latest}\n")

            except Exception:
                log(f"ERROR processing {name}:\n{traceback.format_exc()}")

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
