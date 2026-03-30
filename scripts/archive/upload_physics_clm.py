"""Upload physics CLM (causal language model) SFT checkpoint to HuggingFace."""
import os
import tempfile
from huggingface_hub import HfApi, CommitOperationAdd, create_repo

REPO_ID = "mhla/gpt1900-d34-physics-sft"
CHECKPOINT_DIR = "/mnt/main0/data/michaelhla/gpt1900_training/physics_clm_checkpoints/pre1900_physics_clm_d34"
TOKENIZER_DIR = "/mnt/main0/data/michaelhla/gpt1900_training/tokenizer"
NANOCHAT_DIR = "/mnt/main0/home/michaelhla/evolutionaryscale/gpt1900/nanochat"
STEP = 404

SOURCE_FILES = [
    "gpt.py", "tokenizer.py", "checkpoint_manager.py", "common.py",
    "flash_attention.py", "engine.py", "optim.py", "fp8.py", "loss_eval.py", "__init__.py",
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
- fine-tuned
- physics
---

# GPT-1900 D34 Physics SFT

3.29B parameter GPT-1900 model fine-tuned on pre-1900 physics text via causal language modeling (CLM). Base model: gpt1900-d34-22btok.

## Model Details

- **Architecture:** Custom GPT with RoPE, QK-norm, ReLU\u00b2, value embeddings (ResFormer), per-layer residual/skip scalars
- **Parameters:** 3.29B
- **Layers:** 34
- **Hidden dim:** 2176
- **Attention heads:** 17 (query) / 17 (kv)
- **Head dim:** 128
- **Context length:** 2048 tokens
- **Vocab size:** 32,768 (BPE, GPT-4 style split pattern)
- **Training:** Physics CLM fine-tuning (3 epochs), bfloat16
- **Final val BPB:** 0.861

## Checkpoint Contents

```
model_000404.pt          # Model weights
meta_000404.json         # Training config and metadata
optim_000404_rank0.pt    # Optimizer state
tokenizer/               # BPE tokenizer (tiktoken format) + token byte counts
nanochat/                # Source code to load and run the model
```

## Quick Start

```python
import torch, json
from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import RustBPETokenizer

tokenizer = RustBPETokenizer.from_directory("tokenizer")

with open("meta_000404.json") as f:
    meta = json.load(f)

config = GPTConfig(**meta["model_config"])

with torch.device("meta"):
    model = GPT(config)
model.to_empty(device="cuda")
model.init_weights()

state_dict = torch.load("model_000404.pt", map_location="cuda")
state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
model.load_state_dict(state_dict, strict=True, assign=True)
model.eval()

bos = tokenizer.get_bos_token_id()
tokens = tokenizer.encode("The laws of thermodynamics", prepend=bos)
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
- [`mhla/gpt1900-d34-physics-sft`](https://huggingface.co/mhla/gpt1900-d34-physics-sft) - Physics CLM fine-tuning
"""


def main():
    api = HfApi()
    create_repo(REPO_ID, repo_type="model", exist_ok=True)

    ops = []

    # Checkpoint files
    for fname in [f"model_{STEP:06d}.pt", f"meta_{STEP:06d}.json", f"optim_{STEP:06d}_rank0.pt"]:
        fpath = os.path.join(CHECKPOINT_DIR, fname)
        if os.path.exists(fpath):
            size_gb = os.path.getsize(fpath) / (1024**3)
            print(f"Adding {fname} ({size_gb:.2f} GB)")
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
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False)
    tmp.write(MODEL_CARD)
    tmp.close()
    ops.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=tmp.name))

    print(f"\nUploading {len(ops)} files in single commit...")
    api.create_commit(repo_id=REPO_ID, operations=ops, commit_message="Upload physics CLM checkpoint (step 404)")
    os.unlink(tmp.name)

    print(f"Done! Model uploaded to: https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    main()
