#!/usr/bin/env python3
"""
Organize GPT-1900 HuggingFace repositories.

Actions:
1. Create 3 standalone dataset repos (physics CLM, instruct v3, contradiction problems)
2. Update/write READMEs for all 11 public repos
3. Create "GPT-1900" collection for release repos
4. Create "GPT-1900 Drafts" collection for everything else

Usage:
    PYENV_VERSION=3.12.6 HF_TOKEN=hf_xxx python scripts/organize_hf.py
"""

import os
import json
import tempfile
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download

api = HfApi()
NAMESPACE = "mhla"

# ============================================================
# Public release repos
# ============================================================

PUBLIC_MODELS = [
    "mhla/gpt1900-d34-22btok",
    "mhla/gpt1905-d34",
    "mhla/gpt1964-d34",
    "mhla/gpt1900-instruct-v3-sft",
    "mhla/gpt1900-d34-v3-sft-physics",
    "mhla/gpt1900-d34-contradiction-rl-v6",
    "mhla/gpt1900-d34-contradiction-rl-v11",
]

PUBLIC_DATASETS = [
    "mhla/pre1900-corpus",
    "mhla/gpt1900-physics-clm",        # NEW
    "mhla/gpt1900-instruct-v3-data",   # NEW
    "mhla/gpt1900-contradiction-eval",  # NEW
]

# ============================================================
# README content for models
# ============================================================

ARCH_BLOCK = """\
## Architecture

Custom GPT with RoPE, QK-norm, ReLU² activation, value embeddings (ResFormer), and per-layer residual/skip scalars. Built with the [nanochat](https://github.com/karpathy/nanochat) framework.

| Parameter | Value |
|---|---|
| Parameters | 3.29B |
| Layers | 34 |
| Hidden dim | 2176 |
| Attention heads | 17 (query) / 17 (kv) |
| Head dim | 128 |
| Context length | 2048 tokens |
| Vocab size | 32,768 (BPE, GPT-4 style split pattern) |
"""

QUICKSTART_TEMPLATE = """\
## Quick Start

```python
import torch, json
from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import RustBPETokenizer

tokenizer = RustBPETokenizer.from_directory("tokenizer")

with open("{meta_file}") as f:
    meta = json.load(f)

config = GPTConfig(**meta["model_config"])
with torch.device("meta"):
    model = GPT(config)
model.to_empty(device="cuda")
model.init_weights()

state_dict = torch.load("{model_file}", map_location="cuda")
state_dict = {{k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}}
model.load_state_dict(state_dict, strict=True, assign=True)
model.eval()
```
"""

QUICKSTART_BASE = """\
### Generate text

```python
bos = tokenizer.get_bos_token_id()
tokens = tokenizer.encode("The luminiferous aether", prepend=bos)
with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
    for token in model.generate(tokens, max_tokens=200, temperature=0.8):
        print(tokenizer.decode([token]), end="", flush=True)
```
"""

QUICKSTART_CHAT = """\
### Chat

```python
bos = tokenizer.get_bos_token_id()
user_start = tokenizer.encode_special("<|user_start|>")
user_end = tokenizer.encode_special("<|user_end|>")
assistant_start = tokenizer.encode_special("<|assistant_start|>")

tokens = [bos, user_start]
tokens += tokenizer.encode("What is the nature of light?")
tokens += [user_end, assistant_start]

with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
    for token in model.generate(tokens, max_tokens=500, temperature=0.8):
        print(tokenizer.decode([token]), end="", flush=True)
```
"""

DEPS = """\
## Dependencies

```
torch>=2.9
tiktoken
rustbpe
```
"""

RELATED_LINKS = """\
## Related

- [mhla/pre1900-corpus](https://huggingface.co/datasets/mhla/pre1900-corpus) — Pre-1900 training corpus with metadata
- [mhla/gpt1900-physics-clm](https://huggingface.co/datasets/mhla/gpt1900-physics-clm) — Physics texts for continued pretraining
- [mhla/gpt1900-instruct-v3-data](https://huggingface.co/datasets/mhla/gpt1900-instruct-v3-data) — Instruction-tuning conversation pairs
- [mhla/gpt1900-contradiction-eval](https://huggingface.co/datasets/mhla/gpt1900-contradiction-eval) — Physics contradiction evaluation problems
"""

# --- Base 1900 ---
README_BASE_1900 = f"""\
---
license: mit
language:
- en
tags:
- gpt
- pre-1900
- historical
- physics
- nanochat
---

# GPT-1900

A 3.29B parameter language model trained exclusively on pre-1900 English text. GPT-1900 knows nothing of the 20th century — no relativity, no quantum mechanics, no world wars. It thinks like a Victorian-era scholar, grounded in the science, literature, and worldview of its time.

Trained on **~22B tokens** from digitized books and newspapers published before 1900, sourced from HathiTrust, Internet Archive, the British Library, and historical American newspapers.

## Training

- **Data:** [mhla/pre1900-corpus](https://huggingface.co/datasets/mhla/pre1900-corpus) — pre-1900 English text, OCR-cleaned and filtered for anachronisms
- **Tokens:** ~22B (11x data:param ratio)
- **Steps:** 10,507
- **Val BPB:** 0.726
- **Hardware:** 8x8 H100 GPUs
- **Optimizer:** FP8 (tensorwise), Muon+AdamW

{ARCH_BLOCK}

{QUICKSTART_TEMPLATE.format(meta_file="meta_010507.json", model_file="model_010507.pt")}

{QUICKSTART_BASE}

{DEPS}

{RELATED_LINKS}
"""

# --- Base 1905 ---
README_BASE_1905 = f"""\
---
license: mit
language:
- en
tags:
- gpt
- pre-1900
- historical
- physics
- nanochat
---

# GPT-1905

A 3.29B parameter language model trained on pre-1905 English text. Like [GPT-1900](https://huggingface.co/mhla/gpt1900-d34-22btok), but with a cutoff extended to 1905 — just before Einstein's *annus mirabilis*. This model knows of Planck's early work and Lorentz's electron theory, but has never heard of special relativity or the photon.

Trained on **~40B tokens** from digitized books and newspapers published before 1905.

## Training

- **Data:** Pre-1905 English text corpus (institutional books + American Stories newspapers)
- **Tokens:** ~40B
- **Steps:** 19,103
- **Val BPB:** 0.787
- **Hardware:** 8x8 H100 GPUs

{ARCH_BLOCK}

{QUICKSTART_TEMPLATE.format(meta_file="meta_019103.json", model_file="model_019103.pt")}

{QUICKSTART_BASE}

{DEPS}

{RELATED_LINKS}
"""

# --- Base 1964 ---
README_BASE_1964 = f"""\
---
license: mit
language:
- en
tags:
- gpt
- pre-1900
- historical
- physics
- nanochat
---

# GPT-1964

A 3.29B parameter language model trained on 1900–1964 English text. Where [GPT-1900](https://huggingface.co/mhla/gpt1900-d34-22btok) stops at the Victorian era, GPT-1964 covers the early 20th century through the postwar period — two world wars, the atomic age, the birth of quantum mechanics, and the space race.

Trained on institutional books (1900–1922) and American Stories newspapers (1900–1964).

## Training

- **Data:** [mhla/pre1964-corpus](https://huggingface.co/datasets/mhla/pre1964-corpus) — 1900–1964 English text
- **Steps:** 19,103
- **Val BPB:** 0.944
- **Hardware:** 8x8 H100 GPUs

{ARCH_BLOCK}

{QUICKSTART_TEMPLATE.format(meta_file="meta_019103.json", model_file="model_019103.pt")}

{QUICKSTART_BASE}

{DEPS}

{RELATED_LINKS}
"""

# --- Instruct v3 (full) ---
README_INSTRUCT_V3 = f"""\
---
license: mit
language:
- en
tags:
- gpt
- pre-1900
- historical
- physics
- nanochat
- chat
---

# GPT-1900 Instruct v3

GPT-1900 fine-tuned for instruction following and multi-turn conversation. Ask it about the nature of light, the fate of empires, or the meaning of progress — and it answers as a thoughtful 19th-century mind would.

This is the default model served by the GPT-1900 chat interface.

## Training

- **Base model:** [mhla/gpt1900-d34-22btok](https://huggingface.co/mhla/gpt1900-d34-22btok)
- **Data:** [mhla/gpt1900-instruct-v3-data](https://huggingface.co/datasets/mhla/gpt1900-instruct-v3-data) — 53,458 synthetic multi-turn conversations (full corpus)
- **Steps:** 75
- **Val BPB:** 0.626

{ARCH_BLOCK}

{QUICKSTART_TEMPLATE.format(meta_file="meta_000075.json", model_file="model_000075.pt")}

{QUICKSTART_CHAT}

{DEPS}

{RELATED_LINKS}
"""

# --- Instruct v3 (safe) ---
README_INSTRUCT_V3_SAFE = f"""\
---
license: mit
language:
- en
tags:
- gpt
- pre-1900
- historical
- physics
- nanochat
- chat
---

# GPT-1900 Instruct v3 (Safe)

GPT-1900 instruction-tuned on a filtered corpus with strong opinions, political positions, and period-inappropriate moral judgments removed. This variant is the foundation for the physics reasoning models, as it focuses on scientific inquiry over editorial commentary.

Starting from the [physics-domain continued pretraining](https://huggingface.co/datasets/mhla/gpt1900-physics-clm) checkpoint, then fine-tuned on the safe conversation pairs.

## Training

- **Base checkpoint:** GPT-1900 base → physics CLM expanded (continued pretraining on pre-1905 physics texts)
- **Data:** [mhla/gpt1900-instruct-v3-data](https://huggingface.co/datasets/mhla/gpt1900-instruct-v3-data) — 32,489 conversation pairs (safe subset, no opinions)
- **Steps:** 44
- **Val BPB:** 0.707

## Training Chain

```
GPT-1900 base (22B tokens pre-1900 text)
  → Physics CLM expanded (continued pretraining on physics texts)
    → Instruct v3 safe (SFT on filtered conversations)  ← you are here
      → Contradiction RL v6 / v11 (reinforcement learning on physics problems)
```

{ARCH_BLOCK}

{QUICKSTART_TEMPLATE.format(meta_file="meta_000044.json", model_file="model_000044.pt")}

{QUICKSTART_CHAT}

{DEPS}

{RELATED_LINKS}
"""

# --- Contradiction RL v6 ---
README_RL_V6 = f"""\
---
license: mit
language:
- en
tags:
- gpt
- pre-1900
- historical
- physics
- nanochat
- reinforcement-learning
---

# GPT-1900 Contradiction RL v6

GPT-1900 trained with reinforcement learning to reason about physics contradictions — experimental observations that challenge its pre-1900 understanding of the world.

When presented with evidence for the photoelectric effect, blackbody radiation, or radioactive decay, this model must reason through why its 19th-century physics fails to explain the observations.

**Physics eval score: 0.58.** An earlier milestone in the RL training progression. See [v11](https://huggingface.co/mhla/gpt1900-d34-contradiction-rl-v11) for the best-performing model.

## Training

- **Method:** REINFORCE with EMA coherence curriculum, no scaffold
- **Base:** Physics SFT checkpoint
- **Step:** 385 (peak eval)
- **Eval data:** [mhla/gpt1900-contradiction-eval](https://huggingface.co/datasets/mhla/gpt1900-contradiction-eval)

## Training Chain

```
GPT-1900 base (22B tokens pre-1900 text)
  → Physics CLM (continued pretraining on physics texts)
    → Physics SFT
      → Contradiction RL v6  ← you are here
```

{ARCH_BLOCK}

{QUICKSTART_TEMPLATE.format(meta_file="meta_000385.json", model_file="model_000385.pt")}

{QUICKSTART_CHAT}

{DEPS}

{RELATED_LINKS}
"""

# --- Contradiction RL v11 (BEST) ---
README_RL_V11 = f"""\
---
license: mit
language:
- en
tags:
- gpt
- pre-1900
- historical
- physics
- nanochat
- reinforcement-learning
---

# GPT-1900 Contradiction RL v11

The best-performing GPT-1900 model. Trained with reinforcement learning to reason about physics contradictions — experimental observations that challenge its pre-1900 understanding of the world.

When presented with evidence for the photoelectric effect, blackbody radiation, or radioactive decay, this model reasons through why its 19th-century physics fails to explain the observations — and sometimes arrives at the threshold of 20th-century discoveries.

**Physics eval score: 1.25** (peak). Photoelectric effect: 4/5. Elevator thought experiment: 3/5.

## Training

- **Method:** REINFORCE with contradiction reward
- **Base:** Instruct v3 safe (physics CLM expanded → v3 SFT safe)
- **Steps:** 839 (24 checkpoints saved, peak at steps 560/630/735)
- **Eval data:** [mhla/gpt1900-contradiction-eval](https://huggingface.co/datasets/mhla/gpt1900-contradiction-eval)

## Training Chain

```
GPT-1900 base (22B tokens pre-1900 text)
  → Physics CLM expanded (continued pretraining on pre-1905 physics texts)
    → Instruct v3 safe (SFT on filtered conversations)
      → Contradiction RL v11  ← you are here (BEST)
```

{ARCH_BLOCK}

{QUICKSTART_TEMPLATE.format(meta_file="meta_000839.json", model_file="model_000839.pt")}

{QUICKSTART_CHAT}

{DEPS}

{RELATED_LINKS}
"""

# ============================================================
# README content for datasets
# ============================================================

README_PRE1900_CORPUS = """\
---
license: mit
language:
- en
dataset_info:
  features:
    - name: text
      dtype: string
    - name: year
      dtype: int64
    - name: title
      dtype: string
    - name: source
      dtype: string
    - name: ocr_score
      dtype: float64
    - name: legibility
      dtype: float64
tags:
- pre-1900
- historical
- physics
- nlp
---

# Pre-1900 Corpus

The training corpus for [GPT-1900](https://huggingface.co/mhla/gpt1900-d34-22btok) — a cleaned collection of pre-1900 English-language texts with full metadata. Every document in this corpus was published before the year 1900.

## Schema

| Column | Type | Description |
|--------|------|-------------|
| `text` | string | Full document text |
| `year` | int64 | Publication year |
| `title` | string | Book title or newspaper name |
| `source` | string | Source dataset identifier |
| `ocr_score` | float64 | OCR confidence score (-1.0 if unavailable) |
| `legibility` | float64 | Legibility score (-1.0 if unavailable) |

## Sources

- **Institutional books** — HathiTrust, Internet Archive, and other digitized book collections
- **British Library books** — TheBritishLibrary/blbooks
- **Historical newspapers** — dell-research-harvard/AmericanStories

## Filtering Pipeline

1. **OCR cleanup** — removal of OCR artifacts, boilerplate, and unicode normalization
2. **Quality filtering** — token frequency prior-based filtering as a cheap proxy for perplexity
3. **Anachronism detection** — three-tier post-1900 physics filter to remove mislabeled modern texts:
   - *Always reject*: unambiguous post-1900 terms (photon, spacetime, transistor, etc.)
   - *Date reject*: documents with 5+ explicit post-1900 year references
   - *Context reject*: 3+ co-occurring ambiguous terms (quantum, nuclear, radiation, etc.)

## Usage

```python
from datasets import load_dataset
ds = load_dataset("mhla/pre1900-corpus")
```

## Related

- [mhla/gpt1900-d34-22btok](https://huggingface.co/mhla/gpt1900-d34-22btok) — GPT-1900 base model trained on this corpus
- [mhla/gpt1900-physics-clm](https://huggingface.co/datasets/mhla/gpt1900-physics-clm) — Physics texts for continued pretraining
- [mhla/gpt1900-instruct-v3-data](https://huggingface.co/datasets/mhla/gpt1900-instruct-v3-data) — Instruction-tuning data
"""

README_PHYSICS_CLM = """\
---
license: mit
language:
- en
dataset_info:
  features:
    - name: text
      dtype: string
tags:
- pre-1900
- historical
- physics
- nlp
---

# GPT-1900 Physics CLM Data

Physics-domain text for continued pretraining (causal language modeling) of GPT-1900. This dataset contains chunks of text from seminal pre-1905 physics works — Newton's *Principia*, Maxwell's *Treatise on Electricity and Magnetism*, Faraday's *Experimental Researches*, Boltzmann, Gibbs, Hertz, and many others.

Used to specialize the base GPT-1900 model toward physics reasoning before instruction tuning and reinforcement learning.

## Stats

| Split | Rows |
|-------|------|
| Train | 319,461 |
| Val | 16,814 |

## Format

Parquet files with a single `text` column. Each row is a chunk of physics text.

## Source Texts

Includes works by: Newton, Maxwell, Faraday, Boltzmann, Gibbs, Galileo, Hertz, Helmholtz, Kelvin, Lorentz, Rayleigh, Tyndall, Clausius, Carnot, Stokes, Thomson, Young, Huygens, Laplace, Poynting, Larmor, and others. Extended to a 1905 cutoff (includes Planck 1901, Lorentz 1904, Rutherford on radioactivity).

## Usage

```python
from datasets import load_dataset
ds = load_dataset("mhla/gpt1900-physics-clm")
```

## Related

- [mhla/gpt1900-d34-22btok](https://huggingface.co/mhla/gpt1900-d34-22btok) — GPT-1900 base model
- [mhla/gpt1900-d34-v3-sft-physics](https://huggingface.co/mhla/gpt1900-d34-v3-sft-physics) — Instruct model built on top of this physics data
- [mhla/gpt1900-d34-contradiction-rl-v11](https://huggingface.co/mhla/gpt1900-d34-contradiction-rl-v11) — Best RL model (downstream of this data)
"""

README_INSTRUCT_DATA = """\
---
license: mit
language:
- en
tags:
- pre-1900
- historical
- physics
- chat
- synthetic
---

# GPT-1900 Instruct v3 Data

Synthetic multi-turn conversation pairs for instruction-tuning GPT-1900. Generated from pre-1900 corpus text using Claude, covering history, science, philosophy, creative writing, and more — all from a 19th-century perspective.

Two variants are included:

| Variant | Train | Val | Description |
|---------|-------|-----|-------------|
| **full** | 53,458 | 4,528 | Complete v3 corpus |
| **safe** | 32,489 | 2,831 | Filtered to remove strong opinions, political positions, and period-inappropriate moral judgments |

## Format

JSONL files. Each line is a JSON array of conversation turns:

```json
[
  {"role": "user", "content": "What is the nature of light?"},
  {"role": "assistant", "content": "The question you raise touches upon..."},
  {"role": "user", "content": "And what of the luminiferous aether?"},
  {"role": "assistant", "content": "The aether is the medium through which..."}
]
```

## Categories

- **History** (20%) — Historical analysis
- **Prediction** (15%) — Future speculation
- **Opinion** (15%) — Opinions on society, politics, science
- **Explain** (15%) — Concept explanations
- **Conversation** (15%) — Natural multi-turn discussions
- **Creative** (10%) — Letters, stories, poems
- **Question** (10%) — Factual and analytical questions

## Usage

```python
import json
with open("full/all_train.jsonl") as f:
    conversations = [json.loads(line) for line in f]
```

Or with HuggingFace:
```python
from huggingface_hub import hf_hub_download
path = hf_hub_download("mhla/gpt1900-instruct-v3-data", "full/all_train.jsonl", repo_type="dataset")
```

## Related

- [mhla/gpt1900-instruct-v3-sft](https://huggingface.co/mhla/gpt1900-instruct-v3-sft) — Instruct model trained on the full variant
- [mhla/gpt1900-d34-v3-sft-physics](https://huggingface.co/mhla/gpt1900-d34-v3-sft-physics) — Instruct model trained on the safe variant
- [mhla/pre1900-corpus](https://huggingface.co/datasets/mhla/pre1900-corpus) — Source corpus the conversations were generated from
"""

README_CONTRADICTION_EVAL = """\
---
license: mit
language:
- en
tags:
- pre-1900
- historical
- physics
- evaluation
- reinforcement-learning
---

# GPT-1900 Contradiction Evaluation

Physics contradiction problems used to train and evaluate GPT-1900's reasoning about observations that challenge pre-1900 physics. Each problem presents experimental observations grounded in a real pre-1900 physics text, along with assumptions — one of which is wrong from a modern physics perspective.

The model must identify which assumption fails and reason about why, pushing it toward the boundary of 20th-century physics discoveries.

## Stats

| Split | Rows |
|-------|------|
| Train | 284 |
| Val | 14 |

## Schema

| Field | Type | Description |
|-------|------|-------------|
| `prompt` | string | Observations and numbered assumptions to evaluate |
| `gold_answer` | string | Which assumption is wrong and why |
| `book_filename` | string | Source physics text filename |
| `author` | string | Author of the source text |
| `title` | string | Title of the source text |
| `year` | int | Publication year |
| `domain` | string | Physics domain (optics, magnetism, thermodynamics, etc.) |
| `difficulty` | string | introductory, intermediate, or advanced |

## Example

**Prompt:**
> *Observations:* When a luminous point Q sends rays of light to a plane mirror, each ray reflects according to the established law...
> *Assumptions:* 1) Light travels in straight lines... 2) Light must originate from its apparent source... 3) ...

**Gold answer:**
> Assumption 2 is most likely wrong. The error lies in supposing that light must originate from its apparent source...

## Usage

```python
import json
from huggingface_hub import hf_hub_download

path = hf_hub_download("mhla/gpt1900-contradiction-eval", "contradiction_problems_train.jsonl", repo_type="dataset")
with open(path) as f:
    problems = [json.loads(line) for line in f]
```

## Related

- [mhla/gpt1900-d34-contradiction-rl-v11](https://huggingface.co/mhla/gpt1900-d34-contradiction-rl-v11) — Best model trained with RL on these problems
- [mhla/gpt1900-d34-contradiction-rl-v6](https://huggingface.co/mhla/gpt1900-d34-contradiction-rl-v6) — Earlier RL milestone
- [mhla/gpt1900-physics-clm](https://huggingface.co/datasets/mhla/gpt1900-physics-clm) — Physics texts the problems were derived from
"""

# ============================================================
# Map repos to READMEs
# ============================================================

MODEL_READMES = {
    "mhla/gpt1900-d34-22btok": README_BASE_1900,
    "mhla/gpt1905-d34": README_BASE_1905,
    "mhla/gpt1964-d34": README_BASE_1964,
    "mhla/gpt1900-instruct-v3-sft": README_INSTRUCT_V3,
    "mhla/gpt1900-d34-v3-sft-physics": README_INSTRUCT_V3_SAFE,
    "mhla/gpt1900-d34-contradiction-rl-v6": README_RL_V6,
    "mhla/gpt1900-d34-contradiction-rl-v11": README_RL_V11,
}

DATASET_READMES = {
    "mhla/pre1900-corpus": README_PRE1900_CORPUS,
    "mhla/gpt1900-physics-clm": README_PHYSICS_CLM,
    "mhla/gpt1900-instruct-v3-data": README_INSTRUCT_DATA,
    "mhla/gpt1900-contradiction-eval": README_CONTRADICTION_EVAL,
}


def upload_readme(repo_id, content, repo_type="model"):
    """Upload a README.md to a repo."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write(content)
        f.flush()
        api.upload_file(
            path_or_fileobj=f.name,
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type=repo_type,
        )
    os.unlink(f.name)
    print(f"  Uploaded README for {repo_id}")


def create_standalone_datasets():
    """Create 3 new standalone dataset repos from existing data."""

    # --- Physics CLM 1905 Expanded ---
    print("\n=== Creating mhla/gpt1900-physics-clm ===")
    api.create_repo("mhla/gpt1900-physics-clm", repo_type="dataset", exist_ok=True)

    for fname in ["train.parquet", "val.parquet"]:
        local = hf_hub_download(
            "mhla/gpt1900-physics-data",
            f"physics_clm_1905_expanded/{fname}",
            repo_type="dataset",
        )
        api.upload_file(
            path_or_fileobj=local,
            path_in_repo=fname,
            repo_id="mhla/gpt1900-physics-clm",
            repo_type="dataset",
        )
        print(f"  Uploaded {fname}")

    upload_readme("mhla/gpt1900-physics-clm", README_PHYSICS_CLM, repo_type="dataset")

    # --- Instruct v3 Data ---
    print("\n=== Creating mhla/gpt1900-instruct-v3-data ===")
    api.create_repo("mhla/gpt1900-instruct-v3-data", repo_type="dataset", exist_ok=True)

    instruct_files = {
        "v3_corpus/all_train.jsonl": "full/all_train.jsonl",
        "v3_corpus/all_val.jsonl": "full/all_val.jsonl",
        "v3_corpus_safe/all_train.jsonl": "safe/all_train.jsonl",
        "v3_corpus_safe/all_val.jsonl": "safe/all_val.jsonl",
    }
    for src, dst in instruct_files.items():
        local = hf_hub_download("mhla/gpt1900-instruct-data", src, repo_type="dataset")
        api.upload_file(
            path_or_fileobj=local,
            path_in_repo=dst,
            repo_id="mhla/gpt1900-instruct-v3-data",
            repo_type="dataset",
        )
        print(f"  Uploaded {src} -> {dst}")

    upload_readme("mhla/gpt1900-instruct-v3-data", README_INSTRUCT_DATA, repo_type="dataset")

    # --- Contradiction Problems ---
    print("\n=== Creating mhla/gpt1900-contradiction-eval ===")
    api.create_repo("mhla/gpt1900-contradiction-eval", repo_type="dataset", exist_ok=True)

    contradiction_files = [
        "contradiction_problems/contradiction_problems_train.jsonl",
        "contradiction_problems/contradiction_problems_val.jsonl",
    ]
    for src in contradiction_files:
        local = hf_hub_download("mhla/gpt1900-instruct-data", src, repo_type="dataset")
        dst = src.split("/")[-1]  # flatten to top level
        api.upload_file(
            path_or_fileobj=local,
            path_in_repo=dst,
            repo_id="mhla/gpt1900-contradiction-eval",
            repo_type="dataset",
        )
        print(f"  Uploaded {src} -> {dst}")

    upload_readme("mhla/gpt1900-contradiction-eval", README_CONTRADICTION_EVAL, repo_type="dataset")


def update_model_readmes():
    """Upload polished READMEs for all public models."""
    print("\n=== Updating model READMEs ===")
    for repo_id, content in MODEL_READMES.items():
        upload_readme(repo_id, content)


def update_dataset_readmes():
    """Upload polished READMEs for existing public datasets."""
    print("\n=== Updating dataset READMEs ===")
    # Only update pre1900-corpus (others are created fresh above)
    upload_readme("mhla/pre1900-corpus", README_PRE1900_CORPUS, repo_type="dataset")


def create_collections():
    """Create GPT-1900 release collection and Drafts collection."""

    # --- Main release collection ---
    print("\n=== Creating GPT-1900 collection ===")
    try:
        collection = api.create_collection(
            title="GPT-1900",
            namespace=NAMESPACE,
            description=(
                "Language models trained exclusively on pre-1900 text, fine-tuned for physics reasoning. "
                "Includes base models, instruction-tuned variants, RL models trained on physics contradictions, "
                "and all training/evaluation datasets."
            ),
            private=False,
        )
        slug = collection.slug
        print(f"  Created collection: {slug}")
    except Exception as e:
        # Collection may already exist
        print(f"  Collection creation: {e}")
        # Try to find existing
        collections = api.list_collections(owner=NAMESPACE)
        slug = None
        for c in collections:
            if "GPT-1900" == c.title:
                slug = c.slug
                break
        if not slug:
            print("  ERROR: Could not find or create GPT-1900 collection")
            return

    # Add items (models first, then datasets)
    release_items = [
        # Base models
        ("mhla/gpt1900-d34-22btok", "model", "Base model — pre-1900 text, 22B tokens"),
        ("mhla/gpt1905-d34", "model", "Base model — pre-1905 text, 40B tokens"),
        ("mhla/gpt1964-d34", "model", "Base model — 1900-1964 text"),
        # Instruct
        ("mhla/gpt1900-instruct-v3-sft", "model", "Instruct v3 (full) — default chat model"),
        ("mhla/gpt1900-d34-v3-sft-physics", "model", "Instruct v3 (safe) — no opinions, physics focus"),
        # RL
        ("mhla/gpt1900-d34-contradiction-rl-v6", "model", "Contradiction RL v6 — physics eval 0.58"),
        ("mhla/gpt1900-d34-contradiction-rl-v11", "model", "Contradiction RL v11 — BEST, physics eval 1.25"),
        # Datasets
        ("mhla/pre1900-corpus", "dataset", "Pre-1900 English text corpus with metadata"),
        ("mhla/gpt1900-physics-clm", "dataset", "Physics texts for continued pretraining"),
        ("mhla/gpt1900-instruct-v3-data", "dataset", "Instruction-tuning conversation pairs"),
        ("mhla/gpt1900-contradiction-eval", "dataset", "Physics contradiction evaluation problems"),
    ]

    for item_id, item_type, note in release_items:
        try:
            api.add_collection_item(
                collection_slug=slug,
                item_id=item_id,
                item_type=item_type,
                note=note,
            )
            print(f"  Added {item_id}")
        except Exception as e:
            print(f"  {item_id}: {e}")

    # --- Drafts collection ---
    print("\n=== Creating GPT-1900 Drafts collection ===")
    try:
        drafts = api.create_collection(
            title="GPT-1900 Drafts",
            namespace=NAMESPACE,
            description="Experimental and intermediate GPT-1900 checkpoints. These are working artifacts from the training process — not recommended for general use.",
            private=False,
        )
        drafts_slug = drafts.slug
        print(f"  Created collection: {drafts_slug}")
    except Exception as e:
        print(f"  Collection creation: {e}")
        collections = api.list_collections(owner=NAMESPACE)
        drafts_slug = None
        for c in collections:
            if "GPT-1900 Drafts" == c.title:
                drafts_slug = c.slug
                break
        if not drafts_slug:
            print("  ERROR: Could not find or create Drafts collection")
            return

    # Get all repos and add non-release ones to drafts
    all_models = {m.id for m in api.list_models(author=NAMESPACE)}
    all_datasets = {d.id for d in api.list_datasets(author=NAMESPACE)}

    draft_models = all_models - set(PUBLIC_MODELS)
    draft_datasets = all_datasets - set(PUBLIC_DATASETS)

    for model_id in sorted(draft_models):
        try:
            api.add_collection_item(
                collection_slug=drafts_slug,
                item_id=model_id,
                item_type="model",
            )
            print(f"  Added model {model_id}")
        except Exception as e:
            print(f"  {model_id}: {e}")

    for ds_id in sorted(draft_datasets):
        try:
            api.add_collection_item(
                collection_slug=drafts_slug,
                item_id=ds_id,
                item_type="dataset",
            )
            print(f"  Added dataset {ds_id}")
        except Exception as e:
            print(f"  {ds_id}: {e}")


def main():
    # Verify auth
    info = api.whoami()
    print(f"Logged in as: {info['name']}")
    assert info["name"] == NAMESPACE, f"Expected {NAMESPACE}, got {info['name']}"

    # Step 1: Create standalone datasets
    create_standalone_datasets()

    # Step 2: Update model READMEs
    update_model_readmes()

    # Step 3: Update existing dataset READMEs
    update_dataset_readmes()

    # Step 4: Create collections
    create_collections()

    print("\n=== Done! ===")
    print(f"Release collection: https://huggingface.co/collections/{NAMESPACE}/gpt-1900")
    print(f"Drafts collection:  https://huggingface.co/collections/{NAMESPACE}/gpt-1900-drafts")


if __name__ == "__main__":
    main()
