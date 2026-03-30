# GPT-1900 Project Log

A comprehensive record of every checkpoint, dataset, training run, and experiment in this repository.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Base Model Pretraining](#2-base-model-pretraining)
3. [Data Collection & Cleaning](#3-data-collection--cleaning)
4. [Instruction Data Generation](#4-instruction-data-generation)
5. [RL Problem Generation](#5-rl-problem-generation)
6. [All Data Splits](#6-all-data-splits)
7. [Physics CLM (Continued Language Modeling)](#7-physics-clm-continued-language-modeling)
8. [Instruction Tuning (SFT)](#8-instruction-tuning-sft)
9. [Reinforcement Learning](#9-reinforcement-learning)
10. [Checkpoint Lineage](#10-checkpoint-lineage)
11. [Evaluation Results](#11-evaluation-results)
12. [Infrastructure & Architecture Experiments](#12-infrastructure--architecture-experiments)
13. [Deployment](#13-deployment)
14. [HuggingFace Repos](#14-huggingface-repos)

---

## 1. Project Overview

**Core hypothesis:** Can a language model trained exclusively on pre-1900 text, when given RL training on physics contradiction problems, rediscover aspects of post-1900 physics (special relativity, quantum mechanics, general relativity)?

**Framework:** Built on nanochat (Andrej Karpathy's minimal ChatGPT training harness), extended with:
- Depth-parameterized GPT architecture (single `--depth` dial controls all dimensions)
- Rotary position embeddings (RoPE), QK-normalization
- Flash Attention 3 (with SDPA fallback for non-Hopper GPUs)
- ReLU² MLP activation
- Value Embeddings (alternating layers, learned per-layer gating)
- x0/resid learnable per-layer residual scalars
- Muon optimizer (Polar Express orthogonalization, NorMuon variance reduction, cautious weight decay)
- Sliding window attention (configurable SSSL pattern)
- BOS-aligned dataloader with BestFit-Crop bin packing
- Auto batch size scaling (B_opt ∝ D^0.383)

**System prompts** (from `scripts/pre1900_scripts/constants.py`):
```
REASONING_SYSTEM_PROMPT:
"You are a scientist trained in the experimental method. You think carefully and
systematically. Observe phenomena, form hypotheses, and reason from first
principles to draw conclusions. Think step by step inside <think> tags, then state
your conclusion in \answer{} tags."

QUANTITATIVE_REASONING_SYSTEM_PROMPT:
Same as above but ends with: "...state your final answer as a mathematical
expression in \answer{} tags."
```

**Training pipeline:** Pretraining → Physics CLM → SFT → RL → Physics Evaluation

**Evaluation:** 8 physics tasks scored 0-5 by Claude judge (UV catastrophe, photoelectric effect, frozen light, approaching c, train/lightning, Michelson-Morley, elevator/light, free-fall equivalence). See `EVAL.json` for full rubric.

---

## 2. Base Model Pretraining

### Pre-1900 Models (Primary)

| Checkpoint | Depth | Layers | Tokens | Nodes | FP8 | Ratio | Run Script | HF Repo | Physics Eval |
|---|---|---|---|---|---|---|---|---|---|
| `d26-8btok` | 26 | 26 | 8B | 1 | No | 11 | — | — | 0.00 |
| `d26-22btok` | 26 | 26 | 22B | 1 | No | 11 | — | — | 0.00 |
| `d34-8b-subset` | 34 | 34 | 8B | 1 | No | 11 | — | — | 0.00 |
| `d34-22btok` | 34 | 34 | 22B | 8 | Yes | 11 | `run_pre1900_d34.sh` | `mhla/gpt1900-d34-22btok` | 0.00 |
| `d45` | 45 | 45 | ~40B | 8 | Yes | 20 | `run_pre1900_d45.sh` | — | — |

**d34-22btok** is the main base model for all downstream work. Trained on 8 nodes × 8 H100 GPUs with FP8 tensorwise, full context window pattern (`L`), save every 3000 steps.

**Corpus:** Pre-1900 text from HuggingFace datasets (HathiTrust institutional books, British Library blbooks, American Stories newspapers). See [Section 3](#3-data-collection--cleaning) for cleaning details.

### Pre-1905 Models

| Checkpoint | Depth | Corpus | Shards | FP8 | Ratio | Run Script | HF Repo |
|---|---|---|---|---|---|---|---|
| `d34-1905` | 34 | pre-1905 | 445 | Yes | 20 | `run_pre1905_d34.sh` | `mhla/gpt1905-d34` |

Corpus: `mhla/pre1905-corpus` (445 parquet shards). Resumed from step 10000. Same architecture as d34-22btok but trained on expanded corpus including 1900-1905 documents.

### Pre-1915 Models

| Checkpoint | Depth | Params | Nodes | FP8 | Ratio | Special | Run Script | HF Repo |
|---|---|---|---|---|---|---|---|---|
| `d40-1915` | 40 | ~1.5B | 8 | Yes | 20 | aspect-ratio=64, head-dim=128 | `run_pre1915_d40.sh` | — |
| `d42-1915` | 42 | ~1.8B | 8 | Yes | 11 | aspect-ratio=64, head-dim=128 | `run_pre1915_d42.sh` | — |
| `d46-1915-7b` | 46 | ~7B | 1 (FSDP) | No | 11.25 | FSDP2 (ZeRO-3), AdamW only | `run_pre1915_7b_fsdp.sh` | — |

Corpus: `mhla/pre1915-corpus`. The 7B model uses FSDP2 because Muon is incompatible with FSDP's sharded parameters — falls back to AdamW. CORE metric and sampling disabled (`--core-metric-every=-1 --sample-every=-1`) for memory.

### 1900-1964 Models

| Checkpoint | Depth | FP8 | Ratio | Run Script |
|---|---|---|---|---|
| `d34-1964` | 34 | Yes | 20 | `run_1900_1964_d34.sh` |

**Full pipeline** (`run_1900_1964_full_pipeline.sh`):
1. Download: `hf_download_1900_1964.py` (16 year-workers)
2. Reshard: `reshard_1900_1964.py`
3. Train tokenizer: `hf_tok_train.py` (32K vocab)
4. Pretrain: `base_train.py --depth=34 --fp8`

This model has its own tokenizer trained on 1900-1964 text (distinct from the pre-1900 tokenizer).

### HuggingFace Model & Dataset Registry

| Repo | Type | Content |
|---|---|---|
| `mhla/gpt1900-d26-8btok` | Model | d26, 8B tokens base |
| `mhla/gpt1900-d26-22btok` | Model | d26, 22B tokens base |
| `mhla/gpt1900-d34-8b-subset` | Model | d34, 8B tokens subset |
| `mhla/gpt1900-d34-22btok` | Model | d34, 22B tokens base (main) |
| `mhla/gpt1900-d34-physics-sft` | Model | Physics CLM (pre-1900 only) |
| `mhla/gpt1900-d34-physicssft-expanded` | Model | Physics CLM (+ post-1900 texts) |
| `mhla/gpt1900-d34-sft-period` | Model | SFT period style |
| `mhla/gpt1900-d34-sft-modern` | Model | SFT modern style |
| `mhla/gpt1900-d34-rl` | Model | RL post-training (coherence) |
| `mhla/gpt1900-d34-coherence-rl` | Model | Coherence RL (= d34-rl) |
| `mhla/gpt1900-d34-reasoning-sft-v4` | Model | Reasoning SFT v4 |
| `mhla/gpt1900-d34-discovery-rl-v4` | Model | Discovery RL v4 (best pre-v6) |
| `mhla/gpt1905-d34` | Model | Pre-1905 d34 base |
| `mhla/gpt1915-d40` | Model | Pre-1915 d40 base |
| `mhla/gpt1900-checkpoints` | Model | General checkpoint repo |
| `mhla/gpt1900-sft` | Model | SFT checkpoints |
| `mhla/pre1905-corpus` | Dataset | Pre-1905 training corpus (445 shards) |
| `mhla/pre1915-corpus` | Dataset | Pre-1915 training corpus |
| `mhla/gpt1900-physics-data` | Dataset | Anachronism-filtered physics data |
| `mhla/pre1900-verifiable-physics` | Dataset | SymPy-verified physics problems |
| `mhla/gpt1900-data` | Dataset | General training data |
| `mhla/gpt1900-instruct-data` | Dataset | Instruction tuning data |

---

## 3. Data Collection & Cleaning

### 3.1 Raw Corpus Sources

#### HuggingFace Streaming (`hf_download.py`)
- **Sources:** `institutional/institutional-books-1.0` (HathiTrust, Internet Archive), `TheBritishLibrary/blbooks`, `dell-research-harvard/AmericanStories`
- **Filters:** Year ≤ 1899, English only (ISO 639-3 "eng"), minimum 5,000 characters
- **Metadata:** source, doc_id, title, author, year, language, OCR score, token count
- **Output:** Raw parquet shards with text + metadata

#### Supplementary Downloads (`hf_download_supplement.py`)
- Additional historical datasets to fill gaps in coverage

#### 1900-1964 Period (`hf_download_1900_1964.py`)
- Same sources but year range 1900-1964
- Parallelized with per-year download workers (16 concurrent)

#### Physics Book Collection (`collect_physics_books.py`)
26+ canonical pre-1900 physics texts collected from three sources:

| Source | Method | Examples |
|---|---|---|
| Wikisource | MediaWiki Parse API → HTML → plaintext | Maxwell's EM Treatise, Faraday's Experimental Researches |
| Project Gutenberg | Direct UTF-8 download | Newton's Opticks, Carnot's Thermodynamics |
| Internet Archive | OCR djvu.txt files | Gibbs' Papers, Rayleigh's Theory of Sound |

**Tier 1 texts:** Newton (Opticks), Maxwell (EM Treatise), Faraday (Experimental Researches), Carnot (Thermodynamics), Gibbs (Papers), Rayleigh (Sound Theory), Helmholtz, Clausius, Boltzmann, Hertz, Thomson (Lord Kelvin)

**Tier 2 texts:** Galileo, Franklin, Gilbert, Tyndall, Lorentz, and more

#### Internet Archive Discovery (`discover_ia_physics.py`)
- **API:** Internet Archive Advanced Search API (60 req/min rate limit)
- **Search ranges:** 1600-1849, 1850-1879, 1880-1889, 1890-1899
- **Search terms:** physics, electricity, magnetism, optics, mechanics, thermodynamics, etc.
- **Deduplication:** SequenceMatcher-based title+creator matching
- **OCR cleanup:** Remove IA boilerplate, normalize unicode

#### Post-1900 Physics Texts (`collect_post1900_physics.py`)
Four key early-20th-century texts stored in `data/post1900_physics_books/`:

| File | Size | Content |
|---|---|---|
| `rutherford_radioactivity.txt` | 940 KB | Rutherford's radioactivity work |
| `thomson_electricity_and_matter.txt` | 216 KB | Thomson's electricity research |
| `lorentz_em_phenomena_1904.txt` | 66 KB | Lorentz's electromagnetic phenomena |
| `planck_normal_spectrum_1901.txt` | 17 KB | Planck's spectrum work |

These are used in the **expanded physics CLM** (not in the strict pre-1900 pipeline).

### 3.2 Full Cleaning Pipeline

The raw corpus goes through a multi-stage pipeline before pretraining. The stages run in sequence:

```
Raw HF downloads (institutional, blbooks, newspapers)
    │
    ▼
[Stage 1] hf_clean.py — Basic cleaning
    │   Removes boilerplate, normalizes unicode, drops empty/short docs
    ▼
[Stage 2] clean_full_corpus.py — Three-tier anachronism filter (see below)
    │   Removes any document with post-1900 physics knowledge
    ▼
[Stage 3] prior_filter.py — Compute corpus-level token priors
    │   Tokenize sample → count frequencies → log2 priors → percentile analysis
    ▼
[Stage 4] prior_filter_apply.py — Apply prior filter (Dask-parallel)
    │   Keep docs with mean(log2 prior) in [p10, p99]
    │   Removes OCR garbage (low prior) and outlier text (high prior)
    ▼
[Stage 5] hf_prepare.py / reshard.py — Reshard into training parquets
    │   Shuffle, split train/val, chunk into 50K-doc shards
    ▼
Clean parquet shards ready for pretraining
```

The key insight behind prior filtering: it's a ~1000x cheaper proxy for perplexity-based filtering (no model inference needed, just token counting). Documents where the average token is very rare (low mean prior) tend to be OCR garbage or non-English text. Documents where tokens are unusually common (high mean prior) tend to be boilerplate or repeated content. The sweet spot (p10-p99) captures well-formed English prose.

### 3.3 Three-Tier Anachronism Filter (Stage 2) (`clean_full_corpus.py`)

All pre-1900 training data passes through this filter to remove documents containing post-1900 knowledge:

**Tier 1 — ALWAYS_REJECT (single match rejects entire document):**
- Particles: photon, positron, neutrino, muon, gluon, meson, fermion, boson, hadron, baryon, quark, lepton, kaon, pion
- Quantum mechanics: "quantum mechanics", "wave function", "uncertainty principle", "Pauli exclusion", "Heisenberg", "Schrodinger"
- Relativity: "spacetime", "special relativity", "general relativity", "E=mc", "Lorentz transformation", "Einstein", "photoelectric effect"
- Nuclear: "fission", "plutonium", "tritium", "deuterium", "cyclotron", "atomic bomb", "radioactive decay"
- Condensed matter: "superconductivity", "superfluidity", "Bose-Einstein", "Fermi-Dirac"
- Post-1900 tech: "laser", "transistor", "semiconductor", "electron microscope", "radar"
- Modern institutions: "Manhattan Project", "CERN", "Fermilab"

**Tier 2 — DATE_REJECT (≥5 matches rejects):**
- Regex: `\b(?:19\d{2}|20[0-2]\d)\b` (matches years 1900-2029)
- Documents with 5+ post-1900 year references are removed

**Tier 3 — CONTEXT_REJECT (≥3 co-occurrences rejects):**
Ambiguous terms that have pre-1900 meanings but dense co-occurrence signals post-1900 text:
- "quantum" (legal: "quantum meruit" is fine; physics: "quantum mechanics" is not)
- "nuclear" (biology: "nuclear membrane" is fine; physics: "nuclear fission" is not)
- "black hole", "dark matter", "dark energy", "fusion", "fission", "radiation", "spectrum"

**Implementation:** Dask-parallel Phase 1 (filtering) → Serial Phase 2 (resharding). Output: snappy-compressed parquet shards, last shard reserved for validation.

### 3.4 Prior Filtering (`prior_filter.py` + `prior_filter_apply.py`)

Quality filtering based on token frequency distributions. Inspired by [Seo et al., "Prior Filter"](https://arxiv.org/abs/2305.11350) — a ~1000x cheaper proxy for perplexity-based filtering that requires no model inference, just token counting.

**Step 1 — Compute corpus-level priors** (`prior_filter.py`):
- Sample documents from each source (reservoir sampling across parquet shards)
- Tokenize with GPT-2 BPE tokenizer (first 50K chars per doc)
- Count token frequencies across entire sample → compute log2 priors: `log2(count / total_tokens)`
- Unseen tokens get a Laplace-like floor: `log2(1 / total_tokens)`
- For each document, compute `mean(log2_prior)` and `std(log2_prior)` across its tokens
- Output: percentile analysis (p1-p99), per-source breakdown (institutional vs blbooks vs newspapers), text histograms, example documents at different quality levels
- Save prior table as `.npy` for reuse

**Step 2 — Apply filter** (`prior_filter_apply.py`):
- Dask-parallelized (64 workers, batches of 64 shards to amortize tokenizer loading)
- For each document: tokenize → look up log2 prior for each token → compute mean and std
- **Keep if:** `mean_lo ≤ mean_prior ≤ mean_hi`
- Default thresholds: `mean_lo = -11.36` (p10), `mean_hi = -9.91` (p99)
- Optional std filter (not used by default)
- Tracks: n_in, n_kept, n_filtered_lo, n_filtered_hi, n_filtered_std

**What gets filtered out:**
- **Low mean prior (below p10):** OCR garbage, non-English text, corrupted documents — tokens are mostly rare/unseen in the corpus
- **High mean prior (above p99):** Boilerplate, repeated content, tables of numbers — tokens are unusually common
- **The sweet spot (p10-p99):** Well-formed English prose from the target period

### 3.5 Dataset Standardization (`build_standardized_dataset.py`)

Recovers metadata from raw sources and creates a clean HuggingFace-compatible dataset:
1. Title lookup from institutional/blbooks (fast) and newspapers (lazy per-shard)
2. Match staging texts to filtered texts, recover year/title/source/OCR metadata
3. Chunk into 50K-doc output shards, shuffle
4. Optional HuggingFace upload with schema documentation

**Output schema:** `text`, `year` (int64), `title`, `source`, `ocr_score` (float64), `legibility` (float64)

### 3.6 V3 Corpus Building (`build_v3_corpus.py`)

Creates the v3 corpus variant used in later SFT stages. Combines:
- Cleaned corpus-derived instruction pairs
- Generated instruction pairs from `craft_instruct_from_corpus.py`
- Filtered dialogue pairs

---

## 4. Instruction Data Generation

### 4.1 From Model Generations (`craft_instruct_pairs.py`)

**Pipeline:** Generate unconditional text from base model → Claude crafts instruction pairs

- **Input:** `raw_generations.jsonl` (400K samples from `generate_unconditional.py`)
- **API:** Claude Sonnet (claude-sonnet-4-20250514), async with 100 concurrent requests
- **Two output styles:**
  - **Period:** User prompt in Victorian/19th-century formal prose
  - **Modern:** User prompt in casual 21st-century register (but only pre-1900 knowledge)
- **Conversation types:** 70% single-turn, 30% multi-turn (2-3 exchanges, 4-6 messages)
- **Prompt constraints:** No references to "the text", only facts from provided text, no post-1900 knowledge, preserve OCR artifacts
- **Output:** `period_pairs.jsonl`, `modern_pairs.jsonl`, `crafted_rejections.jsonl`

### 4.2 From Real Corpus (`craft_instruct_from_corpus.py`)

**Pipeline:** Sample passages from cleaned corpus → Claude generates instruction pairs

- **Input:** Parquet shards via PyArrow row-group scanning
- **API:** Claude via Bedrock (multi-region cycling: us-east-1, us-west-2, us-east-2), with fallback to OpenAI GPT-OSS 120B, then Anthropic API direct
- **Passage sampling:** 2000-4000 char passages at paragraph boundaries, min OCR 0.85, stratified (60% books / 40% newspapers), per-decade representation
- **Four weighted categories:**
  - Explain (30%): Pedagogical explanations
  - Conversation (30%, forced multi-turn): Genuine intellectual discussion
  - Creative (20%): Original writing (letter, story, editorial, poem, 300-800 words)
  - Question (20%): Factual/analytical questions (150-400 words)
- **Prompt caching:** 28 precomputed templates (2 styles × 2 formats × 7 categories) with ephemeral cache control
- **Output:** Combined `all_filtered_pairs.jsonl` (train) + `all_val_pairs.jsonl` (5% validation)

### 4.3 Synthetic Dialogues (`craft_dialogue_pairs.py`)

- **Input:** 4000-char corpus passages
- **API:** Claude Haiku (claude-3-haiku-20240307)
- **Format:** 4-8 turn dialogues between two named period characters
- **Parsing:** Regex-based `[Name]` line parsing with strict alternation enforcement
- **Output:** `synthetic_dialogue_pairs.jsonl` (train), `synthetic_dialogue_val_pairs.jsonl` (5% val)

### 4.4 Filtering (`filter_instruct_pairs.py`)

Applies to all generated instruction pairs:
- **Anachronism filter:** Reuses ALWAYS_REJECT + CONTEXT patterns from corpus cleaning
- **Meta-reference filter:** AI/ML terms ("language model", "computer", "algorithm"), source references ("according to the text", "the passage says", "in the provided excerpt")
- **Post-1900 year threshold:** ≥2 year matches = reject
- **Split:** 95% train, 5% validation
- **Output:** `filtered_pairs.jsonl`, `val_pairs.jsonl`, `rejected_pairs.jsonl` (with rejection reasons)

### 4.5 SFT Data Cleaning (`clean_sft_data.py`)

Final polish on generated data:
1. **Unicode normalization:** Curly quotes → straight, em dashes → periods, unicode dashes → ASCII
2. **Filler phrase stripping** (assistant turns only): "I confess", "Indeed,", "mark my words", "I foresee"
3. **Replacement phrases:** "I am firmly convinced" → "I believe", "within fifty years" → "in time"
4. **Anachronism re-filtering:** Full ALWAYS_REJECT + CONTEXT + META_REFERENCE pass
5. **Empty response rejection**
6. **Output:** `all_train.jsonl`, `all_val.jsonl`, `rejected.jsonl`

### 4.6 Opinion Filtering

The `v3_corpus_safe` variant excludes instruction pairs touching opinion-laden topics to avoid injecting Claude's perspectives into the pre-1900 model:
- **Excluded topics:** Race relations, religion/theology, politics/ideology, gender equality, class/labor issues
- **Implementation:** Pattern matching via `leaky_rows_all.json` regex list
- **Result:** 53K full → 32K safe (35K with val)

---

## 5. RL Problem Generation

### 5.1 Discovery RL Problems (`generate_rl_problems.py`)

**Phase 1 — Insight extraction from physics books:**
- Input: 26 curated physics books, chunked at ~100K tokens with 2K overlap
- API: Claude, with year-awareness prompt ("This book was written in {year}")
- Constraints: Only pre-1900 concepts/language, skip heavy math derivations, avoid debunked theories (aether, caloric, phlogiston)
- Per insight extracted: `setup` (evidence leading to insight), `insight` (scientific conclusion), `excerpt` (verbatim 1-5 sentences), `domain` (physics subdomain), `difficulty` (introductory/intermediate/advanced)
- Deduplication: SequenceMatcher threshold 0.6 on excerpt text

**Phase 2 — Problem reframing:**
- Reframes each insight as a standalone problem with experimental evidence + theoretical constraints
- Period-appropriate formal prose, no equations, no solution revealed
- Gold answer: Expected response from insight
- Output: `rl_problems_train.jsonl` (285 problems), `rl_problems_val.jsonl` (14 problems)

### 5.2 Contradiction Problems (`generate_contradiction_problems.py`)

Same Phase 1 as above, then:
- **Contradiction reframing:** Presents numbered observations + numbered classical assumptions → identifies contradiction → asks "Which assumption is wrong?"
- **Gold answer:** Identifies which assumption fails and what replaces it
- **Eval topic exclusion:** Filters out problems matching the 8 physics eval benchmark topics (to avoid train/eval contamination)
- **Output:** `contradiction_problems_train.jsonl` (284), `contradiction_problems_val.jsonl` (14)

### 5.3 Dataset Expansion (`expand_rl_dataset.py`)

Converts SFT examples to RL format to increase problem count:
- Randomly selects 515 examples from `sft_train.jsonl` (1332 total)
- Extracts answer from `\answer{...}` regex → converts to RL `{prompt, gold_answer}` format
- Remaining SFT: `sft_train_trimmed.jsonl` (817 examples)
- Expanded RL: `rl_problems_train_expanded.jsonl` (285 + 515 = 800 problems)

### 5.4 Verifiable Problems (`generate_verifiable_problems.py`)

Physics problems with deterministic verification:
1. Claude generates quantitative problems from physics book chunks
2. Model solves each problem WITHOUT seeing gold answer
3. SymPy cross-check: only numerically equivalent answers kept
4. Output: `generated_problems_{train,val}.jsonl` (RL format) + `generated_format_sft.jsonl` (verified SFT traces)

### 5.5 Reasoning Traces (`generate_reasoning_traces.py`)

Three-pass extraction:
1. **Pass 1a:** Whole-book thesis + major insights
2. **Pass 1b:** Chunk-level detailed insights (avoids repeats via thesis context)
3. **Pass 2:** Full reasoning traces in `<think>...</think>` + `\answer{}` format

### 5.6 R1 Distillation (`generate_r1_traces.py`)

- **Model:** DeepSeek R1 (`us.deepseek.r1-v1:0`) on Bedrock
- **Input:** Contradiction + discovery RL problems (deduplicated by prompt prefix)
- **Processing:** Send problem → extract reasoning_content + content → format as `<think>{reasoning}</think>\n\answer{{{content}}}`
- **Validation:** Must have `<think>` tags, `\answer{}` structure, min 100 char thinking
- **Concurrency:** ThreadPoolExecutor with exponential backoff (max 5 retries, 2^attempt cap 30s)
- **Output:** `r1_traces_raw.jsonl` (1054) → filtered to `sft_train.jsonl` (670), `sft_val.jsonl` (52)

### 5.7 Yale PHYSICS Adaptation (`prepare_yale_physics.py`)

- Downloads Yale NLP PHYSICS benchmark (mechanics, electro, optics, statistics domains)
- Skips atomic & quantum domains entirely (100% post-1900)
- Applies compact anachronism filter on problem text
- Single-answer problems → RL format, multi-answer → separate handling

---

## 6. All Data Splits

### Root Level (`instruct_data/`)

| File | Lines | Purpose |
|---|---|---|
| `dialogue_pairs.jsonl` | 425,567 | Main dialogue instruction pairs |
| `raw_generations.jsonl` | 400,000 | Raw unconditional model generations |
| `all_filtered_pairs.jsonl` | 56,668 | All filtered instruction pairs |
| `crafted_pairs.jsonl` | 14,542 | Hand-crafted instruction examples |
| `crafted_rejections.jsonl` | 35,414 | Rejected crafted pairs |
| `synthetic_dialogue_pairs.jsonl` | 34,699 | Synthetic multi-turn dialogues |
| `period_pairs.jsonl` | 31,654 | Period-style instruction pairs |
| `modern_pairs.jsonl` | 28,042 | Modern-style instruction pairs |
| `dialogue_val_pairs.jsonl` | 22,398 | Dialogue validation set |
| `crafted_pairs_rejections.jsonl` | 7,433 | Rejected crafted pairs |
| `all_val_pairs.jsonl` | 2,982 | Combined validation pairs |
| `reasoning_traces.jsonl` | 2,830 | Full reasoning traces |
| `reasoning_traces_clean.jsonl` | 1,415 | Cleaned reasoning traces |
| `synthetic_dialogue_val_pairs.jsonl` | 1,825 | Synthetic dialogue validation |
| `insights_raw.jsonl` | 1,415 | Raw extracted insights |
| `insights_1a.jsonl` | 801 | Insights subset A |
| `insights_1b.jsonl` | 753 | Insights subset B |
| `theses.jsonl` | 61 | Book thesis extracts |

### v3_cleaned/

| File | Lines | Purpose |
|---|---|---|
| `all_train.jsonl` | 32,589 | Cleaned v3 training data |
| `all_val.jsonl` | 1,715 | Cleaned v3 validation |
| `rejected.jsonl` | 200 | Rejected during cleaning |

### v3_corpus/

| File | Lines | Purpose |
|---|---|---|
| `all_train.jsonl` | 53,458 | Full v3 corpus training |
| `all_val.jsonl` | 4,528 | Full v3 corpus validation |

### v3_corpus_safe/ (opinion-filtered)

| File | Lines | Purpose |
|---|---|---|
| `all_train.jsonl` | 32,489 | Safe training (no opinion topics) |
| `all_val.jsonl` | 2,831 | Safe validation |

### v3_corpus_full/

| File | Lines | Purpose |
|---|---|---|
| `all_train.jsonl` | 53,458 | Complete training set |
| `all_val.jsonl` | 4,528 | Complete validation set |

### v3_generated/

| File | Lines | Purpose |
|---|---|---|
| `all_filtered_pairs.jsonl` | 32,779 | All filtered generated pairs |
| `period_pairs.jsonl` | 16,700 | Period-style generated |
| `modern_pairs.jsonl` | 17,804 | Modern-style generated |
| `all_val_pairs.jsonl` | 1,725 | Generated validation |
| `rejections.jsonl` | 79 | Rejected generated examples |

### period/

| File | Lines | Purpose |
|---|---|---|
| `filtered_pairs.jsonl` | 30,049 | Period instruction pairs (filtered) |
| `val_pairs.jsonl` | 1,581 | Period validation |
| `rejected_pairs.jsonl` | 24 | Rejected period pairs |

### modern/

| File | Lines | Purpose |
|---|---|---|
| `filtered_pairs.jsonl` | 26,619 | Modern instruction pairs (filtered) |
| `val_pairs.jsonl` | 1,401 | Modern validation |
| `rejected_pairs.jsonl` | 22 | Rejected modern pairs |

### reasoning/

| File | Lines | Purpose |
|---|---|---|
| `sft_train.jsonl` | 1,332 | Reasoning SFT training |
| `filtered_pairs.jsonl` | 1,332 | Filtered reasoning pairs |
| `sft_train_trimmed.jsonl` | 817 | Trimmed (after 515 moved to RL) |
| `sft_val.jsonl` | 70 | Reasoning validation (raw) |
| `sft_val_clean.jsonl` | 70 | Reasoning validation (cleaned) |
| `val_pairs.jsonl` | 70 | Reasoning validation pairs |
| `rejected_pairs.jsonl` | 13 | Rejected reasoning pairs |

### rl_problems/

| File | Lines | Purpose |
|---|---|---|
| `rl_problems_train_expanded.jsonl` | 800 | Expanded RL training problems |
| `rl_prompts_sys_train_expanded.jsonl` | 800 | Expanded system prompts (train) |
| `rl_problems_train.jsonl` | 285 | Original RL problems |
| `rl_prompts_sys_train.jsonl` | 285 | Original system prompts |
| `rl_problems_raw.jsonl` | 299 | Raw before filtering |
| `rl_insights_raw.jsonl` | 299 | Raw insights |
| `rl_problems_val.jsonl` | 14 | Validation problems |
| `rl_prompts_sys_val.jsonl` | 14 | Validation system prompts |
| `rl_prompts_sys_val_clean.jsonl` | 14 | Cleaned val system prompts |
| `rl_prompts_train.jsonl` | 285 | Training prompts (no system) |
| `rl_prompts_val.jsonl` | 14 | Validation prompts (no system) |

### contradiction_problems/

| File | Lines | Purpose |
|---|---|---|
| `contradiction_problems_raw.jsonl` | 298 | Raw contradiction problems |
| `contradiction_problems_train.jsonl` | 284 | Contradiction training |
| `contradiction_problems_val.jsonl` | 14 | Contradiction validation |
| `contradiction_prompts_sys_train.jsonl` | 284 | System prompts (train) |
| `contradiction_prompts_sys_val.jsonl` | 14 | System prompts (val) |

### intuitor_prompts/

| File | Lines | Purpose |
|---|---|---|
| `intuitor_prompts_sys_train.jsonl` | 1,663 | Intuitor system prompts (train) |
| `intuitor_prompts_sys_val.jsonl` | 92 | Intuitor system prompts (val) |
| `intuitor_problems_val.jsonl` | 92 | Intuitor validation problems |

### generated_physics/

| File | Lines | Purpose |
|---|---|---|
| `candidates_raw.jsonl` | 3,015 | Raw physics candidates |
| `verified_raw.jsonl` | 3,015 | Verified raw problems |
| `generated_format_sft.jsonl` | 2,031 | Physics SFT format |
| `sft_train.jsonl` | 1,981 | Physics SFT training |
| `combined_problems_train.jsonl` | 1,094 | Combined problems (train) |
| `combined_prompts_sys_train.jsonl` | 1,094 | Combined prompts (train) |
| `generated_problems_train.jsonl` | 951 | Generated problems (train) |
| `generated_prompts_sys_train.jsonl` | 951 | Generated prompts (train) |
| `yale_problems_train.jsonl` | 143 | Yale PHYSICS (train) |
| `yale_prompts_sys_train.jsonl` | 143 | Yale prompts (train) |
| `sft_val.jsonl` | 50 | Physics SFT validation |
| `combined_problems_val.jsonl` | 64 | Combined problems (val) |
| `combined_prompts_sys_val.jsonl` | 64 | Combined prompts (val) |
| `generated_problems_val.jsonl` | 49 | Generated problems (val) |
| `generated_prompts_sys_val.jsonl` | 49 | Generated prompts (val) |
| `yale_problems_val.jsonl` | 15 | Yale PHYSICS (val) |
| `yale_prompts_sys_val.jsonl` | 15 | Yale prompts (val) |

### r1_reasoning/

| File | Lines | Purpose |
|---|---|---|
| `r1_traces_raw.jsonl` | 1,054 | Raw R1 reasoning traces |
| `sft_train.jsonl` | 670 | R1 SFT training |
| `sft_val.jsonl` | 52 | R1 SFT validation |
| `filtered/filtered_pairs.jsonl` | 670 | Filtered reasoning pairs |
| `filtered/rejected_pairs.jsonl` | 231 | Rejected reasoning pairs |
| `filtered/val_pairs.jsonl` | 35 | Filtered validation |

### Post-1900 Physics Books (`data/post1900_physics_books/`)

| File | Size | Lines |
|---|---|---|
| `rutherford_radioactivity.txt` | 940 KB | 23,014 |
| `thomson_electricity_and_matter.txt` | 216 KB | 5,230 |
| `lorentz_em_phenomena_1904.txt` | 66 KB | 7,359 |
| `planck_normal_spectrum_1901.txt` | 17 KB | 226 |

---

## 7. Physics CLM (Continued Language Modeling)

Continued pretraining of the base model on physics textbooks to improve domain knowledge before SFT/RL.

### Checkpoints

| Checkpoint | Base | Physics Data | Run Script | HF Repo |
|---|---|---|---|---|
| `d34-physics-sft` | d34-22btok | Pre-1900 physics books only | `run_physics_clm_d34.sh` | `mhla/gpt1900-d34-physics-sft` |
| `d34-physicssft-expanded` | d34-22btok | Pre-1900 + post-1900 texts (Rutherford, Thomson, Lorentz, Planck) | `run_physics_clm_d34_expanded.sh` | `mhla/gpt1900-d34-physicssft-expanded` |

### Training Config

```
torchrun --nproc_per_node=8 -m scripts.physics_clm --
    --data-dir ${DATA_DIR}
    --num-epochs 3
    --device-batch-size 4
    --total-batch-size 65536
    --matrix-lr 0.005
    --embedding-lr 0.05
    --unembedding-lr 0.001
    --eval-every 50 --save-every 100
```

### Physics CLM Eval Results

| Checkpoint | UV | Photo | Frozen | ApprC | Train | MM | Elev | Fall | Mean |
|---|---|---|---|---|---|---|---|---|---|
| physics-clm-s004800 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.00 |
| physics-clm-s009600 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.00 |
| physics-clm-s014400 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.3 | 0.0 | 0.04 |
| physics-clm-s019200 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.00 |
| physics-clm-s024000 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.3 | 0.04 |

Physics CLM alone barely moves the needle. Domain knowledge helps but reasoning capability requires SFT+RL.

---

## 8. Instruction Tuning (SFT)

### 8.1 Period/Modern Two-Stage Pipeline

The original SFT approach. Pipeline defined in `run_pre1900_sft.sh`:

1. **Generate** 400K unconditional samples from base model
2. **Craft** instruction pairs via Claude (period + modern styles)
3. **Filter** for anachronisms
4. **SFT Stage 4a:** Train on period-style pairs (Victorian voice)
5. **SFT Stage 4b:** Continue training on modern-style pairs

| Checkpoint | Base | Data | Eval |
|---|---|---|---|
| `d34-sft-period` | d34-22btok | 30K period instruction pairs | 0.21 |
| `d34-sft-modern` | d34-sft-period | 26K modern instruction pairs | 0.00 |

Modern SFT on top of period SFT caused a regression to 0.00 — the model "forgot" the period voice.

### 8.2 Reasoning SFT Versions

| Version | Base | Data | Examples | Key Change | Run Script |
|---|---|---|---|---|---|
| v1 | d34-22btok (raw) | `sft_train.jsonl` | 1,332 | First reasoning SFT | `run_reasoning_sft.sh` |
| v2 | v1 SFT (step 20) | `sft_train.jsonl` | 1,332 | Continued from v1 (step 99) | `run_reasoning_sft_v2.sh` |
| v3 | coherence-rl (step 780) | `sft_train_trimmed.jsonl` | 817 | From coherence RL'd model, trimmed data | `run_reasoning_sft_v3.sh` |
| v4 | d34-22btok (raw) | `sft_train_trimmed.jsonl` | 817 | Raw base ablation (skipped all prior SFT/RL) | `run_reasoning_sft_v4.sh` |
| v5 | d34-physics-sft | `sft_train_trimmed.jsonl` | 817 | Physics-pretrained base | `run_reasoning_sft_v5.sh` |

### 8.3 V3 Corpus SFT Variants

| Variant | Base | Data | Train Examples | Run Script |
|---|---|---|---|---|
| `v3_sft_full` | d34-22btok | v3_corpus_full | 53,458 | `run_v3_sft_full.sh` |
| `v3_sft_safe` | d34-22btok | v3_corpus_safe | 32,489 | `run_v3_sft_safe.sh` |
| `v3_sft_physics_base` | physicssft-expanded | v3_corpus_safe | 32,489 | `run_v3_sft_physics_base.sh` |
| `v3_sft_physics_full` | physicssft-expanded | v3_corpus_full | 53,458 | `run_v3_sft_physics_full.sh` |

Config for physics base (used in v11, v12):
```
torchrun --nproc_per_node=8 -m scripts.pre1900_sft --
    --model-tag d34
    --checkpoints-dir physicssft_expanded_checkpoints
    --device-batch-size=4 --total-batch-size=524288 --max-seq-len=2048
    --train-data v3_corpus_safe/all_train.jsonl
    --val-data v3_corpus_safe/all_val.jsonl
```

### 8.4 R1 Reasoning SFT

| Base | Data | Examples | Iterations | Run Script |
|---|---|---|---|---|
| v3_sft_physics | r1_reasoning/sft_train.jsonl | 670 | 100 | Part of `run_contradiction_rl_v12.sh` |

Config:
```
torchrun --nproc_per_node=8 -m scripts.pre1900_sft --
    --model-tag d34
    --checkpoints-dir v3_sft_physics_checkpoints
    --output-dir r1_reasoning_sft_checkpoints
    --device-batch-size=4 --total-batch-size=65536
    --num-iterations 100 --eval-every 25
    --train-data instruct_data/r1_reasoning/sft_train.jsonl
    --val-data instruct_data/r1_reasoning/sft_val.jsonl
```

### 8.5 Corpus SFT

| Base | Data | Run Script |
|---|---|---|
| d34-22btok | v2_corpus (older variant) | `run_corpus_sft.sh` |

---

## 9. Reinforcement Learning

All RL uses REINFORCE (simplified GRPO without trust region, KL, or PPO clipping). DAPO-style token-level reward normalization. Common config: `--device-batch-size 2 --num-samples 4 --examples-per-step 8`.

### 9.1 Coherence RL

| Field | Value |
|---|---|
| **Base** | d34-sft-period |
| **Reward** | Claude 5-point coherence judge |
| **Data** | General instruction pairs |
| **Output** | `coherence-rl` (= `d34-rl`, same checkpoint) |
| **Eval** | 0.12 |
| **Result** | No effect. Coherence reward signal too weak to move the model. |

### 9.2 Discovery RL v2

| Field | Value |
|---|---|
| **Base** | reasoning-sft-v2 (step 99) |
| **Data** | `rl_problems` (285 general physics problems) |
| **Judge** | OpenAI GPT-4 |
| **Max tokens** | 512 |
| **Epochs** | 3 |
| **Key change** | First RL attempt |
| **Run script** | `run_discovery_rl.sh` |

### 9.3 Discovery RL v3

| Field | Value |
|---|---|
| **Base** | reasoning-sft-v3 (from coherence-rl) |
| **Data** | `rl_problems_expanded` (800 problems) |
| **Judge** | gpt-4.1-mini |
| **Max tokens** | 1024 (doubled from v2) |
| **Epochs** | 3 |
| **Key change** | Expanded RL dataset, coherence RL base, trimmed SFT data |
| **Run script** | `run_discovery_rl_v3.sh` |

### 9.4 Discovery RL v4 (Best pre-v6)

| Field | Value |
|---|---|
| **Base** | reasoning-sft-v4 (from raw d34-22btok) |
| **Data** | `rl_problems_expanded` (800 problems) |
| **Judge** | Claude Sonnet |
| **Max tokens** | 1024 |
| **Epochs** | 3 |
| **Eval** | **0.29** |
| **Key change** | Ablation — skipped all prior SFT/RL, raw pretrained base. Switched to Claude judge. |
| **Finding** | Raw base outperforms coherence RL'd and physics CLM'd bases. Scores 1/5 on UV catastrophe, photoelectric effect, and GR elevator. |
| **Run script** | `run_discovery_rl_v4.sh` |

### 9.5 Discovery RL v5 (Regression)

| Field | Value |
|---|---|
| **Base** | reasoning-sft-v5 (from d34-physics-sft) |
| **Data** | `contradiction_problems` (284 problems) |
| **Judge** | Claude Sonnet, contradiction style |
| **Max tokens** | 1024 |
| **Epochs** | 3 |
| **Eval** | Worse than v4 (peak 0.21) |
| **Key change** | Two things changed: physics-pretrained base AND contradiction RL data/judge |
| **Finding** | Reward hacking — model learned to produce well-formatted `<think>...\answer{}` while content degraded |
| **Run script** | `run_discovery_rl_v5.sh` |

### 9.6 Contradiction RL v6

| Field | Value |
|---|---|
| **Base** | d34-physics-sft (direct, no reasoning SFT) |
| **Data** | `contradiction_problems` (284 problems) |
| **Judge** | Claude, contradiction style |
| **Max tokens** | 2048 |
| **Epochs** | 12 |
| **Coherence** | EMA curriculum (alpha=0.05, min weight=0.1) |
| **Key change** | No scaffold (no reasoning SFT warmstart), coherence reward, longer generation |
| **Eval** | Peak 0.58 (s385) |
| **Run script** | `run_contradiction_rl_v6.sh` |

### 9.7 Contradiction RL v7

| Field | Value |
|---|---|
| **Base** | d34-physics-sft |
| **Data** | `contradiction_problems` (284) |
| **Max tokens** | 2048 |
| **Epochs** | 12 |
| **Coherence** | Fixed weight 0.1 (no EMA) |
| **Key change** | Fixed coherence weight instead of EMA curriculum |
| **Run script** | `run_contradiction_rl_v7.sh` |

### 9.8 Contradiction RL v8

| Field | Value |
|---|---|
| **Base** | d34-physics-sft |
| **Data** | `contradiction_problems` (284) |
| **Max tokens** | 2048 |
| **Epochs** | 24 (doubled from v6/v7) |
| **Coherence** | Fixed weight 0.25 |
| **Key change** | Bedrock API, empty responses score 0, higher coherence weight, longer training |
| **Eval** | Peak 0.62 (s315) |
| **Run script** | `run_contradiction_rl_v8.sh` |

### 9.9 Contradiction RL v9

| Field | Value |
|---|---|
| **Base** | d34-physicssft-expanded (includes post-1900 texts) |
| **Data** | `contradiction_problems` (284) |
| **Max tokens** | 2048 |
| **Epochs** | 24 |
| **Coherence** | EMA (alpha=0.05, min=0.1) — like v6 |
| **Key change** | Expanded physics SFT base |
| **Run script** | `run_contradiction_rl_v9.sh` |

### 9.10 Contradiction RL v10

| Field | Value |
|---|---|
| **Base** | d34-physicssft-expanded |
| **Data** | `contradiction_problems` (284) |
| **Max tokens** | 2048 |
| **Epochs** | 24 |
| **Coherence** | EMA + logical style |
| **Key change** | Added `--coherence-style logical` (emphasizes logical flow in coherence judging) |
| **Run script** | `run_contradiction_rl_v10.sh` |

### 9.11 Contradiction RL v11 (Best Overall)

| Field | Value |
|---|---|
| **Chain** | d34-22btok → physicssft-expanded → v3 SFT physics (safe) → contradiction RL |
| **Data** | `contradiction_problems` (284) |
| **Max tokens** | 2048 |
| **Epochs** | 24 |
| **Coherence** | Fixed weight 0.25, logical style |
| **Key change** | v3 SFT physics base (opinion-filtered corpus SFT on physics-expanded model) |
| **Eval** | **Peak 1.25** (s560, s630) |
| **Best scores** | Photoelectric 4/5 (s770), Elevator 3/5 (s630), UV catastrophe 2/5 (s735), Michelson-Morley 2/5 (s455) |
| **Run script** | `run_contradiction_rl_v11.sh` |

### 9.12 Contradiction RL v12

| Field | Value |
|---|---|
| **Chain** | d34-22btok → physicssft-expanded → v3 SFT physics → R1 reasoning SFT → contradiction RL |
| **Data** | `contradiction_problems` (284, v12 prompts without system prompt) |
| **Max tokens** | 2048 |
| **Epochs** | 24 |
| **Coherence** | Fixed 0.25, logical style |
| **Key changes** | R1 distillation SFT before RL. No system prompt anywhere. User prompts end with "Think deeply and step by step." Format reward checks `<think>` and `\answer{}` tags. Physics online eval from EVAL.json. |
| **Run script** | `run_contradiction_rl_v12.sh` |

### 9.13 1905 RL Pipeline

| Field | Value |
|---|---|
| **Chain** | d34-1905 → physics CLM (pre+post-1900 texts, 10 epochs) → v3 SFT safe → contradiction RL (v11-style) |
| **Key difference** | Base model trained on pre-1905 corpus instead of pre-1900. Physics CLM includes post-1900 texts (Rutherford, Thomson, Lorentz, Planck). |
| **Run script** | `run_1905_rl.sh` |

### 9.14 Math RL

| Field | Value |
|---|---|
| **Chain** | R1 reasoning SFT → Math RL |
| **Data** | GSM8K train (7,473 examples, 30%) + MATH train (7,500 examples, 70%) |
| **Rewards** | Format (0.3): `<think>` + `\answer{}` tags. Correctness (1.0): numeric comparison (GSM8K) or SymPy symbolic equivalence (MATH) |
| **Physics eval** | Monitor only (from EVAL.json) |
| **Config** | `--gsm8k-ratio 0.3 --num-epochs 3 --eval-gsm8k-examples 50 --eval-math-examples 50` |
| **Run script** | `run_math_rl.sh` |

### 9.15 Intuitor RL

| Field | Value |
|---|---|
| **Chain** | d34-22btok → physicssft-expanded → Intuitor RL |
| **Reward** | `--reward-mode self-certainty` (intrinsic reward, no gold answers or API calls for training) |
| **Validation** | Claude judge still used for val eval to track actual correctness |
| **Data** | `intuitor_prompts/intuitor_prompts_sys_train.jsonl` (1,663 problems) |
| **Config** | `--num-samples 8 --max-new-tokens 512 --temperature 0.9` |
| **Run script** | `run_intuitor_rl.sh` |

### 9.16 Generated Physics SFT + Verifiable RL

**SFT Stage:**
| Field | Value |
|---|---|
| **Chain** | d34-physics-sft → Generated physics format SFT |
| **Data** | `generated_physics/sft_train.jsonl` (1,981 verified traces), `sft_val.jsonl` (50) |
| **Config** | `--num-iterations 200 --total-batch-size=65536` |
| **HF Dataset** | `mhla/pre1900-verifiable-physics` |
| **Run script** | `run_generated_physics_sft.sh` |

**RL Stage:**
| Field | Value |
|---|---|
| **Chain** | Generated physics SFT → Verifiable RL |
| **Data** | `generated_physics/combined_problems_train.jsonl` (1,094 = 951 generated + 143 Yale) |
| **Reward** | SymPy symbolic equivalence (1.0) + format bonus (0.3) |
| **Config** | `--num-samples 8 --max-new-tokens 2048 --num-epochs 20 --eval-every 136 --save-every 136` |
| **Script** | `scripts/pre1900_scripts/verifiable_rl.py` |
| **Run script** | `run_generated_physics_rl.sh` |

---

## 10. Checkpoint Lineage

```
Pre-1900 Base Models
====================

d26-8btok
d26-22btok

d34-22btok (MAIN BASE) ─┬─ d34-sft-period ──────────── d34-sft-modern
                         │
                         ├─ coherence-rl (= d34-rl)  [eval: 0.12, no effect]
                         │
                         ├─ reasoning-sft-v1
                         │   └─ reasoning-sft-v2
                         │       └─ discovery-rl-v2   [judge: GPT-4, 512 tok]
                         │
                         ├─ coherence-rl
                         │   └─ reasoning-sft-v3
                         │       └─ discovery-rl-v3   [judge: gpt-4.1-mini, 800 problems]
                         │
                         ├─ reasoning-sft-v4          [raw base ablation]
                         │   └─ discovery-rl-v4       [eval: 0.29, best pre-v6]
                         │
                         ├─ d34-physics-sft (CLM) ─┬─ reasoning-sft-v5
                         │                         │   └─ discovery-rl-v5         [regression]
                         │                         ├─ contradiction-rl-v6         [eval: 0.58]
                         │                         ├─ contradiction-rl-v7
                         │                         └─ contradiction-rl-v8         [eval: 0.62]
                         │
                         └─ physicssft-expanded ─┬─ contradiction-rl-v9
                            (CLM + post-1900)    ├─ contradiction-rl-v10
                                                 └─ v3-sft-physics (safe) ─┬─ contradiction-rl-v11  [eval: 1.25, BEST]
                                                                           └─ r1-reasoning-sft ─┬─ contradiction-rl-v12
                                                                                                └─ math-rl

d45 (8-node, pre-1900)

Pre-1905 Models
===============
d34-1905 ─── physics-clm-1905 ─── v3-sft-1905 ─── contradiction-rl-1905

Pre-1915 Models
===============
d40-1915 (8-node, aspect-ratio=64, head-dim=128)
d42-1915 (8-node, aspect-ratio=64, head-dim=128, ratio=11)
d46-1915-7b (FSDP, ZeRO-3, AdamW only)

1900-1964 Models
================
d34-1964 (own tokenizer, full download→reshard→train pipeline)
```

---

## 11. Evaluation Results

### Physics Eval Rubric

8 tasks, each scored 0-5 by Claude judge. Scores averaged across 3 samples per task per checkpoint.

**Tasks:**
1. **UV Catastrophe** — heated cavity radiation, why equipartition fails
2. **Photoelectric Effect** — discrete energy delivery, threshold frequency
3. **Frozen Light** — invariant light speed, no stationary EM wave
4. **Approaching c** — relativistic energy/momentum, asymptotic speed limit
5. **Train/Lightning** — simultaneity is frame-dependent
6. **Michelson-Morley** — null result, no detectable aether
7. **Elevator/Light** — equivalence principle, light bends in gravity
8. **Free-Fall Equivalence** — gravity removable by free-fall, local equivalence

### Base Models (all 0.00)

All base models and early SFT models score 0.00 across all tasks — the model generates text but makes no progress toward post-1900 physics concepts.

### Discovery RL v5

| Checkpoint | UV | Photo | Frozen | ApprC | Train | MM | Elev | Fall | Mean |
|---|---|---|---|---|---|---|---|---|---|
| v5-s030 | 0.0 | 0.0 | 0.0 | 0.3 | 0.0 | 0.3 | 0.3 | 0.3 | **0.17** |
| v5-s060 | 0.0 | 0.3 | 0.0 | 0.0 | 0.0 | 0.3 | 0.7 | 0.3 | **0.21** |
| v5-s090 | 0.0 | 0.0 | 0.0 | 0.3 | 0.0 | 0.0 | 1.0 | 0.0 | **0.17** |
| v5-s104 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.7 | **0.08** |

Score peaked early (s060) then degraded — reward hacking.

### Contradiction RL v6

| Checkpoint | UV | Photo | Frozen | ApprC | Train | MM | Elev | Fall | Mean |
|---|---|---|---|---|---|---|---|---|---|
| v6-s035 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.3 | 0.3 | 0.08 |
| v6-s105 | 0.3 | 0.7 | 0.0 | 0.0 | 0.3 | 0.3 | 0.3 | 0.3 | 0.29 |
| v6-s140 | 0.0 | 1.0 | 0.0 | 0.0 | 0.3 | 0.3 | 0.3 | 0.7 | 0.33 |
| v6-s175 | 1.0 | 0.7 | 0.0 | 0.0 | 0.3 | 0.3 | 0.7 | 1.0 | 0.50 |
| v6-s245 | 0.7 | 1.0 | 0.3 | 0.0 | 0.0 | 1.0 | 0.0 | 1.0 | 0.50 |
| v6-s385 | 1.0 | 1.0 | 0.3 | 0.0 | 0.3 | 0.7 | 0.3 | 1.0 | **0.58** |
| v6-s419 | 1.0 | 1.0 | 0.0 | 0.0 | 0.0 | 1.0 | 0.7 | 0.7 | 0.54 |

### Contradiction RL v8

| Checkpoint | UV | Photo | Frozen | ApprC | Train | MM | Elev | Fall | Mean |
|---|---|---|---|---|---|---|---|---|---|
| v8-s035 | 0.3 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.3 | 0.0 | 0.08 |
| v8-s210 | 0.7 | 1.0 | 0.7 | 0.0 | 0.0 | 0.7 | 0.3 | 0.7 | 0.50 |
| v8-s315 | 0.7 | 1.3 | 0.7 | 0.0 | 0.0 | 1.0 | 1.0 | 0.3 | **0.62** |
| v8-s385 | 1.0 | 1.0 | 0.3 | 0.0 | 0.0 | 0.7 | 0.7 | 0.7 | 0.54 |
| v8-s420 | 1.0 | 1.0 | 0.3 | 0.0 | 0.3 | 0.3 | 0.3 | 0.0 | 0.42 |

### Contradiction RL v11 (Best Overall)

| Checkpoint | UV | Photo | Frozen | ApprC | Train | MM | Elev | Fall | Mean |
|---|---|---|---|---|---|---|---|---|---|
| v11-s035 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 | 0.25 |
| v11-s070 | 0.0 | 1.0 | 1.0 | 0.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.75 |
| v11-s210 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 | 1.0 | 1.0 | 1.0 | 0.88 |
| v11-s315 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 2.0 | 1.0 | **1.12** |
| v11-s350 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 2.0 | 1.0 | **1.12** |
| v11-s455 | 1.0 | 2.0 | 1.0 | 0.0 | 1.0 | 2.0 | 1.0 | 1.0 | **1.12** |
| v11-s525 | 1.0 | 2.0 | 0.0 | 1.0 | 1.0 | 1.0 | 2.0 | 1.0 | **1.12** |
| v11-s560 | 1.0 | 2.0 | 1.0 | 1.0 | 1.0 | 1.0 | 2.0 | 1.0 | **1.25** |
| v11-s630 | 1.0 | 3.0 | 1.0 | 0.0 | 0.0 | 1.0 | 3.0 | 1.0 | **1.25** |
| v11-s735 | 2.0 | 2.0 | 1.0 | 0.0 | 1.0 | 1.0 | 2.0 | 1.0 | **1.25** |
| v11-s770 | 1.0 | **4.0** | 1.0 | 0.0 | 1.0 | 1.0 | 1.0 | 0.0 | 1.12 |
| v11-s839 | 1.0 | 2.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.12 |

### Best Individual Generations (v11)

| Task | Checkpoint | Score | Key Matched Concepts |
|---|---|---|---|
| Photoelectric Effect | v11-s770 | **4/5** | Rejects continuous wave, proposes discrete energy at particular points/instants, explains threshold frequency, immediate emission, brightness vs frequency distinction |
| Elevator/Light | v11-s630 | **3/5** | Acceleration and gravity locally equivalent, light bends in gravitational fields, gravity affects light despite no mass |
| UV Catastrophe | v11-s735 | **2/5** | Not all modes get equal energy, total energy is finite, frequency affects distribution |
| Michelson-Morley | v11-s455 | **2/5** | Null result undermines aether, no need for universal rest frame, light travels at fixed speed |

### General Reasoning Eval

R1-SFT baseline evaluated on 16 problems across math, logic, common sense, science, and everyday reasoning. The model struggles severely with basic math and reasoning tasks — generating incoherent chains of numbers rather than logical steps. RL reasoning capabilities do not transfer well outside the physics domain.

### Checkpoint Registry Steps (from `physics_eval.py`)

All checkpoint steps used in evaluation:

| Checkpoint | HF Repo | Step | Mode |
|---|---|---|---|
| d26-8btok | mhla/gpt1900-d26-8btok | 7226 | completion |
| d26-22btok | mhla/gpt1900-d26-22btok | 17517 | completion |
| d34-8b-subset | mhla/gpt1900-d34-8b-subset | 10507 | completion |
| d34-22btok | mhla/gpt1900-d34-22btok | 10507 | completion |
| d34-physics-sft | mhla/gpt1900-d34-physics-sft | 404 | completion |
| d34-sft-period | mhla/gpt1900-d34-sft-period | 20 | chat |
| d34-sft-modern | mhla/gpt1900-d34-sft-modern | 16 | chat |
| d34-rl | mhla/gpt1900-d34-rl | 780 | chat |
| coherence-rl | mhla/gpt1900-d34-coherence-rl | 780 | chat |
| reasoning-sft-v4 | mhla/gpt1900-d34-reasoning-sft-v4 | 99 | chat |
| discovery-rl-v4 | mhla/gpt1900-d34-discovery-rl-v4 | 180 | chat |
| discovery-rl-v5-* | local | 30-104 | chat |
| discovery-rl-v6-* | local | 35-419 | completion |
| discovery-rl-v7-* | local | 35-419 | completion |
| discovery-rl-v8-* | local | 35-420 | completion |
| discovery-rl-v11-* | local | 35-839 | chat |

Note: v6-v8 use "completion" mode (no system prompt, raw generation), while v5 and v11 use "chat" mode.

### Key Findings

1. **v11 is the best model** — chain: physicssft-expanded → v3 SFT safe → contradiction RL
2. **Raw base outperforms prior RL** — v4 (raw base) beat v3 (coherence RL base)
3. **Coherence RL had no effect** — too weak a reward signal
4. **v5 regressed** — changed base AND data simultaneously; contradiction problems solvable within pre-1900 physics
5. **v6-v8 steady improvement** — no scaffold, coherence curriculum, longer training
6. **v11 breakthrough** — v3 SFT on opinion-filtered corpus provided crucial instruction-following capability
7. **Reward hacking** — v5 learned to produce formatted outputs while content degraded
8. **All models fail the same core way** — can sometimes identify wrong classical assumption but cannot propose a coherent post-1900 alternative
9. **Peak single-sample score: 4/5** on photoelectric effect (v11-s770) — the model almost independently describes discrete energy packets

---

## 12. Infrastructure & Architecture Experiments

Chronological summary from `dev/LOG.md`. All experiments tested at d12-d26 scale unless noted.

### Adopted

| Date | Experiment | Result |
|---|---|---|
| 2026-01-10 | **Polar Express orthogonalization** (Muon) | In noise, kept as default |
| 2026-01-10 | **NorMuon variance reduction** | Small improvement, kept |
| 2026-01-10 | **Cautious weight decay** (only decay where update·weight ≥ 0) | Solid improvement |
| 2026-01-10 | **Weight decay schedule** (linear 1.0→0.0) | Better than static |
| 2026-01-10 | **Weight decay scaling law** (WD ∝ 1/width²) | Consistent across depths |
| 2026-01-11 | **Flash Attention 3** | ~9% tok/sec improvement |
| 2026-01-11 | **Sliding window attention** (SSSL pattern) | Good balance compute/quality |
| 2026-01-11 | **x0/resid per-layer scalars** | Consistent improvement (0.004-0.01 bpb across depths) |
| 2026-01-13 | **BOS-aligned dataloader** (BestFit-Crop) | 100% utilization, 34.6% crop waste |
| 2026-01-13 | **Number token split** ({1,2} optimal for 32K vocab) | Validated |
| 2026-01-16 | **FA3 → SDPA fallback** | Enables CPU/MPS/older GPU support |
| 2026-01-17 | **Value Embeddings** (alternating layers, gated) | Big improvement, reduced optimal ratio to ~4 |
| 2026-01-19-22 | **Optimizer hyperparameter sweep** (320 experiments) | x0_beta1=0.96 only surviving change at target scale |
| 2026-02-02 | **FP8 training** (torchao tensorwise, all Linear layers) | ~5% capability-matched speedup at d24+ |
| 2026-02-05 | **Auto batch size scaling** (B_opt ∝ D^0.383) | Principled batch size for all depths |

### Not Adopted (Negative Results)

| Date | Experiment | Result |
|---|---|---|
| 2026-01-08 | Gradient clipping | No benefit at any scale, removed entirely |
| 2026-01-12 | Multi-token prediction (MTP) | +13GB memory, same/worse quality |
| 2026-01-13 | FP8 for lm_head only | +2GB memory, 1% speedup, not worth complexity |
| 2026-01-13 | Varlen attention (cross-doc boundary prevention) | 0.0002 bpb, not worth code complexity |
| 2026-01-15 | Olmo pretraining mix (dolma3) | CORE 15.5 → 13.8, worse |
| 2026-01-16 | Half-truncated RoPE | No improvement |
| 2026-01-16 | Asymmetric softcap | Slightly worse |
| 2026-01-16 | Smear gate | Negligible |
| 2026-01-16 | Backout / skip connection | No improvement / slightly worse |
| 2026-01-17 | Attention gates | No improvement, +1GB memory |
| 2026-01-18 | Muon custom kernels | ~20% in isolation, washed out in training |
| 2026-01-18 | Fused QKVO Linear | ~Zero impact |
| 2026-01-27 | Engram-lite (bigram hash embeddings) | Works at d12, reverted at d25 (no wall-clock gain) |
| 2026-01-29 | Hyperball/MuonH optimization | Worse than baseline across all variants |
| 2026-02-03 | Skip AdamW every other step | ~2% faster tok/s but worse per-step, net negative |
| 2026-02-03 | Flip Muon MLP LR multiplier | Slightly worse |
| 2026-02-05 | SwiGLU activation | Worse on all axes (steps, wall clock, FLOPs) |

### Key Scaling Results

**Optimal tokens:params ratio:** ~10.5 (Kaplan-style, counting transformer matrices + lm_head)

**Weight decay scaling:** WD_target = WD_reference × (d_reference / d_target)², i.e. WD ∝ 1/width²

**Hyperparameters are scale-dependent:** Elaborate fine-tuning at d12 actively hurts at d20. Only x0_beta1=0.96 survived validation at target scale.

### Serving Infrastructure (Mar 2026)

| Date | Component | Description |
|---|---|---|
| 2026-03-25 | **Continuous Batching Engine** (`nanochat/batch_engine.py`) | Async scheduler for prefilling + batched decode across concurrent requests. KVCacheView (slot-based KV cache pool), request swapping. |
| 2026-03-25 | **Flash Attention 3 Wrapper** (`nanochat/flash_attention.py`) | FA3 on Hopper+ with automatic SDPA fallback for older GPUs, CPU, and MPS. |
| 2026-03-26 | **TTFT Timeout + Request Cancellation** (`scripts/chat_web_batch.py`) | 20-second timeout for first token, graceful cancellation of in-flight requests. |
| 2026-03-29 | **SQLite Chat Logger** (`nanochat/chat_logger.py`) | Request/response persistence with conversation filtering, time-range queries, pagination. |

---

## 13. Deployment

### Serving Infrastructure

**Continuous Batching Server** (`scripts/chat_web_batch.py`, added Mar 2026):
- FastAPI server using `nanochat/batch_engine.py` for async continuous batching
- One process per GPU, each handling many concurrent requests
- Endpoints: `POST /chat/completions` (SSE streaming), `GET /health`, `GET /stats`
- Features: TTFT timeout (20s), request cancellation, single-turn mode, API key auth

**Multi-GPU Orchestration** (`scripts/launch_serving.sh`):
- Launches one `chat_web_batch.py` process per GPU (ports 8001-800N)
- Generates nginx config dynamically for load balancing on port 80
- Least-conn upstream selection, rate limiting (10 req/s per IP, burst 20, max 4 concurrent)
- API key authentication via `BACKEND_API_KEY` env var

**Data Parallelism Server** (`scripts/chat_web.py`):
- Older approach: copies model to N GPUs, distributes requests round-robin
- Each request runs on its own GPU sequentially
- Supports same endpoints as batch server

**Chat Logger** (`nanochat/chat_logger.py`, added Mar 2026):
- SQLite request/response logging with indexed queries
- Schema: conversation_id, messages_json, response_text, temperature, latency_ms, client_ip, GPU tracking
- Supports conversation filtering, time-range queries, pagination

### RunPod Serverless (`deploy/runpod/`)
- `Dockerfile`: Inference container
- `Dockerfile.train`: Training container
- `handler.py`: Serverless request handler (generator-style, streaming via RunPod protocol)
- `download_model.py`: Model download on cold start

### AWS SageMaker (`deploy/sagemaker/`)
- `entry_point.sh`: SageMaker entry point for training/inference

### Vercel Frontend (`deploy/vercel/`)
- `api/chat.js`: Edge Runtime chat endpoint — forwards to self-hosted `BACKEND_URL` with API key auth
- `public/index.html`: Web UI with vintage 19th-century aesthetic, animated "monocle man" mascot with multiple character states (thinking, talking, sleeping, writing, etc.), welcome screen with suggested prompts
- `vercel.json`, `package.json`: Deployment config (no build step, static HTML)
- Redesigned Mar 2026 (commit 76e6e14)

### Nginx (`deploy/nginx/`)
- `nanochat.conf`: Reverse proxy config template (rate limiting, connection limits, API key validation)

### Cloud Providers
- **Lambda Labs**: Primary GPU provider for training (H100 8×GPU nodes)
- **AWS EC2/Bedrock**: Claude API access for judging, R1 distillation
- Launch scripts: `launch_ec2.py`, `launch_runpod.py`, `launch_sagemaker.py`, `interactive_pod.py`

### MCP Integration (`mcp/sms_alert/`)
- SMS alerting system for training monitoring
- `server.py`: MCP server for sending alerts

---

## 14. HuggingFace Repos

### Model Checkpoints

| HF Repo | Type | Description |
|---|---|---|
| `mhla/gpt1900-d26` | Base | d26 base model |
| `mhla/gpt1900-d26-8btok` | Base | d26, 8B tokens |
| `mhla/gpt1900-d26-22btok` | Base | d26, 22B tokens |
| `mhla/gpt1900-d34-8b-subset` | Base | d34, 8B tokens (subset) |
| `mhla/gpt1900-d34-22btok` | Base | d34, 22B tokens (MAIN BASE) |
| `mhla/gpt1900-d34-physics-sft` | CLM | d34 + physics CLM (pre-1900 only) |
| `mhla/gpt1900-d34-physicssft-expanded` | CLM | d34 + physics CLM (pre+post-1900 texts) |
| `mhla/gpt1900-d34-physics-clm-pre1900` | CLM | Physics CLM intermediate |
| `mhla/gpt1900-d34-sft-period` | SFT | Period-style instruction pairs |
| `mhla/gpt1900-d34-sft-modern` | SFT | Modern-style instruction pairs |
| `mhla/gpt1900-d34-sft-dialogue` | SFT | Dialogue SFT |
| `mhla/gpt1900-d34-dialogue-sft` | SFT | Dialogue SFT variant |
| `mhla/gpt1900-d34-dialogue-sft-v2` | SFT | Dialogue SFT v2 |
| `mhla/gpt1900-d34-dialogue-sft-v3` | SFT | Dialogue SFT v3 |
| `mhla/gpt1900-d34-reasoning-sft-v1` | SFT | Reasoning SFT v1 |
| `mhla/gpt1900-d34-reasoning-sft-v2` | SFT | Reasoning SFT v2 |
| `mhla/gpt1900-d34-reasoning-sft-v3` | SFT | Reasoning SFT v3 |
| `mhla/gpt1900-d34-reasoning-sft-v4` | SFT | Reasoning SFT v4 |
| `mhla/gpt1900-d34-r1-reasoning-sft` | SFT | R1 distillation SFT |
| `mhla/gpt1900-d34-v3-sft` | SFT | V3 corpus SFT |
| `mhla/gpt1900-d34-v3-sft-physics` | SFT | V3 SFT on physics base (safe) |
| `mhla/gpt1900-d34-v3-sft-physics-full` | SFT | V3 SFT on physics base (full) |
| `mhla/gpt1900-instruct` | SFT | Instruct model (early) |
| `mhla/gpt1900-instruct-base-sft` | SFT | Instruct base SFT |
| `mhla/gpt1900-instruct-base-sft-deconfessed` | SFT | Instruct base (deconfessed) |
| `mhla/gpt1900-instruct-physics-sft` | SFT | Instruct physics SFT |
| `mhla/gpt1900-instruct-v3-sft` | SFT | **Instruct v3 SFT (recommended for chat)** |
| `mhla/gpt1900-d34-coherence-rl` | RL | Coherence RL |
| `mhla/gpt1900-d34-discovery-rl-v1` | RL | Discovery RL v1 |
| `mhla/gpt1900-d34-discovery-rl-v2` | RL | Discovery RL v2 |
| `mhla/gpt1900-d34-discovery-rl-v3` | RL | Discovery RL v3 |
| `mhla/gpt1900-d34-discovery-rl-v4` | RL | Discovery RL v4 |
| `mhla/gpt1900-d34-discovery-rl-v6` | RL | Discovery RL v6 |
| `mhla/gpt1900-d34-contradiction-rl-v6` | RL | Contradiction RL v6 (eval: 0.58) |
| `mhla/gpt1900-d34-contradiction-rl-v7` | RL | Contradiction RL v7 |
| `mhla/gpt1900-d34-contradiction-rl-v8` | RL | Contradiction RL v8 (eval: 0.62) |
| `mhla/gpt1900-d34-contradiction-rl-v9` | RL | Contradiction RL v9 |
| `mhla/gpt1900-d34-contradiction-rl-v11` | RL | **Contradiction RL v11 (eval: 1.25, BEST)** |
| `mhla/gpt1900-d34-contradiction-rl-v12` | RL | Contradiction RL v12 (R1 base) |
| `mhla/gpt1900-d34-gen-physics-rl` | RL | Generated physics RL |
| `mhla/gpt1900-d34-intuitor-rl` | RL | Intuitor RL (self-certainty) |
| `mhla/gpt1900-d34-math-rl` | RL | Math RL (GSM8K + MATH) |
| `mhla/gpt1900-d34-discovery-rl-1900` | RL | Discovery RL 1900 pipeline |
| `mhla/gpt1900-d34-v3-sft-physics-1900-rl` | RL | V3 SFT physics → 1900 RL |
| `mhla/gpt1905-d34` | Base | Pre-1905 d34 base model |
| `mhla/gpt1964-d34` | Base | 1900-1964 d34 base model |
| `mhla/pre1900_d26_22btok_2epoch` | Base | d26 22B tokens, 2 epochs |

### Datasets

| HF Repo | Description |
|---|---|
| `mhla/pre1900-corpus` | Pre-1900 text corpus (HathiTrust, BL, AmericanStories) |
| `mhla/pre1900-training` | Tokenized pre-1900 training data |
| `mhla/pre1915-corpus` | Pre-1915 text corpus |
| `mhla/pre1964-corpus` | Pre-1964 text corpus |
| `mhla/gpt1900-physics-data` | Physics books and texts |
| `mhla/gpt1900-physics-clm` | Physics CLM training data |
| `mhla/gpt1900-instruct-data` | Instruction pairs (early versions) |
| `mhla/gpt1900-instruct-v3-data` | V3 instruction data (cleaned, opinion-filtered) |
| `mhla/pre1900-verifiable-physics` | Verifiable physics problems (SymPy) |
| `mhla/gpt1900-contradiction-eval` | Contradiction evaluation data |

---

## Run Scripts Index

Active scripts in `runs/` (older versions moved to `runs/archive/`):

| Script | Purpose |
|---|---|
| **Infrastructure** | |
| `chat.sh` | Download model from HF + launch interactive chat or generation |
| `interactive_pod.py` | Interactive GPU pod management |
| `launch_ec2.py` | Launch AWS EC2 instances |
| `launch_runpod.py` | Launch RunPod pods |
| `launch_sagemaker.py` | Launch SageMaker training jobs |
| **SFT** | |
| `run_reasoning_sft_v5.sh` | Reasoning SFT v5 (latest) |
| `run_v3_sft_full.sh` | V3 SFT (full corpus) |
| `run_v3_sft_safe.sh` | V3 SFT (opinion-filtered) |
| `run_v3_sft_physics_base.sh` | V3 SFT on physics base |
| `run_v3_sft_physics_full.sh` | V3 SFT on physics (full) |
| `run_corpus_sft.sh` | Corpus-based SFT |
| `run_openthoughts_sft.sh` | OpenThoughts SFT (prepare + train on physics-expanded base) |
| **RL** | |
| `run_contradiction_rl_v12.sh` | Contradiction RL v12 (latest) |
| `run_discovery_rl_v5.sh` | Discovery RL v5 (latest) |
| `run_1900_rl.sh` | Full 1900 RL pipeline (d34 → CLM → v3 SFT → contradiction RL) |
| `run_1905_rl.sh` | 1905 RL pipeline |
| `run_math_rl.sh` | Math RL (GSM8K + MATH) |
| `run_intuitor_rl.sh` | Intuitor RL (self-certainty reward) |
| `run_openthoughts_rl.sh` | OpenThoughts verifiable RL (SymPy equivalence) |
| **Pretraining** | |
| `run_pre1915_7b_fsdp.sh` | Pre-1915 7B (FSDP, latest large-scale run) |
| **Eval** | |
| `run_physics_eval.sh` | Physics evaluation |

Archived scripts in `runs/archive/` (older experiment versions preserved for reproducibility):
- Pretraining: `run_pre1900*.sh`, `run_pre1905*.sh`, `run_pre1915_d40.sh`, `run_pre1915_d42.sh`, `run_1900_1964*.sh`
- Physics CLM: `run_physics_clm_d34*.sh`
- SFT: `run_pre1900_sft.sh`, `run_reasoning_sft.sh` through `v4.sh`, `run_sft_stages2_4.sh`, `run_generated_physics_sft.sh`
- RL: `run_discovery_rl.sh` through `v4.sh`, `run_contradiction_rl_v5.sh` through `v11.sh`, `run_pre1900_rl.sh`, `run_generated_physics_rl.sh`
- Misc: `speedrun.sh`, `miniseries.sh`, `scaling_laws.sh`, `runcpu.sh`, `profile_d52_memory.sh`, `run_reasoning_pipeline.sh`, `run_reasoning_traces.sh`, `run_llm_clean.sh`
