# Pre-1900 Scripts

Scripts for building the pre-1900 training corpus, generating instruction/RL data, and running RL training loops.

## Data Collection
- `hf_download.py` — Download pre-1900 texts from HuggingFace (institutional-books, blbooks, AmericanStories)
- `hf_download_1900_1964.py` — Download 1900-1964 period texts
- `collect_physics_books.py` — Collect curated pre-1900 physics books (Wikisource, Gutenberg, Internet Archive)
- `collect_post1900_physics.py` — Collect post-1900 physics reference texts (Rutherford, Thomson, etc.)
- `search_physics_books.py` — Search the dataset for specific physics books

## Data Cleaning
- `hf_clean.py` — Clean raw HuggingFace downloads (unicode, boilerplate)
- `clean_full_corpus.py` — Full corpus OCR/quality cleanup (exports shared constants used by other scripts)
- `clean_year_split.py` — Split corpus by year
- `clean_sft_data.py` — Clean SFT training data
- `llm_clean.py` — LLM-assisted cleanup pass

## Data Preparation
- `reshard.py` — Reshard dataset for distributed training
- `reshard_1900_1964.py` — Reshard 1900-1964 data
- `prepare_physics_parquet.py` — Convert physics texts to parquet
- `prepare_reasoning_data.py` — Prepare reasoning SFT data
- `prepare_openthoughts.py` — Prepare OpenThoughts data
- `prior_filter.py` — Build token-frequency prior for quality filtering

## Instruction Data Generation
- `craft_instruct_pairs.py` — Generate instruction/response pairs from corpus
- `filter_instruct_pairs.py` — Filter instruction pairs (exports shared patterns)
- `generate_rl_problems.py` — Generate RL training problems (exports shared utilities)
- `generate_contradiction_problems.py` — Generate contradiction-detection problems
- `generate_reasoning_traces.py` — Generate reasoning traces for SFT pre-training
- `generate_r1_traces.py` — Generate R1-style thinking traces
- `generate_unconditional.py` — Generate unconditional samples from checkpoint

## RL Training
- `verifiable_rl.py` — GRPO with SymPy-verifiable rewards (primary)
- `discovery_rl.py` — GRPO with Claude judge rewards
- `coherence_rl.py` — Coherence-based RL training
- `intuitor_rl.py` — Intuitor-based RL training

## Utilities
- `constants.py` — Shared system prompts (REASONING_SYSTEM_PROMPT, QUANTITATIVE_REASONING_SYSTEM_PROMPT)
- `hf_tok_train.py` — Train BPE tokenizer on corpus

## Archive
One-off data processing, generation, and utility scripts that were used during development but are no longer actively referenced. See `archive/` subdirectory.
