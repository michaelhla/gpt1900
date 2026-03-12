# GPT-1900 Checkpoint Registry

## Base Models (Pretraining)

| Checkpoint | Description | Tokens | Mode | Eval Score |
|---|---|---|---|---|
| `d26-8btok` | 26-layer, 8B tokens | 8B | completion | 0.12 |
| `d26-22btok` | 26-layer, 22B tokens | 22B | completion | 0.12 |
| `d34-8b-subset` | 34-layer, 8B tokens (subset) | 8B | completion | 0.08 |
| `d34-22btok` | 34-layer, 22B tokens | 22B | completion | 0.29 |
| `d34-physics-sft` | d34-22btok + continued pretraining on physics textbooks (CLM) | +physics books | completion | 0.33 |

## Instruction Tuning (SFT)

| Checkpoint | Base | Data | Eval Score |
|---|---|---|---|
| `d34-sft-period` | d34-22btok | Period-style instruction pairs (Victorian voice) | 0.21 |
| `d34-sft-modern` | d34-sft-period | Modern-style instruction pairs (layered on period SFT) | 0.00 |

## Coherence RL

| Checkpoint | Base | Reward | Eval Score | Notes |
|---|---|---|---|---|
| `d34-rl` | d34-sft-period | Claude 5-pt coherence judge | 0.12 | Step 780 |
| `coherence-rl` | d34-sft-period | Claude 5-pt coherence judge | 0.12 | **Identical to d34-rl** (same checkpoint) |

## Reasoning SFT + Discovery RL Versions

### v1 — First reasoning SFT

| Stage | Base | Data | Output |
|---|---|---|---|
| Reasoning SFT | d34-22btok (raw) | `sft_train.jsonl` (1332 examples) | `reasoning_sft_v1` |

No RL stage. No eval.

### v2 — First discovery RL

| Stage | Base | Data | Output |
|---|---|---|---|
| Reasoning SFT | v1 SFT (step 20) | `sft_train.jsonl` (1332 examples) | `reasoning_sft_v2` |
| Discovery RL | v2 SFT (step 99) | `rl_problems` (general physics) | `discovery_rl_v2` |

- **Judge:** OpenAI GPT-4
- **Max tokens:** 512
- **Key change:** First RL attempt. Continued SFT from v1.

### v3 — Coherence RL base + expanded data

| Stage | Base | Data | Output |
|---|---|---|---|
| Reasoning SFT | coherence-rl (step 780) | `sft_train_trimmed.jsonl` (817 examples) | `reasoning_sft_v3` |
| Discovery RL | v3 SFT | `rl_problems_expanded` (800 problems) | `discovery_rl_v3` |

- **Judge:** gpt-4.1-mini
- **Max tokens:** 1024 (doubled from v2)
- **Key change:** Started from coherence RL'd model instead of raw base. Expanded RL problem set. Trimmed SFT data.

### v4 — Raw base ablation (best result)

| Stage | Base | Data | Output | Eval Score |
|---|---|---|---|---|
| Reasoning SFT | d34-22btok (raw base) | `sft_train_trimmed.jsonl` (817 examples) | `reasoning_sft_v4` | 0.12 |
| Discovery RL | v4 SFT | `rl_problems_expanded` (800 problems) | `discovery_rl_v4` | **0.29** |

- **Judge:** Claude Sonnet
- **Max tokens:** 1024
- **Key change:** Ablation — skipped all prior SFT/RL, started from raw pretrained base. Switched from OpenAI to Claude judge.
- **Best overall model.** Scores 1/5 on UV catastrophe, photoelectric effect, and GR elevator.

### v5 — Physics-pretrained base + contradiction RL (regression)

| Stage | Base | Data | Output | Eval Score |
|---|---|---|---|---|
| Reasoning SFT | d34-physics-sft (physics CLM) | `sft_train_trimmed.jsonl` (817 examples) | `reasoning_sft_v5` | — |
| Discovery RL | v5 SFT | `contradiction_problems` (284 problems) | `discovery_rl_v5` | worse than v4 |

- **Judge:** Claude Sonnet with contradiction rubric (`--judge-style contradiction`)
- **Max tokens:** 1024
- **Key changes:** Physics-pretrained base (instead of raw). New contradiction-resolution RL problems + judge.
- **Two things changed at once** — base model AND RL data/judge, making it hard to isolate what caused the regression.

## Eval Notes

- All eval scores are averages on a 0–5 scale across 8 physics tasks × 3 samples.
- Max individual sample score across all models is 1/5. No model has scored above 1 on any single sample.
- Eval tasks: UV catastrophe, photoelectric effect, frozen light, approaching c, train/lightning, Michelson-Morley, elevator/light, free-fall equivalence.
- See `EVAL.json` for full rubric and `results/physics_eval/` for generations and judge scores.

## Key Findings

1. **v4 is the best model** — starting from raw pretrained base outperformed all other base choices (coherence RL, physics CLM).
2. **Coherence RL had no effect** — `coherence-rl` and `d34-rl` produced identical outputs; the coherence reward signal was too weak to move the model.
3. **v5 regressed** — changed both the base (physics CLM) and the RL data (contradiction problems) simultaneously. The 284 contradiction problems are solvable within pre-1900 physics, creating a domain mismatch with the eval (which requires post-1900 breakthroughs).
4. **v5 reward hacks** — the model learned to produce well-formatted `<think>...\answer{}` outputs while the actual content degraded across RL steps (s030 had more correct concepts than s104).
5. **All models fail the same core way** — they can sometimes identify that a classical assumption is wrong but cannot explain *why* or propose a coherent alternative. They are stuck inside the pre-1900 knowledge prior.
