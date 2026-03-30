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

## Contradiction RL (v6–v12)

### v6 — No scaffold, coherence curriculum

| Field | Value |
|---|---|
| **Chain** | d34-physics-sft (direct, no reasoning SFT) → contradiction RL |
| **Data** | `contradiction_problems` (284) |
| **Max tokens** | 2048 |
| **Coherence** | EMA curriculum (alpha=0.05, min weight=0.1) |
| **Eval** | Peak **0.58** (s385) |

### v8 — Higher coherence weight, longer training

| Field | Value |
|---|---|
| **Chain** | d34-physics-sft → contradiction RL |
| **Data** | `contradiction_problems` (284) |
| **Max tokens** | 2048, 24 epochs |
| **Coherence** | Fixed weight 0.25 |
| **Eval** | Peak **0.62** (s315) |

### v11 — Best overall

| Field | Value |
|---|---|
| **Chain** | d34-22btok → physicssft-expanded → v3 SFT physics (safe) → contradiction RL |
| **Data** | `contradiction_problems` (284) |
| **Max tokens** | 2048, 24 epochs |
| **Coherence** | Fixed weight 0.25, logical style |
| **Eval** | Peak **1.25** (s560, s630) |
| **Best scores** | Photoelectric **4/5** (s770), Elevator **3/5** (s630), UV catastrophe **2/5** (s735), Michelson-Morley **2/5** (s455) |
| **HF Repo** | `mhla/gpt1900-d34-contradiction-rl-v11` |

### v12 — R1 distillation base

| Field | Value |
|---|---|
| **Chain** | d34-22btok → physicssft-expanded → v3 SFT physics → R1 reasoning SFT → contradiction RL |
| **Data** | `contradiction_problems` (284, v12 prompts without system prompt) |
| **Max tokens** | 2048, 24 epochs |
| **Key changes** | R1 distillation SFT before RL. No system prompt. Format reward checks `<think>` and `\answer{}` tags. |
| **HF Repo** | `mhla/gpt1900-d34-contradiction-rl-v12` |

## Alternative RL Pipelines

### 1905 RL

| Field | Value |
|---|---|
| **Chain** | d34-1905 → physics CLM (pre+post-1900 texts, 10 epochs) → v3 SFT safe → contradiction RL (v11-style) |
| **Key difference** | Base model trained on pre-1905 corpus. Physics CLM includes post-1900 texts (Rutherford, Thomson, Lorentz, Planck). |
| **Run script** | `run_1905_rl.sh` |

### 1900 RL

| Field | Value |
|---|---|
| **Chain** | d34-22btok → physics CLM → v3 SFT safe → contradiction RL |
| **Run script** | `run_1900_rl.sh` |

### Math RL

| Field | Value |
|---|---|
| **Chain** | R1 reasoning SFT → Math RL |
| **Data** | GSM8K train (7,473 examples, 30%) + MATH train (7,500 examples, 70%) |
| **Rewards** | Format (0.3): `<think>` + `\answer{}` tags. Correctness (1.0): numeric comparison (GSM8K) or SymPy symbolic equivalence (MATH). |
| **Run script** | `run_math_rl.sh` |

### Intuitor RL

| Field | Value |
|---|---|
| **Chain** | d34-22btok → physicssft-expanded → Intuitor RL |
| **Reward** | Self-certainty (intrinsic reward, no gold answers or API calls for training) |
| **Data** | `intuitor_prompts_sys_train.jsonl` (1,663 problems) |
| **Run script** | `run_intuitor_rl.sh` |

### OpenThoughts SFT + Verifiable RL

| Field | Value |
|---|---|
| **SFT Chain** | d34 physicssft-expanded → OpenThoughts SFT |
| **SFT Data** | OpenThoughts3 (math + science, anachronism-filtered) |
| **RL Chain** | OpenThoughts SFT → Verifiable RL |
| **RL Data** | `combined_problems_train.jsonl` (generated + Yale physics) |
| **RL Reward** | SymPy symbolic equivalence (1.0) + format bonus (0.3) |
| **Run scripts** | `run_openthoughts_sft.sh`, `run_openthoughts_rl.sh` |

## Eval Notes

- All eval scores are averages on a 0–5 scale across 8 physics tasks × 3 samples.
- Peak individual sample score: **4/5** on photoelectric effect (v11-s770).
- Eval tasks: UV catastrophe, photoelectric effect, frozen light, approaching c, train/lightning, Michelson-Morley, elevator/light, free-fall equivalence.
- See `EVAL.json` for full rubric and `results/physics_eval/` for generations and judge scores.

## Key Findings

1. **v11 is the best model** (eval: 1.25) — chain: physicssft-expanded → v3 SFT safe → contradiction RL.
2. **Raw base outperforms prior RL** — v4 (raw base) beat v3 (coherence RL base) in early experiments.
3. **Coherence RL had no effect** — `coherence-rl` produced identical outputs to base; too weak a reward signal.
4. **v6-v8 steady improvement** — no reasoning SFT scaffold, coherence curriculum, longer training.
5. **v11 breakthrough** — v3 SFT on opinion-filtered corpus provided crucial instruction-following capability.
6. **Reward hacking** — v5 learned to produce formatted outputs while content degraded across RL steps.
7. **Peak single-sample score: 4/5** on photoelectric effect (v11-s770) — the model almost independently describes discrete energy packets.
8. **3/5 on elevator/light** (v11-s630) — derives equivalence between gravity and acceleration from first principles.
