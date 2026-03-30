"""
General reasoning eval: test whether v12 RL checkpoints show thinking behaviors
on prompts outside the physics/contradiction training domain.

Generates from R1 SFT (pre-RL baseline) and several v12 RL checkpoints
on diverse reasoning prompts (math, logic, common sense, science, everyday).

Usage:
    python -m scripts.general_reasoning_eval
    python -m scripts.general_reasoning_eval --checkpoints r1-sft,v12-s035,v12-s455
    python -m scripts.general_reasoning_eval --num-samples 1 --max-tokens 512
"""

import argparse
import gc
import json
import os
import time

import torch

from nanochat.checkpoint_manager import load_checkpoint, _patch_missing_config_keys, _patch_missing_keys
from nanochat.engine import Engine
from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import RustBPETokenizer

# ---------------------------------------------------------------------------
# Checkpoints to evaluate
# ---------------------------------------------------------------------------

BASE_DIR = "/opt/dlami/nvme/gpt1905_training"
TOKENIZER_DIR = os.path.join(BASE_DIR, "tokenizer")

CHECKPOINT_CONFIGS = {
    "r1-sft":    {"dir": "r1_reasoning_sft_checkpoints/d34",                 "step": 99},
    "v12-s035":  {"dir": "pre1900_discovery_rl_v12_checkpoints/d34",         "step": 35},
    "v12-s105":  {"dir": "pre1900_discovery_rl_v12_checkpoints/d34",         "step": 105},
    "v12-s210":  {"dir": "pre1900_discovery_rl_v12_checkpoints/d34",         "step": 210},
    "v12-s315":  {"dir": "pre1900_discovery_rl_v12_checkpoints/d34",         "step": 315},
    "v12-s455":  {"dir": "pre1900_discovery_rl_v12_checkpoints/d34",         "step": 455},
}

# ---------------------------------------------------------------------------
# Diverse reasoning prompts (outside the physics/contradiction training domain)
# ---------------------------------------------------------------------------

PROMPTS = [
    # --- Math reasoning ---
    {
        "id": "math_divisibility",
        "category": "math",
        "prompt": "Is the number 2^{10} - 1 divisible by 11? Explain your reasoning.",
    },
    {
        "id": "math_probability",
        "category": "math",
        "prompt": "You flip a fair coin 4 times. What is the probability of getting exactly 2 heads? Show your work.",
    },
    {
        "id": "math_geometry",
        "category": "math",
        "prompt": "A right triangle has legs of length 5 and 12. What is the length of the hypotenuse, and what is the area of the triangle?",
    },
    # --- Logic puzzles ---
    {
        "id": "logic_knights_knaves",
        "category": "logic",
        "prompt": "On an island, every person is either a knight (always tells the truth) or a knave (always lies). You meet two people, A and B. A says: 'We are both knaves.' What are A and B?",
    },
    {
        "id": "logic_deduction",
        "category": "logic",
        "prompt": "All roses are flowers. Some flowers fade quickly. Can we conclude that some roses fade quickly? Why or why not?",
    },
    {
        "id": "logic_sequence",
        "category": "logic",
        "prompt": "What is the next number in the sequence: 2, 6, 12, 20, 30, ...? Explain the pattern.",
    },
    # --- Common sense / everyday reasoning ---
    {
        "id": "common_sense_ice",
        "category": "common_sense",
        "prompt": "If you leave a glass of ice water on a table in a warm room, the outside of the glass becomes wet. Why does this happen?",
    },
    {
        "id": "common_sense_shadow",
        "category": "common_sense",
        "prompt": "Why is your shadow longer in the early morning than at noon?",
    },
    {
        "id": "common_sense_cooking",
        "category": "common_sense",
        "prompt": "Water boils at a lower temperature on top of a tall mountain than at sea level. Why? What practical consequence does this have for cooking?",
    },
    # --- Science (non-physics or general) ---
    {
        "id": "science_seasons",
        "category": "science",
        "prompt": "Why does Earth have seasons? Is it because Earth is closer to the Sun in summer?",
    },
    {
        "id": "science_chemistry",
        "category": "science",
        "prompt": "Iron rusts when exposed to air and moisture, but gold does not. What explains this difference?",
    },
    {
        "id": "science_biology",
        "category": "science",
        "prompt": "Why do we see our breath on cold days but not on warm days?",
    },
    # --- Abstract / causal reasoning ---
    {
        "id": "abstract_counterfactual",
        "category": "abstract",
        "prompt": "If the Earth rotated twice as fast as it currently does, what consequences might this have for day length, weather patterns, and the shape of the Earth?",
    },
    {
        "id": "abstract_analogy",
        "category": "abstract",
        "prompt": "In what ways is a biological cell like a factory? In what ways does the analogy break down?",
    },
    # --- Planning / multi-step reasoning ---
    {
        "id": "planning_river",
        "category": "planning",
        "prompt": "A farmer needs to cross a river with a wolf, a goat, and a cabbage. The boat can only carry the farmer and one item at a time. If left alone, the wolf will eat the goat, and the goat will eat the cabbage. How can the farmer get everything across safely?",
    },
    {
        "id": "planning_scheduling",
        "category": "planning",
        "prompt": "You have three tasks: Task A takes 2 hours, Task B takes 3 hours, and Task C takes 1 hour. Task B cannot start until Task A is finished. Task C can be done at any time. What is the shortest total time to complete all three tasks?",
    },
]

PROMPT_SUFFIX = "Think deeply and step by step."

# ---------------------------------------------------------------------------
# Model loading (same as physics_eval.py)
# ---------------------------------------------------------------------------

def build_model_only(checkpoint_dir, step, device):
    model_data, _, meta_data = load_checkpoint(checkpoint_dir, step, device, load_optimizer=False)
    if device.type in {"cpu", "mps"}:
        model_data = {k: v.float() if v.dtype == torch.bfloat16 else v for k, v in model_data.items()}
    model_data = {k.removeprefix("_orig_mod."): v for k, v in model_data.items()}
    model_config_kwargs = meta_data["model_config"]
    _patch_missing_config_keys(model_config_kwargs)
    model_config = GPTConfig(**model_config_kwargs)
    _patch_missing_keys(model_data, model_config)
    with torch.device("meta"):
        model = GPT(model_config)
    model.to_empty(device=device)
    model.init_weights()
    model.load_state_dict(model_data, strict=True, assign=True)
    if device.type == "cuda":
        model.bfloat16()
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_for_checkpoint(ckpt_name, ckpt_cfg, tokenizer, device, num_samples, max_tokens, temperature, top_k):
    ckpt_dir = os.path.join(BASE_DIR, ckpt_cfg["dir"])
    step = ckpt_cfg["step"]
    print(f"\n{'='*60}")
    print(f"Loading {ckpt_name} (step {step}) from {ckpt_dir}")
    print(f"{'='*60}")

    model = build_model_only(ckpt_dir, step, device)
    engine = Engine(model, tokenizer)

    results = []
    for p in PROMPTS:
        prompt_text = p["prompt"].rstrip() + "\n\n" + PROMPT_SUFFIX

        # Use chat mode (render_for_completion) matching v12 training
        conversation = {
            "messages": [
                {"role": "user", "content": prompt_text},
                {"role": "assistant", "content": ""},
            ]
        }
        tokens = tokenizer.render_for_completion(conversation)
        prompt_len = len(tokens)

        all_tokens, _ = engine.generate_batch(
            tokens,
            num_samples=num_samples,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
        )

        completions = []
        for seq in all_tokens:
            gen_tokens = seq[prompt_len:]
            completions.append(tokenizer.decode(gen_tokens))

        result = {
            "checkpoint": ckpt_name,
            "step": step,
            "prompt_id": p["id"],
            "category": p["category"],
            "prompt": prompt_text,
            "completions": completions,
        }
        results.append(result)

        # Print a preview
        avg_len = sum(len(c) for c in completions) / len(completions)
        has_think = sum(1 for c in completions if "<think>" in c)
        has_answer = sum(1 for c in completions if "\\answer{" in c)
        print(f"  {p['id']:30s} | avg {avg_len:5.0f} chars | <think>: {has_think}/{len(completions)} | \\answer: {has_answer}/{len(completions)}")

    del model, engine
    torch.cuda.empty_cache()
    gc.collect()

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="General reasoning eval for v12 checkpoints")
    parser.add_argument("--checkpoints", type=str, default=None,
                        help="Comma-separated checkpoint names (default: all)")
    parser.add_argument("--output-dir", type=str, default="results/general_reasoning_eval")
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    # Select checkpoints
    if args.checkpoints:
        ckpt_names = [s.strip() for s in args.checkpoints.split(",")]
    else:
        ckpt_names = list(CHECKPOINT_CONFIGS.keys())

    # Validate
    for name in ckpt_names:
        if name not in CHECKPOINT_CONFIGS:
            print(f"Unknown checkpoint: {name}")
            print(f"Available: {list(CHECKPOINT_CONFIGS.keys())}")
            return

    # Load tokenizer
    tokenizer = RustBPETokenizer.from_directory(TOKENIZER_DIR)
    device = torch.device(f"cuda:{args.gpu}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Generate for each checkpoint
    all_results = []
    for name in ckpt_names:
        cfg = CHECKPOINT_CONFIGS[name]
        results = generate_for_checkpoint(
            name, cfg, tokenizer, device,
            num_samples=args.num_samples,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )
        all_results.extend(results)

        # Save per-checkpoint results
        out_path = os.path.join(args.output_dir, f"{name}.jsonl")
        with open(out_path, "w") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  Saved to {out_path}")

    # Save combined results
    combined_path = os.path.join(args.output_dir, "all_results.jsonl")
    with open(combined_path, "w") as f:
        for r in all_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nAll results saved to {combined_path}")

    # Print summary table
    print(f"\n{'='*80}")
    print("SUMMARY: Thinking behavior across checkpoints")
    print(f"{'='*80}")
    ans_label = "\\answer rate"
    print(f"{'Checkpoint':20s} | {'<think> rate':>12s} | {ans_label:>12s} | {'Avg length':>10s}")
    print("-" * 60)
    for name in ckpt_names:
        ckpt_results = [r for r in all_results if r["checkpoint"] == name]
        total_completions = sum(len(r["completions"]) for r in ckpt_results)
        think_count = sum(1 for r in ckpt_results for c in r["completions"] if "<think>" in c)
        answer_count = sum(1 for r in ckpt_results for c in r["completions"] if "\\answer{" in c)
        avg_len = sum(len(c) for r in ckpt_results for c in r["completions"]) / max(total_completions, 1)
        print(f"{name:20s} | {think_count:>5d}/{total_completions:<5d} | {answer_count:>5d}/{total_completions:<5d} | {avg_len:>10.0f}")

    # Print per-category breakdown for each checkpoint
    categories = sorted(set(p["category"] for p in PROMPTS))
    print(f"\n{'='*80}")
    print("DETAIL: <think> rate by category")
    print(f"{'='*80}")
    header = f"{'Checkpoint':20s} | " + " | ".join(f"{c:>14s}" for c in categories)
    print(header)
    print("-" * len(header))
    for name in ckpt_names:
        ckpt_results = [r for r in all_results if r["checkpoint"] == name]
        parts = []
        for cat in categories:
            cat_results = [r for r in ckpt_results if r["category"] == cat]
            total = sum(len(r["completions"]) for r in cat_results)
            think = sum(1 for r in cat_results for c in r["completions"] if "<think>" in c)
            parts.append(f"{think:>5d}/{total:<5d}")
        print(f"{name:20s} | " + "    | ".join(parts))


if __name__ == "__main__":
    main()
