"""
Physics reasoning evaluation pipeline for GPT-1900 checkpoints.

Downloads each checkpoint from HuggingFace, generates responses to 8 physics
reasoning tasks, then uses Claude API to judge them on a 0-5 rubric.

Usage:
    python -m scripts.physics_eval --output-dir results/physics_eval
    python -m scripts.physics_eval --only discovery-rl-v4,coherence-rl
    python -m scripts.physics_eval --skip-judge
    python -m scripts.physics_eval --only d34-22btok --skip-judge --num-samples 1
"""

import argparse
import gc
import json
import os
import shutil
import time
from dataclasses import dataclass

import torch
from huggingface_hub import snapshot_download

from nanochat.checkpoint_manager import load_checkpoint, _patch_missing_config_keys, _patch_missing_keys
from nanochat.engine import Engine
from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import RustBPETokenizer


# ---------------------------------------------------------------------------
# Checkpoint registry
# ---------------------------------------------------------------------------

@dataclass
class Checkpoint:
    name: str
    repo: str
    step: int
    mode: str  # "completion" or "chat"


CHECKPOINTS = [
    Checkpoint("d26-8btok",        "mhla/gpt1900-d26-8btok",              7226,  "completion"),
    Checkpoint("d26-22btok",       "mhla/gpt1900-d26-22btok",             17517, "completion"),
    Checkpoint("d34-8b-subset",    "mhla/gpt1900-d34-8b-subset",          10507, "completion"),
    Checkpoint("d34-22btok",       "mhla/gpt1900-d34-22btok",             10507, "completion"),
    Checkpoint("d34-physics-sft",  "mhla/gpt1900-d34-physics-sft",        404,   "completion"),
    Checkpoint("d34-sft-period",   "mhla/gpt1900-d34-sft-period",         20,    "chat"),
    Checkpoint("d34-sft-modern",   "mhla/gpt1900-d34-sft-modern",         16,    "chat"),
    Checkpoint("d34-rl",           "mhla/gpt1900-d34-rl",                 780,   "chat"),
    Checkpoint("coherence-rl",     "mhla/gpt1900-d34-coherence-rl",       780,   "chat"),
    Checkpoint("reasoning-sft-v4", "mhla/gpt1900-d34-reasoning-sft-v4",   99,    "chat"),
    Checkpoint("discovery-rl-v4",  "mhla/gpt1900-d34-discovery-rl-v4",    180,   "chat"),
    Checkpoint("discovery-rl-v5-s030", "local", 30,  "chat"),
    Checkpoint("discovery-rl-v5-s060", "local", 60,  "chat"),
    Checkpoint("discovery-rl-v5-s090", "local", 90,  "chat"),
    Checkpoint("discovery-rl-v5-s104", "local", 104, "chat"),
    Checkpoint("discovery-rl-v6-s035", "local", 35,  "completion"),
    Checkpoint("discovery-rl-v6-s070", "local", 70,  "completion"),
    Checkpoint("discovery-rl-v6-s105", "local", 105, "completion"),
    Checkpoint("discovery-rl-v6-s140", "local", 140, "completion"),
    Checkpoint("discovery-rl-v6-s175", "local", 175, "completion"),
    Checkpoint("discovery-rl-v6-s210", "local", 210, "completion"),
    Checkpoint("discovery-rl-v6-s245", "local", 245, "completion"),
    Checkpoint("discovery-rl-v6-s280", "local", 280, "completion"),
    Checkpoint("discovery-rl-v6-s315", "local", 315, "completion"),
    Checkpoint("discovery-rl-v6-s350", "local", 350, "completion"),
    Checkpoint("discovery-rl-v6-s385", "local", 385, "completion"),
    Checkpoint("discovery-rl-v6-s419", "local", 419, "completion"),
]


# ---------------------------------------------------------------------------
# Model loading (avoids get_tokenizer / NANOCHAT_BASE_DIR dependency)
# ---------------------------------------------------------------------------

def build_model_only(checkpoint_dir, step, device):
    """Load model from checkpoint without loading the tokenizer."""
    model_data, _, meta_data = load_checkpoint(checkpoint_dir, step, device, load_optimizer=False)
    if device.type in {"cpu", "mps"}:
        model_data = {
            k: v.float() if v.dtype == torch.bfloat16 else v
            for k, v in model_data.items()
        }
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
# HuggingFace download
# ---------------------------------------------------------------------------

def download_checkpoint(ckpt: Checkpoint, cache_dir: str) -> str:
    """Download model + meta from HF. Returns local directory path."""
    step_str = f"{ckpt.step:06d}"
    local_dir = os.path.join(cache_dir, ckpt.name)
    if os.path.exists(os.path.join(local_dir, f"model_{step_str}.pt")):
        print(f"  [cached] {ckpt.name}")
        return local_dir
    if ckpt.repo == "local":
        raise FileNotFoundError(f"Local checkpoint not found: {local_dir}/model_{step_str}.pt")
    print(f"  Downloading {ckpt.repo} step={ckpt.step} ...")
    snapshot_download(
        repo_id=ckpt.repo,
        allow_patterns=[f"model_{step_str}.pt", f"meta_{step_str}.json", "tokenizer/*"],
        local_dir=local_dir,
    )
    return local_dir


def download_tokenizer(cache_dir: str) -> RustBPETokenizer:
    """Download tokenizer from first repo and load it."""
    first = CHECKPOINTS[0]
    local_dir = os.path.join(cache_dir, first.name)
    tok_dir = os.path.join(local_dir, "tokenizer")
    if not os.path.exists(tok_dir):
        print(f"  Downloading tokenizer from {first.repo} ...")
        snapshot_download(
            repo_id=first.repo,
            allow_patterns=["tokenizer/*"],
            local_dir=local_dir,
        )
    return RustBPETokenizer.from_directory(tok_dir)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def make_prompt_text(task: dict) -> str:
    """Join prompt_lines into a single string."""
    return "\n".join(task["prompt_lines"])


def generate_for_checkpoint(
    ckpt: Checkpoint,
    tasks: list[dict],
    tokenizer: RustBPETokenizer,
    cache_dir: str,
    device: torch.device,
    num_samples: int = 3,
    max_tokens: int = 1024,
    temperature: float = 0.7,
    top_k: int = 50,
) -> dict:
    """Generate responses for all tasks with one checkpoint. Returns dict of task_id -> completions."""
    local_dir = download_checkpoint(ckpt, cache_dir)
    model = build_model_only(local_dir, ckpt.step, device)
    engine = Engine(model, tokenizer)
    bos = tokenizer.get_bos_token_id()

    results = {}
    for task in tasks:
        prompt_text = make_prompt_text(task)

        if ckpt.mode == "completion":
            tokens = tokenizer.encode(prompt_text, prepend=bos)
        else:  # chat
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
            # Strip the prompt prefix to get only the generated part
            gen_tokens = seq[prompt_len:]
            completions.append(tokenizer.decode(gen_tokens))

        results[task["id"]] = {
            "prompt": prompt_text,
            "mode": ckpt.mode,
            "completions": completions,
        }
        print(f"    {task['id']}: {len(completions)} samples, "
              f"avg {sum(len(c) for c in completions) / len(completions):.0f} chars")

    # Free GPU memory
    del model, engine
    torch.cuda.empty_cache()
    gc.collect()

    return results


# ---------------------------------------------------------------------------
# Claude judge
# ---------------------------------------------------------------------------

def build_judge_messages(eval_config: dict, task: dict, response: str) -> list[dict]:
    """Build the messages for one Claude judge call."""
    checklist = "\n".join(f"- {item}" for item in eval_config["shared_judge_checklist"])
    rules = "\n".join(f"- {item}" for item in eval_config["shared_judge_rules"])
    system_text = (
        f"{eval_config['shared_judge_prompt']}\n\n"
        f"Checklist:\n{checklist}\n\n"
        f"Rules:\n{rules}"
    )

    expected = "\n".join(f"- {c}" for c in task["expected_core_concepts"])
    rubric_lines = "\n".join(f"  {k}: {v}" for k, v in task["scoring_rubric"].items())
    schema = json.dumps(eval_config["judge_output_schema"], indent=2)

    user_text = (
        f"## Original Prompt\n{make_prompt_text(task)}\n\n"
        f"## Judge Focus\n{task['judge_focus']}\n\n"
        f"## Expected Core Concepts\n{expected}\n\n"
        f"## Scoring Rubric\n{rubric_lines}\n\n"
        f"## Model Response\n{response}\n\n"
        f"## Instructions\n"
        f"Score the model response above. Return ONLY a JSON object matching this schema:\n{schema}"
    )

    return [
        {"role": "user", "content": user_text},
    ], system_text


def judge_all(eval_config: dict, tasks: list[dict], generations: dict, delay: float = 0.3) -> dict:
    """Run Claude judge on all generations. Returns judged results."""
    import anthropic

    client = anthropic.Anthropic()
    tasks_by_id = {t["id"]: t for t in tasks}
    judged = {}

    total = sum(
        len(gen_data["completions"])
        for ckpt_gens in generations.values()
        for gen_data in ckpt_gens.values()
    )
    done = 0

    for ckpt_name, ckpt_gens in generations.items():
        judged[ckpt_name] = {}
        for task_id, gen_data in ckpt_gens.items():
            task = tasks_by_id[task_id]
            sample_results = []
            for i, completion in enumerate(gen_data["completions"]):
                messages, system_text = build_judge_messages(eval_config, task, completion)
                try:
                    resp = client.messages.create(
                        model="claude-sonnet-4-20250514",
                        max_tokens=512,
                        system=system_text,
                        messages=messages,
                    )
                    raw = resp.content[0].text
                    # Try to extract JSON from the response
                    raw = raw.strip()
                    if raw.startswith("```"):
                        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
                    judge_output = json.loads(raw)
                except Exception as e:
                    print(f"    [WARN] Judge failed for {ckpt_name}/{task_id}/sample{i}: {e}")
                    judge_output = {"score": 0, "rationale": f"Judge error: {e}", "matched_concepts": [], "missing_concepts": []}

                sample_results.append({
                    "completion": completion,
                    "judge": judge_output,
                })
                done += 1
                if done % 10 == 0 or done == total:
                    print(f"    Judged {done}/{total}")
                time.sleep(delay)

            judged[ckpt_name][task_id] = {
                "prompt": gen_data["prompt"],
                "mode": gen_data["mode"],
                "samples": sample_results,
            }

    return judged


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

# Short labels for the summary table
TASK_SHORT_NAMES = {
    "uv_catastrophe_main": "UV",
    "photoelectric_effect_main": "Photo",
    "sr_frozen_light": "Frozen",
    "sr_approaching_c": "ApprC",
    "sr_train_lightning": "Train",
    "sr_michelson_morley": "MM",
    "gr_main_elevator_light": "Elev",
    "gr_free_fall_equivalence": "Fall",
}


def make_summary(judged: dict, tasks: list[dict]) -> str:
    """Build a human-readable score table."""
    task_ids = [t["id"] for t in tasks]
    headers = [TASK_SHORT_NAMES.get(tid, tid[:6]) for tid in task_ids]

    # Column widths
    name_w = max(len(n) for n in judged) + 2
    col_w = max(len(h) for h in headers) + 2

    # Header row
    lines = []
    header_line = " " * name_w + "".join(h.rjust(col_w) for h in headers) + "  | Mean"
    lines.append(header_line)

    for ckpt_name in judged:
        scores_row = []
        for tid in task_ids:
            task_data = judged[ckpt_name].get(tid)
            if task_data:
                sample_scores = [s["judge"]["score"] for s in task_data["samples"]]
                avg = sum(sample_scores) / len(sample_scores)
            else:
                avg = 0.0
            scores_row.append(avg)

        mean = sum(scores_row) / len(scores_row) if scores_row else 0.0
        row = ckpt_name.ljust(name_w) + "".join(f"{s:.1f}".rjust(col_w) for s in scores_row) + f"  | {mean:.2f}"
        lines.append(row)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Physics reasoning evaluation for GPT-1900 checkpoints")
    parser.add_argument("--output-dir", default="results/physics_eval", help="Output directory")
    parser.add_argument("--only", default=None, help="Comma-separated list of checkpoint names to evaluate")
    parser.add_argument("--skip-judge", action="store_true", help="Skip the Claude judge phase")
    parser.add_argument("--num-samples", type=int, default=3, help="Number of samples per task")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Max tokens per generation")
    parser.add_argument("--cache-dir", default=None, help="HF download cache dir (default: <output-dir>/hf_cache)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    cache_dir = args.cache_dir or os.path.join(args.output_dir, "hf_cache")
    os.makedirs(cache_dir, exist_ok=True)

    # Load eval config
    eval_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "EVAL.json")
    with open(eval_path) as f:
        eval_config = json.load(f)
    tasks = eval_config["tasks"]

    # Filter checkpoints
    checkpoints = CHECKPOINTS
    if args.only:
        names = set(args.only.split(","))
        checkpoints = [c for c in checkpoints if c.name in names]
        missing = names - {c.name for c in checkpoints}
        if missing:
            print(f"WARNING: unknown checkpoint names: {missing}")
    print(f"Evaluating {len(checkpoints)} checkpoints on {len(tasks)} tasks, {args.num_samples} samples each\n")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Download tokenizer once
    print("Loading tokenizer ...")
    tokenizer = download_tokenizer(cache_dir)
    print(f"  Vocab size: {tokenizer.get_vocab_size()}\n")

    # Check for existing generations (crash resilience)
    gen_path = os.path.join(args.output_dir, "generations.json")
    if os.path.exists(gen_path):
        with open(gen_path) as f:
            all_generations = json.load(f)
        print(f"Loaded existing generations ({len(all_generations)} checkpoints)\n")
    else:
        all_generations = {}

    # Generation phase
    print("=" * 60)
    print("GENERATION PHASE")
    print("=" * 60)
    for ckpt in checkpoints:
        if ckpt.name in all_generations:
            print(f"\n[skip] {ckpt.name} (already generated)")
            continue
        print(f"\n[{ckpt.name}] {ckpt.repo} step={ckpt.step} mode={ckpt.mode}")
        ckpt_results = generate_for_checkpoint(
            ckpt, tasks, tokenizer, cache_dir, device,
            num_samples=args.num_samples,
            max_tokens=args.max_tokens,
        )
        all_generations[ckpt.name] = ckpt_results
        # Save after each checkpoint for crash resilience
        with open(gen_path, "w") as f:
            json.dump(all_generations, f, indent=2)
        print(f"  Saved generations to {gen_path}")
        # Free disk space by removing downloaded checkpoint (skip symlinks / local)
        ckpt_cache = os.path.join(cache_dir, ckpt.name)
        if os.path.isdir(ckpt_cache) and not os.path.islink(ckpt_cache):
            shutil.rmtree(ckpt_cache)
            print(f"  Removed cache {ckpt_cache}")

    print(f"\nGeneration complete: {len(all_generations)} checkpoints\n")

    # Judge phase
    if args.skip_judge:
        print("Skipping judge phase (--skip-judge)")
        return

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set. Use --skip-judge to skip judging.")
        return

    print("=" * 60)
    print("JUDGE PHASE")
    print("=" * 60)
    judged = judge_all(eval_config, tasks, all_generations)

    # Save judged results
    judged_path = os.path.join(args.output_dir, "results_judged.json")
    with open(judged_path, "w") as f:
        json.dump(judged, f, indent=2)
    print(f"\nSaved judged results to {judged_path}")

    # Summary
    summary = make_summary(judged, tasks)
    print(f"\n{'=' * 60}")
    print("RESULTS")
    print(f"{'=' * 60}")
    print(summary)

    summary_path = os.path.join(args.output_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(summary + "\n")
    print(f"\nSaved summary to {summary_path}")


if __name__ == "__main__":
    main()
