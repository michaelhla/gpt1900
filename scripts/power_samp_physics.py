"""
Power sampling (p^α) for physics reasoning evaluation.

Implements MCMC-based power sampling from "Reasoning with Sampling: Your Base
Model is Smarter Than You Think" (Karan & Du, 2025), adapted for nanochat.

The idea: sample from p(x)^α (α > 1) to concentrate probability mass on
higher-likelihood sequences. This lets a base model exhibit stronger reasoning
without any fine-tuning, by finding sequences the model itself assigns high
probability to but wouldn't reach via naive sampling.

Usage:
    # Power sampling with α=2 on physics-CLM checkpoint
    python -m scripts.power_samp_physics --alpha 2.0

    # Higher α (sharper), more MCMC steps
    python -m scripts.power_samp_physics --alpha 3.0 --mcmc-steps 5

    # Specify checkpoint explicitly
    python -m scripts.power_samp_physics --model-tag d34 --step 40572 --alpha 2.0

    # Compare against naive sampling baseline
    python -m scripts.power_samp_physics --alpha 2.0 --baseline

    # Use a HuggingFace checkpoint
    python -m scripts.power_samp_physics --repo mhla/gpt1900-d34-22btok --step 10507 --alpha 2.0
"""

import argparse
import gc
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from contextlib import nullcontext
from tqdm import tqdm

from nanochat.checkpoint_manager import load_model
from nanochat.common import compute_init, autodetect_device_type
from nanochat.engine import KVCache, sample_next_token

# Autocast context for cuda inference
_autocast_ctx = None

def get_autocast():
    global _autocast_ctx
    if _autocast_ctx is None:
        return nullcontext()
    return _autocast_ctx


# ---------------------------------------------------------------------------
# Core: autoregressive generation with log-prob tracking
# ---------------------------------------------------------------------------

@torch.inference_mode()
def naive_temp_generate(model, prefix, temp, max_new_tokens, device, top_k=None,
                        rep_penalty=1.0, rep_window=64):
    """
    Autoregressive generation at temperature `temp`, tracking both:
      - log_probs_norm:   log q(x_t)  (proposal distribution, temp-scaled)
      - log_probs_unnorm: α * log p(x_t)  (target distribution, sharpened)

    rep_penalty: repetition penalty (>1.0 penalizes repeats, 1.0 = no penalty)
    rep_window: how many recent tokens to check for repeats

    Returns (tokens, log_probs_norm, log_probs_unnorm).
    """
    alpha = 1.0 / temp
    m = model.config
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    rng = torch.Generator(device=device)
    rng.manual_seed(random.randint(0, 2**31))

    gen = list(prefix)
    log_probs_norm = []
    log_probs_unnorm = []

    # Prefill the prefix
    kv_cache = KVCache(
        batch_size=1,
        num_heads=m.n_kv_head,
        seq_len=min(len(prefix) + max_new_tokens, m.sequence_len),
        head_dim=m.n_embd // m.n_head,
        num_layers=m.n_layer,
        device=device,
        dtype=dtype,
    )
    ids = torch.tensor([prefix], dtype=torch.long, device=device)
    with get_autocast():
        logits = model.forward(ids, kv_cache=kv_cache)  # (1, T, vocab)
    logits = logits[:, -1, :]  # (1, vocab)

    for _ in range(max_new_tokens):
        logits_f = logits[0].float()

        # Log probs under ORIGINAL model (before any penalty) for MH ratio
        log_p = F.log_softmax(logits_f, dim=-1)

        # Apply repetition penalty to logits before temp scaling
        if rep_penalty > 1.0 and len(gen) > len(prefix):
            recent = set(gen[-rep_window:])
            for tok_id in recent:
                if logits_f[tok_id] > 0:
                    logits_f[tok_id] /= rep_penalty
                else:
                    logits_f[tok_id] *= rep_penalty

            # N-gram blocking: if generating this token would complete a 4-gram
            # that already appeared, apply a heavy penalty
            if len(gen) >= 3:
                trigram = tuple(gen[-3:])
                # Collect all tokens that have followed this trigram before
                seen_continuations = set()
                for j in range(len(prefix), len(gen) - 3):
                    if tuple(gen[j:j+3]) == trigram:
                        seen_continuations.add(gen[j+3])
                for tok_id in seen_continuations:
                    logits_f[tok_id] -= 10.0  # strong suppression

        log_q = F.log_softmax(logits_f / temp, dim=-1)

        # Sample from proposal distribution (with penalty baked in)
        penalized_logits = logits_f.unsqueeze(0)
        next_id = sample_next_token(penalized_logits, rng, temperature=temp, top_k=top_k)
        token = next_id[0, 0].item()

        log_probs_norm.append(log_q[token].item())
        log_probs_unnorm.append((alpha * log_p[token]).item())

        gen.append(token)

        # Next step
        with get_autocast():
            logits = model.forward(next_id, kv_cache=kv_cache)[:, -1, :]

    return gen, log_probs_norm, log_probs_unnorm


# ---------------------------------------------------------------------------
# Core: MCMC power sampling
# ---------------------------------------------------------------------------

def mcmc_power_samp(model, prefix, alpha, mcmc_steps, max_new_tokens,
                    num_blocks=8, device=None, top_k=None,
                    rep_penalty=1.0):
    """
    MCMC power sampling from p^α.

    Generates in blocks, running Metropolis-Hastings refinement after each.
    The proposal regenerates a suffix from a random position.

    Returns (tokens, acceptance_ratio).
    """
    temp = 1.0 / alpha
    c = len(prefix)

    if max_new_tokens % num_blocks != 0:
        # Round down to nearest multiple
        max_new_tokens = (max_new_tokens // num_blocks) * num_blocks
    jump_size = max_new_tokens // num_blocks

    print(f"  Power sampling: α={alpha:.1f} (temp={temp:.3f}), "
          f"{num_blocks} blocks × {jump_size} tokens, "
          f"{mcmc_steps} MCMC steps/block"
          + (f", rep_penalty={rep_penalty}" if rep_penalty > 1.0 else ""))

    # Generate first block
    gen, log_probs_norm, log_probs_unnorm = naive_temp_generate(
        model, prefix, temp, jump_size, device, top_k=top_k,
        rep_penalty=rep_penalty,
    )

    attempts = 0
    acceptances = 0

    for block_idx in tqdm(range(1, num_blocks), desc="  blocks"):
        # Extend with another block
        gen_ext, lp_norm_ext, lp_unnorm_ext = naive_temp_generate(
            model, gen, temp, jump_size, device, top_k=top_k,
            rep_penalty=rep_penalty,
        )
        gen = gen_ext
        log_probs_norm.extend(lp_norm_ext)
        log_probs_unnorm.extend(lp_unnorm_ext)

        # MCMC refinement
        for step in range(mcmc_steps):
            attempts += 1
            t = len(gen)

            # Pick random position in generated region
            idx = random.randint(c, t - 1)
            suffix_len = t - idx

            # Generate proposal suffix
            prop, lp_norm_prop, lp_unnorm_prop = naive_temp_generate(
                model, gen[:idx], temp, suffix_len, device, top_k=top_k,
                rep_penalty=rep_penalty,
            )

            # Log probs for current suffix (from idx onward)
            start = idx - c
            end = start + suffix_len

            log_prob_cur = log_probs_norm[start:end]
            target_log_prob_cur = log_probs_unnorm[start:end]

            # Metropolis-Hastings acceptance ratio
            log_r = (sum(lp_unnorm_prop) + sum(log_prob_cur)
                     - sum(target_log_prob_cur) - sum(lp_norm_prop))

            # Accept/reject (clip log_r for numerical stability)
            if np.random.rand() < np.exp(min(log_r, 0)):
                acceptances += 1
                gen = prop
                log_probs_norm[start:] = lp_norm_prop
                log_probs_unnorm[start:] = lp_unnorm_prop

    acceptance_ratio = acceptances / max(attempts, 1)
    return gen, acceptance_ratio


# ---------------------------------------------------------------------------
# Naive baseline sampling (standard temperature)
# ---------------------------------------------------------------------------

@torch.inference_mode()
def naive_generate(model, prefix, temperature, max_new_tokens, device, top_k=50):
    """Standard autoregressive generation (no MCMC). Returns token list."""
    m = model.config
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    rng = torch.Generator(device=device)
    rng.manual_seed(random.randint(0, 2**31))
    bos = 0  # will be overridden

    gen = list(prefix)
    kv_cache = KVCache(
        batch_size=1,
        num_heads=m.n_kv_head,
        seq_len=min(len(prefix) + max_new_tokens, m.sequence_len),
        head_dim=m.n_embd // m.n_head,
        num_layers=m.n_layer,
        device=device,
        dtype=dtype,
    )
    ids = torch.tensor([prefix], dtype=torch.long, device=device)
    with get_autocast():
        logits = model.forward(ids, kv_cache=kv_cache)[:, -1, :]

    for _ in range(max_new_tokens):
        next_id = sample_next_token(logits, rng, temperature=temperature, top_k=top_k)
        gen.append(next_id[0, 0].item())
        with get_autocast():
            logits = model.forward(next_id, kv_cache=kv_cache)[:, -1, :]

    return gen


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Power sampling (p^α) for physics reasoning evaluation"
    )
    # Model selection
    parser.add_argument("--model-tag", type=str, default=None,
                        help="Model tag for load_model (e.g. d34)")
    parser.add_argument("--step", type=int, default=None,
                        help="Checkpoint step (default: latest)")
    parser.add_argument("--repo", type=str, default=None,
                        help="HuggingFace repo ID (alternative to --model-tag)")
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                        help="Direct path to checkpoint parent dir (default: physics_clm_checkpoints)")

    # Power sampling params
    parser.add_argument("--alpha", type=float, default=2.0,
                        help="Sharpening exponent α (default: 2.0, higher = sharper)")
    parser.add_argument("--mcmc-steps", type=int, default=3,
                        help="MCMC refinement steps per block (default: 3)")
    parser.add_argument("--num-blocks", type=int, default=8,
                        help="Number of generation blocks (default: 8)")
    parser.add_argument("--top-k", type=int, default=None,
                        help="Top-k filtering during sampling (default: None)")
    parser.add_argument("--rep-penalty", type=float, default=1.2,
                        help="Repetition penalty (default: 1.2, 1.0 = off)")

    # Generation params
    parser.add_argument("--max-tokens", type=int, default=1024,
                        help="Max tokens to generate (default: 1024)")
    parser.add_argument("--num-samples", type=int, default=3,
                        help="Samples per task (default: 3)")
    parser.add_argument("--baseline", action="store_true",
                        help="Also run naive sampling at temp=0.7 for comparison")
    parser.add_argument("--baseline-temp", type=float, default=0.7,
                        help="Temperature for baseline sampling (default: 0.7)")

    # Output
    parser.add_argument("--output-dir", type=str, default="results/power_samp",
                        help="Output directory (default: results/power_samp)")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Single prompt instead of EVAL.json tasks (for quick testing)")

    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-judge", action="store_true",
                        help="Skip the Claude judge phase")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # Init device
    device_type = autodetect_device_type()
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)

    # Set up autocast for cuda
    global _autocast_ctx
    if device_type == "cuda":
        _autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16)
    else:
        _autocast_ctx = nullcontext()

    # Load model
    if args.repo:
        from nanochat.checkpoint_manager import load_checkpoint, _patch_missing_config_keys, _patch_missing_keys
        from nanochat.gpt import GPT, GPTConfig
        from nanochat.tokenizer import RustBPETokenizer
        from huggingface_hub import snapshot_download

        step_str = f"{args.step:06d}"
        local_dir = os.path.join(args.output_dir, "hf_cache", args.repo.split("/")[-1])
        os.makedirs(local_dir, exist_ok=True)

        if not os.path.exists(os.path.join(local_dir, f"model_{step_str}.pt")):
            print(f"Downloading {args.repo} step={args.step} ...")
            snapshot_download(
                repo_id=args.repo,
                allow_patterns=[f"model_{step_str}.pt", f"meta_{step_str}.json", "tokenizer/*"],
                local_dir=local_dir,
            )

        model_data, _, meta_data = load_checkpoint(local_dir, args.step, device, load_optimizer=False)
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

        tok_dir = os.path.join(local_dir, "tokenizer")
        tokenizer = RustBPETokenizer.from_directory(tok_dir)
    else:
        from nanochat.checkpoint_manager import load_model_from_dir, get_base_dir
        source_dir = args.checkpoint_dir if args.checkpoint_dir else os.path.join(get_base_dir(), "physics_clm_checkpoints")
        model, tokenizer, meta = load_model_from_dir(
            source_dir, device, phase="eval",
            model_tag=args.model_tag, step=args.step
        )

    bos = tokenizer.get_bos_token_id()
    config = model.config
    print(f"\nModel: d{config.n_layer} | seq_len={config.sequence_len} | {device_type}")
    print(f"Power sampling: α={args.alpha}, mcmc_steps={args.mcmc_steps}, "
          f"blocks={args.num_blocks}, max_tokens={args.max_tokens}")
    print("=" * 70)

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Single prompt mode ----
    if args.prompt:
        prefix = tokenizer.encode(args.prompt, prepend=bos)
        print(f"\nPrompt ({len(prefix)} tokens): {args.prompt}\n")

        # Power sampling
        print(f"\n--- Power sampling (α={args.alpha}) ---")
        gen_tokens, acc_ratio = mcmc_power_samp(
            model, prefix, alpha=args.alpha,
            mcmc_steps=args.mcmc_steps,
            max_new_tokens=args.max_tokens,
            num_blocks=args.num_blocks,
            device=device,
            top_k=args.top_k,
            rep_penalty=args.rep_penalty,
        )
        gen_text = tokenizer.decode(gen_tokens[len(prefix):])
        print(f"\n  Acceptance ratio: {acc_ratio:.3f}")
        print(f"  Output:\n{args.prompt}{gen_text}\n")

        # Baseline
        if args.baseline:
            print(f"\n--- Naive sampling (temp={args.baseline_temp}) ---")
            baseline_tokens = naive_generate(
                model, prefix, args.baseline_temp, args.max_tokens, device
            )
            baseline_text = tokenizer.decode(baseline_tokens[len(prefix):])
            print(f"  Output:\n{args.prompt}{baseline_text}\n")

        return

    # ---- Physics eval mode ----
    eval_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "EVAL.json")
    with open(eval_path) as f:
        eval_config = json.load(f)
    tasks = eval_config["tasks"]
    print(f"\nRunning on {len(tasks)} physics tasks, {args.num_samples} samples each\n")

    all_results = {}

    for task in tasks:
        task_id = task["id"]
        prompt_text = "\n".join(task["prompt_lines"])
        prefix = tokenizer.encode(prompt_text, prepend=bos)

        print(f"\n{'='*70}")
        print(f"Task: {task_id} ({len(prefix)} prompt tokens)")
        print(f"{'='*70}")

        task_results = {
            "prompt": prompt_text,
            "mode": "completion",
            "power_sampling": {"alpha": args.alpha, "completions": [], "acceptance_ratios": []},
        }

        # Power sampling
        for sample_idx in range(args.num_samples):
            print(f"\n  Sample {sample_idx + 1}/{args.num_samples} (power α={args.alpha})")
            t0 = time.time()

            gen_tokens, acc_ratio = mcmc_power_samp(
                model, prefix, alpha=args.alpha,
                mcmc_steps=args.mcmc_steps,
                max_new_tokens=args.max_tokens,
                num_blocks=args.num_blocks,
                device=device,
                top_k=args.top_k,
            )
            elapsed = time.time() - t0
            gen_text = tokenizer.decode(gen_tokens[len(prefix):])

            task_results["power_sampling"]["completions"].append(gen_text)
            task_results["power_sampling"]["acceptance_ratios"].append(acc_ratio)

            print(f"  Acceptance: {acc_ratio:.3f} | {elapsed:.1f}s | {len(gen_text)} chars")
            # Print first 300 chars as preview
            preview = gen_text[:300].replace("\n", " ")
            print(f"  Preview: {preview}...")

        # Baseline sampling
        if args.baseline:
            task_results["baseline"] = {"temperature": args.baseline_temp, "completions": []}
            for sample_idx in range(args.num_samples):
                print(f"\n  Sample {sample_idx + 1}/{args.num_samples} (baseline temp={args.baseline_temp})")
                t0 = time.time()
                baseline_tokens = naive_generate(
                    model, prefix, args.baseline_temp, args.max_tokens, device
                )
                elapsed = time.time() - t0
                baseline_text = tokenizer.decode(baseline_tokens[len(prefix):])
                task_results["baseline"]["completions"].append(baseline_text)
                print(f"  {elapsed:.1f}s | {len(baseline_text)} chars")
                preview = baseline_text[:300].replace("\n", " ")
                print(f"  Preview: {preview}...")

        all_results[task_id] = task_results

    # Save results
    tag = f"alpha{args.alpha}_mcmc{args.mcmc_steps}_blocks{args.num_blocks}"
    output_path = os.path.join(args.output_dir, f"power_samp_{tag}.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n\nResults saved to {output_path}")

    # ---- Optional: Claude judge ----
    if not args.skip_judge and os.environ.get("ANTHROPIC_API_KEY"):
        print("\n" + "=" * 70)
        print("JUDGE PHASE")
        print("=" * 70)

        from scripts.physics_eval import build_judge_messages
        import anthropic

        client = anthropic.Anthropic()
        tasks_by_id = {t["id"]: t for t in tasks}

        for task_id, result in all_results.items():
            task = tasks_by_id[task_id]

            # Judge power sampling completions
            for label in ["power_sampling", "baseline"]:
                if label not in result:
                    continue
                for i, completion in enumerate(result[label]["completions"]):
                    messages, system_text = build_judge_messages(eval_config, task, completion)
                    try:
                        resp = client.messages.create(
                            model="claude-sonnet-4-20250514",
                            max_tokens=512,
                            system=system_text,
                            messages=messages,
                        )
                        raw = resp.content[0].text.strip()
                        if raw.startswith("```"):
                            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
                        judge_output = json.loads(raw)
                    except Exception as e:
                        print(f"  [WARN] Judge failed: {e}")
                        judge_output = {"score": 0, "rationale": f"Judge error: {e}"}

                    result[label].setdefault("judge_scores", [])
                    result[label]["judge_scores"].append(judge_output)
                    print(f"  {task_id}/{label}/sample{i}: score={judge_output.get('score', '?')}")
                    time.sleep(0.3)

        # Save judged results
        judged_path = os.path.join(args.output_dir, f"power_samp_{tag}_judged.json")
        with open(judged_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nJudged results saved to {judged_path}")

        # Print summary
        print(f"\n{'='*70}")
        print("SUMMARY")
        print(f"{'='*70}")
        for label in ["power_sampling", "baseline"]:
            scores_by_task = {}
            for task_id, result in all_results.items():
                if label in result and "judge_scores" in result[label]:
                    task_scores = [j["score"] for j in result[label]["judge_scores"]]
                    scores_by_task[task_id] = sum(task_scores) / len(task_scores)

            if scores_by_task:
                mean = sum(scores_by_task.values()) / len(scores_by_task)
                detail = "  ".join(f"{k[:6]}={v:.1f}" for k, v in scores_by_task.items())
                print(f"  {label:20s}: mean={mean:.2f}  ({detail})")

    # Cleanup
    del model
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()
