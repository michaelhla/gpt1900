"""
Math RL: REINFORCE/GRPO on GSM8K + MATH with deterministic rewards.

Trains the R1 SFT model on verifiable math problems to build general
math reasoning. Uses SymPy-based symbolic equivalence for MATH and
numeric comparison for GSM8K. Physics eval from EVAL.json kept as
a side monitor (Claude judge).

Based on verifiable_rl.py structure with GSM8K + MATH data.

1 GPU:
python -m scripts.math_rl

8 GPUs:
torchrun --standalone --nproc_per_node=8 -m scripts.math_rl -- --run=default
"""

import argparse
import os
import re
import asyncio
import itertools
import random
import json
import wandb
import torch
import torch.distributed as dist
from contextlib import nullcontext

import anthropic

from nanochat.common import compute_init, compute_cleanup, print0, get_base_dir, DummyWandb, autodetect_device_type
from nanochat.checkpoint_manager import save_checkpoint, load_model_from_dir
from nanochat.engine import Engine
from tasks.yale_physics import extract_answer_latex, is_symbolically_equivalent
from tasks.gsm8k import extract_answer as gsm8k_extract_answer
from tasks.math_task import MATH, extract_boxed_answer

# -----------------------------------------------------------------------------
# Format reward

FORMAT_REWARD = 0.3  # partial credit for using the correct reasoning format

def compute_format_reward(response: str) -> float:
    """Return partial format reward: 0.15 for <think>...</think>, 0.15 for \\answer{...}."""
    reward = 0.0
    if re.search(r"<think>.*?</think>", response, re.DOTALL):
        reward += FORMAT_REWARD / 2
    if re.search(r"\\answer\{", response):
        reward += FORMAT_REWARD / 2
    return reward


# -----------------------------------------------------------------------------
# Correctness rewards

def compute_gsm8k_reward(response: str, gold_answer: str) -> float:
    """Check GSM8K correctness: extract from \\answer{}, compare numerically."""
    pred = extract_answer_latex(response)
    if pred is None:
        return 0.0
    # Clean up pred for numeric comparison (remove LaTeX formatting)
    pred_clean = pred.strip().replace(",", "").replace("$", "").replace("\\", "")
    gold_clean = gold_answer.strip().replace(",", "")
    # Direct numeric comparison
    try:
        if float(pred_clean) == float(gold_clean):
            return 1.0
    except ValueError:
        pass
    # Fallback: SymPy comparison
    result = is_symbolically_equivalent(pred, gold_clean)
    return 1.0 if result is True else 0.0


def compute_math_reward(response: str, gold_answer: str) -> float:
    """Check MATH correctness: extract from \\answer{}, compare via SymPy."""
    pred = extract_answer_latex(response)
    if pred is None:
        return 0.0
    result = is_symbolically_equivalent(pred, gold_answer)
    return 1.0 if result is True else 0.0


# -----------------------------------------------------------------------------
# CLI arguments
parser = argparse.ArgumentParser(description="Math RL: GSM8K + MATH via REINFORCE/GRPO")
# Logging
parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' disables wandb logging)")
# Runtime
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
parser.add_argument("--dtype", type=str, default="bfloat16", help="float32|bfloat16")
# Model loading
parser.add_argument("--model-tag", type=str, default=None, help="model tag to load from")
parser.add_argument("--model-step", type=int, default=None, help="model step to load from")
parser.add_argument("--resume-step", type=int, default=0, help="resume training from this step")
parser.add_argument("--checkpoints-dir", type=str, default="r1_reasoning_sft_checkpoints", help="source checkpoints directory (relative to base dir)")
# Training horizon
parser.add_argument("--num-epochs", type=int, default=3, help="number of epochs over combined data")
# Batch sizes / sampling
parser.add_argument("--device-batch-size", type=int, default=2, help="max batch size per forward pass")
parser.add_argument("--examples-per-step", type=int, default=8, help="total examples per optimization step across all ranks")
parser.add_argument("--num-samples", type=int, default=4, help="number of samples per example/question")
# Generation
parser.add_argument("--max-new-tokens", type=int, default=2048, help="max tokens to generate per sample")
parser.add_argument("--temperature", type=float, default=1.0, help="sampling temperature")
parser.add_argument("--top-k", type=int, default=50, help="top-k sampling (0 = disabled)")
# Optimization
parser.add_argument("--embedding-lr", type=float, default=0.2, help="learning rate for embedding parameters (Adam)")
parser.add_argument("--unembedding-lr", type=float, default=0.004, help="learning rate for unembedding parameters (Adam)")
parser.add_argument("--matrix-lr", type=float, default=0.02, help="learning rate for matrix parameters (Muon)")
parser.add_argument("--weight-decay", type=float, default=0.0, help="weight decay for embedding/unembedding parameters (Adam)")
parser.add_argument("--init-lr-frac", type=float, default=0.05, help="initial LR as fraction of base LR")
# Data
parser.add_argument("--gsm8k-ratio", type=float, default=0.3, help="fraction of GSM8K in combined training data (rest is MATH)")
parser.add_argument("--math-levels", type=str, default=None, help="comma-separated MATH levels to include (e.g. 'Level 1,Level 2')")
parser.add_argument("--math-types", type=str, default=None, help="comma-separated MATH types to include (e.g. 'Algebra,Geometry')")
# Evaluation / checkpointing
parser.add_argument("--eval-every", type=int, default=30, help="evaluate every N steps")
parser.add_argument("--eval-gsm8k-examples", type=int, default=200, help="number of GSM8K test examples for eval")
parser.add_argument("--eval-math-examples", type=int, default=200, help="number of MATH test examples for eval")
parser.add_argument("--save-every", type=int, default=60, help="save checkpoint every N steps")
# Physics eval (EVAL.json) — uses Claude judge
parser.add_argument("--judge-model", type=str, default="claude-sonnet-4-20250514", help="Anthropic model for physics eval only")
parser.add_argument("--max-concurrent-api", type=int, default=20, help="max concurrent API requests for physics eval")
# Output
parser.add_argument("--output-dir", type=str, default="math_rl_checkpoints", help="output checkpoints directory (relative to base dir)")
args = parser.parse_args()
user_config = vars(args).copy()

# -----------------------------------------------------------------------------

# Init compute/precision
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0
ptdtype = torch.float32 if args.dtype == 'float32' else torch.bfloat16
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()

# wandb logging init
use_dummy_wandb = args.run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat-rl", name=args.run, config=user_config)

# Init model and tokenizer from checkpoints
base_dir = get_base_dir()
checkpoints_dir = os.path.join(base_dir, args.checkpoints_dir)
model, tokenizer, meta = load_model_from_dir(checkpoints_dir, device, phase="eval", model_tag=args.model_tag, step=args.model_step)
engine = Engine(model, tokenizer)

# -----------------------------------------------------------------------------
# Load training data: GSM8K + MATH combined

from datasets import load_dataset as hf_load_dataset

# GSM8K train
print0("Loading GSM8K train...")
gsm8k_ds = hf_load_dataset("openai/gsm8k", "main", split="train").shuffle(seed=42)
gsm8k_train_data = []
for row in gsm8k_ds:
    # Extract gold answer from #### marker
    gold = gsm8k_extract_answer(row["answer"])
    if gold is not None:
        gsm8k_train_data.append({
            "source": "gsm8k",
            "question": row["question"],
            "gold_answer": gold,
        })
print0(f"GSM8K train: {len(gsm8k_train_data)} examples")

# MATH train
print0("Loading MATH train...")
math_levels = args.math_levels.split(",") if args.math_levels else None
math_types = args.math_types.split(",") if args.math_types else None
math_task = MATH("train", levels=math_levels, types=math_types)
math_train_data = []
for i in range(math_task.num_examples()):
    gold = math_task.get_gold_answer(i)
    if gold is not None:
        math_train_data.append({
            "source": "math",
            "question": math_task.get_question(i),
            "gold_answer": gold,
            "metadata": math_task.get_metadata(i),
        })
print0(f"MATH train: {len(math_train_data)} examples (filtered from {math_task.num_examples()})")

# Combine with ratio control
def build_combined_dataset(gsm8k_data, math_data, gsm8k_ratio, seed=42):
    """Combine GSM8K and MATH data with controlled ratio, shuffled."""
    rng = random.Random(seed)
    # Subsample to achieve the desired ratio
    total = len(gsm8k_data) + len(math_data)
    desired_gsm8k = int(total * gsm8k_ratio)
    desired_math = total - desired_gsm8k
    # Cap to available data
    gsm8k_sample = rng.sample(gsm8k_data, min(desired_gsm8k, len(gsm8k_data)))
    math_sample = rng.sample(math_data, min(desired_math, len(math_data)))
    combined = gsm8k_sample + math_sample
    rng.shuffle(combined)
    return combined

train_data = build_combined_dataset(gsm8k_train_data, math_train_data, args.gsm8k_ratio)
actual_gsm8k_count = sum(1 for d in train_data if d["source"] == "gsm8k")
actual_math_count = sum(1 for d in train_data if d["source"] == "math")
print0(f"Combined train: {len(train_data)} examples (GSM8K: {actual_gsm8k_count}, MATH: {actual_math_count})")

num_steps = (len(train_data) // args.examples_per_step) * args.num_epochs
print0(f"Calculated number of steps: {num_steps}")

# Load eval datasets
print0("Loading eval datasets...")
gsm8k_test_ds = hf_load_dataset("openai/gsm8k", "main", split="test").shuffle(seed=42)
gsm8k_test_data = []
for row in gsm8k_test_ds:
    gold = gsm8k_extract_answer(row["answer"])
    if gold is not None:
        gsm8k_test_data.append({
            "source": "gsm8k",
            "question": row["question"],
            "gold_answer": gold,
        })
print0(f"GSM8K test: {len(gsm8k_test_data)} examples")

math_test_task = MATH("test", levels=math_levels, types=math_types)
math_test_data = []
for i in range(math_test_task.num_examples()):
    gold = math_test_task.get_gold_answer(i)
    if gold is not None:
        math_test_data.append({
            "source": "math",
            "question": math_test_task.get_question(i),
            "gold_answer": gold,
            "metadata": math_test_task.get_metadata(i),
        })
print0(f"MATH test: {len(math_test_data)} examples")


# -----------------------------------------------------------------------------
# Helper: build conversation for a training example (no system prompt)

PROMPT_SUFFIX = "\n\nThink deeply and step by step."

def make_conversation(question: str) -> dict:
    """Build a conversation dict for generation (no system prompt)."""
    return {
        "messages": [
            {"role": "user", "content": question.rstrip() + PROMPT_SUFFIX},
            {"role": "assistant", "content": ""},
        ]
    }


# -----------------------------------------------------------------------------
# Rollout / sampling generator loop

@torch.no_grad()
def get_batch():
    assistant_end = tokenizer.encode_special("<|assistant_end|>")
    epoch = 0
    while True:
        # Re-shuffle each epoch with different seed per rank
        rng = random.Random(epoch + ddp_rank)
        all_indices = list(range(len(train_data)))
        rng.shuffle(all_indices)
        rank_indices = all_indices[ddp_rank::ddp_world_size]
        epoch += 1
        for example_idx in rank_indices:

            item = train_data[example_idx]
            conversation = make_conversation(item["question"])
            source = item["source"]
            gold_answer = item["gold_answer"]

            # Tokenize using chat mode
            tokens = tokenizer.render_for_completion(conversation)
            prefix_length = len(tokens)

            # Generate num_samples samples
            model.eval()
            generated_token_sequences = []
            masks = []
            num_sampling_steps = args.num_samples // args.device_batch_size
            for sampling_step in range(num_sampling_steps):
                seed = hash((step, example_idx, sampling_step)) & 0x7FFFFFFF
                with autocast_ctx:
                    generated_token_sequences_batch, masks_batch = engine.generate_batch(
                        tokens,
                        num_samples=args.device_batch_size,
                        max_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        top_k=args.top_k,
                        seed=seed,
                    )
                generated_token_sequences.extend(generated_token_sequences_batch)
                masks.extend(masks_batch)

            # Decode all generated texts
            generated_texts = []
            for sample_tokens in generated_token_sequences:
                generated_tokens = sample_tokens[prefix_length:]
                generated_text = tokenizer.decode(generated_tokens)
                generated_texts.append(generated_text)

            # Score all samples: format reward + correctness reward
            format_rewards = [compute_format_reward(text) for text in generated_texts]
            if source == "gsm8k":
                correctness_rewards = [compute_gsm8k_reward(text, gold_answer) for text in generated_texts]
            else:
                correctness_rewards = [compute_math_reward(text, gold_answer) for text in generated_texts]

            rewards_list = [f + c for f, c in zip(format_rewards, correctness_rewards)]

            # Pad sequences so their lengths match
            max_length = max(len(seq) for seq in generated_token_sequences)
            padded_generated_token_sequences = [seq + [assistant_end] * (max_length - len(seq)) for seq in generated_token_sequences]
            padded_masks = [mask + [0] * (max_length - len(mask)) for mask in masks]
            # Stack into tensors
            ids = torch.tensor(padded_generated_token_sequences, dtype=torch.long, device=device)
            mask_ids = torch.tensor(padded_masks, dtype=torch.long, device=device)
            inputs = ids[:, :-1]
            targets = ids[:, 1:].clone()
            targets[mask_ids[:, 1:] == 0] = -1
            rewards = torch.tensor(rewards_list, dtype=torch.float, device=device)
            # Advantages: subtract per-prompt mean
            mu = rewards.mean()
            advantages = rewards - mu

            # Extra stats for logging
            batch_stats = {
                "source": source,
                "correct_frac": sum(1 for c in correctness_rewards if c > 0) / len(correctness_rewards),
                "format_frac": sum(1 for f in format_rewards if f > 0) / len(format_rewards),
                "parse_fail_frac": sum(1 for text in generated_texts if extract_answer_latex(text) is None) / len(generated_texts),
            }
            yield generated_token_sequences, inputs, targets, rewards, advantages, batch_stats


# -----------------------------------------------------------------------------
# Evaluation: accuracy on held-out GSM8K test + MATH test

def run_math_eval(eval_data, tokenizer, engine, max_examples=None, max_completion_tokens=2048, temperature=0.0, top_k=50):
    """
    Evaluate correctness on held-out data. Each rank processes a subset.
    Returns (scores, parse_fails, per_level_scores) where per_level_scores
    is a dict of level -> list of scores (MATH only).
    """
    max_examples = min(max_examples, len(eval_data)) if max_examples is not None else len(eval_data)
    scores = []
    parse_fails = 0
    per_level_scores = {}  # level -> list of scores (for MATH)

    for idx in range(ddp_rank, max_examples, ddp_world_size):
        item = eval_data[idx]
        conversation = make_conversation(item["question"])
        tokens = tokenizer.render_for_completion(conversation)
        prefix_length = len(tokens)

        generated_token_sequences, _masks = engine.generate_batch(
            tokens,
            num_samples=1,
            max_tokens=max_completion_tokens,
            temperature=temperature,
            top_k=top_k,
        )
        generated_tokens = generated_token_sequences[0][prefix_length:]
        generated_text = tokenizer.decode(generated_tokens)

        # Check correctness
        if item["source"] == "gsm8k":
            reward = compute_gsm8k_reward(generated_text, item["gold_answer"])
        else:
            reward = compute_math_reward(generated_text, item["gold_answer"])

        scores.append(reward)
        if extract_answer_latex(generated_text) is None:
            parse_fails += 1

        # Track per-level for MATH
        if item["source"] == "math" and "metadata" in item:
            level = item["metadata"]["level"]
            if level not in per_level_scores:
                per_level_scores[level] = []
            per_level_scores[level].append(reward)

    return scores, parse_fails, per_level_scores


# -----------------------------------------------------------------------------
# Online physics eval (EVAL.json) — same as verifiable_rl.py

eval_json_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "EVAL.json")
if os.path.exists(eval_json_path):
    with open(eval_json_path) as f:
        _eval_config = json.load(f)
    _physics_eval_tasks = _eval_config["tasks"]
    print0(f"Loaded {len(_physics_eval_tasks)} physics eval tasks from EVAL.json")
else:
    _eval_config = None
    _physics_eval_tasks = []
    print0("EVAL.json not found, skipping physics eval")


def _build_physics_judge_prompt(eval_config, task, response):
    """Build a judge prompt for one EVAL.json task."""
    checklist = "\n".join(f"- {item}" for item in eval_config["shared_judge_checklist"])
    rules = "\n".join(f"- {item}" for item in eval_config["shared_judge_rules"])
    expected = "\n".join(f"- {c}" for c in task["expected_core_concepts"])
    rubric_lines = "\n".join(f"  {k}: {v}" for k, v in task["scoring_rubric"].items())
    prompt_text = "\n".join(task["prompt_lines"])

    return (
        f"{eval_config['shared_judge_prompt']}\n\n"
        f"Checklist:\n{checklist}\n\n"
        f"Rules:\n{rules}\n\n"
        f"## Original Prompt\n{prompt_text}\n\n"
        f"## Judge Focus\n{task['judge_focus']}\n\n"
        f"## Expected Core Concepts\n{expected}\n\n"
        f"## Scoring Rubric\n{rubric_lines}\n\n"
        f"## Model Response\n{response}\n\n"
        f"Respond with ONLY a <score> tag containing your integer score (0-5). Example: <score>3</score>"
    )


async def _judge_physics_eval_batch(prompts_and_responses, eval_config, tasks, model_name, max_concurrent):
    """Judge a batch of (task, response) pairs. Returns list of scores (0-5 scale)."""
    client = anthropic.AsyncAnthropic()
    semaphore = asyncio.Semaphore(max_concurrent)

    async def judge_one(task, response):
        judge_input = _build_physics_judge_prompt(eval_config, task, response)
        async with semaphore:
            try:
                result = await client.messages.create(
                    model=model_name,
                    max_tokens=64,
                    messages=[
                        {"role": "user", "content": judge_input},
                        {"role": "assistant", "content": "<score>"},
                    ],
                )
                response_text = result.content[0].text.strip()
                match = re.match(r"(\d)", response_text)
                if match is None:
                    return 0.0
                return float(max(0, min(5, int(match.group(1)))))
            except Exception as e:
                print0(f"  Physics eval judge error: {e}")
                return 0.0

    judge_tasks = [judge_one(t, r) for t, r in prompts_and_responses]
    return await asyncio.gather(*judge_tasks)


def run_physics_eval(tokenizer, engine, max_tokens=2048, temperature=0.7, top_k=50):
    """Run EVAL.json physics eval on rank 0 only. Returns dict of task_id -> score (0-5)."""
    if not _physics_eval_tasks or _eval_config is None:
        return {}
    if ddp_rank != 0:
        return {}

    model.eval()
    prompts_and_responses = []
    task_ids = []

    for task in _physics_eval_tasks:
        prompt_text = "\n".join(task["prompt_lines"])
        # No system prompt, consistent with math RL training
        conversation = {
            "messages": [
                {"role": "user", "content": prompt_text + PROMPT_SUFFIX},
                {"role": "assistant", "content": ""},
            ]
        }
        tokens = tokenizer.render_for_completion(conversation)
        prefix_length = len(tokens)

        with autocast_ctx:
            generated_seqs, _ = engine.generate_batch(
                tokens, num_samples=1, max_tokens=max_tokens,
                temperature=temperature, top_k=top_k,
            )
        gen_text = tokenizer.decode(generated_seqs[0][prefix_length:])
        prompts_and_responses.append((task, gen_text))
        task_ids.append(task["id"])

    # Judge all responses
    scores = asyncio.run(_judge_physics_eval_batch(
        prompts_and_responses, _eval_config, _physics_eval_tasks,
        args.judge_model, args.max_concurrent_api,
    ))

    return {tid: s for tid, s in zip(task_ids, scores)}


# Short names for wandb logging
_TASK_SHORT_NAMES = {
    "uv_catastrophe_main": "UV",
    "photoelectric_effect_main": "Photo",
    "sr_frozen_light": "Frozen",
    "sr_approaching_c": "ApprC",
    "sr_train_lightning": "Train",
    "sr_michelson_morley": "MM",
    "gr_main_elevator_light": "Elev",
    "gr_free_fall_equivalence": "Fall",
}


# -----------------------------------------------------------------------------
# Training loop

# Init the optimizer
optimizer = model.setup_optimizer(
    unembedding_lr=args.unembedding_lr,
    embedding_lr=args.embedding_lr,
    matrix_lr=args.matrix_lr,
    weight_decay=args.weight_decay,
)

# Restore optimizer state if resuming
if args.resume_step > 0:
    output_dirname = args.model_tag if args.model_tag else f"d{model.config.n_layer}"
    resume_ckpt_dir = os.path.join(base_dir, args.output_dir, output_dirname)
    optim_path = os.path.join(resume_ckpt_dir, f"optim_{args.resume_step:06d}_rank{ddp_rank}.pt")
    if os.path.exists(optim_path):
        optim_state = torch.load(optim_path, map_location=device)
        optimizer.load_state_dict(optim_state)
        print0(f"Restored optimizer state from {optim_path}")
    else:
        print0(f"No optimizer state found at {optim_path}, starting fresh")

# Set the initial learning rate as a fraction of the base learning rate
for group in optimizer.param_groups:
    group["lr"] = group["lr"] * args.init_lr_frac
    group["initial_lr"] = group["lr"]

# Learning rate scheduler: linear rampdown to zero over num_steps
def get_lr_multiplier(it):
    lrm = 1.0 - it / num_steps
    return lrm

# Calculate the number of examples each rank handles
print0(f"Total sequences per step: {args.examples_per_step * args.num_samples}")
assert args.examples_per_step % ddp_world_size == 0, "Desired examples per step must be divisible by the number of ranks"
examples_per_rank = args.examples_per_step // ddp_world_size
print0(f"Calculated examples per rank: {examples_per_rank}")

# Kick off the training loop
batch_iterator = get_batch()
if args.resume_step > 0:
    print0(f"Resuming training from step {args.resume_step}/{num_steps}")
for step in range(args.resume_step, num_steps):

    # Evaluate once in a while
    if step % args.eval_every == 0:
        model.eval()

        # GSM8K eval
        with autocast_ctx:
            gsm8k_scores, gsm8k_parse_fails, _ = run_math_eval(
                gsm8k_test_data, tokenizer, engine,
                max_examples=args.eval_gsm8k_examples, temperature=0.0,
            )
        gsm8k_score_sum = torch.tensor(sum(gsm8k_scores), dtype=torch.float, device=device)
        gsm8k_count = torch.tensor(len(gsm8k_scores), dtype=torch.long, device=device)
        gsm8k_pf = torch.tensor(gsm8k_parse_fails, dtype=torch.long, device=device)
        if ddp:
            dist.all_reduce(gsm8k_score_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(gsm8k_count, op=dist.ReduceOp.SUM)
            dist.all_reduce(gsm8k_pf, op=dist.ReduceOp.SUM)
        gsm8k_acc = (gsm8k_score_sum / gsm8k_count.clamp(min=1)).item()
        gsm8k_pf_rate = (gsm8k_pf.float() / gsm8k_count.clamp(min=1)).item()
        print0(f"Step {step} | GSM8K eval acc: {gsm8k_acc:.4f} | parse fail: {gsm8k_pf_rate:.2%}")

        # MATH eval
        with autocast_ctx:
            math_scores, math_parse_fails, per_level = run_math_eval(
                math_test_data, tokenizer, engine,
                max_examples=args.eval_math_examples, temperature=0.0,
            )
        math_score_sum = torch.tensor(sum(math_scores), dtype=torch.float, device=device)
        math_count = torch.tensor(len(math_scores), dtype=torch.long, device=device)
        math_pf = torch.tensor(math_parse_fails, dtype=torch.long, device=device)
        if ddp:
            dist.all_reduce(math_score_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(math_count, op=dist.ReduceOp.SUM)
            dist.all_reduce(math_pf, op=dist.ReduceOp.SUM)
        math_acc = (math_score_sum / math_count.clamp(min=1)).item()
        math_pf_rate = (math_pf.float() / math_count.clamp(min=1)).item()
        print0(f"Step {step} | MATH eval acc: {math_acc:.4f} | parse fail: {math_pf_rate:.2%}")

        eval_log = {
            "step": step,
            "eval/gsm8k_acc": gsm8k_acc,
            "eval/gsm8k_parse_fail_rate": gsm8k_pf_rate,
            "eval/math_acc": math_acc,
            "eval/math_parse_fail_rate": math_pf_rate,
        }

        # Per-level MATH accuracy (local rank only, approximate)
        for level, level_scores in per_level.items():
            if level_scores:
                level_acc = sum(level_scores) / len(level_scores)
                level_key = level.replace(" ", "_").lower()
                eval_log[f"eval/math_{level_key}_acc"] = level_acc
                print0(f"  MATH {level}: {level_acc:.4f} ({len(level_scores)} examples)")

        wandb_run.log(eval_log)

    # Physics eval (EVAL.json) — aligned with save_every
    if _physics_eval_tasks and step % args.save_every == 0:
        model.eval()
        with autocast_ctx:
            physics_scores = run_physics_eval(tokenizer, engine, max_tokens=args.max_new_tokens)
        if physics_scores:
            physics_log = {"step": step}
            all_scores = []
            for tid, score in physics_scores.items():
                short = _TASK_SHORT_NAMES.get(tid, tid[:6])
                physics_log[f"physics_eval/{short}"] = score
                all_scores.append(score)
                print0(f"  Physics eval {short}: {score:.1f}/5")
            physics_log["physics_eval/mean"] = sum(all_scores) / len(all_scores)
            print0(f"  Physics eval mean: {physics_log['physics_eval/mean']:.2f}/5")
            wandb_run.log(physics_log)

    # Forward/Backward on rollouts over multiple examples
    rewards_list = []
    sequence_lengths = []
    agg_stats = {"gsm8k_correct_frac": 0.0, "math_correct_frac": 0.0, "format_frac": 0.0, "parse_fail_frac": 0.0}
    gsm8k_example_count = 0
    math_example_count = 0
    for example_step in range(examples_per_rank):
        sequences_all, inputs_all, targets_all, rewards_all, advantages_all, batch_stats = next(batch_iterator)
        model.train()

        # Accumulate per-source stats
        if batch_stats["source"] == "gsm8k":
            agg_stats["gsm8k_correct_frac"] += batch_stats["correct_frac"]
            gsm8k_example_count += 1
        else:
            agg_stats["math_correct_frac"] += batch_stats["correct_frac"]
            math_example_count += 1
        agg_stats["format_frac"] += batch_stats["format_frac"]
        agg_stats["parse_fail_frac"] += batch_stats["parse_fail_frac"]

        assert inputs_all.size(0) % args.device_batch_size == 0
        num_passes = inputs_all.size(0) // args.device_batch_size
        for pass_idx in range(num_passes):
            b0, b1 = pass_idx * args.device_batch_size, (pass_idx + 1) * args.device_batch_size
            inputs = inputs_all[b0:b1]
            targets = targets_all[b0:b1]
            rewards = rewards_all[b0:b1]
            advantages = advantages_all[b0:b1]
            with autocast_ctx:
                logp = -model(inputs, targets, loss_reduction='none').view_as(inputs)
            # PG objective with DAPO token-level normalization
            pg_obj = (logp * advantages.unsqueeze(-1)).sum()
            num_valid = (targets >= 0).sum().clamp(min=1)
            pg_obj = pg_obj / (num_valid * num_passes * examples_per_rank)
            loss = -pg_obj
            loss.backward()
            print0(f"Step {step}/{num_steps} | Example step {example_step} | Pass {pass_idx} | loss: {loss.item():.6f} | reward: {rewards.mean().item():.3f}")
        rewards_list.append(rewards_all.mean().item())
        sequence_lengths.extend(len(seq) for seq in sequences_all)

    # Average aggregate stats
    agg_stats["format_frac"] /= examples_per_rank
    agg_stats["parse_fail_frac"] /= examples_per_rank
    if gsm8k_example_count > 0:
        agg_stats["gsm8k_correct_frac"] /= gsm8k_example_count
    if math_example_count > 0:
        agg_stats["math_correct_frac"] /= math_example_count

    # Logging
    mean_reward = sum(rewards_list) / len(rewards_list)
    mean_sequence_length = sum(sequence_lengths) / len(sequence_lengths)
    if ddp:
        mean_reward_tensor = torch.tensor(mean_reward, dtype=torch.float, device=device)
        mean_sequence_length_tensor = torch.tensor(mean_sequence_length, dtype=torch.float, device=device)
        dist.all_reduce(mean_reward_tensor, op=dist.ReduceOp.AVG)
        dist.all_reduce(mean_sequence_length_tensor, op=dist.ReduceOp.AVG)
        mean_reward = mean_reward_tensor.item()
        mean_sequence_length = mean_sequence_length_tensor.item()
    print0(f"Step {step}/{num_steps} | reward: {mean_reward:.3f} | seq_len: {mean_sequence_length:.0f} | gsm8k_correct: {agg_stats['gsm8k_correct_frac']:.2%} | math_correct: {agg_stats['math_correct_frac']:.2%} | format: {agg_stats['format_frac']:.2%}")
    wandb_run.log({
        "step": step,
        "reward": mean_reward,
        "sequence_length": mean_sequence_length,
        "reward/gsm8k_correct_frac": agg_stats["gsm8k_correct_frac"],
        "reward/math_correct_frac": agg_stats["math_correct_frac"],
        "reward/format_frac": agg_stats["format_frac"],
        "reward/parse_fail_frac": agg_stats["parse_fail_frac"],
    })

    # Update model parameters
    lrm = get_lr_multiplier(step)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
    optimizer.step()
    model.zero_grad(set_to_none=True)
    wandb_run.log({
        "step": step,
        "lrm": lrm,
    })

    # Save checkpoints (model + optimizer state per rank)
    if (step > 0 and step % args.save_every == 0) or step == num_steps - 1:
        depth = model.config.n_layer
        output_dirname = args.model_tag if args.model_tag else f"d{depth}"
        checkpoint_dir = os.path.join(base_dir, args.output_dir, output_dirname)
        model_config_kwargs = model.config.__dict__
        save_checkpoint(
            checkpoint_dir,
            step,
            model.state_dict(),
            optimizer.state_dict(),
            {
                "model_config": model_config_kwargs,
            },
            rank=ddp_rank,
        )
        print0(f"Saved model + optimizer checkpoint to {checkpoint_dir}")

# Log to report
from nanochat.report import get_report
get_report().log(section="Math RL", data=[
    user_config,
])

wandb_run.finish()
compute_cleanup()
