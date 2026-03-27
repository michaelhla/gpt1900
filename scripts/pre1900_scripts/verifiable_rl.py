"""
Verifiable RL for the pre-1900 model via GRPO/REINFORCE with SymPy rewards.

Uses deterministic SymPy-based symbolic equivalence checking instead of
Claude as a correctness judge. Rewards are binary (0 or 1) for single-answer
problems, partial credit for multi-answer.

Fork of discovery_rl.py with key changes:
  - Replace Claude judge with SymPy reward (no API calls for training rewards)
  - Keep format reward (0.1 for correct <think> + \\answer{} tags)
  - Load gold_answers as arrays (not single strings)
  - Keep EVAL.json physics eval (orthogonal qualitative eval via Claude judge)
  - Add logging: reward/correct_frac, reward/sympy_parse_fail_frac, reward/format_frac

1 GPU:
python -m scripts.pre1900_scripts.verifiable_rl --model-tag d34

8 GPUs:
torchrun --standalone --nproc_per_node=8 -m scripts.pre1900_scripts.verifiable_rl -- --run=default
"""

import argparse
import copy
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
from nanochat.checkpoint_manager import save_checkpoint, load_checkpoint, load_model_from_dir
from nanochat.engine import Engine
from tasks.customjson import CustomJSON
from tasks.yale_physics import extract_answer_latex, is_symbolically_equivalent, compute_reward

from scripts.pre1900_scripts.constants import QUANTITATIVE_REASONING_SYSTEM_PROMPT

# -----------------------------------------------------------------------------
# Format reward

FORMAT_REWARD = 0.1  # partial credit for using the correct reasoning format

def compute_format_reward(response: str) -> float:
    """Return partial format reward: 0.15 for <think>...</think>, 0.15 for \\answer{...}."""
    reward = 0.0
    if re.search(r"<think>.*?</think>", response, re.DOTALL):
        reward += FORMAT_REWARD / 2
    if re.search(r"\\answer\{", response):
        reward += FORMAT_REWARD / 2
    return reward


# -----------------------------------------------------------------------------
# CLI arguments
parser = argparse.ArgumentParser(description="Verifiable RL for pre-1900 model (SymPy rewards)")
# Logging
parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' disables wandb logging)")
# Runtime
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
parser.add_argument("--dtype", type=str, default="bfloat16", help="float32|bfloat16")
# Model loading
parser.add_argument("--model-tag", type=str, default=None, help="model tag to load from")
parser.add_argument("--model-step", type=int, default=None, help="model step to load from")
parser.add_argument("--resume-step", type=int, default=0, help="resume training from this step (skips earlier steps, fast-forwards data iterator)")
parser.add_argument("--checkpoints-dir", type=str, default="pre1900_reasoning_sft_checkpoints", help="source checkpoints directory (relative to base dir)")
# Training horizon
parser.add_argument("--num-epochs", type=int, default=1, help="number of epochs over training data")
# Batch sizes / sampling
parser.add_argument("--device-batch-size", type=int, default=8, help="max batch size per forward pass")
parser.add_argument("--examples-per-step", type=int, default=8, help="total examples per optimization step across all ranks")
parser.add_argument("--num-samples", type=int, default=8, help="number of samples per example/question")
# Generation
parser.add_argument("--max-new-tokens", type=int, default=1024, help="max tokens to generate per sample")
parser.add_argument("--temperature", type=float, default=1.0, help="sampling temperature")
parser.add_argument("--top-k", type=int, default=50, help="top-k sampling (0 = disabled)")
# Optimization
parser.add_argument("--embedding-lr", type=float, default=0.2, help="learning rate for embedding parameters (Adam)")
parser.add_argument("--unembedding-lr", type=float, default=0.004, help="learning rate for unembedding parameters (Adam)")
parser.add_argument("--matrix-lr", type=float, default=0.02, help="learning rate for matrix parameters (Muon)")
parser.add_argument("--weight-decay", type=float, default=0.0, help="weight decay for embedding/unembedding parameters (Adam)")
parser.add_argument("--kl-coeff", type=float, default=0.0, help="KL penalty coefficient (0 = disabled). Penalizes divergence from frozen reference model.")
parser.add_argument("--init-lr-frac", type=float, default=0.05, help="initial LR as fraction of base LR")
# Evaluation / checkpointing
parser.add_argument("--eval-every", type=int, default=60, help="evaluate correctness every N steps")
parser.add_argument("--eval-examples", type=int, default=50, help="number of examples for correctness evaluation")
parser.add_argument("--save-every", type=int, default=60, help="save checkpoint every N steps")
# Data
parser.add_argument("--train-data", type=str, default="instruct_data/yale_physics/yale_prompts_sys_train.jsonl", help="training prompts JSONL")
parser.add_argument("--val-data", type=str, default="instruct_data/yale_physics/yale_prompts_sys_val.jsonl", help="validation prompts JSONL")
parser.add_argument("--problems-data", type=str, default="instruct_data/yale_physics/yale_problems_train.jsonl", help="training problems with gold answers JSONL")
parser.add_argument("--problems-val-data", type=str, default="instruct_data/yale_physics/yale_problems_val.jsonl", help="validation problems with gold answers JSONL")
# Physics eval (EVAL.json) — uses Claude judge for qualitative eval
parser.add_argument("--judge-model", type=str, default="claude-sonnet-4-20250514", help="Anthropic model for EVAL.json physics eval only")
parser.add_argument("--max-concurrent-api", type=int, default=20, help="max concurrent API requests for EVAL.json eval")
# Output
parser.add_argument("--output-dir", type=str, default="pre1900_verifiable_rl_checkpoints", help="output checkpoints directory (relative to base dir)")
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

# Frozen reference model for KL penalty
if args.kl_coeff > 0:
    ref_model = copy.deepcopy(model)
    for p in ref_model.parameters():
        p.requires_grad = False
    ref_model.eval()
    print0(f"Created frozen reference model for KL penalty (β={args.kl_coeff})")
else:
    ref_model = None

# -----------------------------------------------------------------------------
# Load training and validation data (prompts + gold answers)

train_filepath = os.path.join(base_dir, args.train_data)
val_filepath = os.path.join(base_dir, args.val_data)
train_task = CustomJSON(filepath=train_filepath)
val_task = CustomJSON(filepath=val_filepath)
print0(f"Train dataset: {len(train_task)} conversations from {train_filepath}")
print0(f"Val dataset: {len(val_task)} conversations from {val_filepath}")
num_steps = (len(train_task) // args.examples_per_step) * args.num_epochs
print0(f"Calculated number of steps: {num_steps}")

# Load gold answers as arrays
def load_gold_answers(filepath):
    answers = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            answers.append(obj["gold_answers"])  # list of strings
    return answers

train_gold_answers = load_gold_answers(os.path.join(base_dir, args.problems_data))
val_gold_answers = load_gold_answers(os.path.join(base_dir, args.problems_val_data))
assert len(train_gold_answers) == len(train_task), f"Gold answers ({len(train_gold_answers)}) != prompts ({len(train_task)})"
assert len(val_gold_answers) == len(val_task), f"Val gold answers ({len(val_gold_answers)}) != val prompts ({len(val_task)})"
print0(f"Loaded {len(train_gold_answers)} train gold answers, {len(val_gold_answers)} val gold answers")


def extract_last_user_prompt(conversation):
    """Extract the last user message from a conversation for logging."""
    messages = conversation["messages"]
    for msg in reversed(messages):
        if msg["role"] == "user":
            return msg["content"]
    return ""


# -----------------------------------------------------------------------------
# Rollout / sampling generator loop that yields batches of examples for training

bos_token = tokenizer.get_bos_token_id()

@torch.no_grad()
def get_batch():
    assistant_end = tokenizer.encode_special("<|assistant_end|>")
    epoch = 0
    while True:
        # Shuffle all indices, then each rank takes its slice
        all_indices = list(range(len(train_task)))
        random.Random(epoch + ddp_rank).shuffle(all_indices)
        rank_indices = all_indices[ddp_rank::ddp_world_size]
        epoch += 1
        for example_idx in rank_indices:

            # Get the full conversation and gold answers
            conversation = train_task[example_idx]
            gold_answers = train_gold_answers[example_idx]  # list of strings

            # Tokenize using chat mode
            tokens = tokenizer.render_for_completion(conversation)
            prefix_length = len(tokens)

            # Generate num_samples samples using batched generation
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

            # Score all samples: format reward + SymPy correctness reward
            format_rewards = [compute_format_reward(text) for text in generated_texts]
            correctness_rewards = [compute_reward(text, gold_answers) for text in generated_texts]

            # Track parse failures for logging
            parse_fails = sum(1 for text in generated_texts if extract_answer_latex(text) is None)

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
            # Advantages: subtract per-prompt mean (not z-score)
            mu = rewards.mean()
            advantages = rewards - mu

            # Extra stats for logging
            batch_stats = {
                "correct_frac": sum(1 for c in correctness_rewards if c > 0) / len(correctness_rewards),
                "format_frac": sum(1 for f in format_rewards if f > 0) / len(format_rewards),
                "parse_fail_frac": parse_fails / len(generated_texts),
            }
            yield generated_token_sequences, inputs, targets, rewards, advantages, batch_stats


# -----------------------------------------------------------------------------
# Evaluation: mean correctness score on held-out val set (SymPy-based)

def run_correctness_eval(task, gold_answers, tokenizer, engine, max_examples=None, max_completion_tokens=1024, temperature=0.0, top_k=50):
    """
    Evaluate correctness on held-out data using SymPy rewards.
    Each rank processes a subset, caller must reduce across ranks.
    """
    max_examples = min(max_examples, len(task)) if max_examples is not None else len(task)
    scores = []
    parse_fails = 0
    for idx in range(ddp_rank, max_examples, ddp_world_size):
        conversation = task[idx]
        gold_ans = gold_answers[idx]
        tokens = tokenizer.render_for_completion(conversation)
        prefix_length = len(tokens)
        # Generate a single sample for evaluation
        generated_token_sequences, _masks = engine.generate_batch(
            tokens,
            num_samples=1,
            max_tokens=max_completion_tokens,
            temperature=temperature,
            top_k=top_k,
        )
        generated_tokens = generated_token_sequences[0][prefix_length:]
        generated_text = tokenizer.decode(generated_tokens)
        reward = compute_reward(generated_text, gold_ans)
        scores.append(reward)
        if extract_answer_latex(generated_text) is None:
            parse_fails += 1

    return scores, parse_fails


# -----------------------------------------------------------------------------
# Online physics eval (EVAL.json) — unchanged from discovery_rl.py

eval_json_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "EVAL.json")
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
        conversation = {
            "messages": [
                {"role": "system", "content": QUANTITATIVE_REASONING_SYSTEM_PROMPT},
                {"role": "user", "content": prompt_text},
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

    results = {tid: s for tid, s in zip(task_ids, scores)}

    # Save generations to disk
    generations_dir = os.path.join(base_dir, args.output_dir, "eval_generations")
    os.makedirs(generations_dir, exist_ok=True)
    gen_path = os.path.join(generations_dir, f"physics_eval_s{step:05d}.json")
    gen_data = []
    for (task, gen_text), score in zip(prompts_and_responses, scores):
        gen_data.append({
            "task_id": task["id"],
            "score": score,
            "generation": gen_text,
        })
    with open(gen_path, "w") as f:
        json.dump(gen_data, f, indent=2, ensure_ascii=False)
    print0(f"  Saved physics eval generations to {gen_path}")

    return results


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
    resume_ckpt_dir = os.path.join(base_dir, args.checkpoints_dir, output_dirname)
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

    # Evaluate correctness once in a while (SymPy-based)
    if step % args.eval_every == 0:
        model.eval()
        with autocast_ctx:
            scores, eval_parse_fails = run_correctness_eval(
                val_task, val_gold_answers, tokenizer, engine,
                max_examples=args.eval_examples, temperature=1.0,
            )
        # Reduce across ranks
        score_sum = torch.tensor(sum(scores), dtype=torch.float, device=device)
        score_count = torch.tensor(len(scores), dtype=torch.long, device=device)
        parse_fail_sum = torch.tensor(eval_parse_fails, dtype=torch.long, device=device)
        if ddp:
            dist.all_reduce(score_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(score_count, op=dist.ReduceOp.SUM)
            dist.all_reduce(parse_fail_sum, op=dist.ReduceOp.SUM)
        mean_correctness = (score_sum / score_count.clamp(min=1)).item()
        parse_fail_rate = (parse_fail_sum.float() / score_count.clamp(min=1)).item()
        print0(f"Step {step} | Eval correctness: {mean_correctness:.4f} | Parse fail rate: {parse_fail_rate:.2%}")
        wandb_run.log({
            "step": step,
            "eval/correctness_score": mean_correctness,
            "eval/parse_fail_rate": parse_fail_rate,
        })

    # Physics eval (EVAL.json) — once per epoch (aligned with save_every)
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
    kl_accumulator = 0.0
    agg_stats = {"correct_frac": 0.0, "format_frac": 0.0, "parse_fail_frac": 0.0}
    for example_step in range(examples_per_rank):
        sequences_all, inputs_all, targets_all, rewards_all, advantages_all, batch_stats = next(batch_iterator)
        model.train()
        for k in agg_stats:
            agg_stats[k] += batch_stats[k]
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
            # KL penalty: penalize divergence from frozen reference model
            if ref_model is not None:
                with torch.no_grad(), autocast_ctx:
                    ref_logp = -ref_model(inputs, targets, loss_reduction='none').view_as(inputs)
                # Per-token KL: logp_policy - logp_ref (only on valid tokens)
                valid_mask = (targets >= 0).float()
                kl_per_token = (logp - ref_logp) * valid_mask
                kl_penalty = kl_per_token.sum() / (num_valid * num_passes * examples_per_rank)
                loss = loss + args.kl_coeff * kl_penalty
                kl_accumulator += kl_penalty.item()
            loss.backward()
            print0(f"Step {step}/{num_steps} | Example step {example_step} | Pass {pass_idx} | loss: {loss.item():.6f} | Average reward: {rewards.mean().item()}")
        rewards_list.append(rewards_all.mean().item())
        sequence_lengths.extend(len(seq) for seq in sequences_all)

    # Average aggregate stats over examples_per_rank
    for k in agg_stats:
        agg_stats[k] /= examples_per_rank

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
    print0(f"Step {step}/{num_steps} | Average reward: {mean_reward} | Average sequence length: {mean_sequence_length:.2f} | Correct: {agg_stats['correct_frac']:.2%} | Format: {agg_stats['format_frac']:.2%}")
    log_dict = {
        "step": step,
        "reward": mean_reward,
        "sequence_length": mean_sequence_length,
        "reward/correct_frac": agg_stats["correct_frac"],
        "reward/format_frac": agg_stats["format_frac"],
        "reward/sympy_parse_fail_frac": agg_stats["parse_fail_frac"],
    }
    if ref_model is not None:
        log_dict["kl_penalty"] = kl_accumulator
    wandb_run.log(log_dict)

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
get_report().log(section="Pre-1900 Verifiable RL", data=[
    user_config,
])

wandb_run.finish()
compute_cleanup()
