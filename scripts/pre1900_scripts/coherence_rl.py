"""
Coherence RL for the pre-1900 model via GRPO/REINFORCE.

Uses Claude as a coherence judge to score generations on a 5-point scale,
then applies policy gradient updates to improve logical coherence.

Mirrors scripts/chat_rl.py with key differences:
  - Task: CustomJSON (filtered_pairs.jsonl) instead of GSM8K
  - Reward: 5-point coherence score via async Claude API (normalized to [0,1])
  - Model source: pre1900_sft_checkpoints/ instead of chatsft_checkpoints/
  - Output: pre1900_rl_checkpoints/

1 GPU:
python -m scripts.pre1900_scripts.coherence_rl --model-tag d26

8 GPUs:
torchrun --standalone --nproc_per_node=8 -m scripts.pre1900_scripts.coherence_rl -- --run=default
"""

import argparse
import os
import asyncio
import itertools
import wandb
import torch
import torch.distributed as dist
from contextlib import nullcontext

import anthropic

from nanochat.common import compute_init, compute_cleanup, print0, get_base_dir, DummyWandb, autodetect_device_type
from nanochat.checkpoint_manager import save_checkpoint, load_model_from_dir
from nanochat.engine import Engine
from tasks.customjson import CustomJSON

# -----------------------------------------------------------------------------
# Claude coherence judge

COHERENCE_JUDGE_PROMPT = """\
You are evaluating the coherence of a language model's response. The model is trained on pre-1900 text and may use archaic language, vocabulary, and writing conventions. Do NOT penalize archaic style, period-appropriate vocabulary, or pre-1900 phrasing.

Rate the response on these 5 criteria (each 0 or 1):
1. Logical coherence: Does the response logically follow from the question/prompt?
2. Internal consistency: Is the response free of self-contradictions?
3. Topical relevance: Does the response stay on topic?
4. Structural clarity: Are sentences well-formed and organized?
5. Fluency: Is the text free of garbled words, excessive repetition, or abrupt endings?

User prompt:
{prompt}

Model response:
{response}

Return ONLY a single integer from 1 to 5 representing the total score (sum of the 5 criteria). Nothing else."""


async def judge_coherence_single(
    client: anthropic.AsyncAnthropic,
    prompt: str,
    response: str,
    model: str,
    semaphore: asyncio.Semaphore,
) -> float:
    """Score a single prompt-response pair using Claude. Returns normalized score in [0, 1]."""
    judge_input = COHERENCE_JUDGE_PROMPT.format(prompt=prompt, response=response)
    async with semaphore:
        try:
            result = await client.messages.create(
                model=model,
                max_tokens=8,
                messages=[{"role": "user", "content": judge_input}],
            )
            score_text = result.content[0].text.strip()
            score = int(score_text)
            score = max(1, min(5, score))  # clamp to [1, 5]
            return (score - 1) / 4.0  # normalize to [0, 1]
        except Exception as e:
            print0(f"  Judge API error: {e}")
            return 0.5  # neutral fallback on error


async def batch_judge_coherence(
    prompts: list[str],
    responses: list[str],
    model: str,
    max_concurrent: int,
) -> list[float]:
    """Score a batch of prompt-response pairs concurrently."""
    client = anthropic.AsyncAnthropic()
    semaphore = asyncio.Semaphore(max_concurrent)
    tasks = [
        judge_coherence_single(client, p, r, model, semaphore)
        for p, r in zip(prompts, responses)
    ]
    return await asyncio.gather(*tasks)


# -----------------------------------------------------------------------------
# CLI arguments
parser = argparse.ArgumentParser(description="Coherence RL for pre-1900 model")
# Logging
parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' disables wandb logging)")
# Runtime
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
parser.add_argument("--dtype", type=str, default="bfloat16", help="float32|bfloat16")
# Model loading
parser.add_argument("--model-tag", type=str, default=None, help="model tag to load from")
parser.add_argument("--model-step", type=int, default=None, help="model step to load from")
parser.add_argument("--checkpoints-dir", type=str, default="pre1900_sft_checkpoints", help="source checkpoints directory (relative to base dir)")
# Training horizon
parser.add_argument("--num-epochs", type=int, default=1, help="number of epochs over training data")
# Batch sizes / sampling
parser.add_argument("--device-batch-size", type=int, default=8, help="max batch size per forward pass")
parser.add_argument("--examples-per-step", type=int, default=8, help="total examples per optimization step across all ranks")
parser.add_argument("--num-samples", type=int, default=8, help="number of samples per example/question")
# Generation
parser.add_argument("--max-new-tokens", type=int, default=256, help="max tokens to generate per sample")
parser.add_argument("--temperature", type=float, default=1.0, help="sampling temperature")
parser.add_argument("--top-k", type=int, default=50, help="top-k sampling (0 = disabled)")
# Optimization
parser.add_argument("--embedding-lr", type=float, default=0.2, help="learning rate for embedding parameters (Adam)")
parser.add_argument("--unembedding-lr", type=float, default=0.004, help="learning rate for unembedding parameters (Adam)")
parser.add_argument("--matrix-lr", type=float, default=0.02, help="learning rate for matrix parameters (Muon)")
parser.add_argument("--weight-decay", type=float, default=0.0, help="weight decay for embedding/unembedding parameters (Adam)")
parser.add_argument("--init-lr-frac", type=float, default=0.05, help="initial LR as fraction of base LR")
# Evaluation / checkpointing
parser.add_argument("--eval-every", type=int, default=60, help="evaluate coherence every N steps")
parser.add_argument("--eval-examples", type=int, default=50, help="number of examples for coherence evaluation")
parser.add_argument("--save-every", type=int, default=60, help="save checkpoint every N steps")
# Data
parser.add_argument("--train-data", type=str, default="instruct_data/filtered_pairs.jsonl", help="training data JSONL")
parser.add_argument("--val-data", type=str, default="instruct_data/val_pairs.jsonl", help="validation data JSONL")
# Claude judge
parser.add_argument("--judge-model", type=str, default="claude-sonnet-4-20250514", help="Anthropic model for coherence judging")
parser.add_argument("--max-concurrent-api", type=int, default=20, help="max concurrent API requests per rank")
# Output
parser.add_argument("--output-dir", type=str, default="pre1900_rl_checkpoints", help="output checkpoints directory (relative to base dir)")
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

# Init model and tokenizer from pre1900_sft_checkpoints
base_dir = get_base_dir()
checkpoints_dir = os.path.join(base_dir, args.checkpoints_dir)
model, tokenizer, meta = load_model_from_dir(checkpoints_dir, device, phase="eval", model_tag=args.model_tag, step=args.model_step)
engine = Engine(model, tokenizer)

# -----------------------------------------------------------------------------
# Load training and validation data

train_filepath = os.path.join(base_dir, args.train_data)
val_filepath = os.path.join(base_dir, args.val_data)
train_task = CustomJSON(filepath=train_filepath)
val_task = CustomJSON(filepath=val_filepath)
print0(f"Train dataset: {len(train_task)} conversations from {train_filepath}")
print0(f"Val dataset: {len(val_task)} conversations from {val_filepath}")
num_steps = (len(train_task) // args.examples_per_step) * args.num_epochs
print0(f"Calculated number of steps: {num_steps}")


def extract_last_user_prompt(conversation):
    """Extract the last user message from a conversation for judging."""
    messages = conversation["messages"]
    for msg in reversed(messages):
        if msg["role"] == "user":
            return msg["content"]
    return ""


# -----------------------------------------------------------------------------
# Rollout / sampling generator loop that yields batches of examples for training

@torch.no_grad()
def get_batch():
    assistant_end = tokenizer.encode_special("<|assistant_end|>")
    rank_indices = range(ddp_rank, len(train_task), ddp_world_size)
    for example_idx in itertools.cycle(rank_indices):

        # Get the full conversation
        conversation = train_task[example_idx]

        # Tokenize, deleting the last Assistant message and priming for completion
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
        prompt_text = extract_last_user_prompt(conversation)
        generated_texts = []
        for sample_tokens in generated_token_sequences:
            generated_tokens = sample_tokens[prefix_length:]
            generated_text = tokenizer.decode(generated_tokens)
            generated_texts.append(generated_text)

        # Score all samples via async Claude API calls
        rewards_list = asyncio.run(batch_judge_coherence(
            [prompt_text] * len(generated_texts),
            generated_texts,
            args.judge_model,
            args.max_concurrent_api,
        ))

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
        yield generated_token_sequences, inputs, targets, rewards, advantages


# -----------------------------------------------------------------------------
# Evaluation: mean coherence score on held-out val set

def run_coherence_eval(task, tokenizer, engine, max_examples=None, max_completion_tokens=256, temperature=0.0, top_k=50):
    """
    Evaluate coherence on held-out data. Each rank processes a subset,
    caller must reduce across ranks.
    """
    max_examples = min(max_examples, len(task)) if max_examples is not None else len(task)
    all_prompts = []
    all_responses = []
    for idx in range(ddp_rank, max_examples, ddp_world_size):
        conversation = task[idx]
        tokens = tokenizer.render_for_completion(conversation)
        prefix_length = len(tokens)
        # Generate a single sample for evaluation
        generated_token_sequences, masks = engine.generate_batch(
            tokens,
            num_samples=1,
            max_tokens=max_completion_tokens,
            temperature=temperature,
            top_k=top_k,
        )
        generated_tokens = generated_token_sequences[0][prefix_length:]
        generated_text = tokenizer.decode(generated_tokens)
        prompt_text = extract_last_user_prompt(conversation)
        all_prompts.append(prompt_text)
        all_responses.append(generated_text)

    # Batch judge all responses at once
    if all_prompts:
        scores = asyncio.run(batch_judge_coherence(
            all_prompts, all_responses, args.judge_model, args.max_concurrent_api,
        ))
    else:
        scores = []
    return scores


# -----------------------------------------------------------------------------
# Training loop

# Init the optimizer
optimizer = model.setup_optimizer(
    unembedding_lr=args.unembedding_lr,
    embedding_lr=args.embedding_lr,
    matrix_lr=args.matrix_lr,
    weight_decay=args.weight_decay,
)

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
for step in range(num_steps):

    # Evaluate coherence once in a while
    if step % args.eval_every == 0:
        model.eval()
        with autocast_ctx:
            scores = run_coherence_eval(val_task, tokenizer, engine, max_examples=args.eval_examples, temperature=1.0)
        # Reduce across ranks
        score_sum = torch.tensor(sum(scores), dtype=torch.float, device=device)
        score_count = torch.tensor(len(scores), dtype=torch.long, device=device)
        if ddp:
            dist.all_reduce(score_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(score_count, op=dist.ReduceOp.SUM)
        mean_coherence = (score_sum / score_count.clamp(min=1)).item()
        print0(f"Step {step} | Eval coherence score: {mean_coherence:.4f}")
        wandb_run.log({
            "step": step,
            "eval/coherence_score": mean_coherence,
        })

    # Forward/Backward on rollouts over multiple examples
    rewards_list = []
    sequence_lengths = []
    for example_step in range(examples_per_rank):
        sequences_all, inputs_all, targets_all, rewards_all, advantages_all = next(batch_iterator)
        model.train()
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
            print0(f"Step {step}/{num_steps} | Example step {example_step} | Pass {pass_idx} | loss: {loss.item():.6f} | Average reward: {rewards.mean().item()}")
        rewards_list.append(rewards_all.mean().item())
        sequence_lengths.extend(len(seq) for seq in sequences_all)

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
    print0(f"Step {step}/{num_steps} | Average reward: {mean_reward} | Average sequence length: {mean_sequence_length:.2f}")
    wandb_run.log({
        "step": step,
        "reward": mean_reward,
        "sequence_length": mean_sequence_length,
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

    # Save checkpoints
    if master_process and ((step > 0 and step % args.save_every == 0) or step == num_steps - 1):
        depth = model.config.n_layer
        output_dirname = args.model_tag if args.model_tag else f"d{depth}"
        checkpoint_dir = os.path.join(base_dir, args.output_dir, output_dirname)
        model_config_kwargs = model.config.__dict__
        save_checkpoint(
            checkpoint_dir,
            step,
            model.state_dict(),
            None,
            {
                "model_config": model_config_kwargs,
            }
        )
        print(f"Saved model checkpoint to {checkpoint_dir}")

# Log to report
from nanochat.report import get_report
get_report().log(section="Pre-1900 Coherence RL", data=[
    user_config,
])

wandb_run.finish()
compute_cleanup()
