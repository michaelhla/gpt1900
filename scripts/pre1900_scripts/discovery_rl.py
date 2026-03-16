"""
Discovery RL for the pre-1900 model via GRPO/REINFORCE.

Uses Claude as a correctness judge to score generations against gold answers,
then applies policy gradient updates to improve reasoning quality.

Fork of coherence_rl.py with key differences:
  - Reward: binary correctness score via Claude judge + gold answers (0 or 1)
  - Gold answers loaded from a parallel JSONL file
  - Default --max-new-tokens 512 (reasoning needs more tokens)
  - Model source: pre1900_reasoning_sft_checkpoints/ (reasoning SFT output)
  - Output: pre1900_discovery_rl_checkpoints/

1 GPU:
python -m scripts.pre1900_scripts.discovery_rl --model-tag d34

8 GPUs:
torchrun --standalone --nproc_per_node=8 -m scripts.pre1900_scripts.discovery_rl -- --run=default
"""

import argparse
import os
import re
import asyncio
import itertools
import json
import wandb
import torch
import torch.distributed as dist
from contextlib import nullcontext

import anthropic

from nanochat.common import compute_init, compute_cleanup, print0, get_base_dir, DummyWandb, autodetect_device_type

# -----------------------------------------------------------------------------
# Bedrock client with region cycling on rate limits

_BEDROCK_REGIONS = ["us-east-1", "us-west-2", "us-east-2"]

_ANTHROPIC_TO_BEDROCK = {
    "claude-sonnet-4-20250514": "us.anthropic.claude-sonnet-4-20250514-v1:0",
    "claude-opus-4-6": "us.anthropic.claude-opus-4-6-v1",
    "claude-opus-4-20250514": "us.anthropic.claude-opus-4-20250514-v1:0",
    "claude-haiku-4-5-20251001": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
}


class _MessagesProxy:
    """Proxy that intercepts messages.create() and cycles regions on rate limits."""
    def __init__(self, parent):
        self._parent = parent

    async def create(self, **kwargs):
        import asyncio as _asyncio
        p = self._parent
        model = kwargs.pop("model", None)
        bedrock_model = _ANTHROPIC_TO_BEDROCK.get(model, model)
        last_err = None

        # Try Bedrock regions with retries (3 rounds, backoff between rounds)
        for attempt in range(3):
            for _ in range(len(p._region_clients)):
                idx = p._current_region_idx
                client = p._region_clients[idx]
                region = p._regions[idx]
                try:
                    kwargs["model"] = bedrock_model
                    result = await client.messages.create(**kwargs)
                    return result
                except anthropic.RateLimitError as e:
                    last_err = e
                    p._current_region_idx = (idx + 1) % len(p._region_clients)
                    print(f"  Rate limited on {region}, rotating to {p._regions[p._current_region_idx]}")
                except Exception as e:
                    last_err = e
                    p._current_region_idx = (idx + 1) % len(p._region_clients)
                    print(f"  Error on {region}: {e}, rotating to {p._regions[p._current_region_idx]}")
            # Back off before next round of retries
            if attempt < 2:
                delay = 2 ** attempt  # 1s, 2s
                await _asyncio.sleep(delay)

        # All Bedrock regions failed after retries — fall back to Anthropic API
        try:
            kwargs["model"] = model  # Use original Anthropic model ID
            result = await p._anthropic_client.messages.create(**kwargs)
            if not p._anthropic_fallback_logged:
                print("  All Bedrock regions exhausted after retries, falling back to Anthropic API")
                p._anthropic_fallback_logged = True
            return result
        except Exception as e:
            print(f"  Anthropic API fallback also failed: {e}")
            raise last_err


class BedrockFallbackClient:
    """Bedrock client with region cycling + Anthropic API fallback."""

    def __init__(self, regions=None):
        self._region_clients = []
        self._regions = []
        self._current_region_idx = 0
        self._anthropic_client = anthropic.AsyncAnthropic()
        self._anthropic_fallback_logged = False
        for region in (regions or _BEDROCK_REGIONS):
            self._region_clients.append(anthropic.AsyncAnthropicBedrock(aws_region=region))
            self._regions.append(region)
        self.messages = _MessagesProxy(self)


from nanochat.checkpoint_manager import save_checkpoint, load_model_from_dir
from nanochat.engine import Engine
from tasks.customjson import CustomJSON

from scripts.pre1900_scripts.constants import REASONING_SYSTEM_PROMPT

# -----------------------------------------------------------------------------
# Claude discovery judge

DISCOVERY_JUDGE_PROMPT = """\
You are a scientific accuracy judge. Given a problem, the correct answer, and a \
model's response, rate the model's response on a scale from 0 to 5.

IMPORTANT: If the model response is empty or contains no meaningful content, \
you MUST score it 0. Do not score the gold answer — only score the model's response.

Rubric:
0 = Completely wrong, irrelevant, or empty response
1 = Shows some relevant reasoning but reaches an incorrect conclusion
2 = Identifies part of the correct phenomenon but misses the key insight
3 = Gets the general direction right but with significant errors or gaps
4 = Mostly correct with minor inaccuracies or incomplete reasoning
5 = Fully correct, captures the core physical insight

Respond with ONLY a <score> tag containing your integer score. Example: <score>3</score>

Problem:
{prompt}

Correct answer:
{gold_answer}

Model response:
{response}"""


CONTRADICTION_JUDGE_PROMPT = """\
You are a scientific reasoning judge evaluating a model's response to a \
contradiction-resolution problem. Rate the response from 0 to 5.

IMPORTANT: If the model response is empty or contains no meaningful content, \
you MUST score it 0. Do not score the gold answer — only score the model's response.

Focus on whether the response:
1. Identifies the core classical assumption that fails or must be revised
2. Proposes a coherent replacement principle or interpretation
3. Explains the main observations rather than merely restating them
4. Avoids preserving the original contradiction
5. Shows conceptually correct understanding even if wording is informal

Do not require exact historical phrasing or equations. A response can earn a \
high score with simple language if the concepts are right. A response should \
lose points if it uses fancy words but the reasoning is confused or wrong.

Rubric:
0 = Incoherent, irrelevant, or does not engage with the contradiction
1 = Shows some relevant reasoning but does not identify which assumption fails
2 = Identifies that something is wrong but gives a vague or ad hoc fix
3 = Correctly identifies the failing assumption but the replacement is incomplete
4 = Clearly identifies the failing assumption and proposes a mostly correct replacement
5 = Excellent: identifies the failing assumption, proposes a correct replacement, \
and explains why it resolves the contradiction

Respond with ONLY a <score> tag containing your integer score. Example: <score>3</score>

Problem:
{prompt}

Correct answer:
{gold_answer}

Model response:
{response}"""


async def judge_discovery_single(
    client,
    prompt: str,
    response: str,
    gold_answer: str,
    model: str,
    semaphore: asyncio.Semaphore,
    judge_prompt_template: str = DISCOVERY_JUDGE_PROMPT,
) -> float:
    """Score a single response against the gold answer. Returns 0.0 to 1.0."""
    # Empty responses automatically score 0 — don't waste an API call
    if not response or not response.strip():
        return 0.0
    judge_input = judge_prompt_template.format(prompt=prompt, response=response, gold_answer=gold_answer)
    async with semaphore:
        try:
            result = await client.messages.create(
                model=model,
                max_tokens=64,
                messages=[
                    {"role": "user", "content": judge_input},
                    {"role": "assistant", "content": "<score>"},
                ],
            )
            response_text = result.content[0].text.strip()
            match = re.match(r"(\d)", response_text)
            if match is None:
                print0(f"  Judge parse error: {response_text[:100]}")
                return 0.0
            score = int(match.group(1))
            return float(max(0, min(5, score))) / 5.0  # normalize to [0, 1]
        except Exception as e:
            print0(f"  Judge API error: {e}")
            return 0.0  # conservative fallback on error


async def batch_judge_discovery(
    prompts: list[str],
    responses: list[str],
    gold_answers: list[str],
    model: str,
    max_concurrent: int,
    judge_prompt_template: str = DISCOVERY_JUDGE_PROMPT,
) -> list[float]:
    """Score a batch of prompt-response pairs concurrently against gold answers."""
    client = BedrockFallbackClient()
    semaphore = asyncio.Semaphore(max_concurrent)
    tasks = [
        judge_discovery_single(client, p, r, g, model, semaphore, judge_prompt_template)
        for p, r, g in zip(prompts, responses, gold_answers)
    ]
    return await asyncio.gather(*tasks)


COHERENCE_JUDGE_PROMPT = """\
You are an engagement-quality judge. Given a problem and a model's response, \
rate how coherently the response engages with the question on a scale from 0 to 5.

You are NOT judging correctness. You are judging whether the response:
1. Actually addresses the question asked (vs continuing unrelated text)
2. Has a logical reasoning flow (vs rambling, repeating, or nonsensical text)
3. Attempts to analyze the problem (vs ignoring it entirely)

Rubric:
0 = Does not engage with the question at all (random text, unrelated continuation)
1 = Barely engages — mentions a keyword but does not attempt to address the question
2 = Partially engages but reasoning is disorganized or mostly off-topic
3 = Addresses the question with some logical structure but has significant gaps
4 = Clearly engages with the question and reasoning flow is mostly logical
5 = Fully engages with the question, reasoning is well-structured and focused

Respond with ONLY a <score> tag containing your integer score. Example: <score>3</score>

Problem:
{prompt}

Model response:
{response}"""

COHERENCE_LOGICAL_JUDGE_PROMPT = """\
You are a reasoning-quality judge. Given a problem and a model's response, \
rate the coherence and logical flow of the response on a scale from 0 to 5.

You are NOT judging correctness. A response can be logically well-structured \
even if its conclusion is wrong. You ARE judging:
1. Engagement — does the response address the question asked, or does it ignore it / continue with unrelated text?
2. Sequential reasoning — does each claim build on the previous one, or does it jump between unconnected ideas?
3. Internal consistency — does the response contradict itself, make claims that conflict with its own earlier statements, or reach a conclusion that contradicts its reasoning? Self-contradiction is a strong signal of low coherence.
4. Progression toward a conclusion — does the reasoning move forward (observation → analysis → synthesis), or does it stall, repeat, or ramble?
5. No repetition or looping — does the response say the same thing multiple times in different words, or get stuck repeating a phrase/idea? Repetition and looping are strong signals of low coherence and should be penalized heavily.

Rubric:
0 = Does not engage with the question at all (random text, unrelated continuation, or empty)
1 = Mentions relevant concepts but no logical connections between them; OR heavily repetitive/looping (restates the same point 3+ times)
2 = Engages with the question but jumps between ideas, contradicts itself, loops back to restate earlier points, or is mostly disorganized
3 = Recognizable reasoning chain that addresses the question, but with gaps, some repetition, minor self-contradictions, or unsupported leaps
4 = Clearly engages with the question and builds a logical progression with no contradictions, no repetition, and only minor gaps
5 = Fully addresses the question with a tight, internally consistent reasoning chain — every claim connects to the prior one, building toward a coherent conclusion with no contradictions, redundancy, or looping

Respond with ONLY a <score> tag containing your integer score. Example: <score>3</score>

Problem:
{prompt}

Model response:
{response}"""


async def batch_judge_coherence(
    prompts: list[str],
    responses: list[str],
    model: str,
    max_concurrent: int,
    coherence_prompt: str = COHERENCE_JUDGE_PROMPT,
) -> list[float]:
    """Score a batch of responses for coherence/engagement quality."""
    client = BedrockFallbackClient()
    semaphore = asyncio.Semaphore(max_concurrent)

    async def judge_one(prompt, response):
        # Empty responses automatically score 0
        if not response or not response.strip():
            return 0.0
        judge_input = coherence_prompt.format(prompt=prompt, response=response)
        async with semaphore:
            try:
                result = await client.messages.create(
                    model=model,
                    max_tokens=64,
                    messages=[
                        {"role": "user", "content": judge_input},
                        {"role": "assistant", "content": "<score>"},
                    ],
                )
                response_text = result.content[0].text.strip()
                match = re.match(r"(\d)", response_text)
                if match is None:
                    print0(f"  Coherence judge parse error: {response_text[:100]}")
                    return 0.0
                score = int(match.group(1))
                return float(max(0, min(5, score))) / 5.0
            except Exception as e:
                print0(f"  Coherence judge API error: {e}")
                return 0.0

    tasks = [judge_one(p, r) for p, r in zip(prompts, responses)]
    return await asyncio.gather(*tasks)


FORMAT_REWARD = 0.3  # partial credit for using the correct reasoning format

def compute_format_reward(response: str) -> float:
    """Return FORMAT_REWARD if the response uses <think>...</think> and \\answer{...} tags, else 0."""
    has_think = bool(re.search(r"<think>.*?</think>", response, re.DOTALL))
    has_answer = bool(re.search(r"\\answer\{", response))
    return FORMAT_REWARD if (has_think and has_answer) else 0.0


# -----------------------------------------------------------------------------
# CLI arguments
parser = argparse.ArgumentParser(description="Discovery RL for pre-1900 model")
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
parser.add_argument("--max-new-tokens", type=int, default=512, help="max tokens to generate per sample")
parser.add_argument("--temperature", type=float, default=1.0, help="sampling temperature")
parser.add_argument("--top-k", type=int, default=50, help="top-k sampling (0 = disabled)")
# Optimization
parser.add_argument("--embedding-lr", type=float, default=0.2, help="learning rate for embedding parameters (Adam)")
parser.add_argument("--unembedding-lr", type=float, default=0.004, help="learning rate for unembedding parameters (Adam)")
parser.add_argument("--matrix-lr", type=float, default=0.02, help="learning rate for matrix parameters (Muon)")
parser.add_argument("--weight-decay", type=float, default=0.0, help="weight decay for embedding/unembedding parameters (Adam)")
parser.add_argument("--init-lr-frac", type=float, default=0.05, help="initial LR as fraction of base LR")
# Evaluation / checkpointing
parser.add_argument("--eval-every", type=int, default=60, help="evaluate correctness every N steps")
parser.add_argument("--eval-examples", type=int, default=50, help="number of examples for correctness evaluation")
parser.add_argument("--save-every", type=int, default=60, help="save checkpoint every N steps")
# Data
parser.add_argument("--train-data", type=str, default="instruct_data/rl_problems/rl_prompts_sys_train.jsonl", help="training prompts JSONL")
parser.add_argument("--val-data", type=str, default="instruct_data/rl_problems/rl_prompts_sys_val.jsonl", help="validation prompts JSONL")
parser.add_argument("--problems-data", type=str, default="instruct_data/rl_problems/rl_problems_train.jsonl", help="training problems with gold answers JSONL")
parser.add_argument("--problems-val-data", type=str, default="instruct_data/rl_problems/rl_problems_val.jsonl", help="validation problems with gold answers JSONL")
# Claude judge
parser.add_argument("--judge-style", type=str, default="general", choices=["general", "contradiction"], help="Judge prompt style: 'general' (existing) or 'contradiction' (assumption-identification)")
parser.add_argument("--judge-model", type=str, default="claude-sonnet-4-20250514", help="Anthropic model for correctness judging")
parser.add_argument("--max-concurrent-api", type=int, default=20, help="max concurrent API requests per rank")
# v6: no-scaffold mode
parser.add_argument("--no-scaffold", action="store_true", help="No system prompt, no format reward, completion-mode generation")
parser.add_argument("--coherence-reward", action="store_true", help="Enable coherence reward with dynamic weighting curriculum")
parser.add_argument("--ema-alpha", type=float, default=0.05, help="EMA smoothing factor for coherence tracking (lower = slower)")
parser.add_argument("--min-coherence-weight", type=float, default=0.1, help="Floor for coherence weight to prevent regression")
parser.add_argument("--fixed-coherence-weight", type=float, default=None, help="If set, use fixed coherence/correctness weights instead of EMA curriculum (e.g. 0.1 means 0.1*coherence + 0.9*correctness)")
parser.add_argument("--coherence-style", type=str, default="engagement", choices=["engagement", "logical"], help="Coherence judge style: 'engagement' (v6-v9) or 'logical' (v10+, emphasizes reasoning flow)")
# Output
parser.add_argument("--output-dir", type=str, default="pre1900_discovery_rl_checkpoints", help="output checkpoints directory (relative to base dir)")
args = parser.parse_args()
user_config = vars(args).copy()

# Select judge prompt based on --judge-style
active_judge_prompt = CONTRADICTION_JUDGE_PROMPT if args.judge_style == "contradiction" else DISCOVERY_JUDGE_PROMPT
active_coherence_prompt = COHERENCE_LOGICAL_JUDGE_PROMPT if args.coherence_style == "logical" else COHERENCE_JUDGE_PROMPT

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

# Init model and tokenizer from reasoning SFT checkpoints
base_dir = get_base_dir()
checkpoints_dir = os.path.join(base_dir, args.checkpoints_dir)
model, tokenizer, meta = load_model_from_dir(checkpoints_dir, device, phase="eval", model_tag=args.model_tag, step=args.model_step)
engine = Engine(model, tokenizer)

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

# Load gold answers (parallel to prompts)
def load_gold_answers(filepath):
    answers = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            answers.append(obj["gold_answer"])
    return answers

train_gold_answers = load_gold_answers(os.path.join(base_dir, args.problems_data))
val_gold_answers = load_gold_answers(os.path.join(base_dir, args.problems_val_data))
assert len(train_gold_answers) == len(train_task), f"Gold answers ({len(train_gold_answers)}) != prompts ({len(train_task)})"
assert len(val_gold_answers) == len(val_task), f"Val gold answers ({len(val_gold_answers)}) != val prompts ({len(val_task)})"
print0(f"Loaded {len(train_gold_answers)} train gold answers, {len(val_gold_answers)} val gold answers")


def extract_last_user_prompt(conversation):
    """Extract the last user message from a conversation for judging."""
    messages = conversation["messages"]
    for msg in reversed(messages):
        if msg["role"] == "user":
            return msg["content"]
    return ""


# -----------------------------------------------------------------------------
# Rollout / sampling generator loop that yields batches of examples for training

bos_token = tokenizer.get_bos_token_id()

# EMA state for coherence curriculum (v6)
ema_coherence = 0.0

@torch.no_grad()
def get_batch():
    global ema_coherence
    assistant_end = tokenizer.encode_special("<|assistant_end|>")
    rank_indices = range(ddp_rank, len(train_task), ddp_world_size)
    for example_idx in itertools.cycle(rank_indices):

        # Get the full conversation and gold answer
        conversation = train_task[example_idx]
        gold_answer = train_gold_answers[example_idx]
        prompt_text = extract_last_user_prompt(conversation)

        # Tokenize: completion mode (no-scaffold) or chat mode
        if args.no_scaffold:
            tokens = tokenizer.encode(prompt_text, prepend=bos_token)
        else:
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

        # Score all samples: format reward + correctness reward (+ optional coherence reward)
        if args.no_scaffold:
            format_rewards = [0.0] * len(generated_texts)  # no format reward in no-scaffold mode
        else:
            format_rewards = [compute_format_reward(text) for text in generated_texts]

        # Judge correctness (and optionally coherence) concurrently
        if args.coherence_reward:
            async def judge_both():
                correctness_task = batch_judge_discovery(
                    [prompt_text] * len(generated_texts),
                    generated_texts,
                    [gold_answer] * len(generated_texts),
                    args.judge_model,
                    args.max_concurrent_api,
                    active_judge_prompt,
                )
                coherence_task = batch_judge_coherence(
                    [prompt_text] * len(generated_texts),
                    generated_texts,
                    args.judge_model,
                    args.max_concurrent_api,
                    active_coherence_prompt,
                )
                return await asyncio.gather(correctness_task, coherence_task)
            correctness_rewards, coherence_rewards = asyncio.run(judge_both())

            if args.fixed_coherence_weight is not None:
                # v7: Fixed weighting — correctness drives learning, coherence prevents collapse
                coherence_weight = args.fixed_coherence_weight
                correctness_weight = 1.0 - args.fixed_coherence_weight
            else:
                # v6: Dynamic EMA weighting curriculum
                batch_mean_coherence = sum(coherence_rewards) / len(coherence_rewards)
                ema_coherence = (1 - args.ema_alpha) * ema_coherence + args.ema_alpha * batch_mean_coherence
                coherence_weight = max(args.min_coherence_weight, 1.0 - ema_coherence)
                correctness_weight = ema_coherence
            rewards_list = [
                f + coherence_weight * coh + correctness_weight * cor
                for f, coh, cor in zip(format_rewards, coherence_rewards, correctness_rewards)
            ]
        else:
            correctness_rewards = asyncio.run(batch_judge_discovery(
                [prompt_text] * len(generated_texts),
                generated_texts,
                [gold_answer] * len(generated_texts),
                args.judge_model,
                args.max_concurrent_api,
                active_judge_prompt,
            ))
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
        yield generated_token_sequences, inputs, targets, rewards, advantages


# -----------------------------------------------------------------------------
# Evaluation: mean correctness score on held-out val set

def run_correctness_eval(task, gold_answers, tokenizer, engine, max_examples=None, max_completion_tokens=512, temperature=0.0, top_k=50):
    """
    Evaluate correctness on held-out data. Each rank processes a subset,
    caller must reduce across ranks.
    """
    max_examples = min(max_examples, len(task)) if max_examples is not None else len(task)
    all_prompts = []
    all_responses = []
    all_golds = []
    for idx in range(ddp_rank, max_examples, ddp_world_size):
        conversation = task[idx]
        gold_answer = gold_answers[idx]
        if args.no_scaffold:
            prompt_text = extract_last_user_prompt(conversation)
            tokens = tokenizer.encode(prompt_text, prepend=bos_token)
        else:
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
        all_golds.append(gold_answer)

    # Batch judge all responses at once
    if all_prompts:
        scores = asyncio.run(batch_judge_discovery(
            all_prompts, all_responses, all_golds, args.judge_model, args.max_concurrent_api,
            active_judge_prompt,
        ))
    else:
        scores = []
    return scores


# -----------------------------------------------------------------------------
# Online physics eval (EVAL.json)

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
    """Build a judge prompt for one EVAL.json task, returning a single string for the async judge."""
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
    """Judge a batch of (task, response) pairs. Returns list of scores (0-5 scale, not normalized)."""
    client = BedrockFallbackClient()
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
        if args.no_scaffold:
            tokens = tokenizer.encode(prompt_text, prepend=bos_token)
        else:
            conversation = {
                "messages": [
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

    # Evaluate correctness once in a while
    if step % args.eval_every == 0:
        model.eval()
        with autocast_ctx:
            scores = run_correctness_eval(val_task, val_gold_answers, tokenizer, engine, max_examples=args.eval_examples, temperature=1.0)
        # Reduce across ranks
        score_sum = torch.tensor(sum(scores), dtype=torch.float, device=device)
        score_count = torch.tensor(len(scores), dtype=torch.long, device=device)
        if ddp:
            dist.all_reduce(score_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(score_count, op=dist.ReduceOp.SUM)
        mean_correctness = (score_sum / score_count.clamp(min=1)).item()
        print0(f"Step {step} | Eval correctness score: {mean_correctness:.4f}")
        wandb_run.log({
            "step": step,
            "eval/correctness_score": mean_correctness,
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
    log_dict = {
        "step": step,
        "reward": mean_reward,
        "sequence_length": mean_sequence_length,
    }
    if args.coherence_reward:
        if args.fixed_coherence_weight is not None:
            log_dict["reward/coherence_weight"] = args.fixed_coherence_weight
            log_dict["reward/correctness_weight"] = 1.0 - args.fixed_coherence_weight
        else:
            log_dict["reward/ema_coherence"] = ema_coherence
            log_dict["reward/coherence_weight"] = max(args.min_coherence_weight, 1.0 - ema_coherence)
            log_dict["reward/correctness_weight"] = ema_coherence
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

    # Save checkpoints
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
get_report().log(section="Pre-1900 Discovery RL", data=[
    user_config,
])

wandb_run.finish()
compute_cleanup()
