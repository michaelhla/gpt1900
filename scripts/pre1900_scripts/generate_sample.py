"""
Quick sanity check: generate from the reasoning SFT model to see if it follows
the <think>...</think> \\answer{} format.

Usage (single GPU):
    pixi run python3 -m scripts.pre1900_scripts.generate_sample
"""
import torch
from contextlib import nullcontext

from nanochat.common import autodetect_device_type, get_base_dir
from nanochat.engine import Engine
from nanochat.checkpoint_manager import load_model_from_dir

import os

SYSTEM_PROMPT = (
    "You are a scientist trained in the experimental method. You think carefully and "
    "systematically — observing phenomena, forming hypotheses, and reasoning from first "
    "principles to draw conclusions. Think step by step inside <think> tags, then state "
    "your conclusion in \\answer{} tags."
)

SAMPLE_QUESTIONS = [
    # From training distribution
    (
        "A wire carrying an electrical current is placed near a compass needle. "
        "When the current flows, the needle is deflected; when the current ceases, "
        "the needle returns to its original orientation. What does this observation "
        "reveal about the relationship between electrical currents and magnetism?"
    ),
    # Slightly out-of-distribution but same style
    (
        "A beam of light passes through a glass prism and separates into a spectrum "
        "of colours. The violet rays are bent most strongly, the red rays least. "
        "What does this tell us about the nature of white light and the dependence "
        "of refraction upon the colour of the light?"
    ),
]

MAX_NEW_TOKENS = 1024
TOP_K = 50

def run_sample(engine, tokenizer, conversation, autocast_ctx, temperature, label=""):
    tokens = tokenizer.render_for_completion(conversation)
    bos = tokenizer.get_bos_token_id()

    print(f"\nASSISTANT (temp={temperature}):", flush=True)
    full_response = []

    with autocast_ctx:
        for token_column, _masks in engine.generate(
            tokens,
            num_samples=1,
            max_tokens=MAX_NEW_TOKENS,
            temperature=temperature,
            top_k=TOP_K,
        ):
            tok = token_column[0]
            if tok == bos:
                break
            text = tokenizer.decode([tok])
            print(text, end="", flush=True)
            full_response.append(text)

    full_text = "".join(full_response)
    print("\n")

    has_think = "<think>" in full_text and "</think>" in full_text
    has_answer = "\\answer{" in full_text
    print(f"  [Format check] <think>...</think>: {has_think} | \\answer{{}}: {has_answer}")
    return full_text

def main():
    device_type = autodetect_device_type()
    device = torch.device(device_type)
    ptdtype = torch.bfloat16
    autocast_ctx = (
        torch.amp.autocast(device_type=device_type, dtype=ptdtype)
        if device_type == "cuda"
        else nullcontext()
    )

    base_dir = get_base_dir()
    checkpoints_dir = os.path.join(base_dir, "pre1900_discovery_rl_v2_checkpoints")

    print(f"Loading model from {checkpoints_dir} ...")
    model, tokenizer, meta = load_model_from_dir(
        checkpoints_dir, device, phase="eval", model_tag="d34", step=90
    )
    engine = Engine(model, tokenizer)
    print(f"Model loaded: {meta.get('model_config', {})}\n")

    # -------------------------------------------------------------------------
    # Test 1: a verbatim training example (greedy decode)
    print("=" * 70)
    print("TEST 1: Verbatim training example (first question from sft_train.jsonl)")
    import json
    train_path = os.path.join(base_dir, "instruct_data/reasoning/sft_train.jsonl")
    with open(train_path) as f:
        first_example = json.loads(f.readline())
    # first_example is a list of message dicts
    conversation_train = {"messages": first_example}
    user_content = first_example[1]["content"]  # index 1 since 0 is system
    print(f"USER: {user_content[:300]}...\n")
    run_sample(engine, tokenizer, conversation_train, autocast_ctx, temperature=0.0, label="train_ex_greedy")

    # -------------------------------------------------------------------------
    # Test 2: new questions at different temperatures
    for i, question in enumerate(SAMPLE_QUESTIONS):
        print("=" * 70)
        print(f"\n--- Sample {i+1} ---")
        print(f"USER: {question}\n")

        conversation = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",      "content": question},
                {"role": "assistant", "content": "placeholder"},  # stripped
            ]
        }

        for temp in [0.0, 0.5]:
            run_sample(engine, tokenizer, conversation, autocast_ctx, temperature=temp)

    print("=" * 70)


if __name__ == "__main__":
    main()
