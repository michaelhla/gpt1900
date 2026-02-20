"""
Generate text from a base model checkpoint on a single GPU.

Usage:
    # Interactive mode - keeps prompting for input
    python -m scripts.generate --step 5000

    # Single prompt
    python -m scripts.generate --step 5000 -p "The theory of evolution"

    # Adjust generation params
    python -m scripts.generate --step 5000 -t 0.8 -k 100 --max-tokens 512

Requires NANOCHAT_BASE_DIR to be set if checkpoints aren't in ~/.cache/nanochat.
"""
import argparse
import torch
from contextlib import nullcontext
from nanochat.common import compute_init, autodetect_device_type
from nanochat.engine import Engine
from nanochat.checkpoint_manager import load_model

parser = argparse.ArgumentParser(description='Generate from a base model checkpoint')
parser.add_argument('-s', '--step', type=int, default=None, help='Checkpoint step to load (default: latest)')
parser.add_argument('-g', '--model-tag', type=str, default=None, help='Model tag, e.g. d26 (default: largest)')
parser.add_argument('-p', '--prompt', type=str, default=None, help='Single prompt, then exit. Omit for interactive mode.')
parser.add_argument('-t', '--temperature', type=float, default=0.8, help='Sampling temperature (default: 0.8)')
parser.add_argument('-k', '--top-k', type=int, default=50, help='Top-k sampling (default: 50)')
parser.add_argument('--max-tokens', type=int, default=512, help='Max tokens to generate (default: 512)')
parser.add_argument('--device-type', type=str, default='', choices=['cuda', 'cpu', 'mps'], help='Device type (default: autodetect)')
parser.add_argument('-d', '--dtype', type=str, default='bfloat16', choices=['float32', 'bfloat16'])
args = parser.parse_args()

# Init
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
ptdtype = torch.float32 if args.dtype == 'float32' else torch.bfloat16
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()

# Load model
model, tokenizer, meta = load_model("base", device, phase="eval", model_tag=args.model_tag, step=args.step)
engine = Engine(model, tokenizer)
bos = tokenizer.get_bos_token_id()

config = meta.get("model_config", {})
print(f"\nModel: d{config.get('n_layer', '?')} | step {args.step or 'latest'} | {device_type}")
print(f"Params: temperature={args.temperature}, top_k={args.top_k}, max_tokens={args.max_tokens}")
print("-" * 60)

def generate_from_prompt(prompt_text):
    tokens = tokenizer.encode(prompt_text, prepend=bos)
    print(prompt_text, end="", flush=True)
    with autocast_ctx:
        for token_column, token_masks in engine.generate(tokens, num_samples=1, max_tokens=args.max_tokens,
                                                          temperature=args.temperature, top_k=args.top_k):
            tok = token_column[0]
            # stop on bos (model thinks it's done)
            if tok == bos:
                break
            text = tokenizer.decode([tok])
            print(text, end="", flush=True)
    print("\n")

if args.prompt:
    generate_from_prompt(args.prompt)
else:
    print("Interactive mode. Enter a prompt and the model will continue it.")
    print("Commands: 'quit'/'exit' to stop, 'temp=X' to change temperature, 'topk=X' to change top_k")
    print("-" * 60)
    while True:
        try:
            user_input = input("\nPrompt: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ['quit', 'exit']:
            print("Goodbye!")
            break

        # Allow on-the-fly param changes
        if user_input.startswith("temp="):
            args.temperature = float(user_input.split("=")[1])
            print(f"Temperature set to {args.temperature}")
            continue
        if user_input.startswith("topk="):
            args.top_k = int(user_input.split("=")[1])
            print(f"Top-k set to {args.top_k}")
            continue
        if user_input.startswith("maxtok="):
            args.max_tokens = int(user_input.split("=")[1])
            print(f"Max tokens set to {args.max_tokens}")
            continue

        generate_from_prompt(user_input)
