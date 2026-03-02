"""
RunPod serverless handler for nanochat inference.

Generator handler that loads the model once at module level (persists across jobs)
and yields token chunks per job for RunPod's streaming protocol.
"""
import os
import random
import torch
import runpod

# Point nanochat at /app/model so get_tokenizer() finds /app/model/tokenizer/
os.environ["NANOCHAT_BASE_DIR"] = "/app/model"

from nanochat.checkpoint_manager import build_model, find_last_step
from nanochat.engine import Engine

# ---------------------------------------------------------------------------
# Load model ONCE at module level (persists across RunPod jobs on same worker)
# ---------------------------------------------------------------------------
device = torch.device("cuda")
model_dir = "/app/model"
step = find_last_step(model_dir)
model, tokenizer, meta = build_model(model_dir, step, device, phase="eval")
engine = Engine(model, tokenizer)
print(f"Model loaded: step={step}, config={meta['model_config']}")


def handler(job):
    """RunPod generator handler — yields {"text": ...} chunks."""
    inp = job["input"]
    messages = inp.get("messages", [])
    temperature = inp.get("temperature", 0.8)
    top_k = inp.get("top_k", 50)
    max_tokens = inp.get("max_tokens", 512)

    # Build conversation token sequence (mirrors chat_web.py:331-349)
    bos = tokenizer.get_bos_token_id()
    user_start = tokenizer.encode_special("<|user_start|>")
    user_end = tokenizer.encode_special("<|user_end|>")
    assistant_start = tokenizer.encode_special("<|assistant_start|>")
    assistant_end = tokenizer.encode_special("<|assistant_end|>")

    tokens = [bos]
    for msg in messages:
        if msg["role"] == "user":
            tokens.append(user_start)
            tokens.extend(tokenizer.encode(msg["content"]))
            tokens.append(user_end)
        elif msg["role"] == "assistant":
            tokens.append(assistant_start)
            tokens.extend(tokenizer.encode(msg["content"]))
            tokens.append(assistant_end)
    tokens.append(assistant_start)

    # Stream generation (mirrors chat_web.py:262-311)
    accumulated = []
    last_clean = ""

    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        for token_column, _ in engine.generate(
            tokens,
            num_samples=1,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            seed=random.randint(0, 2**31 - 1),
        ):
            tok = token_column[0]
            if tok == assistant_end or tok == bos:
                break

            accumulated.append(tok)
            text = tokenizer.decode(accumulated)

            # Only emit when no incomplete UTF-8 replacement char
            if not text.endswith("\ufffd"):
                new = text[len(last_clean):]
                if new:
                    yield {"text": new}
                    last_clean = text


runpod.serverless.start({"handler": handler, "return_aggregate_stream": True})
