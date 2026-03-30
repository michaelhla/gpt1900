"""
Clean reasoning data and rebuild SFT + RL splits.

1. Strip newlines from <think>...</think> blocks in assistant responses
2. Update the system prompt to the new version
3. Rebuild sft_train_trimmed.jsonl (817 examples)
4. Rebuild rl_problems_train_expanded.jsonl + rl_prompts_sys_train_expanded.jsonl (800 examples)
5. Also clean sft_val.jsonl and rl_prompts_sys_val.jsonl

Usage:
    pixi run python3 -m scripts.pre1900_scripts.clean_and_rebuild_data
"""

import json
import re
import random
from pathlib import Path

from nanochat.common import get_base_dir

SEED = 42
N_CONVERT = 515

SYSTEM_PROMPT = (
    "You are a scientist trained in the experimental method. You think carefully and "
    "systematically. Observe phenomena, form hypotheses, and reason from first "
    "principles to draw conclusions. Think step by step inside <think> tags, then state "
    "your conclusion in \\answer{} tags."
)


def strip_newlines_in_think(text: str) -> str:
    """Replace newlines inside <think>...</think> with spaces, collapse whitespace."""
    def _clean_think(m):
        inner = m.group(1)
        inner = inner.replace("\n", " ")
        inner = re.sub(r"  +", " ", inner).strip()
        return f"<think>{inner}</think>"
    return re.sub(r"<think>(.*?)</think>", _clean_think, text, flags=re.DOTALL)


def clean_conversation(msgs: list[dict]) -> list[dict]:
    """Update system prompt and strip newlines from assistant think blocks."""
    out = []
    for msg in msgs:
        msg = dict(msg)
        if msg["role"] == "system":
            msg["content"] = SYSTEM_PROMPT
        elif msg["role"] == "assistant":
            msg["content"] = strip_newlines_in_think(msg["content"])
        out.append(msg)
    return out


def extract_answer(text: str) -> str | None:
    m = re.search(r'\\answer\{(.*?)\}', text, re.DOTALL)
    return m.group(1).strip() if m else None


def main():
    base = Path(get_base_dir())
    sft_dir = base / "instruct_data" / "reasoning"
    rl_dir  = base / "instruct_data" / "rl_problems"

    # -------------------------------------------------------------------------
    # Load and clean all SFT train examples
    with open(sft_dir / "sft_train.jsonl") as f:
        sft_raw = [json.loads(l) for l in f]
    print(f"Loaded {len(sft_raw)} SFT train examples")

    sft_cleaned = [clean_conversation(ex) for ex in sft_raw]

    # Verify cleaning worked
    sample = sft_cleaned[0][2]["content"]
    think_m = re.search(r"<think>(.*?)</think>", sample, re.DOTALL)
    if think_m:
        newlines_after = think_m.group(1).count("\n")
        print(f"  Sample <think> newlines after cleaning: {newlines_after}")

    # -------------------------------------------------------------------------
    # Split: 817 SFT keep, 515 → RL
    rng = random.Random(SEED)
    indices = list(range(len(sft_cleaned)))
    rng.shuffle(indices)

    convert_idx = set(indices[:N_CONVERT])
    keep_idx    = set(indices[N_CONVERT:])

    sft_keep    = [sft_cleaned[i] for i in sorted(keep_idx)]
    sft_convert = [sft_cleaned[i] for i in sorted(convert_idx)]

    print(f"  SFT keep   : {len(sft_keep)}")
    print(f"  SFT convert: {len(sft_convert)}")

    # -------------------------------------------------------------------------
    # Build RL entries from converted SFT
    new_rl_problems = []
    new_rl_prompts  = []
    skipped = 0

    for ex in sft_convert:
        user_content = ex[1]["content"]
        gold = extract_answer(ex[2]["content"])
        if gold is None:
            skipped += 1
            continue
        new_rl_problems.append({
            "_source": "sft_converted",
            "prompt": user_content,
            "gold_answer": gold,
        })
        new_rl_prompts.append([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
            {"role": "assistant", "content": "placeholder"},
        ])

    print(f"  Converted  : {len(new_rl_problems)} ({skipped} skipped)")

    # -------------------------------------------------------------------------
    # Load existing RL data and clean prompts
    with open(rl_dir / "rl_problems_train.jsonl") as f:
        existing_problems = [json.loads(l) for l in f]
    with open(rl_dir / "rl_prompts_sys_train.jsonl") as f:
        existing_prompts = [json.loads(l) for l in f]

    # Update system prompt in existing RL prompts
    for conv in existing_prompts:
        conv[0]["content"] = SYSTEM_PROMPT

    print(f"\nExisting RL train: {len(existing_problems)}")

    # -------------------------------------------------------------------------
    # Clean and write SFT val
    with open(sft_dir / "sft_val.jsonl") as f:
        sft_val_raw = [json.loads(l) for l in f]
    sft_val_cleaned = [clean_conversation(ex) for ex in sft_val_raw]

    out_sft_val = sft_dir / "sft_val_clean.jsonl"
    with open(out_sft_val, "w") as f:
        for ex in sft_val_cleaned:
            f.write(json.dumps(ex) + "\n")
    print(f"Wrote {len(sft_val_cleaned):4d} examples → {out_sft_val.name}")

    # -------------------------------------------------------------------------
    # Clean and write RL val prompts
    with open(rl_dir / "rl_prompts_sys_val.jsonl") as f:
        rl_val_prompts = [json.loads(l) for l in f]
    for conv in rl_val_prompts:
        conv[0]["content"] = SYSTEM_PROMPT

    out_rl_val = rl_dir / "rl_prompts_sys_val_clean.jsonl"
    with open(out_rl_val, "w") as f:
        for conv in rl_val_prompts:
            f.write(json.dumps(conv) + "\n")
    print(f"Wrote {len(rl_val_prompts):4d} examples → {out_rl_val.name}")

    # -------------------------------------------------------------------------
    # Write train outputs
    out_sft = sft_dir / "sft_train_trimmed.jsonl"
    with open(out_sft, "w") as f:
        for ex in sft_keep:
            f.write(json.dumps(ex) + "\n")
    print(f"Wrote {len(sft_keep):4d} examples → {out_sft.name}")

    combined_problems = existing_problems + new_rl_problems
    out_problems = rl_dir / "rl_problems_train_expanded.jsonl"
    with open(out_problems, "w") as f:
        for ex in combined_problems:
            f.write(json.dumps(ex) + "\n")
    print(f"Wrote {len(combined_problems):4d} examples → {out_problems.name}")

    combined_prompts = existing_prompts + new_rl_prompts
    out_prompts = rl_dir / "rl_prompts_sys_train_expanded.jsonl"
    with open(out_prompts, "w") as f:
        for conv in combined_prompts:
            f.write(json.dumps(conv) + "\n")
    print(f"Wrote {len(combined_prompts):4d} examples → {out_prompts.name}")

    print("\nDone. All files cleaned and rebuilt.")


if __name__ == "__main__":
    main()
