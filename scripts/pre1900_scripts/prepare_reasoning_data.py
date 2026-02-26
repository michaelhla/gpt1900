"""
Prepend the reasoning system prompt to existing SFT and RL data files.

Reads filtered_pairs.jsonl / val_pairs.jsonl (reasoning SFT) and
rl_prompts_train/val.jsonl (RL prompts), prepends a system message,
and writes the augmented versions.

Usage:
    python -m scripts.pre1900_scripts.prepare_reasoning_data --base-dir $NANOCHAT_BASE_DIR
"""

import argparse
import json
import os

from scripts.pre1900_scripts.constants import REASONING_SYSTEM_PROMPT

SYSTEM_MESSAGE = {"role": "system", "content": REASONING_SYSTEM_PROMPT}

# (input_relative_path, output_relative_path)
FILE_PAIRS = [
    ("instruct_data/reasoning/filtered_pairs.jsonl", "instruct_data/reasoning/sft_train.jsonl"),
    ("instruct_data/reasoning/val_pairs.jsonl", "instruct_data/reasoning/sft_val.jsonl"),
    ("instruct_data/rl_problems/rl_prompts_train.jsonl", "instruct_data/rl_problems/rl_prompts_sys_train.jsonl"),
    ("instruct_data/rl_problems/rl_prompts_val.jsonl", "instruct_data/rl_problems/rl_prompts_sys_val.jsonl"),
]


def prepend_system_prompt(input_path: str, output_path: str) -> int:
    """Read a JSONL file, prepend the system message to each conversation, write output. Returns line count."""
    count = 0
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            messages = json.loads(line)
            # Skip if system message already present
            if messages[0]["role"] == "system":
                augmented = messages
            else:
                augmented = [SYSTEM_MESSAGE] + messages
            fout.write(json.dumps(augmented, ensure_ascii=False) + "\n")
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description="Prepend reasoning system prompt to data files")
    parser.add_argument("--base-dir", type=str, default=None, help="Base directory (defaults to NANOCHAT_BASE_DIR)")
    args = parser.parse_args()

    base_dir = args.base_dir or os.environ.get("NANOCHAT_BASE_DIR")
    if not base_dir:
        from nanochat.common import get_base_dir
        base_dir = get_base_dir()

    print(f"Base directory: {base_dir}")
    print(f"System prompt: {REASONING_SYSTEM_PROMPT[:80]}...")

    for input_rel, output_rel in FILE_PAIRS:
        input_path = os.path.join(base_dir, input_rel)
        output_path = os.path.join(base_dir, output_rel)
        if not os.path.exists(input_path):
            print(f"  SKIP (not found): {input_path}")
            continue
        count = prepend_system_prompt(input_path, output_path)
        print(f"  {count:>5d} lines: {input_rel} -> {output_rel}")

    print("Done.")


if __name__ == "__main__":
    main()
