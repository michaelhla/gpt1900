"""
Expand the RL dataset by converting SFT examples into RL problems.

Takes 515 examples from sft_train.jsonl, extracts the \\answer{} as the gold
answer, and appends them to the RL problems files. The remaining 817 SFT
examples go to sft_train_trimmed.jsonl (does not overwrite originals).

Final counts:
  SFT train:  1332 - 515 = 817  (sft_train_trimmed.jsonl)
  RL train:    285 + 515 = 800  (rl_problems_train_expanded.jsonl,
                                  rl_prompts_sys_train_expanded.jsonl)

Usage:
    pixi run python3 -m scripts.pre1900_scripts.expand_rl_dataset
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


def extract_answer(assistant_content: str) -> str | None:
    m = re.search(r'\\answer\{(.*?)\}', assistant_content, re.DOTALL)
    return m.group(1).strip() if m else None


def main():
    base = Path(get_base_dir())
    sft_dir = base / "instruct_data" / "reasoning"
    rl_dir  = base / "instruct_data" / "rl_problems"

    # -------------------------------------------------------------------------
    # Load and split SFT train
    with open(sft_dir / "sft_train.jsonl") as f:
        sft_examples = [json.loads(l) for l in f]
    print(f"Loaded {len(sft_examples)} SFT train examples")

    rng = random.Random(SEED)
    indices = list(range(len(sft_examples)))
    rng.shuffle(indices)

    convert_idx = set(indices[:N_CONVERT])
    keep_idx    = set(indices[N_CONVERT:])

    sft_keep    = [sft_examples[i] for i in sorted(keep_idx)]
    sft_convert = [sft_examples[i] for i in sorted(convert_idx)]

    print(f"  SFT keep   : {len(sft_keep)}")
    print(f"  SFT convert: {len(sft_convert)}")

    # -------------------------------------------------------------------------
    # Build new RL entries from converted SFT examples
    new_rl_problems = []
    new_rl_prompts  = []
    skipped = 0

    for ex in sft_convert:
        # ex = [system_msg, user_msg, assistant_msg]
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
        ])

    print(f"  Converted  : {len(new_rl_problems)} ({skipped} skipped)")

    # -------------------------------------------------------------------------
    # Load existing RL files
    with open(rl_dir / "rl_problems_train.jsonl") as f:
        existing_problems = [json.loads(l) for l in f]
    with open(rl_dir / "rl_prompts_sys_train.jsonl") as f:
        existing_prompts = [json.loads(l) for l in f]

    print(f"\nExisting RL train: {len(existing_problems)} problems / {len(existing_prompts)} prompts")

    # -------------------------------------------------------------------------
    # Write outputs (new filenames — originals untouched)
    out_sft = sft_dir / "sft_train_trimmed.jsonl"
    with open(out_sft, "w") as f:
        for ex in sft_keep:
            f.write(json.dumps(ex) + "\n")
    print(f"\nWrote {len(sft_keep):4d} examples → {out_sft.name}")

    out_problems = rl_dir / "rl_problems_train_expanded.jsonl"
    combined_problems = existing_problems + new_rl_problems
    with open(out_problems, "w") as f:
        for ex in combined_problems:
            f.write(json.dumps(ex) + "\n")
    print(f"Wrote {len(combined_problems):4d} examples → {out_problems.name}")

    out_prompts = rl_dir / "rl_prompts_sys_train_expanded.jsonl"
    combined_prompts = existing_prompts + new_rl_prompts
    with open(out_prompts, "w") as f:
        for ex in combined_prompts:
            f.write(json.dumps(ex) + "\n")
    print(f"Wrote {len(combined_prompts):4d} examples → {out_prompts.name}")

    print("\nAll done. Val files are unchanged.")


if __name__ == "__main__":
    main()
