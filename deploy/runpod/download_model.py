"""
Download model checkpoint + tokenizer from HuggingFace Hub at Docker build time.

Only downloads a single model_*.pt, its meta_*.json, and the tokenizer/ dir.
Skips optimizer shards to keep the image small.

Usage:
    python download_model.py --repo mhla/gpt1900-d34-22btok --dest /app/model
    python download_model.py --repo mhla/gpt1900-d34-22btok --step 10507 --dest /app/model
"""
import argparse
import os
from huggingface_hub import snapshot_download, list_repo_tree


def find_latest_step(repo_id):
    """Find the highest model step in a HF repo."""
    steps = []
    for entry in list_repo_tree(repo_id):
        name = entry.path
        if name.startswith("model_") and name.endswith(".pt"):
            step = int(name.removeprefix("model_").removesuffix(".pt"))
            steps.append(step)
    if not steps:
        raise ValueError(f"No model_*.pt files found in {repo_id}")
    return max(steps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, help="HF repo id")
    parser.add_argument("--step", type=int, default=None, help="Checkpoint step (default: latest)")
    parser.add_argument("--dest", required=True, help="Local destination directory")
    args = parser.parse_args()

    os.makedirs(args.dest, exist_ok=True)

    step = args.step or find_latest_step(args.repo)
    step_str = f"{step:06d}"
    print(f"Downloading {args.repo} step={step} -> {args.dest}")

    snapshot_download(
        repo_id=args.repo,
        allow_patterns=[f"model_{step_str}.pt", f"meta_{step_str}.json", "tokenizer/*"],
        local_dir=args.dest,
    )

    # List what we downloaded
    for root, dirs, files in os.walk(args.dest):
        for f in sorted(files):
            path = os.path.join(root, f)
            size_mb = os.path.getsize(path) / 1e6
            print(f"  {path} ({size_mb:.1f} MB)")

    print("Download complete.")


if __name__ == "__main__":
    main()
