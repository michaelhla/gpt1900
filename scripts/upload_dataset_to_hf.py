"""Upload all project data to HuggingFace as separate private dataset repos."""

import argparse
import sys
from pathlib import Path
from huggingface_hub import HfApi


def upload_physics_data(api: HfApi, dry_run: bool = False):
    repo_id = "mhla/gpt1900-physics-data"
    print(f"\n{'='*60}")
    print(f"Uploading: {repo_id}")
    print(f"{'='*60}")

    if dry_run:
        print("[DRY RUN] Would create repo and upload physics data")
        return

    api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)

    local_root = Path("/mnt/main0/home/michaelhla/evolutionaryscale/gpt1900/data")

    print("  Uploading physics_books/...")
    api.upload_folder(
        folder_path=str(local_root / "physics_books"),
        path_in_repo="physics_books",
        repo_id=repo_id,
        repo_type="dataset",
    )

    print("  Uploading core_physics_books/...")
    api.upload_folder(
        folder_path=str(local_root / "core_physics_books"),
        path_in_repo="core_physics_books",
        repo_id=repo_id,
        repo_type="dataset",
    )

    clm_path = Path("/mnt/main0/data/michaelhla/gpt1900_training/physics_clm_data")
    print("  Uploading physics_clm/ (train.parquet + val.parquet)...")
    api.upload_folder(
        folder_path=str(clm_path),
        path_in_repo="physics_clm",
        repo_id=repo_id,
        repo_type="dataset",
    )

    print(f"  Done: https://huggingface.co/datasets/{repo_id}")


def upload_instruct_data(api: HfApi, dry_run: bool = False):
    repo_id = "mhla/gpt1900-instruct-data"
    print(f"\n{'='*60}")
    print(f"Uploading: {repo_id}")
    print(f"{'='*60}")

    if dry_run:
        print("[DRY RUN] Would create repo and upload instruct data")
        return

    api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)

    instruct_dir = Path("/mnt/main0/data/michaelhla/gpt1900_training/instruct_data")

    print("  Uploading full instruct_data/ directory tree...")
    api.upload_folder(
        folder_path=str(instruct_dir),
        repo_id=repo_id,
        repo_type="dataset",
    )

    print(f"  Done: https://huggingface.co/datasets/{repo_id}")


def upload_pre1905(api: HfApi, dry_run: bool = False):
    repo_id = "mhla/pre1905-corpus"
    source = Path("/mnt/main0/data/michaelhla/pre1905_full_clean")
    print(f"\n{'='*60}")
    print(f"Uploading: {repo_id}")
    print(f"  Source: {source} (68 GB, 445 shards)")
    print(f"{'='*60}")

    if dry_run:
        print("[DRY RUN] Would create repo and upload pre-1905 corpus")
        return

    api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)

    print("  Uploading all parquet shards (this will take a while)...")
    api.upload_folder(
        folder_path=str(source),
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns="*.parquet",
    )

    print(f"  Done: https://huggingface.co/datasets/{repo_id}")


def upload_pre1915(api: HfApi, dry_run: bool = False):
    repo_id = "mhla/pre1915-corpus"
    source = Path("/mnt/main0/data/michaelhla/pre1915_full_clean")
    print(f"\n{'='*60}")
    print(f"Uploading: {repo_id}")
    print(f"  Source: {source} (197 GB, 1116 shards)")
    print(f"{'='*60}")

    if dry_run:
        print("[DRY RUN] Would create repo and upload pre-1915 corpus")
        return

    api.create_repo(repo_id, repo_type="dataset", private=False, exist_ok=True)

    print("  Uploading all parquet shards (this will take a long while)...")
    api.upload_folder(
        folder_path=str(source),
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns="*.parquet",
    )

    print(f"  Done: https://huggingface.co/datasets/{repo_id}")


TARGETS = {
    "physics": upload_physics_data,
    "instruct": upload_instruct_data,
    "pre1905": upload_pre1905,
    "pre1915": upload_pre1915,
}


def main():
    parser = argparse.ArgumentParser(description="Upload data to HuggingFace")
    parser.add_argument(
        "targets",
        nargs="*",
        default=None,
        help="Which datasets to upload: physics, instruct, pre1905, pre1915 (default: all)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print what would be done without uploading")
    args = parser.parse_args()

    api = HfApi()
    user = api.whoami()["name"]
    print(f"Authenticated as: {user}")

    targets = args.targets or list(TARGETS.keys())
    for target in targets:
        if target not in TARGETS:
            print(f"Unknown target: {target}. Choose from: {list(TARGETS.keys())}")
            sys.exit(1)
        TARGETS[target](api, dry_run=args.dry_run)

    print(f"\n{'='*60}")
    print("All uploads complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
