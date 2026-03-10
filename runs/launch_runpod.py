#!/usr/bin/env python3
"""
Launch a training job on RunPod.

Usage:
    # Run a shell script (full pipeline)
    python runs/launch_runpod.py --script runs/speedrun.sh
    python runs/launch_runpod.py --script runs/run_pre1900_sft.sh
    python runs/launch_runpod.py --script runs/run_discovery_rl.sh

    # Run an arbitrary command (single stage)
    python runs/launch_runpod.py --cmd "torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=26 --fp8"
    python runs/launch_runpod.py --cmd "torchrun --standalone --nproc_per_node=8 -m scripts.pre1900_sft -- --run=my_sft"
    python runs/launch_runpod.py --cmd "torchrun --standalone --nproc_per_node=8 -m scripts.pre1900_scripts.discovery_rl -- --run=my_rl"

    # With data/checkpoints from HuggingFace
    python runs/launch_runpod.py --script runs/run_discovery_rl.sh \\
        --hf-data-repo mhla/gpt1900-data \\
        --env WANDB_API_KEY HF_TOKEN OPENAI_API_KEY

    # Dry run
    python runs/launch_runpod.py --script runs/speedrun.sh --dry-run
"""

import argparse
import base64
import json
import os
import re
import signal
import sys
import time

REPO_URL = "https://github.com/michaelhla/gpt1900"
DEFAULT_IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
DEFAULT_GPU_TYPE = "NVIDIA H100 80GB HBM3"
DEFAULT_GPU_COUNT = 8
DEFAULT_CONTAINER_DISK_GB = 50
DEFAULT_VOLUME_GB = 200
DEFAULT_HF_REPO = "mhla/gpt1900-checkpoints"
WORK_DIR = "/workspace/gpt1900"
# All training scripts resolve data/checkpoints relative to this via nanochat.common.get_base_dir()
NANOCHAT_BASE_DIR = "$HOME/.cache/nanochat"

POLL_INTERVAL = 30  # seconds


def _escape_for_graphql(cmd: str) -> str:
    """Escape double quotes for RunPod's GraphQL API.

    The runpod SDK interpolates docker_args into GraphQL as:
        dockerArgs: "{docker_args}"
    so any " in our string breaks the query.
    """
    return cmd.replace("\\", "\\\\").replace('"', '\\"')


# Lines in SLURM scripts that need to be stripped for RunPod.
# The launcher's setup steps already handle cd, PATH, venv activation, etc.
SLURM_STRIP_PATTERNS = [
    re.compile(r"^#SBATCH\b"),                  # SBATCH directives
    re.compile(r"^cd /mnt/"),                    # hardcoded SLURM working dir
    re.compile(r'^export PATH=".*pixi'),         # pixi-based PATH override (.pixi)
    re.compile(r"^export CONDA_PREFIX="),         # SLURM conda prefix
    re.compile(r"^export PYTHONPATH=/mnt/"),      # SLURM PYTHONPATH
    re.compile(r"^export NANOCHAT_BASE_DIR=/mnt/"),  # hardcoded SLURM base dir
]


def sanitize_script(script_path: str) -> str:
    """Read a shell script and strip SLURM-specific preamble lines.

    Returns a sanitized inline script string that can run on RunPod.
    NANOCHAT_BASE_DIR, OMP_NUM_THREADS, and venv activation are set by the
    launcher's setup steps, so we strip duplicates from the script.
    """
    with open(script_path) as f:
        lines = f.readlines()

    out = []
    for line in lines:
        stripped = line.strip()
        # Skip lines matching SLURM patterns
        if any(p.match(stripped) for p in SLURM_STRIP_PATTERNS):
            continue
        out.append(line)

    return "".join(out)


def build_startup_command(
    *,
    script: str | None,
    cmd: str | None,
    hf_repo: str,
    hf_data_repo: str | None,
    setup_cmd: str | None,
) -> str:
    """Build the shell command that runs inside the pod."""

    setup_steps = [
        # Clone repo
        f"git clone {REPO_URL} {WORK_DIR}",
        f"cd {WORK_DIR}",
        # Install uv + deps
        "command -v uv || (curl -LsSf https://astral.sh/uv/install.sh | sh)",
        "export PATH=/root/.local/bin:$PATH",
        "uv sync --extra gpu",
        # Activate venv so torchrun/python use project deps
        "source .venv/bin/activate",
        # Set env vars that all training scripts expect
        "export OMP_NUM_THREADS=1",
        f"export NANOCHAT_BASE_DIR={NANOCHAT_BASE_DIR}",
        f"mkdir -p {NANOCHAT_BASE_DIR}",
    ]

    # Download training data / checkpoints from HuggingFace if requested.
    # SFT/RL scripts expect data at {NANOCHAT_BASE_DIR}/instruct_data/...
    # and checkpoints at {NANOCHAT_BASE_DIR}/{checkpoints_dir}/...
    if hf_data_repo:
        setup_steps.append(f"echo Downloading data from {hf_data_repo}...")
        setup_steps.append(
            f"huggingface-cli download {hf_data_repo} --local-dir {NANOCHAT_BASE_DIR}"
        )

    # Optional user-defined setup command (e.g. download specific checkpoint)
    if setup_cmd:
        setup_steps.append(setup_cmd)

    teardown_steps = [
        # Upload checkpoints to HuggingFace
        "echo Uploading checkpoints to HuggingFace...",
        f"huggingface-cli upload {hf_repo} {NANOCHAT_BASE_DIR}/ --include '*.pt' '*.json' 'tokenizer/*' || echo HF upload failed",
        # Self-terminate
        "echo Training complete. Terminating pod...",
        "runpodctl stop pod $RUNPOD_POD_ID",
    ]

    if cmd:
        # Direct command — just sandwich between setup and teardown
        all_steps = setup_steps + [cmd] + teardown_steps
        return " && ".join(all_steps)
    elif script:
        # Shell script — sanitize, base64-encode, and decode on pod.
        # This avoids all quoting issues (heredoc + single/double quotes
        # break inside bash -c '...' which is inside GraphQL "...").
        sanitized = sanitize_script(script)
        b64 = base64.b64encode(sanitized.encode()).decode()
        all_steps = setup_steps + [
            f"echo {b64} | base64 -d > /tmp/train.sh",
            "bash /tmp/train.sh",
        ] + teardown_steps
        return " && ".join(all_steps)
    else:
        raise ValueError("Either --script or --cmd must be provided")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Launch a training job on RunPod",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline via shell script
  python runs/launch_runpod.py --script runs/speedrun.sh
  python runs/launch_runpod.py --script runs/run_pre1900_sft.sh
  python runs/launch_runpod.py --script runs/run_discovery_rl.sh

  # Single-stage via direct command
  python runs/launch_runpod.py --cmd "torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=26 --fp8"
  python runs/launch_runpod.py --cmd "torchrun --standalone --nproc_per_node=8 -m scripts.pre1900_sft -- --run=my_sft"

  # SFT/RL: download training data + prior checkpoints from HuggingFace first
  python runs/launch_runpod.py --script runs/run_discovery_rl.sh \\
      --hf-data-repo mhla/gpt1900-data \\
      --env WANDB_API_KEY HF_TOKEN OPENAI_API_KEY

  # Custom setup before training (e.g. download a specific checkpoint)
  python runs/launch_runpod.py --cmd "torchrun ..." \\
      --setup-cmd "huggingface-cli download mhla/gpt1900-sft --local-dir $HOME/.cache/nanochat/pre1900_sft_checkpoints"

  # Options
  python runs/launch_runpod.py --script runs/speedrun.sh --gpu-count 4 --disk 100
  python runs/launch_runpod.py --script runs/speedrun.sh --env WANDB_API_KEY HF_TOKEN --dry-run
        """,
    )
    cmd_group = parser.add_mutually_exclusive_group(required=True)
    cmd_group.add_argument("--script", help="Shell script to run (e.g. runs/speedrun.sh, runs/run_pre1900_sft.sh)")
    cmd_group.add_argument("--cmd", help="Arbitrary command to run (e.g. 'torchrun ... -m scripts.base_train -- ...')")
    parser.add_argument("--gpu-type", default=DEFAULT_GPU_TYPE, help=f"GPU type (default: {DEFAULT_GPU_TYPE})")
    parser.add_argument("--gpu-count", type=int, default=DEFAULT_GPU_COUNT, help=f"Number of GPUs (default: {DEFAULT_GPU_COUNT})")
    parser.add_argument("--container-disk", type=int, default=DEFAULT_CONTAINER_DISK_GB, help=f"Container disk in GB for deps/code (default: {DEFAULT_CONTAINER_DISK_GB})")
    parser.add_argument("--volume", type=int, default=DEFAULT_VOLUME_GB, help=f"Network volume in GB for data/checkpoints (default: {DEFAULT_VOLUME_GB})")
    parser.add_argument("--image", default=DEFAULT_IMAGE, help="Docker image name")
    parser.add_argument("--hf-repo", default=DEFAULT_HF_REPO, help=f"HuggingFace repo for checkpoint upload (default: {DEFAULT_HF_REPO})")
    parser.add_argument("--hf-data-repo", default=None, help="HuggingFace repo to download training data/checkpoints from before running")
    parser.add_argument("--setup-cmd", default=None, help="Custom shell command to run after clone but before training (e.g. download checkpoints)")
    parser.add_argument("--env", nargs="*", default=[], metavar="VAR", help="Environment variables to forward from local env (e.g. WANDB_API_KEY HF_TOKEN)")
    parser.add_argument("--dry-run", action="store_true", help="Print config without creating pod")
    parser.add_argument("--name", default="gpt1900-train", help="Pod name (default: gpt1900-train)")
    return parser.parse_args()


def collect_env_vars(var_names: list[str]) -> dict[str, str]:
    """Read env vars from local environment, warn if missing."""
    env = {}
    for name in var_names:
        val = os.environ.get(name)
        if val:
            env[name] = val
        else:
            print(f"  WARNING: --env {name} requested but not set in local environment, skipping")
    return env


def create_pod(args):
    startup_cmd = build_startup_command(
        script=args.script,
        cmd=args.cmd,
        hf_repo=args.hf_repo,
        hf_data_repo=args.hf_data_repo,
        setup_cmd=args.setup_cmd,
    )
    env_vars = collect_env_vars(args.env)

    config = {
        "name": args.name,
        "image_name": args.image,
        "gpu_type_id": args.gpu_type,
        "gpu_count": args.gpu_count,
        "container_disk_in_gb": args.container_disk,
        "volume_in_gb": args.volume,
        "docker_args": _escape_for_graphql(f"bash -c '{startup_cmd}'"),
        "env": env_vars,
        "start_ssh": True,
    }

    if args.dry_run:
        print("\n=== DRY RUN — Pod config ===")
        display_config = {**config, "env": {k: f"{v[:4]}..." if len(v) > 4 else "***" for k, v in env_vars.items()}}
        print(json.dumps(display_config, indent=2))
        print(f"\nStartup command:\n  {startup_cmd}")
        return

    try:
        import runpod
    except ImportError:
        print("ERROR: 'runpod' package not installed. Run: pip install runpod")
        sys.exit(1)

    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        print("ERROR: RUNPOD_API_KEY not set in environment")
        sys.exit(1)

    runpod.api_key = api_key

    print(f"Creating pod: {args.name}")
    print(f"  GPU: {args.gpu_count}x {args.gpu_type}")
    print(f"  Container disk: {args.container_disk} GB")
    print(f"  Volume: {args.volume} GB")
    print(f"  Image: {args.image}")
    print(f"  Command: {args.script or args.cmd}")
    if args.hf_data_repo:
        print(f"  Data repo: {args.hf_data_repo}")
    print(f"  Env vars: {list(env_vars.keys())}")

    pod = runpod.create_pod(**config)
    pod_id = pod["id"]
    print(f"\nPod created: {pod_id}")
    print(f"Dashboard:   https://www.runpod.io/console/pods/{pod_id}")

    return pod_id


def poll_pod(pod_id: str):
    """Poll pod status until it exits."""
    import runpod

    start_time = time.time()

    def handle_interrupt(sig, frame):
        elapsed = time.time() - start_time
        print(f"\n\nInterrupted after {elapsed/60:.1f} min")
        resp = input("Terminate pod? [y/N] ").strip().lower()
        if resp == "y":
            print(f"Terminating pod {pod_id}...")
            runpod.terminate_pod(pod_id)
            print("Pod terminated.")
        else:
            print(f"Pod {pod_id} left running. Monitor at:")
            print(f"  https://www.runpod.io/console/pods/{pod_id}")
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_interrupt)

    print(f"\nPolling pod status every {POLL_INTERVAL}s (Ctrl+C to interrupt)...\n")
    last_status = None

    while True:
        try:
            pod = runpod.get_pod(pod_id)
        except Exception as e:
            print(f"  Poll error: {e}")
            time.sleep(POLL_INTERVAL)
            continue

        status = pod.get("desiredStatus", "UNKNOWN")
        runtime = pod.get("runtime")
        gpu_count = pod.get("gpuCount", "?")

        if status != last_status:
            elapsed = time.time() - start_time
            print(f"  [{elapsed/60:5.1f} min] Status: {status} (GPUs: {gpu_count})")
            last_status = status

        if status in ("EXITED", "TERMINATED", "STOPPED"):
            break

        time.sleep(POLL_INTERVAL)

    elapsed = time.time() - start_time
    print(f"\nPod {pod_id} finished.")
    print(f"  Duration: {elapsed/60:.1f} min ({elapsed/3600:.2f} hr)")
    # Rough cost estimate: H100 ~$3.5/hr/gpu on RunPod
    cost_per_gpu_hr = 3.5
    gpu_count_num = int(gpu_count) if str(gpu_count).isdigit() else 8
    est_cost = (elapsed / 3600) * gpu_count_num * cost_per_gpu_hr
    print(f"  Estimated cost: ~${est_cost:.2f} ({gpu_count_num} GPUs @ ~${cost_per_gpu_hr}/hr)")


def main():
    args = parse_args()
    pod_id = create_pod(args)
    if pod_id:
        poll_pod(pod_id)


if __name__ == "__main__":
    main()
