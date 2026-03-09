#!/usr/bin/env python3
"""
Spin up an interactive RunPod GPU pod with the repo + deps pre-installed, then SSH in.

Usage:
    # Default: 1x H100
    python runs/interactive_pod.py

    # Custom GPU
    python runs/interactive_pod.py --gpu-type "NVIDIA A100 80GB PCIe" --gpu-count 2

    # With checkpoints/data pre-loaded
    python runs/interactive_pod.py --hf-data-repo mhla/gpt1900-data

    # Dry run
    python runs/interactive_pod.py --dry-run

    # Terminate a running pod
    python runs/interactive_pod.py --terminate <pod_id>
"""

import argparse
import base64
import json
import os
import pathlib
import signal
import subprocess
import sys
import time

REPO_URL = "https://github.com/michaelhla/gpt1900"
DEFAULT_IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
DEFAULT_GPU_TYPE = "NVIDIA H100 80GB HBM3"
DEFAULT_GPU_COUNT = 1
DEFAULT_CONTAINER_DISK_GB = 50
DEFAULT_VOLUME_GB = 100
WORK_DIR = "/workspace/gpt1900"
NANOCHAT_BASE_DIR = "$HOME/.cache/nanochat"

POLL_INTERVAL = 5  # seconds — faster polling for interactive use


def _escape_for_graphql(cmd: str) -> str:
    """Escape double quotes for RunPod's GraphQL API.

    The runpod SDK interpolates docker_args into GraphQL as:
        dockerArgs: "{docker_args}"
    so any " in our string breaks the query.
    """
    return cmd.replace("\\", "\\\\").replace('"', '\\"')


def _get_ssh_pubkey() -> str | None:
    """Read the user's SSH public key for RunPod SSH access."""
    for key_name in ["id_ed25519.pub", "id_rsa.pub"]:
        key_path = pathlib.Path.home() / ".ssh" / key_name
        if key_path.exists():
            return key_path.read_text().strip()
    return None


def build_startup_command(hf_data_repo: str | None) -> str:
    """Build startup command that clones repo, installs deps, then sleeps (keeps pod alive)."""
    steps = [
        f"git clone {REPO_URL} {WORK_DIR}",
        f"cd {WORK_DIR}",
        "command -v uv || (curl -LsSf https://astral.sh/uv/install.sh | sh)",
        "export PATH=/root/.local/bin:$PATH",
        "uv sync --extra gpu",
        "source .venv/bin/activate",
        "export OMP_NUM_THREADS=1",
        f"export NANOCHAT_BASE_DIR={NANOCHAT_BASE_DIR}",
        f"mkdir -p {NANOCHAT_BASE_DIR}",
    ]

    if hf_data_repo:
        steps.append(f"echo Downloading data from {hf_data_repo}...")
        steps.append(f"huggingface-cli download {hf_data_repo} --local-dir {NANOCHAT_BASE_DIR}")

    # Write a bashrc snippet so SSH sessions get the venv + env vars automatically.
    # Base64-encode to avoid all quoting issues.
    bashrc_snippet = (
        f"cd {WORK_DIR} && source .venv/bin/activate && "
        f"export NANOCHAT_BASE_DIR={NANOCHAT_BASE_DIR} && "
        "export OMP_NUM_THREADS=1\n"
    )
    bashrc_b64 = base64.b64encode(bashrc_snippet.encode()).decode()
    steps.append(f"echo {bashrc_b64} | base64 -d >> /root/.bashrc")

    # Write sentinel file so the launcher knows setup is done
    steps.append("touch /tmp/.setup_complete")
    steps.append("echo === Pod ready for SSH ===")
    steps.append("sleep infinity")

    return " && ".join(steps)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Spin up an interactive RunPod GPU pod and SSH in",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python runs/interactive_pod.py                        # 1x H100, SSH in
  python runs/interactive_pod.py --gpu-count 8          # 8x H100
  python runs/interactive_pod.py --hf-data-repo mhla/gpt1900-data  # pre-load data
  python runs/interactive_pod.py --terminate abc123     # kill a pod
  python runs/interactive_pod.py --dry-run
        """,
    )
    parser.add_argument("--gpu-type", default=DEFAULT_GPU_TYPE, help=f"GPU type (default: {DEFAULT_GPU_TYPE})")
    parser.add_argument("--gpu-count", type=int, default=DEFAULT_GPU_COUNT, help=f"Number of GPUs (default: {DEFAULT_GPU_COUNT})")
    parser.add_argument("--container-disk", type=int, default=DEFAULT_CONTAINER_DISK_GB, help=f"Container disk in GB for deps/code (default: {DEFAULT_CONTAINER_DISK_GB})")
    parser.add_argument("--volume", type=int, default=DEFAULT_VOLUME_GB, help=f"Network volume in GB for data/checkpoints (default: {DEFAULT_VOLUME_GB})")
    parser.add_argument("--image", default=DEFAULT_IMAGE, help="Docker image name")
    parser.add_argument("--hf-data-repo", default=None, help="HuggingFace repo to download data/checkpoints from")
    parser.add_argument("--env", nargs="*", default=[], metavar="VAR", help="Environment variables to forward (e.g. HF_TOKEN WANDB_API_KEY)")
    parser.add_argument("--dry-run", action="store_true", help="Print config without creating pod")
    parser.add_argument("--name", default="gpt1900-interactive", help="Pod name")
    parser.add_argument("--terminate", metavar="POD_ID", help="Terminate a running pod by ID")
    parser.add_argument("--no-ssh", action="store_true", help="Don't auto-SSH, just print connection info")
    parser.add_argument("--ssh-key", default=None, help="Path to SSH private key (default: auto-detect ~/.ssh/id_ed25519)")
    return parser.parse_args()


def collect_env_vars(var_names: list[str]) -> dict[str, str]:
    env = {}
    for name in var_names:
        val = os.environ.get(name)
        if val:
            env[name] = val
        else:
            print(f"  WARNING: --env {name} not set locally, skipping")
    return env


def get_runpod():
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
    return runpod


def terminate_pod(pod_id: str):
    runpod = get_runpod()
    print(f"Terminating pod {pod_id}...")
    runpod.terminate_pod(pod_id)
    print("Done.")


def wait_for_running(runpod, pod_id: str) -> dict:
    """Poll until the pod is RUNNING."""
    print("Waiting for pod to start...", end="", flush=True)

    while True:
        pod = runpod.get_pod(pod_id)
        status = pod.get("desiredStatus", "UNKNOWN")

        if status == "RUNNING":
            print(" running!")
            return pod

        print(".", end="", flush=True)
        time.sleep(POLL_INTERVAL)


def get_ssh_command(pod_host_id: str, ssh_key_path: str | None) -> str:
    """Build SSH command using RunPod's SSH proxy."""
    key_flag = f"-i {ssh_key_path}" if ssh_key_path else "-i ~/.ssh/id_ed25519"
    return f"ssh {pod_host_id}@ssh.runpod.io {key_flag} -o StrictHostKeyChecking=no"


def wait_for_setup(pod_host_id: str, ssh_key_path: str) -> None:
    """Poll the pod via SSH until /tmp/.setup_complete exists."""
    print("Waiting for setup to finish (clone, deps, venv)...", end="", flush=True)
    key_flag = f"-i {ssh_key_path}"
    check_cmd = (
        f"ssh {pod_host_id}@ssh.runpod.io {key_flag} "
        f"-o StrictHostKeyChecking=no -o ConnectTimeout=5 "
        f"test -f /tmp/.setup_complete"
    )
    while True:
        result = subprocess.run(check_cmd, shell=True, capture_output=True)
        if result.returncode == 0:
            print(" done!")
            return
        print(".", end="", flush=True)
        time.sleep(POLL_INTERVAL)


def detect_ssh_key() -> str:
    """Find the user's SSH private key."""
    for key_name in ["id_ed25519", "id_rsa"]:
        key_path = pathlib.Path.home() / ".ssh" / key_name
        if key_path.exists():
            return str(key_path)
    return "~/.ssh/id_ed25519"


def main():
    args = parse_args()

    if args.terminate:
        terminate_pod(args.terminate)
        return

    startup_cmd = build_startup_command(args.hf_data_repo)
    env_vars = collect_env_vars(args.env)

    # Add SSH public key so RunPod sets up SSH access
    pubkey = _get_ssh_pubkey()
    if pubkey:
        env_vars["PUBLIC_KEY"] = pubkey
    else:
        print("  WARNING: No SSH public key found in ~/.ssh/. SSH access may not work.")
        print("  Run: ssh-keygen -t ed25519")

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
        display_config = {**config, "env": {k: ("***" if k == "PUBLIC_KEY" else (f"{v[:4]}..." if len(v) > 4 else "***")) for k, v in env_vars.items()}}
        print(json.dumps(display_config, indent=2))
        return

    runpod = get_runpod()

    print(f"Creating interactive pod: {args.name}")
    print(f"  GPU: {args.gpu_count}x {args.gpu_type}")
    print(f"  Container disk: {args.container_disk} GB")
    print(f"  Volume: {args.volume} GB")

    pod = runpod.create_pod(**config)
    pod_id = pod["id"]
    # Get the SSH host ID from the create response
    pod_host_id = pod.get("machine", {}).get("podHostId", pod_id)

    print(f"  Pod ID: {pod_id}")
    print(f"  Dashboard: https://www.runpod.io/console/pods/{pod_id}")

    # Handle Ctrl+C: offer to terminate
    def handle_interrupt(sig, frame):
        print(f"\n\nTerminate pod {pod_id}? [y/N] ", end="", flush=True)
        try:
            resp = input().strip().lower()
        except EOFError:
            resp = "n"
        if resp == "y":
            runpod.terminate_pod(pod_id)
            print("Pod terminated.")
        else:
            print(f"Pod left running. Terminate later with:")
            print(f"  python runs/interactive_pod.py --terminate {pod_id}")
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_interrupt)

    # Wait for pod to be running
    wait_for_running(runpod, pod_id)

    ssh_key_path = args.ssh_key or detect_ssh_key()
    ssh_cmd = get_ssh_command(pod_host_id, ssh_key_path)

    print(f"\n  SSH command: {ssh_cmd}")
    print(f"  Terminate:   python runs/interactive_pod.py --terminate {pod_id}")

    # Wait for setup script to finish before SSH'ing in
    wait_for_setup(pod_host_id, ssh_key_path)

    if args.no_ssh:
        print(f"\nPod is ready. Connect manually with the SSH command above.")
        print(f"Press Ctrl+C to terminate the pod when done.")
        while True:
            time.sleep(60)
    else:
        print(f"\nConnecting via SSH... (exit shell to return here)\n")
        result = subprocess.run(ssh_cmd, shell=True)

        print(f"\nSSH session ended.")
        print(f"Pod {pod_id} is still running (~${3.5 * args.gpu_count:.1f}/hr).")
        resp = input("Terminate pod? [Y/n] ").strip().lower()
        if resp != "n":
            runpod.terminate_pod(pod_id)
            print("Pod terminated.")
        else:
            print(f"Pod left running. Reconnect:")
            print(f"  {ssh_cmd}")
            print(f"Terminate later:")
            print(f"  python runs/interactive_pod.py --terminate {pod_id}")


if __name__ == "__main__":
    main()
