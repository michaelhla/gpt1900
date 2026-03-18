#!/usr/bin/env python3
"""
Spin up an interactive RunPod GPU pod with the repo + deps pre-installed, then SSH in.

Usage:
    # Default: 1x H100
    python runs/interactive_pod.py

    # Custom GPU count
    python runs/interactive_pod.py --gpu-count 8

    # With checkpoints/data pre-loaded
    python runs/interactive_pod.py --hf-data-repo mhla/gpt1900-data

    # Forward env vars
    python runs/interactive_pod.py --env HF_TOKEN WANDB_API_KEY ANTHROPIC_API_KEY

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


def build_startup_command(hf_data_repo: str | None, setup_cmd: str | None) -> str:
    """Build the shell command that runs inside the pod.

    Modeled on launch_runpod.py — clone, install deps, optionally download data,
    then sleep infinity to keep the pod alive for SSH access.
    """
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

    if setup_cmd:
        steps.append(setup_cmd)

    # Write a bashrc snippet so SSH sessions get the venv + env vars automatically.
    # Base64-encode to avoid all quoting issues.
    bashrc_snippet = (
        f"cd {WORK_DIR} && source .venv/bin/activate && "
        f"export NANOCHAT_BASE_DIR={NANOCHAT_BASE_DIR} && "
        "export OMP_NUM_THREADS=1\n"
    )
    bashrc_b64 = base64.b64encode(bashrc_snippet.encode()).decode()
    steps.append(f"echo {bashrc_b64} | base64 -d >> /root/.bashrc")

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
    parser.add_argument("--container-disk", type=int, default=DEFAULT_CONTAINER_DISK_GB, help=f"Container disk in GB (default: {DEFAULT_CONTAINER_DISK_GB})")
    parser.add_argument("--volume", type=int, default=DEFAULT_VOLUME_GB, help=f"Network volume in GB (default: {DEFAULT_VOLUME_GB})")
    parser.add_argument("--image", default=DEFAULT_IMAGE, help="Docker image name")
    parser.add_argument("--hf-data-repo", default=None, help="HuggingFace repo to download data/checkpoints from")
    parser.add_argument("--setup-cmd", default=None, help="Custom shell command to run after clone (e.g. download checkpoints)")
    parser.add_argument("--env", nargs="*", default=[], metavar="VAR", help="Environment variables to forward (e.g. HF_TOKEN WANDB_API_KEY)")
    parser.add_argument("--dry-run", action="store_true", help="Print config without creating pod")
    parser.add_argument("--name", default="gpt1900-interactive", help="Pod name")
    parser.add_argument("--terminate", metavar="POD_ID", help="Terminate a running pod by ID")
    parser.add_argument("--no-ssh", action="store_true", help="Don't auto-SSH, just print connection info")
    parser.add_argument("--ssh-key", default=None, help="Path to SSH private key (default: auto-detect)")
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


def detect_ssh_key() -> str:
    """Find the user's SSH private key."""
    for key_name in ["id_ed25519", "id_rsa"]:
        key_path = pathlib.Path.home() / ".ssh" / key_name
        if key_path.exists():
            return str(key_path)
    return "~/.ssh/id_ed25519"


def create_pod(args):
    """Create the pod — same pattern as launch_runpod.py."""
    startup_cmd = build_startup_command(args.hf_data_repo, args.setup_cmd)
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
        print(f"\nStartup command:\n  {startup_cmd}")
        return None

    runpod = get_runpod()

    print(f"Creating interactive pod: {args.name}")
    print(f"  GPU: {args.gpu_count}x {args.gpu_type}")
    print(f"  Container disk: {args.container_disk} GB")
    print(f"  Volume: {args.volume} GB")
    print(f"  Image: {args.image}")
    if args.hf_data_repo:
        print(f"  Data repo: {args.hf_data_repo}")
    print(f"  Env vars: {list(env_vars.keys())}")

    pod = runpod.create_pod(**config)
    pod_id = pod["id"]
    pod_host_id = pod.get("machine", {}).get("podHostId", pod_id)
    print(f"\nPod created: {pod_id}")
    print(f"Pod host ID: {pod_host_id}")
    print(f"Dashboard:   https://www.runpod.io/console/pods/{pod_id}")

    return pod_id, pod_host_id


def wait_and_ssh(pod_id: str, pod_host_id: str, args):
    """Poll until RUNNING, then SSH in — combining launch_runpod.py polling with SSH."""
    runpod = get_runpod()
    start_time = time.time()

    def handle_interrupt(sig, frame):
        elapsed = time.time() - start_time
        print(f"\n\nInterrupted after {elapsed/60:.1f} min")
        resp = input(f"Terminate pod {pod_id}? [y/N] ").strip().lower()
        if resp == "y":
            runpod.terminate_pod(pod_id)
            print("Pod terminated.")
        else:
            print(f"Pod left running. Terminate later with:")
            print(f"  python runs/interactive_pod.py --terminate {pod_id}")
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_interrupt)

    # Phase 1: Wait for pod to be RUNNING
    print(f"\nPolling pod status every {POLL_INTERVAL}s (Ctrl+C to interrupt)...")
    last_status = None
    while True:
        try:
            pod = runpod.get_pod(pod_id)
        except Exception as e:
            print(f"  Poll error: {e}")
            time.sleep(POLL_INTERVAL)
            continue

        status = pod.get("desiredStatus", "UNKNOWN")
        if status != last_status:
            elapsed = time.time() - start_time
            print(f"  [{elapsed/60:5.1f} min] Status: {status}")
            last_status = status

        if status == "RUNNING":
            break
        if status in ("EXITED", "TERMINATED", "STOPPED"):
            print(f"Pod exited unexpectedly with status: {status}")
            return

        time.sleep(POLL_INTERVAL)

    # Phase 2: Wait for setup to complete via SSH sentinel check
    ssh_key_path = args.ssh_key or detect_ssh_key()
    # RunPod SSH uses podHostId (not pod_id) as the SSH user
    ssh_host = f"{pod_host_id}@ssh.runpod.io"
    key_flag = f"-i {ssh_key_path}"
    ssh_base = f"ssh {ssh_host} {key_flag} -o StrictHostKeyChecking=no"

    print(f"\nSSH command: {ssh_base}")
    print("Waiting for setup to finish (clone, deps, data)...", end="", flush=True)

    while True:
        check_cmd = f"{ssh_base} -o ConnectTimeout=5 test -f /tmp/.setup_complete 2>/dev/null"
        # Also check if the bashrc has been written (alternate sentinel)
        result = subprocess.run(check_cmd, shell=True, capture_output=True)
        if result.returncode == 0:
            print(" done!")
            break
        print(".", end="", flush=True)
        time.sleep(POLL_INTERVAL)

    print(f"\nTerminate later: python runs/interactive_pod.py --terminate {pod_id}")

    if args.no_ssh:
        print(f"\nPod is ready. Connect manually:")
        print(f"  {ssh_base}")
        print(f"\nPress Ctrl+C to terminate the pod when done.")
        while True:
            time.sleep(60)
    else:
        print(f"\nConnecting via SSH... (exit shell to return here)\n")
        subprocess.run(ssh_base, shell=True)

        gpu_count = args.gpu_count
        print(f"\nSSH session ended.")
        print(f"Pod {pod_id} is still running (~${3.5 * gpu_count:.1f}/hr).")
        resp = input("Terminate pod? [Y/n] ").strip().lower()
        if resp != "n":
            runpod.terminate_pod(pod_id)
            print("Pod terminated.")
        else:
            print(f"Pod left running. Reconnect:")
            print(f"  {ssh_base}")
            print(f"Terminate later:")
            print(f"  python runs/interactive_pod.py --terminate {pod_id}")


def main():
    args = parse_args()

    if args.terminate:
        runpod = get_runpod()
        print(f"Terminating pod {args.terminate}...")
        runpod.terminate_pod(args.terminate)
        print("Done.")
        return

    result = create_pod(args)
    if result:
        pod_id, pod_host_id = result
        wait_and_ssh(pod_id, pod_host_id, args)


if __name__ == "__main__":
    main()
