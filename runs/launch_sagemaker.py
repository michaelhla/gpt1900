#!/usr/bin/env python3
"""
Launch a training job on AWS SageMaker.

Usage:
    # Run a shell script (full pipeline)
    python runs/launch_sagemaker.py --script runs/run_discovery_rl.sh

    # Run an arbitrary command
    python runs/launch_sagemaker.py --cmd "torchrun --standalone --nproc_per_node=8 -m scripts.pre1900_scripts.discovery_rl -- --run=my_rl"

    # With data from S3
    python runs/launch_sagemaker.py --script runs/run_discovery_rl.sh \
        --s3-data s3://gpt1900/data \
        --env ANTHROPIC_API_KEY WANDB_API_KEY HF_TOKEN

    # With data from HuggingFace
    python runs/launch_sagemaker.py --script runs/run_discovery_rl.sh \
        --hf-data-repo mhla/gpt1900-data \
        --env ANTHROPIC_API_KEY WANDB_API_KEY

    # Spot instances (~60% cheaper)
    python runs/launch_sagemaker.py --script runs/run_discovery_rl.sh --spot

    # Dry run
    python runs/launch_sagemaker.py --script runs/run_discovery_rl.sh --dry-run
"""

import argparse
import base64
import json
import os
import re
import signal
import sys
import time

# Reuse SLURM stripping from RunPod launcher
sys.path.insert(0, os.path.dirname(__file__))
from launch_runpod import sanitize_script

REPO_URL = "https://github.com/michaelhla/gpt1900"
DEFAULT_INSTANCE_TYPE = "ml.p5.48xlarge"  # 8x H100
DEFAULT_S3_OUTPUT = "s3://gpt1900/output"
DEFAULT_VOLUME_SIZE = 200  # GB
DEFAULT_MAX_RUN = 86400  # 24 hours
POLL_INTERVAL = 60  # seconds

# SageMaker PyTorch DLC — matches the GPU container SageMaker provides
FRAMEWORK_VERSION = "2.4.0"
PY_VERSION = "py311"

# Cost estimates (on-demand $/hr) for common instance types
COST_PER_HOUR = {
    "ml.p5.48xlarge": 98.32,     # 8x H100 80GB
    "ml.p4d.24xlarge": 32.77,    # 8x A100 40GB
    "ml.p4de.24xlarge": 40.97,   # 8x A100 80GB
    "ml.g5.48xlarge": 16.29,     # 8x A10G
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Launch a training job on AWS SageMaker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline via shell script
  python runs/launch_sagemaker.py --script runs/run_discovery_rl.sh

  # With S3 data channel
  python runs/launch_sagemaker.py --script runs/run_discovery_rl.sh \\
      --s3-data s3://gpt1900/data \\
      --env ANTHROPIC_API_KEY WANDB_API_KEY HF_TOKEN

  # Spot instances for ~60% savings
  python runs/launch_sagemaker.py --script runs/run_discovery_rl.sh --spot

  # Dry run to preview config
  python runs/launch_sagemaker.py --script runs/run_discovery_rl.sh --dry-run
        """,
    )
    cmd_group = parser.add_mutually_exclusive_group(required=True)
    cmd_group.add_argument("--script", help="Shell script to run (e.g. runs/run_discovery_rl.sh)")
    cmd_group.add_argument("--cmd", help="Arbitrary command to run")
    parser.add_argument("--instance-type", default=DEFAULT_INSTANCE_TYPE,
                        help=f"SageMaker instance type (default: {DEFAULT_INSTANCE_TYPE})")
    parser.add_argument("--s3-data", default=None,
                        help="S3 URI for training data (mounted as input channel)")
    parser.add_argument("--s3-output", default=DEFAULT_S3_OUTPUT,
                        help=f"S3 URI for output (default: {DEFAULT_S3_OUTPUT})")
    parser.add_argument("--hf-data-repo", default=None,
                        help="HuggingFace repo to download data from at startup")
    parser.add_argument("--env", nargs="*", default=[], metavar="VAR",
                        help="Environment variables to forward (e.g. ANTHROPIC_API_KEY WANDB_API_KEY)")
    parser.add_argument("--role", default=None,
                        help="SageMaker IAM role ARN (default: from SAGEMAKER_ROLE env var)")
    parser.add_argument("--spot", action="store_true",
                        help="Use managed spot instances (~60%% cheaper)")
    parser.add_argument("--volume-size", type=int, default=DEFAULT_VOLUME_SIZE,
                        help=f"EBS volume size in GB (default: {DEFAULT_VOLUME_SIZE})")
    parser.add_argument("--max-run", type=int, default=DEFAULT_MAX_RUN,
                        help=f"Max training time in seconds (default: {DEFAULT_MAX_RUN})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print config without launching")
    parser.add_argument("--name", default="gpt1900-train",
                        help="Training job name prefix (default: gpt1900-train)")
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


def build_hyperparameters(*, script: str | None, cmd: str | None,
                          hf_data_repo: str | None) -> dict[str, str]:
    """Build SageMaker hyperparameters dict.

    We pass the training command/script through env vars rather than
    hyperparameters, but hyperparameters are logged in the SageMaker
    console which is useful for debugging.
    """
    params = {}
    if script:
        params["script_name"] = os.path.basename(script)
    if cmd:
        params["training_cmd"] = cmd[:256]  # truncate for display
    if hf_data_repo:
        params["hf_data_repo"] = hf_data_repo
    return params


def build_environment(*, script: str | None, cmd: str | None,
                      hf_data_repo: str | None,
                      user_env: dict[str, str]) -> dict[str, str]:
    """Build the environment dict for the SageMaker estimator."""
    env = dict(user_env)
    env["REPO_URL"] = REPO_URL

    if cmd:
        env["TRAINING_CMD"] = cmd
    elif script:
        sanitized = sanitize_script(script)
        env["TRAINING_SCRIPT_B64"] = base64.b64encode(sanitized.encode()).decode()

    if hf_data_repo:
        env["HF_DATA_REPO"] = hf_data_repo

    return env


def get_role(args) -> str:
    """Resolve the SageMaker execution role."""
    if args.role:
        return args.role
    role = os.environ.get("SAGEMAKER_ROLE")
    if role:
        return role
    print("ERROR: No SageMaker role specified.")
    print("  Set --role or SAGEMAKER_ROLE environment variable.")
    print("  Example: arn:aws:iam::123456789012:role/SageMakerExecutionRole")
    sys.exit(1)


def create_estimator(args):
    """Create and return a SageMaker PyTorch estimator (and optionally launch it)."""
    role = get_role(args)
    user_env = collect_env_vars(args.env)
    environment = build_environment(
        script=args.script, cmd=args.cmd,
        hf_data_repo=args.hf_data_repo,
        user_env=user_env,
    )
    hyperparameters = build_hyperparameters(
        script=args.script, cmd=args.cmd,
        hf_data_repo=args.hf_data_repo,
    )

    # Config summary
    spot_label = " (SPOT)" if args.spot else ""
    config = {
        "job_name_prefix": args.name,
        "instance_type": args.instance_type + spot_label,
        "volume_size_gb": args.volume_size,
        "max_run_seconds": args.max_run,
        "s3_output": args.s3_output,
        "s3_data": args.s3_data,
        "hf_data_repo": args.hf_data_repo,
        "training": args.script or args.cmd,
        "env_vars": list(user_env.keys()),
        "role": role[:20] + "..." if len(role) > 20 else role,
    }

    if args.dry_run:
        print("\n=== DRY RUN — SageMaker config ===")
        print(json.dumps(config, indent=2))
        # Show the environment (redact secrets)
        display_env = {}
        for k, v in environment.items():
            if any(secret in k.upper() for secret in ("KEY", "TOKEN", "SECRET")):
                display_env[k] = f"{v[:4]}..." if len(v) > 4 else "***"
            elif k == "TRAINING_SCRIPT_B64":
                display_env[k] = f"<{len(v)} chars base64>"
            else:
                display_env[k] = v
        print(f"\nEnvironment: {json.dumps(display_env, indent=2)}")
        print(f"\nHyperparameters: {json.dumps(hyperparameters, indent=2)}")
        return None

    try:
        from sagemaker.pytorch import PyTorch
    except ImportError:
        print("ERROR: 'sagemaker' package not installed. Run: pip install 'nanochat[sagemaker]'")
        sys.exit(1)

    # Resolve source_dir relative to repo root
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    source_dir = os.path.join(repo_root, "deploy", "sagemaker")

    estimator = PyTorch(
        entry_point="entry_point.sh",
        source_dir=source_dir,
        role=role,
        instance_type=args.instance_type,
        instance_count=1,
        framework_version=FRAMEWORK_VERSION,
        py_version=PY_VERSION,
        volume_size=args.volume_size,
        max_run=args.max_run,
        output_path=args.s3_output,
        environment=environment,
        hyperparameters=hyperparameters,
        use_spot_instances=args.spot,
        max_wait=args.max_run * 2 if args.spot else None,
        base_job_name=args.name,
        enable_network_isolation=False,  # need outbound for Claude API
        keep_alive_period_in_seconds=0,
    )

    # Build input channels
    inputs = None
    if args.s3_data:
        inputs = {"training": args.s3_data}

    print(f"\nLaunching SageMaker training job...")
    print(f"  Instance:  {args.instance_type}{spot_label}")
    print(f"  Volume:    {args.volume_size} GB")
    print(f"  Max run:   {args.max_run}s ({args.max_run/3600:.1f}h)")
    print(f"  Output:    {args.s3_output}")
    print(f"  Training:  {args.script or args.cmd}")
    if args.s3_data:
        print(f"  S3 data:   {args.s3_data}")
    if args.hf_data_repo:
        print(f"  HF data:   {args.hf_data_repo}")
    print(f"  Env vars:  {list(user_env.keys())}")

    estimator.fit(inputs=inputs, wait=False)

    job_name = estimator.latest_training_job.name
    print(f"\nTraining job created: {job_name}")
    print(f"Console: https://console.aws.amazon.com/sagemaker/home#/jobs/{job_name}")

    return estimator


def poll_training_job(estimator):
    """Poll training job status until completion."""
    import boto3

    job_name = estimator.latest_training_job.name
    sm_client = boto3.client("sagemaker")
    start_time = time.time()

    def handle_interrupt(sig, frame):
        elapsed = time.time() - start_time
        print(f"\n\nInterrupted after {elapsed/60:.1f} min")
        resp = input("Stop training job? [y/N] ").strip().lower()
        if resp == "y":
            print(f"Stopping job {job_name}...")
            sm_client.stop_training_job(TrainingJobName=job_name)
            print("Stop requested (may take a minute to terminate).")
        else:
            print(f"Job {job_name} left running. Monitor at:")
            print(f"  https://console.aws.amazon.com/sagemaker/home#/jobs/{job_name}")
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_interrupt)

    instance_type = estimator.instance_type
    cost_rate = COST_PER_HOUR.get(instance_type, 0)
    if estimator.use_spot_instances:
        cost_rate *= 0.4  # rough spot discount

    print(f"\nPolling job status every {POLL_INTERVAL}s (Ctrl+C to interrupt)...")
    if cost_rate:
        print(f"  Cost rate: ~${cost_rate:.2f}/hr ({instance_type})\n")

    last_status = None

    while True:
        try:
            desc = sm_client.describe_training_job(TrainingJobName=job_name)
        except Exception as e:
            print(f"  Poll error: {e}")
            time.sleep(POLL_INTERVAL)
            continue

        status = desc["TrainingJobStatus"]
        secondary = desc.get("SecondaryStatus", "")

        if status != last_status or secondary:
            elapsed = time.time() - start_time
            cost_so_far = f" (~${elapsed/3600 * cost_rate:.2f})" if cost_rate else ""
            detail = f" [{secondary}]" if secondary else ""
            print(f"  [{elapsed/60:5.1f} min] {status}{detail}{cost_so_far}")
            last_status = status

        if status in ("Completed", "Failed", "Stopped"):
            break

        time.sleep(POLL_INTERVAL)

    elapsed = time.time() - start_time
    print(f"\nJob {job_name} finished: {status}")
    print(f"  Duration: {elapsed/60:.1f} min ({elapsed/3600:.2f} hr)")
    if cost_rate:
        est_cost = (elapsed / 3600) * cost_rate
        print(f"  Estimated cost: ~${est_cost:.2f}")

    if status == "Failed":
        failure_reason = desc.get("FailureReason", "unknown")
        print(f"  Failure reason: {failure_reason}")
        sys.exit(1)

    if status == "Completed":
        model_artifacts = desc.get("ModelArtifacts", {}).get("S3ModelArtifacts", "")
        if model_artifacts:
            print(f"  Model artifacts: {model_artifacts}")


def main():
    args = parse_args()
    estimator = create_estimator(args)
    if estimator:
        poll_training_job(estimator)


if __name__ == "__main__":
    main()
