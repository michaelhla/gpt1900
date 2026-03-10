#!/usr/bin/env python3
"""
Launch a training job on an EC2 GPU instance.

Two modes:
  - Capacity block (--capacity-reservation): Instance stays alive after training.
    You've already paid — SSH in, run more jobs, inspect results. No auto-shutdown.
  - On-demand / spot: Instance auto-terminates when training finishes or crashes.

Usage:
    # Capacity block (recommended with AWS credits)
    python runs/launch_ec2.py --script runs/run_contradiction_rl_v5.sh \
        --region us-east-2 \
        --capacity-reservation cr-050645b9aebb222c7 \
        --hf-data-repo mhla/gpt1900-data \
        --key-name gpt1900 \
        --env ANTHROPIC_API_KEY WANDB_API_KEY HF_TOKEN

    # On-demand (auto-terminates after training)
    python runs/launch_ec2.py --script runs/run_discovery_rl.sh \
        --hf-data-repo mhla/gpt1900-data \
        --env ANTHROPIC_API_KEY WANDB_API_KEY HF_TOKEN

    # Dry run
    python runs/launch_ec2.py --script runs/run_discovery_rl.sh --dry-run
"""

import argparse
import base64
import json
import os
import signal
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from launch_runpod import sanitize_script

REPO_URL = "https://github.com/michaelhla/gpt1900"
DEFAULT_INSTANCE_TYPE = "p5.48xlarge"  # 8x H100
DEFAULT_VOLUME_SIZE = 200  # GB root EBS
DEFAULT_HF_REPO = "mhla/gpt1900-checkpoints"
DEFAULT_REGION = "us-east-1"
NANOCHAT_BASE_DIR = "/root/.cache/nanochat"
WORK_DIR = "/root/gpt1900"
POLL_INTERVAL = 30  # seconds

# On-demand $/hr for cost estimates
COST_PER_HOUR = {
    "p5.48xlarge": 98.32,    # 8x H100 80GB
    "p4d.24xlarge": 32.77,   # 8x A100 40GB
    "p4de.24xlarge": 40.97,  # 8x A100 80GB
    "g5.48xlarge": 16.29,    # 8x A10G
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Launch a training job on EC2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Capacity block reservation (instance stays alive after training)
  python runs/launch_ec2.py --script runs/run_contradiction_rl_v5.sh \\
      --region us-east-2 \\
      --capacity-reservation cr-050645b9aebb222c7 \\
      --hf-data-repo mhla/gpt1900-data \\
      --key-name gpt1900 \\
      --env ANTHROPIC_API_KEY WANDB_API_KEY HF_TOKEN

  # On-demand (auto-terminates when done)
  python runs/launch_ec2.py --script runs/run_discovery_rl.sh \\
      --hf-data-repo mhla/gpt1900-data \\
      --env ANTHROPIC_API_KEY WANDB_API_KEY HF_TOKEN

  python runs/launch_ec2.py --script runs/run_discovery_rl.sh --dry-run
        """,
    )
    cmd_group = parser.add_mutually_exclusive_group(required=True)
    cmd_group.add_argument("--script", help="Shell script to run")
    cmd_group.add_argument("--cmd", help="Arbitrary command to run")
    parser.add_argument("--instance-type", default=DEFAULT_INSTANCE_TYPE,
                        help=f"EC2 instance type (default: {DEFAULT_INSTANCE_TYPE})")
    parser.add_argument("--region", default=DEFAULT_REGION,
                        help=f"AWS region (default: {DEFAULT_REGION})")
    parser.add_argument("--ami", default=None,
                        help="AMI ID (default: auto-detect Deep Learning AMI)")
    parser.add_argument("--key-name", default=None,
                        help="EC2 key pair name for SSH access")
    parser.add_argument("--security-group", default=None,
                        help="Security group ID (default: auto-create)")
    parser.add_argument("--hf-repo", default=DEFAULT_HF_REPO,
                        help=f"HuggingFace repo for checkpoint upload (default: {DEFAULT_HF_REPO})")
    parser.add_argument("--hf-data-repo", default=None,
                        help="HuggingFace repo to download data from at startup")
    parser.add_argument("--setup-cmd", default=None,
                        help="Custom command to run before training")
    parser.add_argument("--env", nargs="*", default=[], metavar="VAR",
                        help="Environment variables to forward (e.g. ANTHROPIC_API_KEY WANDB_API_KEY)")
    parser.add_argument("--volume-size", type=int, default=DEFAULT_VOLUME_SIZE,
                        help=f"Root EBS volume size in GB (default: {DEFAULT_VOLUME_SIZE})")
    parser.add_argument("--capacity-reservation", default=None, metavar="CR_ID",
                        help="Capacity reservation ID — instance stays alive after training")
    parser.add_argument("--spot", action="store_true",
                        help="Request spot instance (~60%% cheaper, may be interrupted)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print config without launching")
    parser.add_argument("--name", default="gpt1900-train",
                        help="Instance name tag (default: gpt1900-train)")
    return parser.parse_args()


def collect_env_vars(var_names: list[str]) -> dict[str, str]:
    """Read env vars from local environment, warn if missing."""
    env = {}
    for name in var_names:
        val = os.environ.get(name)
        if val:
            env[name] = val
        else:
            print(f"  WARNING: --env {name} requested but not set locally, skipping")
    return env


def find_dl_ami(ec2_client, region: str) -> str:
    """Find the latest NVIDIA Deep Learning AMI (Ubuntu)."""
    response = ec2_client.describe_images(
        Owners=["amazon"],
        Filters=[
            {"Name": "name", "Values": ["Deep Learning AMI GPU PyTorch *Ubuntu 22.04*"]},
            {"Name": "state", "Values": ["available"]},
            {"Name": "architecture", "Values": ["x86_64"]},
        ],
    )
    images = response["Images"]
    if not images:
        # Fallback: try broader search
        response = ec2_client.describe_images(
            Owners=["amazon"],
            Filters=[
                {"Name": "name", "Values": ["Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)*"]},
                {"Name": "state", "Values": ["available"]},
                {"Name": "architecture", "Values": ["x86_64"]},
            ],
        )
        images = response["Images"]
    if not images:
        print("ERROR: Could not find a Deep Learning AMI in this region.")
        print(f"  Specify --ami manually. Region: {region}")
        sys.exit(1)
    # Sort by creation date, newest first
    images.sort(key=lambda x: x["CreationDate"], reverse=True)
    ami = images[0]
    print(f"  AMI: {ami['ImageId']} ({ami['Name'][:80]})")
    return ami["ImageId"]


def ensure_security_group(ec2_client) -> str:
    """Find or create a security group that allows SSH + all outbound."""
    sg_name = "gpt1900-training"
    try:
        resp = ec2_client.describe_security_groups(
            Filters=[{"Name": "group-name", "Values": [sg_name]}]
        )
        if resp["SecurityGroups"]:
            sg_id = resp["SecurityGroups"][0]["GroupId"]
            print(f"  Security group: {sg_id} (existing: {sg_name})")
            return sg_id
    except Exception:
        pass

    print(f"  Creating security group: {sg_name}")
    resp = ec2_client.create_security_group(
        GroupName=sg_name,
        Description="SSH access for gpt1900 training instances",
    )
    sg_id = resp["GroupId"]
    ec2_client.authorize_security_group_ingress(
        GroupId=sg_id,
        IpPermissions=[{
            "IpProtocol": "tcp",
            "FromPort": 22,
            "ToPort": 22,
            "IpRanges": [{"CidrIp": "0.0.0.0/0", "Description": "SSH"}],
        }],
    )
    print(f"  Security group created: {sg_id}")
    return sg_id


def build_user_data(*, script: str | None, cmd: str | None,
                    hf_repo: str, hf_data_repo: str | None,
                    setup_cmd: str | None,
                    env_vars: dict[str, str],
                    capacity_reservation: bool) -> str:
    """Build the cloud-init user-data script.

    With capacity reservation: no auto-shutdown, instance stays alive.
    Without: auto-terminates when training finishes or crashes.
    """
    lines = [
        "#!/bin/bash",
        "set -euo pipefail",
        "",
    ]

    if not capacity_reservation:
        # On-demand/spot: auto-shutdown on exit
        lines.extend([
            "# Trap: ensure instance shuts down even on error",
            "trap 'echo \"Script exited (code $?). Shutting down in 60s...\"; sleep 60; shutdown -h now' EXIT",
            "",
        ])

    lines.extend([
        "exec > >(tee /var/log/training.log) 2>&1",
        f'echo "=== Training started at $(date) ==="',
        "",
    ])

    # Export environment variables
    for k, v in env_vars.items():
        b64_val = base64.b64encode(v.encode()).decode()
        lines.append(f'export {k}="$(echo {b64_val} | base64 -d)"')
    lines.append("")

    # Write env vars to a file so they persist across SSH sessions
    if capacity_reservation:
        lines.extend([
            "# Persist env vars for SSH sessions",
            "cat >> /root/.bashrc << 'ENVBLOCK'",
        ])
        for k in env_vars:
            lines.append(f'export {k}="${k}"')
        lines.extend([
            f"export NANOCHAT_BASE_DIR={NANOCHAT_BASE_DIR}",
            "export OMP_NUM_THREADS=1",
            f"export PATH=/root/.local/bin:$PATH",
            f"cd {WORK_DIR} 2>/dev/null && source .venv/bin/activate 2>/dev/null || true",
            "ENVBLOCK",
            "",
        ])

    # Set up workspace
    lines.extend([
        f"export NANOCHAT_BASE_DIR={NANOCHAT_BASE_DIR}",
        f"mkdir -p {NANOCHAT_BASE_DIR}",
        "export OMP_NUM_THREADS=1",
        "",
        "# Clone repo + install deps",
        f"git clone {REPO_URL} {WORK_DIR}",
        f"cd {WORK_DIR}",
        'command -v uv || (curl -LsSf https://astral.sh/uv/install.sh | sh)',
        "export PATH=/root/.local/bin:$PATH",
        "uv sync --extra gpu",
        "source .venv/bin/activate",
        "",
    ])

    # Download data from HuggingFace
    if hf_data_repo:
        lines.extend([
            f"echo 'Downloading data from {hf_data_repo}...'",
            f"huggingface-cli download {hf_data_repo} --local-dir {NANOCHAT_BASE_DIR}",
            "",
        ])

    # Custom setup command
    if setup_cmd:
        lines.extend([setup_cmd, ""])

    # Tell checkpoint_manager to upload to HF after every save
    lines.extend([
        f"export HF_UPLOAD_REPO={hf_repo}",
        "",
    ])

    # Training command
    if script:
        sanitized = sanitize_script(script)
        b64 = base64.b64encode(sanitized.encode()).decode()
        lines.extend([
            "# Decode and run training script",
            f"echo '{b64}' | base64 -d > /tmp/train.sh",
            "bash /tmp/train.sh",
            "",
        ])
    elif cmd:
        lines.extend([
            "# Run training command",
            cmd,
            "",
        ])

    # Final upload
    lines.extend([
        "echo 'Final checkpoint upload...'",
        f"huggingface-cli upload {hf_repo} {NANOCHAT_BASE_DIR}/ --include '*.pt' '*.json' 'tokenizer/*' || echo 'HF upload failed'",
        "",
        'echo "=== Training complete at $(date) ==="',
    ])

    if capacity_reservation:
        lines.append('echo "Instance staying alive — SSH in to inspect or run more jobs."')
    else:
        lines.append("# EXIT trap will shut down the instance")

    return "\n".join(lines)


def launch(args):
    """Launch the EC2 instance."""
    import boto3

    ec2 = boto3.client("ec2", region_name=args.region)
    env_vars = collect_env_vars(args.env)

    is_capacity_block = bool(args.capacity_reservation)

    user_data = build_user_data(
        script=args.script, cmd=args.cmd,
        hf_repo=args.hf_repo, hf_data_repo=args.hf_data_repo,
        setup_cmd=args.setup_cmd, env_vars=env_vars,
        capacity_reservation=is_capacity_block,
    )

    if args.dry_run:
        config = {
            "instance_type": args.instance_type,
            "region": args.region,
            "volume_size_gb": args.volume_size,
            "spot": args.spot,
            "capacity_reservation": args.capacity_reservation,
            "auto_terminate": not is_capacity_block,
            "training": args.script or args.cmd,
            "hf_data_repo": args.hf_data_repo,
            "env_vars": list(env_vars.keys()),
        }
        print("\n=== DRY RUN — EC2 config ===")
        print(json.dumps(config, indent=2))
        print(f"\nUser-data script ({len(user_data)} bytes):")
        # Redact secret values in display
        display = user_data
        for k, v in env_vars.items():
            if v and len(v) > 4:
                display = display.replace(
                    base64.b64encode(v.encode()).decode(),
                    f"<{k} redacted>"
                )
        print(display)
        return None

    print(f"Resolving AMI...")
    ami_id = args.ami or find_dl_ami(ec2, args.region)

    sg_id = args.security_group or ensure_security_group(ec2)

    # With capacity block: no auto-terminate (instance stays alive)
    shutdown_behavior = "stop" if is_capacity_block else "terminate"

    launch_params = {
        "ImageId": ami_id,
        "InstanceType": args.instance_type,
        "MinCount": 1,
        "MaxCount": 1,
        "UserData": base64.b64encode(user_data.encode()).decode(),
        "SecurityGroupIds": [sg_id],
        "InstanceInitiatedShutdownBehavior": shutdown_behavior,
        "BlockDeviceMappings": [{
            "DeviceName": "/dev/sda1",
            "Ebs": {
                "VolumeSize": args.volume_size,
                "VolumeType": "gp3",
                "DeleteOnTermination": True,
            },
        }],
        "TagSpecifications": [{
            "ResourceType": "instance",
            "Tags": [
                {"Key": "Name", "Value": args.name},
                {"Key": "Project", "Value": "gpt1900"},
            ],
        }],
    }

    if args.key_name:
        launch_params["KeyName"] = args.key_name

    # Capacity block reservation (targeted)
    if args.capacity_reservation:
        launch_params["CapacityReservationSpecification"] = {
            "CapacityReservationTarget": {
                "CapacityReservationId": args.capacity_reservation,
            },
        }

    # Spot vs on-demand
    if args.spot:
        launch_params["InstanceMarketOptions"] = {
            "MarketType": "spot",
            "SpotOptions": {
                "SpotInstanceType": "one-time",
                "InstanceInterruptionBehavior": "terminate",
            },
        }

    mode = "CAPACITY BLOCK" if is_capacity_block else ("SPOT" if args.spot else "ON-DEMAND")
    print(f"\nLaunching EC2 instance ({mode})...")
    print(f"  Type:     {args.instance_type}")
    print(f"  Volume:   {args.volume_size} GB gp3")
    print(f"  Training: {args.script or args.cmd}")
    if args.hf_data_repo:
        print(f"  HF data:  {args.hf_data_repo}")
    print(f"  Env vars: {list(env_vars.keys())}")
    if args.key_name:
        print(f"  Key pair: {args.key_name}")
    if args.capacity_reservation:
        print(f"  Capacity: {args.capacity_reservation}")
    if is_capacity_block:
        print(f"  Auto-terminate: NO (instance stays alive after training)")
    else:
        print(f"  Auto-terminate: YES")

    response = ec2.run_instances(**launch_params)
    instance_id = response["Instances"][0]["InstanceId"]

    print(f"\nInstance launched: {instance_id}")
    print(f"Console: https://{args.region}.console.aws.amazon.com/ec2/home?region={args.region}#Instances:instanceId={instance_id}")

    # Wait for public IP
    print("Waiting for instance to start...")
    waiter = ec2.get_waiter("instance_running")
    waiter.wait(InstanceIds=[instance_id])
    desc = ec2.describe_instances(InstanceIds=[instance_id])
    inst = desc["Reservations"][0]["Instances"][0]
    public_ip = inst.get("PublicIpAddress")
    if public_ip and args.key_name:
        print(f"\n  SSH:  ssh -i ~/.ssh/{args.key_name}.pem root@{public_ip}")
        print(f"  Logs: ssh in and tail -f /var/log/training.log")

    return instance_id


def poll_instance(instance_id: str, region: str, instance_type: str, spot: bool):
    """Poll instance status until it terminates."""
    import boto3

    ec2 = boto3.client("ec2", region_name=region)
    start_time = time.time()
    cost_rate = COST_PER_HOUR.get(instance_type, 0)
    if spot:
        cost_rate *= 0.4

    def handle_interrupt(sig, frame):
        elapsed = time.time() - start_time
        print(f"\n\nInterrupted after {elapsed/60:.1f} min")
        resp = input("Terminate instance? [y/N] ").strip().lower()
        if resp == "y":
            print(f"Terminating {instance_id}...")
            ec2.terminate_instances(InstanceIds=[instance_id])
            print("Terminate requested.")
        else:
            print(f"Instance {instance_id} left running.")
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_interrupt)

    print(f"\nPolling instance every {POLL_INTERVAL}s (Ctrl+C to detach)...")
    if cost_rate:
        print(f"  Cost rate: ~${cost_rate:.2f}/hr ({instance_type})\n")

    last_state = None
    while True:
        try:
            desc = ec2.describe_instances(InstanceIds=[instance_id])
            state = desc["Reservations"][0]["Instances"][0]["State"]["Name"]
        except Exception as e:
            print(f"  Poll error: {e}")
            time.sleep(POLL_INTERVAL)
            continue

        if state != last_state:
            elapsed = time.time() - start_time
            cost_so_far = f" (~${elapsed/3600 * cost_rate:.2f})" if cost_rate else ""
            print(f"  [{elapsed/60:5.1f} min] {state}{cost_so_far}")
            last_state = state

        if state in ("terminated", "stopped"):
            break

        time.sleep(POLL_INTERVAL)

    elapsed = time.time() - start_time
    print(f"\nInstance {instance_id} {last_state}.")
    print(f"  Duration: {elapsed/60:.1f} min ({elapsed/3600:.2f} hr)")
    if cost_rate:
        est_cost = (elapsed / 3600) * cost_rate
        print(f"  Estimated cost: ~${est_cost:.2f}")


def main():
    args = parse_args()
    instance_id = launch(args)
    if not instance_id:
        return
    # For capacity blocks, don't poll — just print SSH info and exit.
    # The instance stays alive; user can Ctrl+C and SSH in whenever.
    if args.capacity_reservation:
        print("\nCapacity block — instance will stay alive after training.")
        print("Ctrl+C to detach. SSH in to monitor, run more jobs, or inspect results.")
        print("The reservation auto-expires and terminates the instance at the end date.")
        try:
            poll_instance(instance_id, args.region, args.instance_type, args.spot)
        except KeyboardInterrupt:
            print(f"\nDetached. Instance {instance_id} still running.")
    else:
        poll_instance(instance_id, args.region, args.instance_type, args.spot)


if __name__ == "__main__":
    main()
