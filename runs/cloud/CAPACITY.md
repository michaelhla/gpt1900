# GPT-1900 Cloud Capacity Planning

## Model Sizes & Compute Requirements

The nanochat architecture uses `depth` as the single complexity dial. All other
hyperparameters are derived automatically.

| Depth | Params | model_dim | Tokens (ratio=20) | Tokens (ratio=11) |
|-------|--------|-----------|--------------------|--------------------|
| d12   | 91M    | 768       | 1.4B               | 0.8B               |
| d20   | 372M   | 1280      | 6.8B               | 3.7B               |
| d26   | 918M   | 1664      | 17.6B              | 9.7B               |
| d34   | 2.1B   | 2176      | 42B                | 23B                |
| d42   | 4.0B   | 2688      | 80B                | 44B                |
| d45   | 5.2B   | 2880      | 104B               | 57B                |
| d52   | 8.0B   | 3328      | 160B               | 88B                |

## Training Time Estimates

### Pretraining on 8xH100 (single node, FP8)

| Depth | ratio=20 | ratio=11 | VRAM per GPU |
|-------|----------|----------|--------------|
| d12   | ~5 min   | ~3 min   | ~8 GB        |
| d20   | ~1.5 hr  | ~50 min  | ~20 GB       |
| d26   | ~6 hr    | ~3 hr    | ~35 GB       |
| d34   | ~24 hr   | ~12 hr   | ~55 GB       |
| d42   | ~72 hr   | ~36 hr   | ~75 GB (tight) |
| d45   | OOM      | OOM      | >80 GB       |

### Pretraining on 8xH100 (multi-node, FP8)

| Depth | Nodes | ratio=20 | ratio=11 |
|-------|-------|----------|----------|
| d34   | 4     | ~6 hr    | ~3 hr    |
| d34   | 8     | ~3 hr    | ~1.5 hr  |
| d42   | 4     | ~18 hr   | ~9 hr    |
| d42   | 8     | ~10 hr   | ~5 hr    |
| d45   | 8     | ~14 hr   | ~7 hr    |
| d52   | 8     | ~30 hr   | ~15 hr   |

### Post-training (SFT, RL) on 8xH100

Post-training stages are much cheaper than pretraining:

| Stage            | d26    | d34    | d42    |
|------------------|--------|--------|--------|
| SFT              | ~15 min| ~30 min| ~1 hr  |
| Reasoning SFT    | ~10 min| ~20 min| ~45 min|
| Discovery RL     | ~1 hr  | ~2-4 hr| ~6-8 hr|

## Cost Estimates

### AWS Pricing (us-east-1, as of early 2026)

| Instance      | GPUs         | On-Demand | Spot (~60-70% off) |
|---------------|--------------|-----------|---------------------|
| p5.48xlarge   | 8x H100 80GB | ~$98/hr   | ~$30-40/hr          |
| p4de.24xlarge | 8x A100 80GB | ~$40/hr   | ~$15-20/hr          |
| p4d.24xlarge  | 8x A100 40GB | ~$32/hr   | ~$12-15/hr          |

### RunPod Pricing (approximate)

| GPU            | Per GPU/hr (Secure) | Per GPU/hr (Community) |
|----------------|--------------------|-----------------------|
| H100 80GB SXM  | ~$3.50-4.00        | ~$2.50-3.00           |
| A100 80GB PCIe | ~$1.60-2.00        | ~$1.20-1.50           |
| A100 80GB SXM  | ~$1.80-2.20        | ~$1.40-1.70           |

### Total Cost for Full Training Runs

| Model | Stage          | AWS Spot (8xH100) | RunPod Secure (8xH100) |
|-------|----------------|--------------------|-----------------------|
| d26   | Pretrain (r=20)| ~$180              | ~$170                  |
| d26   | Pretrain (r=11)| ~$90               | ~$85                   |
| d34   | Pretrain (r=20)| ~$720              | ~$670                  |
| d34   | Pretrain (r=11)| ~$360              | ~$340                  |
| d34   | Pretrain 8-node| ~$360 (r=20)       | N/A (single-node only) |
| d34   | Full pipeline  | ~$420 (r=11)       | ~$400                  |
| d42   | Pretrain 8-node| ~$600 (r=11)       | N/A                    |
| d45   | Pretrain 8-node| ~$840 (r=11)       | N/A                    |

## Choosing AWS vs RunPod

### Use RunPod when:
- **Single-node training** (d12-d34): simpler setup, competitive pricing
- **Quick experiments**: faster pod spin-up (~2 min vs ~5 min)
- **Budget-conscious**: community cloud can be 30% cheaper than AWS spot
- **AWS capacity unavailable**: RunPod often has H100s when AWS doesn't
- **No multi-node needed**: RunPod doesn't natively support multi-node torchrun

### Use AWS when:
- **Multi-node training required** (d42+): EFA + placement groups for fast NCCL
- **Spot interruption tolerance**: spot instances with auto-restart from checkpoint
- **S3 checkpoint backup**: automatic sync to durable storage
- **Institutional accounts**: existing AWS credits or enterprise agreements
- **Production runs**: better reliability guarantees with on-demand instances

## Recommended Configurations

### Quick experiments (d12-d20)
```bash
# RunPod, single node, < $5
RUNPOD_API_KEY=xxx DEPTH=12 GPUS_PER_NODE=8 \
  bash runs/cloud/runpod_launch.sh
```

### Standard training (d26)
```bash
# RunPod or AWS, single node, ~$85-$180
DEPTH=26 TRAIN_STAGE=pretrain bash runs/launch_cloud.sh
```

### Large model (d34)
```bash
# AWS single node, ~12 hrs, ~$360
DEPTH=34 NUM_NODES=1 TARGET_PARAM_DATA_RATIO=11 \
  TRAIN_STAGE=pretrain bash runs/launch_cloud.sh

# AWS multi-node for faster turnaround, ~3 hrs, ~$360
DEPTH=34 NUM_NODES=8 TARGET_PARAM_DATA_RATIO=11 \
  TRAIN_STAGE=pretrain bash runs/launch_cloud.sh
```

### Very large model (d42-d45, requires multi-node)
```bash
# AWS 8-node, d42, ~5-9 hrs, ~$600
DEPTH=42 NUM_NODES=8 TARGET_PARAM_DATA_RATIO=11 \
  DEVICE_BATCH_SIZE=1 TRAIN_STAGE=pretrain bash runs/launch_cloud.sh
```

### Full pipeline (pretrain + SFT + reasoning + RL)
```bash
# d34 full pipeline, ~14 hrs total
DEPTH=34 TRAIN_STAGE=full_pipeline TARGET_PARAM_DATA_RATIO=11 \
  WANDB_RUN=pre1900_d34_full bash runs/launch_cloud.sh
```

## Checkpoint Resume

All stages support resuming from checkpoints. If a spot instance is interrupted
or training fails, restart with `--resume-from-step`:

```bash
# Add to base_train.py invocations
--resume-from-step=<last_saved_step>
```

For AWS spot instances, checkpoints are auto-synced to S3 every 30 minutes when
`AWS_S3_BUCKET` is set. To restore on a new instance:

```bash
aws s3 sync s3://$AWS_S3_BUCKET/checkpoints/ $NANOCHAT_BASE_DIR/
```

## GPU Memory Requirements

| Depth | device_batch_size=32 | dbs=16 | dbs=4 | dbs=1 |
|-------|---------------------|--------|-------|-------|
| d12   | ~15 GB              | ~10 GB | ~8 GB | ~7 GB |
| d20   | ~45 GB              | ~28 GB | ~15 GB| ~12 GB|
| d26   | ~72 GB              | ~42 GB | ~22 GB| ~16 GB|
| d34   | OOM                 | OOM    | ~55 GB| ~35 GB|
| d42   | OOM                 | OOM    | OOM   | ~65 GB|
| d45   | OOM                 | OOM    | OOM   | ~78 GB|

With `--activation-checkpointing`, memory is roughly halved at the cost of ~30%
slower training. Use this to fit larger models on fewer GPUs.

With FP8 (`--fp8`, H100 only), memory is reduced by ~30% and throughput improves
by ~40-60%.
