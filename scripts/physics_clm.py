"""
Continued pretraining (CLM) on physics books.

Standard causal language modeling on all tokens — no instruction formatting,
no user/assistant masking. Loads the D34 base model and trains on chunked
physics text parquets.

Run as:
    python -m scripts.physics_clm

Or torchrun for multi-GPU training:
    torchrun --standalone --nproc_per_node=8 -m scripts.physics_clm -- \
        --data-dir $NANOCHAT_BASE_DIR/physics_clm_data \
        --device-batch-size 8
"""

import argparse
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import time
import wandb
import torch
import pyarrow.parquet as pq
from contextlib import nullcontext
from nanochat.common import compute_init, compute_cleanup, print0, DummyWandb, get_base_dir, autodetect_device_type
from nanochat.tokenizer import get_token_bytes
from nanochat.checkpoint_manager import save_checkpoint
from nanochat.loss_eval import evaluate_bpb
from nanochat.checkpoint_manager import load_model
import torch.distributed as dist

# -----------------------------------------------------------------------------
# CLI arguments
parser = argparse.ArgumentParser(description="Continued pretraining (CLM) on physics books")
# Logging
parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' disables wandb logging)")
# Runtime
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
parser.add_argument("--dtype", type=str, default="bfloat16", help="float32|bfloat16")
# Model loading
parser.add_argument("--model-tag", type=str, default=None, help="model tag to load from")
parser.add_argument("--model-step", type=int, default=None, help="model step to load from")
# Data
parser.add_argument("--data-dir", type=str, required=True, help="directory with train.parquet and val.parquet")
# Training horizon
parser.add_argument("--num-epochs", type=int, default=5, help="number of epochs over the dataset")
parser.add_argument("--num-iterations", type=int, default=-1, help="override: stop after N steps (-1 = use num-epochs)")
# Batch sizes
parser.add_argument("--max-seq-len", type=int, default=2048, help="max context length")
parser.add_argument("--device-batch-size", type=int, default=8, help="per-device batch size")
parser.add_argument("--total-batch-size", type=int, default=65536, help="total batch size in tokens")
# Optimization
parser.add_argument("--embedding-lr", type=float, default=0.05, help="learning rate for embedding parameters (Adam)")
parser.add_argument("--unembedding-lr", type=float, default=0.001, help="learning rate for unembedding parameters (Adam)")
parser.add_argument("--matrix-lr", type=float, default=0.005, help="learning rate for matrix parameters (Muon)")
parser.add_argument("--weight-decay", type=float, default=0.0, help="weight decay for embedding/unembedding parameters (Adam)")
# Evaluation & saving
parser.add_argument("--eval-every", type=int, default=50, help="evaluate val bpb every N steps (-1 = disable)")
parser.add_argument("--eval-tokens", type=int, default=10*65536, help="number of tokens to evaluate val loss on")
parser.add_argument("--save-every", type=int, default=200, help="save checkpoint every N steps (-1 = only final)")
# Output
parser.add_argument("--output-tag", type=str, default=None, help="checkpoint save name (default: model-tag or d<depth>)")
parser.add_argument("--dry-run", action="store_true", help="log to wandb but skip checkpoints/report")
args = parser.parse_args()
user_config = vars(args).copy()
# -----------------------------------------------------------------------------

# Compute init
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0
ptdtype = torch.float32 if args.dtype == 'float32' else torch.bfloat16
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0

# wandb logging init
use_dummy_wandb = args.run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat-clm", name=args.run, config=user_config)

# Load the model and tokenizer
model, tokenizer, meta = load_model("base", device, phase="train", model_tag=args.model_tag, step=args.model_step)
pretrain_batch_size = meta.get("device_batch_size", None)
if pretrain_batch_size is not None and args.device_batch_size > pretrain_batch_size:
    print0(f"FOOTGUN WARNING: base model training used device_batch_size {pretrain_batch_size}, did you pass in a good --device-batch-size to this script?")
orig_model = model
model = torch.compile(model, dynamic=False)
depth = model.config.n_layer
num_flops_per_token = model.estimate_flops()
tokens_per_fwdbwd = args.device_batch_size * args.max_seq_len
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size
assert args.total_batch_size % world_tokens_per_fwdbwd == 0
grad_accum_steps = args.total_batch_size // world_tokens_per_fwdbwd
print0(f"Tokens / micro-batch / rank: {args.device_batch_size} x {args.max_seq_len} = {tokens_per_fwdbwd:,}")
print0(f"Tokens / micro-batch: {world_tokens_per_fwdbwd:,}")
print0(f"Total batch size {args.total_batch_size:,} => gradient accumulation steps: {grad_accum_steps}")
token_bytes = get_token_bytes(device=device)

# Initialize the Optimizer (combined MuonAdamW: Muon for matrix params, AdamW for rest)
optimizer = model.setup_optimizer(unembedding_lr=args.unembedding_lr, embedding_lr=args.embedding_lr, matrix_lr=args.matrix_lr, weight_decay=args.weight_decay)
for group in optimizer.param_groups:
    group["initial_lr"] = group["lr"]

# -----------------------------------------------------------------------------
# Load physics parquet data
def load_parquet_texts(filepath):
    """Load all text rows from a parquet file."""
    table = pq.read_table(filepath, columns=["text"])
    return table.column("text").to_pylist()

train_texts = load_parquet_texts(os.path.join(args.data_dir, "train.parquet"))
val_texts = load_parquet_texts(os.path.join(args.data_dir, "val.parquet"))
print0(f"Train documents: {len(train_texts)}")
print0(f"Val documents: {len(val_texts)}")

# Estimate total steps for LR scheduling
# Tokenize a sample to estimate tokens/doc ratio
sample_size = min(100, len(train_texts))
sample_token_counts = [len(tokenizer.encode(train_texts[i], prepend=tokenizer.get_bos_token_id())) for i in range(sample_size)]
avg_tokens_per_doc = sum(sample_token_counts) / len(sample_token_counts)
total_train_tokens = avg_tokens_per_doc * len(train_texts) * args.num_epochs
est_total_steps = int(total_train_tokens / args.total_batch_size)
print0(f"Avg tokens/doc: {avg_tokens_per_doc:.0f}")
print0(f"Est. total train tokens: {total_train_tokens:,.0f}")
print0(f"Est. total steps: {est_total_steps}")

# -----------------------------------------------------------------------------
# CLM data generator with BOS-aligned bestfit packing
last_step = False
approx_progress = 0.0
current_epoch = 1

def clm_data_generator_bos_bestfit(split, buffer_size=100):
    """
    BOS-aligned dataloader for CLM with bestfit-pad packing.

    Each row starts with BOS. Documents are packed using best-fit algorithm.
    When no document fits, the row is padded with BOS tokens (targets masked to -1).
    Loops over the dataset for --num-epochs.
    All tokens are trained on (no user/assistant masking).
    """
    global last_step, approx_progress, current_epoch
    assert split in {"train", "val"}, "split must be 'train' or 'val'"
    texts = train_texts if split == "train" else val_texts
    dataset_size = len(texts)
    assert dataset_size > 0
    row_capacity = args.max_seq_len + 1  # +1 for target at last position
    bos_token = tokenizer.get_bos_token_id()

    # Document buffer: list of token lists
    doc_buffer = []
    cursor = ddp_rank  # Each rank processes different documents
    consumed = ddp_rank  # Track actual consumption
    epoch = 1
    it = 0

    def refill_buffer():
        nonlocal cursor, epoch
        while len(doc_buffer) < buffer_size:
            text = texts[cursor % dataset_size]
            ids = tokenizer.encode(text, prepend=bos_token)
            doc_buffer.append(ids)
            cursor += ddp_world_size
            # Track epoch transitions
            if cursor // ddp_world_size >= dataset_size * epoch:
                epoch += 1

    while True:
        rows = []
        row_lengths = []
        for _ in range(args.device_batch_size):
            row = []
            padded = False
            while len(row) < row_capacity:
                # Ensure buffer has documents
                while len(doc_buffer) < buffer_size:
                    refill_buffer()

                remaining = row_capacity - len(row)

                # Find largest document that fits entirely
                best_idx = -1
                best_len = 0
                for i, doc in enumerate(doc_buffer):
                    doc_len = len(doc)
                    if doc_len <= remaining and doc_len > best_len:
                        best_idx = i
                        best_len = doc_len

                if best_idx >= 0:
                    # Found a document that fits — use it entirely
                    doc = doc_buffer.pop(best_idx)
                    row.extend(doc)
                    consumed += ddp_world_size
                else:
                    # No document fits — pad the remainder
                    content_len = len(row)
                    row.extend([bos_token] * remaining)
                    padded = True
                    break

            if padded:
                row_lengths.append(content_len)
            else:
                row_lengths.append(row_capacity)
            rows.append(row[:row_capacity])

        it += 1

        # Stopping conditions
        if split == "train":
            current_epoch = epoch
            # Use token-based step count for progress (reliable with packing/padding)
            total_steps = args.num_iterations if args.num_iterations > 0 else est_total_steps
            approx_progress = min(it / max(total_steps, 1), 1.0)
            if it >= total_steps:
                last_step = True

        # Build tensors
        use_cuda = device_type == "cuda"
        batch_tensor = torch.tensor(rows, dtype=torch.long, pin_memory=use_cuda)
        inputs = batch_tensor[:, :-1].to(device=device, dtype=torch.int32, non_blocking=use_cuda)
        targets = batch_tensor[:, 1:].to(device=device, dtype=torch.int64, non_blocking=use_cuda)

        # Mask out padding positions in targets
        for i, content_len in enumerate(row_lengths):
            if content_len < row_capacity:
                targets[i, content_len-1:] = -1

        yield inputs, targets


train_loader = clm_data_generator_bos_bestfit("train")
build_val_loader = lambda: clm_data_generator_bos_bestfit("val")
progress = 0

# Learning rate scheduler: warmup first 10%, linear decay last 30%
def get_lr_multiplier(progress):
    if progress < 0.1:
        # Linear warmup
        return progress / 0.1
    elif progress < 0.7:
        # Constant
        return 1.0
    else:
        # Linear decay to 0
        return max(0.0, 1.0 - (progress - 0.7) / 0.3)

# Momentum scheduler for Muon optimizer
def get_muon_momentum(it):
    frac = min(it / 300, 1)
    momentum = (1 - frac) * 0.85 + frac * 0.95
    return momentum

# Helper to save checkpoint
base_dir = get_base_dir()
def do_save_checkpoint(step, val_bpb_val=None):
    if not master_process or args.dry_run:
        return
    output_dirname = args.output_tag or args.model_tag or f"d{depth}"
    checkpoint_dir = os.path.join(base_dir, "physics_clm_checkpoints", output_dirname)
    save_checkpoint(
        checkpoint_dir,
        step,
        orig_model.state_dict(),
        optimizer.state_dict(),
        {
            "step": step,
            "val_bpb": val_bpb_val,
            "model_config": {
                "sequence_len": args.max_seq_len,
                "vocab_size": tokenizer.get_vocab_size(),
                "n_layer": depth,
                "n_head": model.config.n_head,
                "n_kv_head": model.config.n_kv_head,
                "n_embd": model.config.n_embd,
                "window_pattern": model.config.window_pattern,
            },
            "user_config": user_config,
        }
    )

# -----------------------------------------------------------------------------
# Training loop
x, y = next(train_loader)
min_val_bpb = float("inf")
smooth_train_loss = 0
ema_beta = 0.9
total_training_time = 0
step = 0
while True:
    flops_so_far = num_flops_per_token * args.total_batch_size * step

    # Synchronize last_step across all ranks
    if ddp:
        last_step_tensor = torch.tensor(last_step, dtype=torch.int32, device=device)
        dist.all_reduce(last_step_tensor, op=dist.ReduceOp.MAX)
        last_step = bool(last_step_tensor.item())

    # Evaluate val bpb periodically
    val_bpb = None
    if last_step or (args.eval_every > 0 and step % args.eval_every == 0):
        model.eval()
        val_loader = build_val_loader()
        eval_steps = args.eval_tokens // (args.device_batch_size * args.max_seq_len * ddp_world_size)
        eval_steps = max(eval_steps, 1)
        with autocast_ctx:
            val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
        print0(f"Step {step:05d} | Validation bpb: {val_bpb:.4f}")
        if val_bpb < min_val_bpb:
            min_val_bpb = val_bpb
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "val/bpb": val_bpb,
        })
        model.train()

    # Save checkpoint periodically
    if args.save_every > 0 and step > 0 and step % args.save_every == 0:
        do_save_checkpoint(step, val_bpb)

    # Save final checkpoint
    if last_step:
        do_save_checkpoint(step, val_bpb)
        break

    # -------------------------------------------------------------------------
    # Single training step
    synchronize()
    t0 = time.time()
    for micro_step in range(grad_accum_steps):
        with autocast_ctx:
            loss = model(x, y)
        train_loss = loss.detach()
        loss = loss / grad_accum_steps
        loss.backward()
        x, y = next(train_loader)
        progress = max(progress, approx_progress)
    # Step the optimizer
    lrm = get_lr_multiplier(progress)
    muon_momentum = get_muon_momentum(step)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
        if group['kind'] == 'muon':
            group["momentum"] = muon_momentum
    optimizer.step()
    model.zero_grad(set_to_none=True)
    synchronize()
    t1 = time.time()
    dt = t1 - t0
    # -------------------------------------------------------------------------

    step += 1

    # Logging
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss.item()
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1))
    pct_done = 100 * progress
    tok_per_sec = int(args.total_batch_size / dt)
    flops_per_sec = num_flops_per_token * args.total_batch_size / dt
    promised_flops_per_sec_h100 = 989e12 * ddp_world_size
    mfu = 100 * flops_per_sec / promised_flops_per_sec_h100
    if step > 10:
        total_training_time += dt
    print0(f"step {step:05d} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.4f} | dt: {dt * 1000:.2f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.2f} | epoch: {current_epoch} | total time: {total_training_time/60:.2f}m")
    if step % 10 == 0:
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "train/loss": debiased_smooth_loss,
            "train/lrm": lrm,
            "train/dt": dt,
            "train/tok_per_sec": tok_per_sec,
            "train/mfu": mfu,
            "train/epoch": current_epoch,
        })

# Print final stats
print0(f"Peak memory usage: {get_max_memory() / 1024 / 1024:.2f}MiB")
print0(f"Total training time: {total_training_time/60:.2f}m")
print0(f"Minimum validation bpb: {min_val_bpb:.4f}")

# Log to report
if not args.dry_run:
    from nanochat.report import get_report
    get_report().log(section="Physics CLM", data=[
        user_config,
        {
            "Number of iterations": step,
            "DDP world size": ddp_world_size,
        },
        {
            "Minimum validation bpb": min_val_bpb,
        }
    ])

# Cleanup
wandb_run.finish()
compute_cleanup()
