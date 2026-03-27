# Server Setup Guide

How to stand up the gpt1900 chat server on a new GPU machine.

## 1. Prerequisites

- Python 3.10+, [uv](https://docs.astral.sh/uv/) package manager
- NVIDIA GPU(s) with CUDA 12.8+ (or CPU/MPS for dev)
- NVMe storage mounted (typically `/opt/dlami/nvme` on AWS DLAMI)

## 2. Install dependencies

```bash
cd /home/ubuntu/gpt1900
uv sync --extra gpu          # GPU machine (CUDA 12.8)
# uv sync --extra cpu        # CPU-only dev machine
```

## 3. Environment variables

```bash
# Required for eval that uses Claude judge
export ANTHROPIC_API_KEY=...

# Required for W&B logging
export WANDB_API_KEY=...
```

Do **not** hardcode keys in scripts — use `.env` or export them in your shell. `.env` is gitignored.

## 4. Model checkpoints (NVMe paths)

The server expects checkpoints on fast local storage. The env var `NANOCHAT_BASE_DIR` controls where nanochat looks (defaults to `~/.cache/nanochat` if unset).

### Directory layout on this machine

```
/opt/dlami/nvme/
├── nanochat_models/                    # Chat-serving models (~10 GB)
│   └── chatsft_checkpoints/
│       └── d34/
│           ├── model_000020.pt
│           ├── meta_000020.json
│           └── tokenizer/
│
├── gpt1900_v12_checkpoint/             # Standalone v12 checkpoint (~10 GB)
│   ├── model_000455.pt
│   ├── meta_000455.json
│   └── tokenizer/
│
├── gpt1900_openthoughts/               # OpenThoughts training tree (~79 GB)
│   ├── tokenizer/
│   ├── base_checkpoints/
│   ├── pre1900_openthoughts_sft_checkpoints/         (~12 GB)
│   ├── pre1900_openthoughts_format_sft_checkpoints/  (~12 GB)
│   ├── pre1900_openthoughts_rl_checkpoints/          (~24 GB)
│   ├── pre1900_openthoughts_rl_v5_checkpoints/
│   ├── physicssft_expanded_checkpoints/
│   └── instruct_data/                                (~2.3 GB)
│       └── openthoughts/
│           ├── rl_{prompts,problems}_{train,val}.jsonl
│           └── sft_{train,val}.jsonl
│
├── hf_cache/                           # HuggingFace cache (~74 GB)
└── chat_logs.db                        # SQLite chat log DB
```

### Reproducing on a new machine

These are **not** in git. To restore:

1. **nanochat_models** — download the chat SFT checkpoint from HuggingFace or copy from another machine
2. **gpt1900_openthoughts** — all training checkpoints and data. Upload/download via HF or `rsync`
3. **gpt1900_v12_checkpoint** — standalone eval checkpoint
4. **hf_cache** — re-downloads automatically (set `HF_HOME=/opt/dlami/nvme/hf_cache` to cache here)

## 5. Starting the chat server

### Single GPU (simplest)

```bash
export NANOCHAT_BASE_DIR=/opt/dlami/nvme/nanochat_models
python -m scripts.chat_web
```

### Multi-GPU (data parallel)

```bash
export NANOCHAT_BASE_DIR=/opt/dlami/nvme/nanochat_models
python -m scripts.chat_web --num-gpus 4
```

### Serve a specific checkpoint

```bash
python -m scripts.chat_web -i sft -g d34 -s 20
```

### Key server flags

| Flag | Default | Description |
|------|---------|-------------|
| `-n, --num-gpus` | 1 | Number of GPUs for data parallelism |
| `-i, --source` | `sft` | Model source: `sft`, `rl`, or `base` |
| `-g, --model-tag` | auto (largest) | Model tag subdirectory |
| `-s, --step` | auto (latest) | Checkpoint step number |
| `-p, --port` | 8000 | Server port |
| `-t, --temperature` | 0.8 | Default generation temperature |
| `-k, --top-k` | 50 | Default top-k sampling |
| `-m, --max-tokens` | 512 | Default max tokens |
| `--log-db` | auto (NVMe) | Path to SQLite chat log DB |
| `--host` | 0.0.0.0 | Bind address |

### Model source resolution

`load_model(source)` maps source names to checkpoint dirs:
- `base` → `{NANOCHAT_BASE_DIR}/base_checkpoints/`
- `sft` → `{NANOCHAT_BASE_DIR}/chatsft_checkpoints/`
- `rl` → `{NANOCHAT_BASE_DIR}/chatrl_checkpoints/`

## 6. Production (nginx reverse proxy)

For multi-GPU serving behind nginx (e.g., 8x H100):

```bash
# Start 8 workers on ports 8001-8008
for i in $(seq 1 8); do
    python -m scripts.chat_web --num-gpus 1 --port $((8000 + i)) &
done

# Start nginx with the included config
sudo nginx -c /path/to/gpt1900/deploy/nginx/nanochat.conf
```

The nginx config at `deploy/nginx/nanochat.conf` includes:
- Least-connections load balancing across 8 workers
- Rate limiting (10 req/s per IP, burst 20)
- Connection limiting (4 concurrent per IP)
- SSE-compatible proxy settings (buffering off)

## 7. Frontend (Vercel)

The web frontend is in `deploy/vercel/`. It's a standalone Vercel app that talks to the backend API.

## 8. Chat log DB

The server logs all requests/responses to SQLite at the path resolved by `--log-db`:
- `/opt/dlami/nvme/chat_logs.db` (this machine, auto-detected)
- Falls back to `./chat_logs.db` if no NVMe found

## 9. Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Chat UI |
| `/chat/completions` | POST | Chat API (SSE streaming) |
| `/health` | GET | Health check with worker pool status |
| `/stats` | GET | Worker pool stats and GPU utilization |
