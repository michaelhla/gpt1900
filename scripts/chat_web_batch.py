#!/usr/bin/env python3
"""
Continuous-batching web chat server — one process per GPU.

Uses BatchEngine to serve many concurrent requests on a single GPU by batching
decode steps across all active requests.

Launch examples:

- Single GPU (default):
python -m scripts.chat_web_batch --port 8001

- Specific GPU:
python -m scripts.chat_web_batch --gpu-id 3 --port 8004

Endpoints:
  GET  /           - Chat UI
  POST /chat/completions - Chat API (streaming only)
  GET  /health     - Health check with batch engine stats
  GET  /stats      - Detailed batch engine statistics
"""

import argparse
import json
import os
import torch
import asyncio
import logging
import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse
from pydantic import BaseModel
from typing import List, Optional, AsyncGenerator
from nanochat.common import autodetect_device_type
from nanochat.checkpoint_manager import load_model, build_model, find_last_step
from nanochat.batch_engine import BatchEngine

# TTFT timeout: if first token doesn't arrive within this many seconds, return 503
TTFT_TIMEOUT = 20

# Abuse prevention limits (same as chat_web.py)
MAX_MESSAGES_PER_REQUEST = 500
MAX_MESSAGE_LENGTH = 8000
MAX_TOTAL_CONVERSATION_LENGTH = 32000
MIN_TEMPERATURE = 0.0
MAX_TEMPERATURE = 2.0
MIN_TOP_K = 0
MAX_TOP_K = 200
MIN_MAX_TOKENS = 1
MAX_MAX_TOKENS = 4096

parser = argparse.ArgumentParser(description='NanoChat Batch Server (one per GPU)')
parser.add_argument('--gpu-id', type=int, default=0, help='GPU device index')
parser.add_argument('-i', '--source', type=str, default="sft", help="Source of the model: sft|rl")
parser.add_argument('--model-dir', type=str, default=None, help='Load directly from this directory (bypasses -i/-g)')
parser.add_argument('-t', '--temperature', type=float, default=0.6, help='Default temperature')
parser.add_argument('-k', '--top-k', type=int, default=50, help='Default top-k')
parser.add_argument('-m', '--max-tokens', type=int, default=512, help='Default max tokens')
parser.add_argument('-g', '--model-tag', type=str, default=None, help='Model tag to load')
parser.add_argument('-s', '--step', type=int, default=None, help='Step to load')
parser.add_argument('-p', '--port', type=int, default=8001, help='Port')
parser.add_argument('--host', type=str, default='0.0.0.0', help='Host')
parser.add_argument('--api-key', type=str, default=os.environ.get('BACKEND_API_KEY', ''), help='API key for authentication (default: $BACKEND_API_KEY)')
parser.add_argument('--max-batch', type=int, default=64, help='Max concurrent requests per GPU')
parser.add_argument('-d', '--dtype', type=str, default='bfloat16', choices=['float32', 'bfloat16'])
args = parser.parse_args()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

device_type = autodetect_device_type()
device = torch.device(f"cuda:{args.gpu_id}" if device_type == "cuda" else device_type)
ptdtype = torch.float32 if args.dtype == 'float32' else torch.bfloat16


class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_k: Optional[int] = None


def validate_chat_request(request: ChatRequest):
    """Validate chat request to prevent abuse. Since we're single-turn, only the last user message matters."""
    if len(request.messages) == 0:
        raise HTTPException(status_code=400, detail="At least one message is required")

    # Find the last user message — that's all we'll use
    last_user_msg = None
    for message in reversed(request.messages):
        if message.role == "user":
            last_user_msg = message
            break
    if last_user_msg is None:
        raise HTTPException(status_code=400, detail="No user message found")
    if not last_user_msg.content:
        raise HTTPException(status_code=400, detail="User message has empty content")
    if len(last_user_msg.content) > MAX_MESSAGE_LENGTH:
        raise HTTPException(status_code=400, detail=f"Message too long. Max {MAX_MESSAGE_LENGTH} chars")

    if request.temperature is not None and not (MIN_TEMPERATURE <= request.temperature <= MAX_TEMPERATURE):
        raise HTTPException(status_code=400, detail=f"Temperature must be {MIN_TEMPERATURE}-{MAX_TEMPERATURE}")
    if request.top_k is not None and not (MIN_TOP_K <= request.top_k <= MAX_TOP_K):
        raise HTTPException(status_code=400, detail=f"top_k must be {MIN_TOP_K}-{MAX_TOP_K}")
    if request.max_tokens is not None and not (MIN_MAX_TOKENS <= request.max_tokens <= MAX_MAX_TOKENS):
        raise HTTPException(status_code=400, detail=f"max_tokens must be {MIN_MAX_TOKENS}-{MAX_MAX_TOKENS}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and start batch engine on startup."""
    print(f"Loading model on GPU {args.gpu_id}...")
    if args.model_dir:
        step = args.step if args.step is not None else find_last_step(args.model_dir)
        model, tokenizer, _ = build_model(args.model_dir, step, device, phase="eval")
    else:
        model, tokenizer, _ = load_model(args.source, device, phase="eval", model_tag=args.model_tag, step=args.step)

    engine = BatchEngine(
        model=model,
        tokenizer=tokenizer,
        device=device,
        dtype=ptdtype,
        max_batch=args.max_batch,
    )
    app.state.engine = engine
    app.state.tokenizer = tokenizer

    # Start the scheduler as a background task
    scheduler_task = asyncio.create_task(engine.run())
    print(f"Batch engine ready on GPU {args.gpu_id} (max_batch={args.max_batch})")
    print(f"Server ready at http://localhost:{args.port}")

    yield

    # Shutdown
    engine.stop()
    scheduler_task.cancel()


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# API key authentication middleware
OPEN_PATHS = {"/", "/logo.svg", "/health"}

class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if not args.api_key or request.url.path in OPEN_PATHS:
            return await call_next(request)
        auth = request.headers.get("authorization", "")
        if auth != f"Bearer {args.api_key}":
            return Response("Unauthorized", status_code=401)
        return await call_next(request)

app.add_middleware(AuthMiddleware)


@app.get("/")
async def root():
    """Serve the chat UI."""
    ui_html_path = os.path.join("nanochat", "ui.html")
    with open(ui_html_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    html_content = html_content.replace(
        "const API_URL = `http://${window.location.hostname}:8000`;",
        "const API_URL = '';"
    )
    return HTMLResponse(content=html_content)


@app.get("/logo.svg")
async def logo():
    return FileResponse(os.path.join("nanochat", "logo.svg"), media_type="image/svg+xml")


async def stream_from_queue(output_queue: asyncio.Queue, tokenizer, request_id: str,
                            engine: BatchEngine) -> AsyncGenerator[str, None]:
    """Read tokens from the batch engine's output queue and stream as SSE."""
    accumulated_tokens = []
    last_clean_text = ""
    first_token = True
    while True:
        try:
            # Apply TTFT timeout only for the first token
            timeout = TTFT_TIMEOUT if first_token else None
            msg = await asyncio.wait_for(output_queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            # TTFT timeout — cancel the request so the GPU doesn't waste work on it
            engine.cancel_request(request_id)
            yield f"data: {json.dumps({'error': 'Server is busy. Please try again in a few seconds.'})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
            return
        first_token = False
        if msg.get("done"):
            yield f"data: {json.dumps({'done': True})}\n\n"
            break
        token_id = msg["token_id"]
        accumulated_tokens.append(token_id)
        current_text = tokenizer.decode(accumulated_tokens)
        # Handle incomplete UTF-8 sequences (same logic as chat_web.py)
        if not current_text.endswith('\ufffd'):
            new_text = current_text[len(last_clean_text):]
            if new_text:
                yield f"data: {json.dumps({'token': new_text}, ensure_ascii=False)}\n\n"
                last_clean_text = current_text

    # Flush any remaining tokens buffered due to incomplete UTF-8
    if accumulated_tokens:
        final_text = tokenizer.decode(accumulated_tokens)
        remaining = final_text[len(last_clean_text):]
        if remaining and remaining != '\ufffd':
            yield f"data: {json.dumps({'token': remaining.replace('\ufffd', '')}, ensure_ascii=False)}\n\n"


@app.post("/chat/completions")
async def chat_completions(request: ChatRequest):
    """Chat completion endpoint with continuous batching."""
    validate_chat_request(request)

    engine: BatchEngine = app.state.engine
    tokenizer = app.state.tokenizer

    # Check capacity
    if engine.num_available <= 0 and engine.pending.qsize() > args.max_batch:
        raise HTTPException(status_code=503, detail="Server at capacity. Try again later.")

    # Log
    logger.info("=" * 20)
    for msg in request.messages:
        logger.info(f"[{msg.role.upper()}]: {msg.content}")
    logger.info("-" * 20)

    # Build conversation tokens (single-turn: only use the last user message)
    bos = tokenizer.get_bos_token_id()
    user_start = tokenizer.encode_special("<|user_start|>")
    user_end = tokenizer.encode_special("<|user_end|>")
    assistant_start = tokenizer.encode_special("<|assistant_start|>")

    # Extract last user message (already validated to exist)
    last_user_content = next(m.content for m in reversed(request.messages) if m.role == "user")

    conversation_tokens = [bos, user_start]
    conversation_tokens.extend(tokenizer.encode(last_user_content))
    conversation_tokens.extend([user_end, assistant_start])

    # Submit to batch engine
    request_id = str(uuid.uuid4())
    temperature = request.temperature if request.temperature is not None else args.temperature
    top_k = request.top_k if request.top_k is not None else args.top_k
    max_tokens = request.max_tokens if request.max_tokens is not None else args.max_tokens

    output_queue = engine.add_request(
        request_id=request_id,
        prompt_tokens=conversation_tokens,
        temperature=temperature,
        top_k=top_k,
        max_tokens=max_tokens,
    )

    return StreamingResponse(
        stream_from_queue(output_queue, tokenizer, request_id, engine),
        media_type="text/event-stream",
    )


@app.get("/health")
async def health():
    engine: BatchEngine = app.state.engine
    return {
        "status": "ok",
        "gpu_id": args.gpu_id,
        "active_requests": engine.num_active,
        "available_slots": engine.num_available,
        "max_batch": engine.max_batch,
    }


@app.get("/stats")
async def stats():
    engine: BatchEngine = app.state.engine
    return engine.stats()


if __name__ == "__main__":
    import uvicorn
    print(f"Starting NanoChat Batch Server on GPU {args.gpu_id}")
    print(f"Temperature: {args.temperature}, Top-k: {args.top_k}, Max tokens: {args.max_tokens}")
    print(f"Max batch: {args.max_batch}")
    uvicorn.run(app, host=args.host, port=args.port)
