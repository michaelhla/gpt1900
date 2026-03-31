"""
Continuous batching engine for serving.

One model per GPU, but batches decode steps across many concurrent requests.
Active requests are kept contiguous at indices 0..num_active-1 in the KV cache pool.
When a request completes, the last active slot is swapped into its position.

Design:
- KVCachePool: pre-allocated KV cache with slot management
- BatchEngine: async scheduler that prefills new requests and runs batched decode
- Requests are added/removed dynamically between decode steps
"""

import torch
import torch.nn.functional as F
import asyncio
import random
from dataclasses import dataclass


@torch.inference_mode()
def sample_next_token(logits, rng, temperature=1.0, top_k=None):
    """Sample a single next token from logits of shape (1, vocab_size). Returns int."""
    assert logits.shape[0] == 1
    if temperature == 0.0:
        return torch.argmax(logits, dim=-1).item()
    if top_k is not None and top_k > 0:
        k = min(top_k, logits.size(-1))
        vals, idx = torch.topk(logits, k, dim=-1)
        vals = vals / temperature
        probs = F.softmax(vals, dim=-1)
        choice = torch.multinomial(probs, num_samples=1, generator=rng)
        return idx.gather(1, choice).item()
    else:
        logits = logits / temperature
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1, generator=rng).item()


class KVCacheView:
    """
    A view into the KVCachePool for a contiguous range of slots.
    Implements the same interface as engine.py:KVCache so the model's forward() works unchanged.
    Uses slicing (not fancy indexing) so FA3 in-place updates propagate to the pool.
    """

    def __init__(self, pool, start, count):
        self.pool = pool
        self.start = start
        self.count = count
        self.n_layers = pool.n_layers

    def get_pos(self):
        """Get position of first element (used for rotary offset in uniform-position case)."""
        return self.pool.cache_seqlens[self.start].item()

    def get_positions(self):
        """Get per-element positions for continuous batching (non-uniform positions)."""
        return self.pool.cache_seqlens[self.start:self.start + self.count]

    def get_layer_cache(self, layer_idx):
        """Return (k_cache, v_cache) views for a specific layer. These are real views, not copies."""
        s, e = self.start, self.start + self.count
        return self.pool.k_cache[layer_idx, s:e], self.pool.v_cache[layer_idx, s:e]

    @property
    def cache_seqlens(self):
        return self.pool.cache_seqlens[self.start:self.start + self.count]

    def advance(self, num_tokens):
        self.pool.cache_seqlens[self.start:self.start + self.count] += num_tokens


class KVCachePool:
    """
    Pre-allocated KV cache pool for continuous batching.
    Active slots are always at indices 0..num_active-1 (kept contiguous via swapping).
    """

    def __init__(self, max_slots, num_layers, seq_len, num_heads, head_dim, device, dtype):
        self.max_slots = max_slots
        self.n_layers = num_layers
        self.seq_len = seq_len
        # Pre-allocate: (n_layers, max_slots, seq_len, n_heads, head_dim)
        self.k_cache = torch.zeros(num_layers, max_slots, seq_len, num_heads, head_dim, device=device, dtype=dtype)
        self.v_cache = torch.zeros(num_layers, max_slots, seq_len, num_heads, head_dim, device=device, dtype=dtype)
        self.cache_seqlens = torch.zeros(max_slots, dtype=torch.int32, device=device)
        self.num_active = 0

    def has_free_slot(self):
        return self.num_active < self.max_slots

    def allocate(self) -> int:
        """Allocate the next slot. Returns batch index (always == num_active)."""
        assert self.has_free_slot(), "No free KV cache slots"
        idx = self.num_active
        self.cache_seqlens[idx] = 0
        self.num_active += 1
        return idx

    def free(self, batch_idx):
        """Free a slot by swapping it with the last active slot."""
        assert 0 <= batch_idx < self.num_active
        last = self.num_active - 1
        if batch_idx != last:
            # Swap KV cache data
            self.k_cache[:, batch_idx].copy_(self.k_cache[:, last])
            self.v_cache[:, batch_idx].copy_(self.v_cache[:, last])
            self.cache_seqlens[batch_idx] = self.cache_seqlens[last]
        self.num_active -= 1

    def get_prefill_view(self, batch_idx):
        """View for a single slot (for prefill with batch=1)."""
        return KVCacheView(self, batch_idx, 1)

    def get_decode_view(self):
        """View for all active slots (for batched decode)."""
        return KVCacheView(self, 0, self.num_active)


@dataclass
class RequestState:
    """State for one active request in the batch."""
    request_id: str
    batch_idx: int  # position in the KV cache pool (0..num_active-1)
    output_queue: asyncio.Queue  # tokens are put here for the SSE stream to read
    temperature: float = 0.8
    top_k: int = 50
    max_tokens: int = 512
    tokens_generated: int = 0
    last_token: int = 0  # the most recently generated token (used as decode input)
    completed: bool = False


class BatchEngine:
    """
    Continuous batching scheduler.

    Usage:
        engine = BatchEngine(model, tokenizer, device, max_batch=64)
        asyncio.create_task(engine.run())
        # For each request:
        queue = engine.add_request(request_id, prompt_tokens, temperature, top_k, max_tokens)
        async for token_data in read_queue(queue): ...
    """

    def __init__(self, model, tokenizer, device, dtype=torch.bfloat16, max_batch=64):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.dtype = dtype
        self.max_batch = max_batch

        # Stop tokens — any special token should terminate generation
        self.stop_tokens = {
            tokenizer.get_bos_token_id(),
            tokenizer.encode_special("<|user_start|>"),
            tokenizer.encode_special("<|user_end|>"),
            tokenizer.encode_special("<|assistant_start|>"),
            tokenizer.encode_special("<|assistant_end|>"),
            tokenizer.encode_special("<|python_start|>"),
            tokenizer.encode_special("<|python_end|>"),
            tokenizer.encode_special("<|output_start|>"),
            tokenizer.encode_special("<|output_end|>"),
        }
        self.stop_tokens.discard(None)  # in case any special token doesn't exist

        # RNG for sampling
        self.rng = torch.Generator(device=device)

        # KV cache pool
        m = model.config
        self.kv_pool = KVCachePool(
            max_slots=max_batch,
            num_layers=m.n_layer,
            seq_len=m.sequence_len,
            num_heads=m.n_kv_head,
            head_dim=m.n_embd // m.n_head,
            device=device,
            dtype=dtype,
        )

        # Active requests: list of RequestState, indices match KV pool positions
        self.active: list[RequestState] = []

        # Pending requests: (request_id, prompt_tokens, temperature, top_k, max_tokens, output_queue)
        self.pending: asyncio.Queue = asyncio.Queue()

        # Cancelled request IDs — checked during prefill to skip cancelled pending requests
        self.cancelled: set[str] = set()

        # Event to wake up the scheduler when there's work
        self.has_work = asyncio.Event()

        self._running = True

    @property
    def num_active(self):
        return len(self.active)

    @property
    def num_available(self):
        return self.max_batch - self.num_active

    def add_request(self, request_id, prompt_tokens, temperature=0.8, top_k=50, max_tokens=512):
        """Add a new request. Returns an asyncio.Queue that will receive token dicts."""
        output_queue = asyncio.Queue()
        self.pending.put_nowait((request_id, prompt_tokens, temperature, top_k, max_tokens, output_queue))
        self.has_work.set()
        return output_queue

    def cancel_request(self, request_id):
        """Cancel a request — works for both pending and active requests."""
        # Mark as cancelled so pending requests get skipped during prefill
        self.cancelled.add(request_id)
        # If already active, mark completed so it gets cleaned up
        for req in self.active:
            if req.request_id == request_id:
                req.completed = True
                req.output_queue.put_nowait({"done": True})
                return

    @torch.inference_mode()
    def _prefill_one(self, prompt_tokens, temperature, top_k, max_tokens, request_id, output_queue):
        """Prefill a single new request and add it to the active set."""
        batch_idx = self.kv_pool.allocate()
        kv_view = self.kv_pool.get_prefill_view(batch_idx)

        # Run model forward with full prompt
        ids = torch.tensor([prompt_tokens], dtype=torch.long, device=self.device)
        with torch.amp.autocast(device_type=self.device.type, dtype=self.dtype):
            logits = self.model.forward(ids, kv_cache=kv_view)
        # logits shape: (1, prompt_len, vocab_size) — we want the last position
        last_logits = logits[:, -1, :]

        # Sample first token
        self.rng.manual_seed(random.randint(0, 2**31 - 1))
        first_token = sample_next_token(last_logits, self.rng, temperature, top_k)

        # Create request state
        is_done = first_token in self.stop_tokens
        req = RequestState(
            request_id=request_id,
            batch_idx=batch_idx,
            output_queue=output_queue,
            temperature=temperature,
            top_k=top_k,
            max_tokens=max_tokens,
            tokens_generated=1,
            last_token=first_token,
            completed=is_done,
        )
        self.active.append(req)

        # Send to output
        if is_done:
            output_queue.put_nowait({"done": True})
        else:
            output_queue.put_nowait({"token_id": first_token})

    @torch.inference_mode()
    def _decode_step(self):
        """Run one batched decode step for all active (non-completed) requests."""
        # Filter to non-completed only
        active = [r for r in self.active if not r.completed]
        if not active:
            return

        n = len(active)
        kv_view = self.kv_pool.get_decode_view()

        # Build input: each request's last token
        input_ids = torch.tensor(
            [[req.last_token] for req in active],
            dtype=torch.long, device=self.device
        )  # (N, 1)

        # Model forward — the model handles per-element rotary positions when T==1
        with torch.amp.autocast(device_type=self.device.type, dtype=self.dtype):
            logits = self.model.forward(input_ids, kv_cache=kv_view)
        # logits: (N, 1, vocab_size)
        logits = logits[:, -1, :]  # (N, vocab_size)

        # Sample per-request (different params per request)
        for i, req in enumerate(active):
            self.rng.manual_seed(random.randint(0, 2**31 - 1))
            token = sample_next_token(logits[i:i+1], self.rng, req.temperature, req.top_k)
            req.last_token = token
            req.tokens_generated += 1

            hit_stop = token in self.stop_tokens
            hit_limit = req.tokens_generated >= req.max_tokens
            if hit_stop or hit_limit:
                req.completed = True
                if hit_limit and not hit_stop:
                    req.output_queue.put_nowait({"max_tokens_reached": True})
                req.output_queue.put_nowait({"done": True})
            else:
                req.output_queue.put_nowait({"token_id": token})

    def _cleanup_completed(self):
        """Remove completed requests and compact the KV cache pool."""
        # Iterate in reverse so swaps don't affect earlier indices
        i = len(self.active) - 1
        while i >= 0:
            req = self.active[i]
            if req.completed:
                last_idx = len(self.active) - 1
                if i != last_idx:
                    # Swap with last active: update the swapped request's batch_idx
                    self.active[last_idx].batch_idx = i
                    self.active[i] = self.active[last_idx]
                self.active.pop()
                # Free the KV cache slot (also swaps internally)
                self.kv_pool.free(i)
            i -= 1

    async def run(self):
        """Main scheduler loop. Run as an asyncio task."""
        while self._running:
            # 1. Prefill at most 1 new request per iteration (limits decode latency for existing requests)
            while not self.pending.empty() and self.kv_pool.has_free_slot():
                request_id, prompt_tokens, temperature, top_k, max_tokens, output_queue = self.pending.get_nowait()
                # Skip cancelled requests — don't waste GPU on prefill
                if request_id in self.cancelled:
                    self.cancelled.discard(request_id)
                    output_queue.put_nowait({"done": True})
                    continue
                self._prefill_one(prompt_tokens, temperature, top_k, max_tokens, request_id, output_queue)
                break  # only prefill 1 per iteration

            # 2. Run one batched decode step
            self._decode_step()

            # 3. Clean up completed requests
            self._cleanup_completed()

            # 4. Yield to event loop
            if self.active or not self.pending.empty():
                await asyncio.sleep(0)
            else:
                # No work — wait until a request arrives
                self.has_work.clear()
                await self.has_work.wait()

    def stop(self):
        self._running = False
        self.has_work.set()

    def stats(self):
        return {
            "active_requests": self.num_active,
            "available_slots": self.num_available,
            "max_batch": self.max_batch,
            "pending_requests": self.pending.qsize(),
        }
