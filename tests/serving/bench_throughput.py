#!/usr/bin/env python3
"""
Throughput and latency benchmark for the batch serving engine.

Measures user-experienced TTFT and latency (from submit time, not HTTP send time).
Supports both simultaneous and staggered request patterns.

Usage:
    # Against a single GPU worker
    python tests/serving/bench_throughput.py --url http://localhost:8001 --concurrent 100

    # Against nginx load balancer (8 GPUs)
    python tests/serving/bench_throughput.py --url http://localhost:8080 --concurrent 1000

    # Staggered arrivals over 60 seconds
    python tests/serving/bench_throughput.py --url http://localhost:8080 --concurrent 5000 --stagger 60

    # Custom max_tokens
    python tests/serving/bench_throughput.py --url http://localhost:8080 --concurrent 500 --max-tokens 256
"""

import argparse
import time
import requests
import json
import random
import concurrent.futures

QUESTIONS = [
    'What is fire?', 'What is water?', 'What is air?', 'What is earth?',
    'What is light?', 'What is sound?', 'What is heat?', 'What is gravity?',
    'What is magnetism?', 'What is electricity?', 'How does a compass work?',
    'How does a telescope work?', 'What causes rain?', 'What causes thunder?',
    'What causes earthquakes?', 'What is the moon made of?', 'How far is the sun?',
    'What is a star?', 'What is a comet?', 'What causes tides?',
]


def send_request(i, question, url, max_tokens, delay=0):
    if delay > 0:
        time.sleep(delay)
    my_submit = time.time()
    try:
        r = requests.post(f"{url}/chat/completions", json={
            'messages': [{'role': 'user', 'content': question}],
            'max_tokens': max_tokens, 'temperature': 0.7
        }, stream=True, timeout=600)
        tokens = []
        user_ttft = None
        got_error = False
        for line in r.iter_lines(decode_unicode=True):
            if line.startswith('data: '):
                data = json.loads(line[6:])
                if 'error' in data:
                    got_error = True
                elif 'token' in data:
                    tokens.append(data['token'])
                    if user_ttft is None:
                        user_ttft = time.time() - my_submit
        return {
            'tokens': len(tokens), 'user_ttft': user_ttft or 0,
            'user_latency': time.time() - my_submit,
            'ok': True, 'timed_out': got_error,
            'text': ''.join(tokens), 'question': question,
            'status': r.status_code,
        }
    except Exception as e:
        return {
            'tokens': 0, 'user_ttft': 0,
            'user_latency': time.time() - my_submit,
            'ok': False, 'timed_out': False, 'error': str(e),
            'text': '', 'question': question, 'status': 0,
        }


def run_benchmark(url, n, max_tokens, stagger, threads):
    delays = sorted([random.uniform(0, stagger) for _ in range(n)]) if stagger > 0 else [0] * n
    reqs = [(i, QUESTIONS[i % len(QUESTIONS)], delays[i]) for i in range(n)]

    t0 = time.time()
    max_workers = min(threads, n)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(send_request, i, q, url, max_tokens, d) for i, q, d in reqs]
        results = [f.result() for f in futures]
    wall = time.time() - t0

    served = [r for r in results if r['tokens'] > 0 and not r.get('timed_out')]
    timed_out = [r for r in results if r.get('timed_out')]
    rejected = [r for r in results if r.get('status') == 503]
    http_err = [r for r in results if 'error' in r]
    zero_tok = [r for r in results if r['tokens'] == 0 and not r.get('timed_out') and 'error' not in r and r.get('status') != 503]

    total_tok = sum(r['tokens'] for r in served)

    print(f"=== N={n}, max_tokens={max_tokens}, stagger={stagger}s, threads={max_workers} ===")
    print(f"  Served:     {len(served)} ({len(served)/n*100:.1f}%)")
    if timed_out:
        print(f"  Timed out:  {len(timed_out)} ({len(timed_out)/n*100:.1f}%)")
    if rejected:
        print(f"  503 reject: {len(rejected)} ({len(rejected)/n*100:.1f}%)")
    if http_err:
        print(f"  HTTP errors:{len(http_err)}")
    if zero_tok:
        print(f"  Zero-tok:   {len(zero_tok)}")
    print(f"  Wall time:  {wall:.1f}s")
    print(f"  Throughput: {total_tok/wall:.0f} tok/s")
    if served:
        avg_tok = total_tok / len(served)
        print(f"  Avg tok/req:{avg_tok:.0f}")

    if served:
        ttfts = sorted([r['user_ttft'] for r in served if r['user_ttft'] > 0])
        lats = sorted([r['user_latency'] for r in served])
        if ttfts:
            print(f"  User TTFT:    p50={ttfts[len(ttfts)//2]:.1f}s  p95={ttfts[int(len(ttfts)*0.95)]:.1f}s  max={max(ttfts):.1f}s")
        if lats:
            print(f"  User Latency: p50={lats[len(lats)//2]:.1f}s  p95={lats[int(len(lats)*0.95)]:.1f}s  max={max(lats):.1f}s")

    if timed_out:
        to_lats = sorted([r['user_latency'] for r in timed_out])
        print(f"  Timeout wait: p50={to_lats[len(to_lats)//2]:.1f}s  max={max(to_lats):.1f}s")

    # Spot check
    if served:
        print()
        samples = random.sample(served, min(3, len(served)))
        for s in samples:
            print(f"  [{s['tokens']} tok, TTFT {s['user_ttft']:.1f}s] {s['question']}")
            print(f"  {s['text'][:120]}")
            print()

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch serving benchmark")
    parser.add_argument("--url", default="http://localhost:8080", help="Server URL")
    parser.add_argument("--concurrent", type=int, nargs="+", default=[100, 500, 1000],
                        help="Number of concurrent requests (can specify multiple)")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max tokens per request")
    parser.add_argument("--stagger", type=float, default=0, help="Spread requests over N seconds (0=simultaneous)")
    parser.add_argument("--threads", type=int, default=500, help="Max HTTP threads")
    args = parser.parse_args()

    # Verify server is up
    try:
        r = requests.get(f"{args.url}/health", timeout=5)
        print(f"Server OK: {r.json()}")
    except Exception as e:
        print(f"Server not reachable at {args.url}: {e}")
        exit(1)
    print()

    for n in args.concurrent:
        run_benchmark(args.url, n, args.max_tokens, args.stagger, args.threads)
        print()
        time.sleep(1)
