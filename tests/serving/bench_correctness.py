#!/usr/bin/env python3
"""
Correctness test for the batch serving engine.

Verifies that:
1. Solo and batched generation produce identical output (temp=0)
2. Staggered requests batch correctly and produce coherent output
3. All responses are non-empty and well-formed

Usage:
    python tests/serving/bench_correctness.py --url http://localhost:8001
"""

import argparse
import time
import requests
import json
import threading
import concurrent.futures


def send_request(url, question, temperature=0.0, max_tokens=128):
    r = requests.post(f"{url}/chat/completions", json={
        'messages': [{'role': 'user', 'content': question}],
        'max_tokens': max_tokens, 'temperature': temperature,
    }, stream=True, timeout=60)
    tokens = []
    for line in r.iter_lines(decode_unicode=True):
        if line.startswith('data: '):
            data = json.loads(line[6:])
            if 'token' in data:
                tokens.append(data['token'])
    return ''.join(tokens)


def test_solo_vs_batched(url):
    """Same prompt should produce identical output solo vs batched (temp=0)."""
    print("=== Test 1: Solo vs Batched (temp=0, deterministic) ===")
    prompt = "What is the chemical composition of water?"

    # Solo
    solo = send_request(url, prompt, temperature=0.0)
    time.sleep(0.5)

    # Batched: same prompt + distractor
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        f1 = pool.submit(send_request, url, prompt, 0.0)
        f2 = pool.submit(send_request, url, "Tell me about the history of France.", 0.0)
        batched = f1.result()
        distractor = f2.result()

    match = solo == batched
    print(f"  Solo:      {solo[:100]}")
    print(f"  Batched:   {batched[:100]}")
    print(f"  Match:     {match}")
    print(f"  Distractor:{distractor[:80]}")
    assert match, "FAIL: Solo and batched outputs differ!"
    print("  PASS")
    print()


def test_staggered_batching(url):
    """Requests arriving at different times should still batch and produce coherent output."""
    print("=== Test 2: Staggered requests batch correctly ===")
    results = {}

    def send_delayed(name, question, delay):
        time.sleep(delay)
        results[name] = send_request(url, question, temperature=0.7, max_tokens=256)

    threads = [
        threading.Thread(target=send_delayed, args=('A', 'What is fire?', 0)),
        threading.Thread(target=send_delayed, args=('B', 'What is water?', 1)),
        threading.Thread(target=send_delayed, args=('C', 'What is earth?', 2)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    all_ok = True
    for name in ['A', 'B', 'C']:
        text = results[name]
        ok = len(text) > 20
        status = "PASS" if ok else "FAIL"
        print(f"  [{name}] ({len(text.split())} words) {status}: {text[:100]}")
        if not ok:
            all_ok = False

    assert all_ok, "FAIL: Some staggered responses were too short"
    print("  PASS")
    print()


def test_concurrent_quality(url):
    """All concurrent responses should be non-empty and coherent."""
    print("=== Test 3: Concurrent response quality ===")
    questions = [
        'What is fire?', 'What is water?', 'What is air?', 'What is earth?',
        'What is light?', 'What is sound?', 'What is heat?', 'What is gravity?',
        'What is magnetism?', 'What is electricity?',
    ]

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
        futures = {pool.submit(send_request, url, q, 0.7, 64): q for q in questions}
        results = []
        for f in concurrent.futures.as_completed(futures):
            q = futures[f]
            text = f.result()
            results.append((q, text))

    empty = [(q, t) for q, t in results if len(t) < 5]
    print(f"  {len(results)} responses, {len(empty)} empty/short")
    for q, t in results[:3]:
        print(f"  [{q}] {t[:80]}")
    assert len(empty) == 0, f"FAIL: {len(empty)} responses were empty"
    print("  PASS")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch serving correctness tests")
    parser.add_argument("--url", default="http://localhost:8001", help="Server URL")
    args = parser.parse_args()

    try:
        r = requests.get(f"{args.url}/health", timeout=5)
        print(f"Server OK: {r.json()}")
    except Exception as e:
        print(f"Server not reachable at {args.url}: {e}")
        exit(1)
    print()

    test_solo_vs_batched(args.url)
    test_staggered_batching(args.url)
    test_concurrent_quality(args.url)

    print("All tests passed!")
