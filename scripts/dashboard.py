#!/usr/bin/env python3
"""
Live terminal dashboard for monitoring nanochat serving cluster.

Usage:
    python -m scripts.dashboard                    # 8 GPUs, default DB
    python -m scripts.dashboard --num-gpus 4       # 4 GPUs
    python -m scripts.dashboard --refresh 2        # 2 second refresh
"""

import argparse
import curses
import json
import os
import sqlite3
import time
import urllib.request
from datetime import datetime, timezone, timedelta

parser = argparse.ArgumentParser(description="NanoChat Live Dashboard")
parser.add_argument("--num-gpus", type=int, default=8)
parser.add_argument("--base-port", type=int, default=8001)
parser.add_argument("--refresh", type=float, default=1.0, help="Refresh interval in seconds")
parser.add_argument("--log-db", type=str, default=None, help="Path to chat_logs.db")
args = parser.parse_args()

if args.log_db is None:
    for nvme in ["/opt/dlami/nvme", "/workspace", "/mnt/nvme", "/local_nvme"]:
        if os.path.isdir(nvme):
            args.log_db = os.path.join(nvme, "chat_logs.db")
            break
    else:
        args.log_db = "chat_logs.db"


# ── helpers ──────────────────────────────────────────────────────────────────

def fetch_json(url, timeout=2):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return json.loads(resp.read())
    except Exception:
        return None


def get_db_connection():
    if not os.path.exists(args.log_db):
        return None
    conn = sqlite3.connect(args.log_db, timeout=2)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def query_db_stats(conn):
    """Query aggregate stats from the chat logs DB."""
    now = datetime.now(timezone.utc)
    stats = {}

    # Request counts over time windows
    for label, minutes in [("1m", 1), ("5m", 5), ("15m", 15), ("1h", 60), ("24h", 1440)]:
        since = (now - timedelta(minutes=minutes)).isoformat()
        row = conn.execute(
            "SELECT COUNT(*) as cnt FROM chat_logs WHERE timestamp >= ?", (since,)
        ).fetchone()
        stats[f"reqs_{label}"] = row["cnt"]

    # Error count in last hour
    since_1h = (now - timedelta(hours=1)).isoformat()
    row = conn.execute(
        "SELECT COUNT(*) as cnt FROM chat_logs WHERE timestamp >= ? AND error IS NOT NULL", (since_1h,)
    ).fetchone()
    stats["errors_1h"] = row["cnt"]

    # Latency & TTFT percentiles from last 100 requests
    rows = conn.execute(
        "SELECT latency_ms, ttft_ms, completion_tokens, prompt_tokens FROM chat_logs "
        "WHERE error IS NULL ORDER BY id DESC LIMIT 100"
    ).fetchall()

    if rows:
        latencies = sorted([r["latency_ms"] for r in rows if r["latency_ms"] is not None])
        ttfts = sorted([r["ttft_ms"] for r in rows if r["ttft_ms"] is not None])
        comp_tokens = [r["completion_tokens"] for r in rows if r["completion_tokens"] is not None]
        prompt_tokens = [r["prompt_tokens"] for r in rows if r["prompt_tokens"] is not None]

        def percentile(data, p):
            if not data:
                return None
            idx = int(len(data) * p / 100)
            idx = min(idx, len(data) - 1)
            return data[idx]

        stats["lat_p50"] = percentile(latencies, 50)
        stats["lat_p95"] = percentile(latencies, 95)
        stats["lat_p99"] = percentile(latencies, 99)
        stats["lat_avg"] = sum(latencies) / len(latencies) if latencies else None

        stats["ttft_p50"] = percentile(ttfts, 50)
        stats["ttft_p95"] = percentile(ttfts, 95)
        stats["ttft_p99"] = percentile(ttfts, 99)
        stats["ttft_avg"] = sum(ttfts) / len(ttfts) if ttfts else None

        stats["avg_comp_tokens"] = sum(comp_tokens) / len(comp_tokens) if comp_tokens else None
        stats["avg_prompt_tokens"] = sum(prompt_tokens) / len(prompt_tokens) if prompt_tokens else None

        # Tokens per second (completion_tokens / latency_s)
        tps_list = []
        for r in rows:
            if r["latency_ms"] and r["completion_tokens"] and r["latency_ms"] > 0:
                tps_list.append(r["completion_tokens"] / (r["latency_ms"] / 1000))
        stats["avg_tps"] = sum(tps_list) / len(tps_list) if tps_list else None
    else:
        for k in ["lat_p50", "lat_p95", "lat_p99", "lat_avg",
                   "ttft_p50", "ttft_p95", "ttft_p99", "ttft_avg",
                   "avg_comp_tokens", "avg_prompt_tokens", "avg_tps"]:
            stats[k] = None

    # Total requests all time
    row = conn.execute("SELECT COUNT(*) as cnt FROM chat_logs").fetchone()
    stats["total_requests"] = row["cnt"]

    # Recent requests for the live feed
    recent = conn.execute(
        "SELECT timestamp, latency_ms, ttft_ms, completion_tokens, prompt_tokens, "
        "gpu_id, client_ip, messages_json, response_text, error "
        "FROM chat_logs ORDER BY id DESC LIMIT 8"
    ).fetchall()
    stats["recent"] = [dict(r) for r in recent]

    return stats


# ── rendering ────────────────────────────────────────────────────────────────

def fmt_ms(val):
    if val is None:
        return "  --  "
    if val >= 1000:
        return f"{val/1000:.1f}s"
    return f"{val:.0f}ms"


def fmt_float(val, unit=""):
    if val is None:
        return "--"
    return f"{val:.1f}{unit}"


def safe_addstr(win, y, x, text, *attrs):
    """addstr that silently truncates if it would overflow the window."""
    h, w = win.getmaxyx()
    if y < 0 or y >= h or x >= w:
        return
    max_len = w - x - 1
    if max_len <= 0:
        return
    try:
        win.addstr(y, x, text[:max_len], *attrs)
    except curses.error:
        pass


def draw(stdscr, gpu_data, db_stats, start_time):
    h, w = stdscr.getmaxyx()
    stdscr.erase()

    # Colors
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_GREEN, -1)     # healthy
    curses.init_pair(2, curses.COLOR_RED, -1)        # down/error
    curses.init_pair(3, curses.COLOR_YELLOW, -1)     # warning
    curses.init_pair(4, curses.COLOR_CYAN, -1)       # headers
    curses.init_pair(5, curses.COLOR_WHITE, -1)      # normal
    curses.init_pair(6, curses.COLOR_MAGENTA, -1)    # accent

    GREEN = curses.color_pair(1) | curses.A_BOLD
    RED = curses.color_pair(2) | curses.A_BOLD
    YELLOW = curses.color_pair(3) | curses.A_BOLD
    CYAN = curses.color_pair(4) | curses.A_BOLD
    DIM = curses.color_pair(5) | curses.A_DIM
    BOLD = curses.A_BOLD
    MAGENTA = curses.color_pair(6) | curses.A_BOLD

    row = 0

    # ── Header ───────────────────────────────────────────────────────────
    uptime_s = time.time() - start_time
    uptime_str = f"{int(uptime_s//3600)}h {int((uptime_s%3600)//60)}m {int(uptime_s%60)}s"
    now_str = datetime.now().strftime("%H:%M:%S")

    title = " NANOCHAT DASHBOARD "
    safe_addstr(stdscr, row, 0, "─" * w, DIM)
    safe_addstr(stdscr, row, (w - len(title)) // 2, title, CYAN)
    row += 1
    safe_addstr(stdscr, row, 2, f"Time: {now_str}    Uptime: {uptime_str}    DB: {args.log_db}", DIM)
    row += 2

    # ── GPU Grid ─────────────────────────────────────────────────────────
    safe_addstr(stdscr, row, 2, "GPU CLUSTER", CYAN)
    row += 1

    total_active = 0
    total_capacity = 0
    total_pending = 0
    gpus_up = 0

    for i, gd in enumerate(gpu_data):
        col = 2 + (i % 4) * (w // 4)
        r = row + (i // 4) * 3

        if gd is None:
            safe_addstr(stdscr, r, col, f"GPU {i}", RED)
            safe_addstr(stdscr, r + 1, col, "  DOWN", RED)
        else:
            active = gd.get("active_requests", 0)
            avail = gd.get("available_slots", 0)
            capacity = gd.get("max_batch", 64)
            pending = gd.get("pending_requests", 0)
            total_active += active
            total_capacity += capacity
            total_pending += pending
            gpus_up += 1

            util = (active / capacity * 100) if capacity > 0 else 0
            color = GREEN if active == 0 else (YELLOW if util < 75 else RED)

            safe_addstr(stdscr, r, col, f"GPU {i}", BOLD)
            # Utilization bar
            bar_w = max(w // 4 - 8, 10)
            filled = int(bar_w * active / capacity) if capacity > 0 else 0
            bar = "█" * filled + "░" * (bar_w - filled)
            safe_addstr(stdscr, r + 1, col, f"  {bar}", color)
            safe_addstr(stdscr, r + 1, col + 2 + bar_w + 1, f"{active}/{capacity}", color)

    gpu_rows = ((args.num_gpus - 1) // 4 + 1) * 3
    row += gpu_rows + 1

    # ── Aggregate ────────────────────────────────────────────────────────
    safe_addstr(stdscr, row, 2, "CLUSTER SUMMARY", CYAN)
    row += 1
    util_pct = (total_active / total_capacity * 100) if total_capacity > 0 else 0
    util_color = GREEN if util_pct < 50 else (YELLOW if util_pct < 80 else RED)

    safe_addstr(stdscr, row, 2, f"GPUs: ", DIM)
    safe_addstr(stdscr, row, 8, f"{gpus_up}/{args.num_gpus} up", GREEN if gpus_up == args.num_gpus else RED)
    safe_addstr(stdscr, row, 22, f"Active: ", DIM)
    safe_addstr(stdscr, row, 30, f"{total_active}", util_color)
    safe_addstr(stdscr, row, 22 + 12, f"Capacity: {total_capacity}", DIM)
    safe_addstr(stdscr, row, 22 + 28, f"Util: ", DIM)
    safe_addstr(stdscr, row, 22 + 34, f"{util_pct:.0f}%", util_color)
    safe_addstr(stdscr, row, 22 + 42, f"Queue: ", DIM)
    q_color = GREEN if total_pending == 0 else (YELLOW if total_pending < 10 else RED)
    safe_addstr(stdscr, row, 22 + 49, f"{total_pending}", q_color)
    row += 2

    if db_stats is None:
        safe_addstr(stdscr, row, 2, f"DB not found: {args.log_db}", RED)
        row += 2
    else:
        # ── Request Volume ───────────────────────────────────────────────
        safe_addstr(stdscr, row, 2, "REQUEST VOLUME", CYAN)
        safe_addstr(stdscr, row, 30, "LATENCY (last 100)", CYAN)
        safe_addstr(stdscr, row, 60, "TTFT (last 100)", CYAN)
        row += 1

        # Volume column
        safe_addstr(stdscr, row, 2, f"Last 1m:  {db_stats['reqs_1m']}", BOLD)
        safe_addstr(stdscr, row + 1, 2, f"Last 5m:  {db_stats['reqs_5m']}", DIM)
        safe_addstr(stdscr, row + 2, 2, f"Last 15m: {db_stats['reqs_15m']}", DIM)
        safe_addstr(stdscr, row + 3, 2, f"Last 1h:  {db_stats['reqs_1h']}", DIM)
        safe_addstr(stdscr, row + 4, 2, f"Last 24h: {db_stats['reqs_24h']}", DIM)
        safe_addstr(stdscr, row + 5, 2, f"Total:    {db_stats['total_requests']}", DIM)
        errs = db_stats["errors_1h"]
        safe_addstr(stdscr, row + 6, 2, f"Errors/1h: {errs}", RED if errs > 0 else DIM)

        # Latency column
        safe_addstr(stdscr, row, 30, f"avg:  {fmt_ms(db_stats['lat_avg'])}", DIM)
        safe_addstr(stdscr, row + 1, 30, f"p50:  {fmt_ms(db_stats['lat_p50'])}", BOLD)
        safe_addstr(stdscr, row + 2, 30, f"p95:  {fmt_ms(db_stats['lat_p95'])}", YELLOW)
        safe_addstr(stdscr, row + 3, 30, f"p99:  {fmt_ms(db_stats['lat_p99'])}", RED)

        # TTFT column
        safe_addstr(stdscr, row, 60, f"avg:  {fmt_ms(db_stats['ttft_avg'])}", DIM)
        safe_addstr(stdscr, row + 1, 60, f"p50:  {fmt_ms(db_stats['ttft_p50'])}", BOLD)
        safe_addstr(stdscr, row + 2, 60, f"p95:  {fmt_ms(db_stats['ttft_p95'])}", YELLOW)
        safe_addstr(stdscr, row + 3, 60, f"p99:  {fmt_ms(db_stats['ttft_p99'])}", RED)

        # Throughput
        safe_addstr(stdscr, row + 5, 30, f"Avg tok/s:     {fmt_float(db_stats['avg_tps'])}", MAGENTA)
        safe_addstr(stdscr, row + 6, 30, f"Avg prompt:    {fmt_float(db_stats['avg_prompt_tokens'], ' tok')}", DIM)
        safe_addstr(stdscr, row + 5, 60, f"Avg completion: {fmt_float(db_stats['avg_comp_tokens'], ' tok')}", DIM)

        row += 8

        # ── Recent Requests ──────────────────────────────────────────────
        safe_addstr(stdscr, row, 2, "RECENT REQUESTS", CYAN)
        row += 1

        header = f"  {'Time':>8}  {'GPU':>3}  {'TTFT':>6}  {'Latency':>7}  {'Tok':>4}  {'IP':<15}  {'Prompt'}"
        safe_addstr(stdscr, row, 0, header, DIM | curses.A_UNDERLINE)
        row += 1

        for req in db_stats.get("recent", []):
            if row >= h - 2:
                break
            try:
                ts = datetime.fromisoformat(req["timestamp"]).strftime("%H:%M:%S")
            except Exception:
                ts = "??:??:??"
            gpu = req.get("gpu_id", "?")
            ttft = fmt_ms(req.get("ttft_ms"))
            lat = fmt_ms(req.get("latency_ms"))
            tok = req.get("completion_tokens") or 0
            ip = (req.get("client_ip") or "?")[:15]

            # Extract user prompt (truncated)
            prompt = ""
            try:
                msgs = json.loads(req.get("messages_json", "[]"))
                for m in reversed(msgs):
                    if m.get("role") == "user":
                        prompt = m["content"]
                        break
            except Exception:
                pass
            max_prompt = w - 55
            if len(prompt) > max_prompt:
                prompt = prompt[:max_prompt - 1] + "…"

            color = RED if req.get("error") else curses.A_NORMAL
            line = f"  {ts:>8}  {gpu:>3}  {ttft:>6}  {lat:>7}  {tok:>4}  {ip:<15}  {prompt}"
            safe_addstr(stdscr, row, 0, line, color)
            row += 1

    # Footer
    safe_addstr(stdscr, h - 1, 2, "Press 'q' to quit | Refreshing every {:.0f}s".format(args.refresh), DIM)

    stdscr.refresh()


# ── main loop ────────────────────────────────────────────────────────────────

def main(stdscr):
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.timeout(int(args.refresh * 1000))

    dash_start = time.time()
    db_conn = get_db_connection()

    while True:
        # Poll GPU health
        gpu_data = []
        for i in range(args.num_gpus):
            port = args.base_port + i
            data = fetch_json(f"http://127.0.0.1:{port}/stats")
            if data is None:
                data = fetch_json(f"http://127.0.0.1:{port}/health")
            gpu_data.append(data)

        # Query DB
        db_stats = None
        if db_conn:
            try:
                db_stats = query_db_stats(db_conn)
            except Exception:
                # DB might be locked briefly, skip this cycle
                pass

        # Draw
        try:
            draw(stdscr, gpu_data, db_stats, dash_start)
        except curses.error:
            pass

        # Handle input
        key = stdscr.getch()
        if key == ord("q") or key == ord("Q"):
            break

    if db_conn:
        db_conn.close()


if __name__ == "__main__":
    curses.wrapper(main)
