from flask import Flask, request, jsonify, render_template_string
import threading
import time
import os
import queue
from datetime import datetime, timezone
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import random
import shutil

print("Node path:", shutil.which("node"))
print("NodeJS path:", shutil.which("nodejs"))

app = Flask(__name__)

TOKEN_EXPIRY_SECONDS = 14 * 60
POOL_TARGET = int(os.environ.get("POOL_TARGET", "1000"))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "10"))
API_PORT = int(os.environ.get("PORT", "8080"))

token_pool = deque()
served_tokens = set()
pool_lock = threading.Lock()
served_lock = threading.Lock()
pool_running = threading.Event()

stats = {
    "total_requests": 0,
    "single_requests": 0,
    "tokens_generated": 0,
    "tokens_served": 0,
    "tokens_expired": 0,
    "generation_failures": 0,
    "active_workers": 0,
    "pool_size": 0,
    "peak_queue": 0,
    "duplicates_rejected": 0,
    "tokens_flushed": 0,
    "last_received": "---",
    "last_served": "---",
    "start_time": datetime.now(timezone.utc).isoformat(),
}
stats_lock = threading.Lock()


def is_token_fresh(entry):
    age = time.time() - entry["generated_at"]
    return age < TOKEN_EXPIRY_SECONDS


def prune_expired():
    removed = 0
    with pool_lock:
        fresh = deque()
        while token_pool:
            entry = token_pool.popleft()
            if is_token_fresh(entry):
                fresh.append(entry)
            else:
                removed += 1
        token_pool.extend(fresh)
    if removed:
        with stats_lock:
            stats["tokens_expired"] += removed
            stats["pool_size"] = len(token_pool)
    return removed


def take_token():
    prune_expired()
    with pool_lock:
        while token_pool:
            entry = token_pool.popleft()
            if not is_token_fresh(entry):
                with stats_lock:
                    stats["tokens_expired"] += 1
                continue
            token_val = entry["token"]
            with served_lock:
                if token_val in served_tokens:
                    continue
                served_tokens.add(token_val)
            with stats_lock:
                stats["tokens_served"] += 1
                stats["pool_size"] = len(token_pool)
                stats["last_served"] = datetime.now(timezone.utc).isoformat()
            return entry
    return None


def cleanup_served_tokens():
    with served_lock:
        if len(served_tokens) > 10000:
            excess = len(served_tokens) - 5000
            for _ in range(excess):
                served_tokens.pop()


def generate_one_token(thread_id):
    from yidun import Dun163, DUN163_DOMAINS, ID, REFERER, FP_H
    from fake_useragent import UserAgent

    ua = UserAgent().random
    domain = DUN163_DOMAINS[thread_id % len(DUN163_DOMAINS)]

    d = Dun163(
        id_=ID,
        referer=REFERER,
        fp_h=FP_H,
        ua=ua,
        thread_id=thread_id,
        domain=domain,
    )

    get_conf_data = d.request_getconf()
    if not get_conf_data:
        return None

    dt = get_conf_data.get("dt")
    ac_data = get_conf_data.get("ac", {})
    ac_token = ac_data.get("token")
    bid = ac_data.get("bid")

    ir_data = get_conf_data.get("ir", {})
    ir_token = ir_data.get("token") if ir_data.get("enable") else None

    get_data = d.request_get(dt, bid, ac_token, ir_token)
    if not get_data:
        return None

    captcha_type = get_data.get("type", 7)
    token = get_data.get("token")
    if not token:
        return None

    if captcha_type != 7:
        return None

    bg_urls = get_data.get("bg", [])
    if not bg_urls:
        return None

    click_points, img_time = d.handle_click_captcha_hybrid(bg_urls[0], token)
    resp_json, js_time = d.request_check(dt, bid, token=token, captcha_type=7, click_data=click_points)

    if resp_json.get("result") == True:
        validate_raw = resp_json.get("validate", "")
        validate_decoded = ""
        if validate_raw and d.ctx:
            try:
                validate_decoded = d.ctx.call("do_onVerify", validate_raw, d.fp)
            except:
                pass

        if validate_decoded and len(validate_decoded.strip()) > 10:
            return validate_decoded.strip()

    return None


def pool_worker(worker_id):
    import yidun
    from loguru import logger

    logger.info(f"[Worker {worker_id}] Initializing...")

    model = yidun.initialize_global_model()
    if not model:
        logger.error(f"[Worker {worker_id}] Model not available, exiting")
        return

    js_ctx = yidun.get_compiled_js("dun163.js")
    if not js_ctx:
        logger.error(f"[Worker {worker_id}] JS context not available, exiting")
        return

    yidun.get_sift_detector()
    logger.success(f"[Worker {worker_id}] Ready, generating tokens...")

    while pool_running.is_set():
        with pool_lock:
            current_size = len(token_pool)
        if current_size >= POOL_TARGET:
            time.sleep(2)
            prune_expired()
            continue

        with stats_lock:
            stats["active_workers"] += 1

        try:
            logger.info(f"[Worker {worker_id}] Calling generator...")
            token_value = generate_one_token(worker_id)

            if token_value:
                logger.info(f"[Worker {worker_id}] Generator returned a result")
            else:
                logger.error(f"[Worker {worker_id}] Generator returned None")

            if token_value:
                with served_lock:
                    if token_value in served_tokens:
                        logger.debug(f"[Worker {worker_id}] Duplicate token skipped")
                        with stats_lock:
                            stats["duplicates_rejected"] += 1
                        continue

                entry = {
                    "token": token_value,
                    "generated_at": time.time(),
                    "worker": worker_id,
                }
                with pool_lock:
                    already_in_pool = any(t["token"] == token_value for t in token_pool)
                    if not already_in_pool:
                        token_pool.append(entry)
                        with stats_lock:
                            stats["tokens_generated"] += 1
                            stats["pool_size"] = len(token_pool)
                            stats["last_received"] = datetime.now(timezone.utc).isoformat()
                            if stats["pool_size"] > stats["peak_queue"]:
                                stats["peak_queue"] = stats["pool_size"]
                        logger.success(f"[Worker {worker_id}] Token added (pool: {len(token_pool)}/{POOL_TARGET})")
                    else:
                        with stats_lock:
                            stats["duplicates_rejected"] += 1
                        logger.debug(f"[Worker {worker_id}] Duplicate in pool, skipped")
            else:
                with stats_lock:
                    stats["generation_failures"] += 1
        except Exception as e:
            with stats_lock:
                stats["generation_failures"] += 1
            logger.error(f"[Worker {worker_id}] Error: {e}")
        finally:
            with stats_lock:
                stats["active_workers"] -= 1

        cleanup_served_tokens()
        time.sleep(random.uniform(0.5, 1.5))


def start_pool():
    pool_running.set()
    for i in range(1, NUM_WORKERS + 1):
        t = threading.Thread(target=pool_worker, args=(i,), daemon=True)
        t.start()
    print(f"  Token pool started: {NUM_WORKERS} workers, target {POOL_TARGET}")


def pruner_loop():
    while pool_running.is_set():
        prune_expired()
        time.sleep(30)


FRONTEND_HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CN31 Token Server</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
  *, *::before, *::after { margin: 0; padding: 0; box-sizing: border-box; }

  :root {
    --bg: #16101a;
    --bg2: #1e1524;
    --surface: rgba(36, 26, 44, 0.75);
    --surface-solid: #241a2c;
    --surface2: rgba(46, 34, 56, 0.8);
    --surface-hover: rgba(56, 40, 66, 0.9);
    --border: rgba(237, 130, 160, 0.08);
    --border-bright: rgba(237, 130, 160, 0.2);
    --border-glow: rgba(248, 140, 170, 0.35);
    --rose: #f8789c;
    --rose-bright: #fcadc4;
    --rose-dim: rgba(248, 120, 156, 0.5);
    --amber: #f5a623;
    --amber-bright: #fcc96e;
    --amber-dim: rgba(245, 166, 35, 0.5);
    --coral: #fb7a5c;
    --coral-dim: rgba(251, 122, 92, 0.5);
    --red: #f87171;
    --red-dim: rgba(248, 113, 113, 0.5);
    --violet: #c77dff;
    --violet-dim: rgba(199, 125, 255, 0.5);
    --gold: #fbbf24;
    --green: #10b981;
    --green-dim: rgba(16, 185, 129, 0.5);
    --text: #f5f1f2;
    --text-secondary: #c4b5bb;
    --text-muted: #8a7580;
    --text-dim: #4a3848;
    --font: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    --mono: 'JetBrains Mono', 'Fira Code', monospace;
    --glass: blur(20px) saturate(1.5);
    --radius: 16px;
    --radius-sm: 10px;
    --radius-xs: 6px;
  }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--font);
    min-height: 100vh;
    overflow-x: hidden;
    position: relative;
  }

  .orb {
    position: fixed;
    border-radius: 50%;
    filter: blur(100px);
    opacity: 0.15;
    pointer-events: none;
    z-index: 0;
  }
  .orb-1 {
    width: 600px; height: 600px;
    background: radial-gradient(circle, #f8789c, transparent 70%);
    top: -200px; left: -100px;
    animation: orbFloat1 20s ease-in-out infinite;
  }
  .orb-2 {
    width: 500px; height: 500px;
    background: radial-gradient(circle, #c77dff, transparent 70%);
    bottom: -150px; right: -100px;
    animation: orbFloat2 25s ease-in-out infinite;
  }
  .orb-3 {
    width: 400px; height: 400px;
    background: radial-gradient(circle, #f5a623, transparent 70%);
    top: 40%; left: 50%;
    animation: orbFloat3 18s ease-in-out infinite;
  }

  @keyframes orbFloat1 {
    0%, 100% { transform: translate(0, 0) scale(1); }
    33% { transform: translate(80px, 50px) scale(1.1); }
    66% { transform: translate(-40px, 80px) scale(0.9); }
  }
  @keyframes orbFloat2 {
    0%, 100% { transform: translate(0, 0) scale(1); }
    33% { transform: translate(-60px, -40px) scale(1.15); }
    66% { transform: translate(50px, -70px) scale(0.85); }
  }
  @keyframes orbFloat3 {
    0%, 100% { transform: translate(-50%, 0) scale(1); opacity: 0.1; }
    50% { transform: translate(-50%, -60px) scale(1.2); opacity: 0.18; }
  }

  body::after {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
      linear-gradient(rgba(248,120,156,0.02) 1px, transparent 1px),
      linear-gradient(90deg, rgba(248,120,156,0.02) 1px, transparent 1px);
    background-size: 60px 60px;
    pointer-events: none;
    z-index: 0;
    mask-image: radial-gradient(ellipse at 50% 50%, black 30%, transparent 80%);
    -webkit-mask-image: radial-gradient(ellipse at 50% 50%, black 30%, transparent 80%);
  }

  body::before {
    content: '';
    position: fixed;
    inset: 0;
    opacity: 0.03;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E");
    pointer-events: none;
    z-index: 1;
  }

  .wrap {
    position: relative;
    z-index: 2;
    max-width: 1200px;
    margin: 0 auto;
    padding: 40px 36px 60px;
  }

  .header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 44px;
    flex-wrap: wrap;
    gap: 20px;
  }

  .header-left {
    display: flex;
    align-items: center;
    gap: 16px;
  }

  .logo-box {
    width: 48px; height: 48px;
    border-radius: 14px;
    background: linear-gradient(135deg, rgba(248,120,156,0.15), rgba(199,125,255,0.1));
    border: 1px solid rgba(248,120,156,0.2);
    display: flex;
    align-items: center;
    justify-content: center;
    position: relative;
    overflow: hidden;
    flex-shrink: 0;
  }

  .logo-box::before {
    content: '';
    position: absolute;
    inset: -1px;
    border-radius: 14px;
    padding: 1px;
    background: linear-gradient(135deg, var(--rose), var(--violet));
    -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
    mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
    -webkit-mask-composite: xor;
    mask-composite: exclude;
    opacity: 0.4;
  }

  .logo-box svg { position: relative; z-index: 1; filter: drop-shadow(0 0 6px rgba(248,120,156,0.4)); }

  .header-title h1 {
    font-size: 24px;
    font-weight: 800;
    letter-spacing: -0.5px;
    background: linear-gradient(135deg, #f5f1f2 30%, var(--rose-bright));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.2;
  }

  .header-title .sub {
    font-family: var(--mono);
    font-size: 11px;
    color: var(--text-muted);
    letter-spacing: 0.5px;
    margin-top: 2px;
  }

  .header-title .sub span { color: var(--rose-dim); }

  .header-right {
    display: flex;
    align-items: center;
    gap: 12px;
  }

  .mode-badge {
    display: flex;
    align-items: center;
    gap: 8px;
    background: rgba(251, 122, 92, 0.06);
    border: 1px solid rgba(251, 122, 92, 0.15);
    border-radius: 100px;
    padding: 8px 16px;
    font-family: var(--mono);
    font-size: 10px;
    font-weight: 600;
    color: var(--coral);
    letter-spacing: 1px;
    text-transform: uppercase;
  }

  .status-badge {
    display: flex;
    align-items: center;
    gap: 8px;
    background: rgba(245, 166, 35, 0.06);
    border: 1px solid rgba(245, 166, 35, 0.15);
    border-radius: 100px;
    padding: 8px 20px 8px 14px;
    font-family: var(--mono);
    font-size: 11px;
    font-weight: 600;
    color: var(--amber);
    letter-spacing: 1px;
    text-transform: uppercase;
  }

  .live-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: var(--amber);
    position: relative;
    flex-shrink: 0;
  }
  .live-dot::after {
    content: '';
    position: absolute;
    inset: -4px;
    border-radius: 50%;
    background: var(--amber);
    opacity: 0.3;
    animation: livePulse 2s ease-in-out infinite;
  }
  @keyframes livePulse {
    0%, 100% { transform: scale(1); opacity: 0.3; }
    50% { transform: scale(1.8); opacity: 0; }
  }

  .time-badge {
    font-family: var(--mono);
    font-size: 11px;
    color: var(--text-muted);
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 100px;
    padding: 8px 16px;
    backdrop-filter: var(--glass);
  }

  .grid-4 {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin-bottom: 16px;
  }

  .card {
    background: var(--surface);
    backdrop-filter: var(--glass);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 24px;
    position: relative;
    overflow: hidden;
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
  }
  .card:hover {
    border-color: var(--border-bright);
    background: var(--surface-hover);
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.3), 0 0 0 1px var(--border-bright);
  }

  .card::before {
    content: '';
    position: absolute;
    top: 0; left: 20%; right: 20%;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--accent, var(--rose)), transparent);
    opacity: 0;
    transition: opacity 0.4s;
  }
  .card:hover::before { opacity: 0.6; }

  .card::after {
    content: '';
    position: absolute;
    top: -40px; right: -40px;
    width: 120px; height: 120px;
    border-radius: 50%;
    background: var(--accent, var(--rose));
    opacity: 0;
    filter: blur(40px);
    transition: opacity 0.4s;
    pointer-events: none;
  }
  .card:hover::after { opacity: 0.07; }

  .card.amber { --accent: var(--amber); }
  .card.rose { --accent: var(--rose); }
  .card.coral { --accent: var(--coral); }
  .card.red { --accent: var(--red); }
  .card.violet { --accent: var(--violet); }
  .card.gold { --accent: var(--gold); }
  .card.green { --accent: var(--green); }

  .card-icon {
    width: 36px; height: 36px;
    border-radius: var(--radius-sm);
    background: linear-gradient(135deg, color-mix(in srgb, var(--accent, var(--rose)) 12%, transparent), transparent);
    border: 1px solid color-mix(in srgb, var(--accent, var(--rose)) 15%, transparent);
    display: flex;
    align-items: center;
    justify-content: center;
    margin-bottom: 16px;
  }

  .card-label {
    font-family: var(--mono);
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: var(--text-muted);
    margin-bottom: 10px;
  }

  .card-value {
    font-size: 40px;
    font-weight: 900;
    color: var(--text);
    line-height: 1;
    letter-spacing: -2px;
    font-variant-numeric: tabular-nums;
    position: relative;
  }

  .card-value .highlight {
    background: linear-gradient(135deg, var(--accent, var(--rose)), color-mix(in srgb, var(--accent, var(--rose)) 60%, white));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    transition: opacity 0.3s ease;
  }

  .card-sub {
    font-family: var(--mono);
    font-size: 10px;
    color: var(--text-muted);
    margin-top: 8px;
    display: flex;
    align-items: center;
    gap: 6px;
  }

  .card-sub .dot {
    width: 4px; height: 4px;
    border-radius: 50%;
    background: var(--accent, var(--rose));
    opacity: 0.5;
  }

  .section {
    background: var(--surface);
    backdrop-filter: var(--glass);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 28px;
    margin-bottom: 16px;
    position: relative;
    overflow: hidden;
  }

  .section-hdr {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 24px;
  }

  .section-hdr .icon {
    width: 28px; height: 28px;
    border-radius: 8px;
    background: rgba(248,120,156,0.08);
    border: 1px solid rgba(248,120,156,0.1);
    display: flex;
    align-items: center;
    justify-content: center;
  }

  .section-hdr h2 {
    font-size: 13px;
    font-weight: 700;
    letter-spacing: -0.2px;
    color: var(--text);
  }

  .section-hdr .line {
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, var(--border-bright), transparent);
  }

  .bar-group { margin-bottom: 18px; }
  .bar-group:last-child { margin-bottom: 0; }

  .bar-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 10px;
  }

  .bar-title {
    font-family: var(--mono);
    font-size: 11px;
    font-weight: 500;
    color: var(--text-secondary);
  }

  .bar-value {
    font-family: var(--mono);
    font-size: 11px;
    font-weight: 700;
    color: var(--text);
  }

  .bar-track {
    background: rgba(70, 45, 60, 0.5);
    border-radius: 100px;
    height: 8px;
    overflow: hidden;
    position: relative;
  }

  .bar-fill {
    height: 100%;
    border-radius: 100px;
    position: relative;
    transition: width 1s cubic-bezier(0.4, 0, 0.2, 1);
  }

  .bar-fill.gradient-rose {
    background: linear-gradient(90deg, #e0527a, #f8789c, #fcadc4);
  }
  .bar-fill.gradient-amber {
    background: linear-gradient(90deg, #d48a10, #f5a623, #fcc96e);
  }
  .bar-fill.gradient-violet {
    background: linear-gradient(90deg, #9b4de0, #c77dff, #e9d5ff);
  }

  .bar-fill::after {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(90deg, transparent 60%, rgba(255,255,255,0.2));
    border-radius: 100px;
  }

  .bar-fill::before {
    content: '';
    position: absolute;
    top: 0; left: -100%; bottom: 0;
    width: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
    animation: shimmer 3s ease-in-out infinite;
  }

  @keyframes shimmer {
    0% { left: -100%; }
    100% { left: 200%; }
  }

  .grid-2 {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
    margin-bottom: 16px;
  }

  .runtime-wrap {
    display: flex;
    gap: 28px;
    align-items: center;
  }

  .ring-container {
    position: relative;
    width: 100px; height: 100px;
    flex-shrink: 0;
  }

  .ring-container svg { transform: rotate(-90deg); }

  .ring-center {
    position: absolute;
    inset: 0;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
  }

  .ring-number {
    font-size: 22px;
    font-weight: 800;
    background: linear-gradient(135deg, var(--rose), var(--amber));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1;
    letter-spacing: -0.5px;
  }

  .ring-unit {
    font-family: var(--mono);
    font-size: 9px;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-top: 3px;
  }

  .stat-pills {
    display: flex;
    flex-direction: column;
    gap: 8px;
    flex: 1;
  }

  .stat-pill {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: rgba(55, 35, 48, 0.6);
    border: 1px solid var(--border);
    border-radius: var(--radius-xs);
    padding: 10px 14px;
    transition: border-color 0.3s, background 0.3s;
  }
  .stat-pill:hover {
    border-color: var(--border-bright);
    background: rgba(55, 35, 48, 0.9);
  }

  .stat-pill-label {
    font-family: var(--mono);
    font-size: 10px;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 1px;
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .stat-pill-label .indicator {
    width: 6px; height: 6px;
    border-radius: 2px;
  }

  .stat-pill-value {
    font-family: var(--mono);
    font-size: 13px;
    font-weight: 700;
  }

  .c-rose { color: var(--rose); }
  .c-amber { color: var(--amber); }
  .c-coral { color: var(--coral); }
  .c-violet { color: var(--violet); }
  .c-gold { color: var(--gold); }
  .c-green { color: var(--green); }

  .endpoint-table {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .ep-row {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 12px 14px;
    border-radius: var(--radius-xs);
    background: rgba(55, 35, 48, 0.3);
    border: 1px solid transparent;
    transition: all 0.3s;
    cursor: pointer;
  }
  .ep-row:hover {
    background: rgba(55, 35, 48, 0.7);
    border-color: var(--border);
    transform: translateX(4px);
  }

  .ep-method {
    font-family: var(--mono);
    font-size: 10px;
    font-weight: 700;
    padding: 4px 10px;
    border-radius: 4px;
    letter-spacing: 0.5px;
    min-width: 50px;
    text-align: center;
    flex-shrink: 0;
  }

  .ep-method.get {
    background: rgba(245,166,35,0.08);
    color: var(--amber);
    border: 1px solid rgba(245,166,35,0.2);
  }

  .ep-path {
    font-family: var(--mono);
    font-size: 12px;
    color: var(--text);
    font-weight: 500;
    flex: 1;
  }

  .ep-desc {
    font-size: 11px;
    color: var(--text-muted);
    text-align: right;
  }

  .ts-strip {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
    margin-bottom: 16px;
  }

  .ts-card {
    display: flex;
    align-items: center;
    gap: 14px;
    background: var(--surface);
    backdrop-filter: var(--glass);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 16px 20px;
    transition: border-color 0.3s;
  }
  .ts-card:hover { border-color: var(--border-bright); }

  .ts-icon-wrap {
    width: 40px; height: 40px;
    border-radius: var(--radius-sm);
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    position: relative;
  }

  .ts-icon-wrap.recv {
    background: linear-gradient(135deg, rgba(248,120,156,0.1), rgba(248,120,156,0.02));
    border: 1px solid rgba(248,120,156,0.12);
  }
  .ts-icon-wrap.serv {
    background: linear-gradient(135deg, rgba(245,166,35,0.1), rgba(245,166,35,0.02));
    border: 1px solid rgba(245,166,35,0.12);
  }

  .ts-info .ts-label {
    font-family: var(--mono);
    font-size: 9px;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: var(--text-muted);
    margin-bottom: 4px;
  }

  .ts-info .ts-val {
    font-family: var(--mono);
    font-size: 12px;
    font-weight: 600;
    color: var(--text);
  }

  .footer {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-top: 36px;
    padding-top: 24px;
    border-top: 1px solid var(--border);
    flex-wrap: wrap;
    gap: 12px;
  }

  .footer-l, .footer-r {
    font-family: var(--mono);
    font-size: 10px;
    color: var(--text-dim);
    letter-spacing: 0.5px;
  }

  .footer-r span { color: var(--rose-dim); }

  .particles {
    position: fixed;
    inset: 0;
    pointer-events: none;
    z-index: 0;
    overflow: hidden;
  }

  .particle {
    position: absolute;
    width: 2px;
    height: 2px;
    background: var(--rose);
    border-radius: 50%;
    opacity: 0;
    animation: particleFloat linear infinite;
  }

  @keyframes particleFloat {
    0% { opacity: 0; transform: translateY(100vh) scale(0); }
    10% { opacity: 0.6; }
    90% { opacity: 0.6; }
    100% { opacity: 0; transform: translateY(-10vh) scale(1); }
  }

  .token-actions {
    margin-bottom: 16px;
  }

  .action-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
  }

  .action-card {
    background: var(--surface);
    backdrop-filter: var(--glass);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 24px;
    position: relative;
    overflow: hidden;
  }

  .action-card .action-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 12px;
    border-radius: 20px;
    font-family: var(--mono);
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 14px;
  }

  .action-badge.single-badge {
    background: rgba(16, 185, 129, 0.08);
    color: var(--green);
    border: 1px solid rgba(16, 185, 129, 0.2);
  }

  .action-card h3 {
    font-size: 16px;
    font-weight: 700;
    margin-bottom: 6px;
    color: var(--text);
  }
  .action-card p {
    font-size: 12px;
    color: var(--text-muted);
    margin-bottom: 16px;
    line-height: 1.5;
  }

  .fetch-btn {
    width: 100%;
    padding: 12px;
    border: none;
    border-radius: var(--radius-sm);
    font-size: 13px;
    font-weight: 700;
    cursor: pointer;
    transition: all 0.3s;
    font-family: var(--font);
    color: white;
    position: relative;
  }

  .fetch-btn.btn-single {
    background: linear-gradient(135deg, #059669, #10b981);
  }
  .fetch-btn.btn-single:hover {
    box-shadow: 0 0 25px rgba(16, 185, 129, 0.4);
    transform: translateY(-1px);
  }

  .fetch-btn:active { transform: translateY(0); }
  .fetch-btn:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }
  .fetch-btn.loading { color: transparent; }
  .fetch-btn.loading::after {
    content: '';
    position: absolute;
    top: 50%; left: 50%;
    width: 20px; height: 20px;
    margin: -10px 0 0 -10px;
    border: 2px solid rgba(255,255,255,0.3);
    border-top-color: white;
    border-radius: 50%;
    animation: spin 0.6s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  .result-section {
    margin-bottom: 16px;
    display: none;
  }
  .result-section.visible { display: block; }

  .result-box {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    overflow: hidden;
  }

  .result-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 16px 20px;
    border-bottom: 1px solid var(--border);
  }
  .result-header h4 {
    font-size: 13px;
    font-weight: 700;
    color: var(--text);
  }
  .result-count {
    font-family: var(--mono);
    font-size: 11px;
    color: var(--rose);
    font-weight: 600;
  }

  .token-item {
    padding: 14px 20px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    transition: background 0.15s;
  }
  .token-item:last-child { border-bottom: none; }
  .token-item:hover { background: var(--surface-hover); }

  .token-index {
    font-family: var(--mono);
    font-size: 11px;
    color: var(--text-muted);
    min-width: 24px;
  }
  .token-value {
    flex: 1;
    font-family: var(--mono);
    font-size: 11px;
    color: var(--text);
    word-break: break-all;
    line-height: 1.5;
  }
  .token-copy {
    background: none;
    border: 1px solid var(--border);
    color: var(--text-muted);
    padding: 4px 10px;
    border-radius: var(--radius-xs);
    cursor: pointer;
    font-size: 10px;
    font-family: var(--mono);
    transition: all 0.2s;
    white-space: nowrap;
  }
  .token-copy:hover { border-color: var(--green); color: var(--green); }
  .token-copy.copied { border-color: var(--green); color: var(--green); }

  .copy-all-btn {
    padding: 10px 20px;
    border: 1px solid var(--border);
    background: rgba(248,120,156,0.06);
    color: var(--rose);
    border-radius: var(--radius-sm);
    font-family: var(--mono);
    font-size: 11px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.2s;
    margin: 12px 20px;
    display: inline-block;
  }
  .copy-all-btn:hover { border-color: var(--rose); background: rgba(248,120,156,0.12); }

  .error-box {
    margin: 16px 20px;
    background: rgba(248, 113, 113, 0.06);
    border: 1px solid rgba(248, 113, 113, 0.15);
    border-radius: var(--radius-sm);
    padding: 16px 20px;
    color: var(--red);
    font-size: 12px;
    font-family: var(--mono);
  }

  @media (max-width: 900px) {
    .grid-4 { grid-template-columns: repeat(2, 1fr); }
    .grid-2, .ts-strip, .action-grid { grid-template-columns: 1fr; }
    .wrap { padding: 24px 18px 40px; }
    .card-value { font-size: 32px; }
  }

  @media (max-width: 500px) {
    .grid-4 { grid-template-columns: 1fr; }
    .header { flex-direction: column; align-items: flex-start; }
    .header-right { width: 100%; justify-content: flex-start; }
  }

  .fade-up {
    animation: fadeUp 0.6s cubic-bezier(0.16, 1, 0.3, 1) both;
  }
  @keyframes fadeUp {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
  }

  .fade-up:nth-child(1) { animation-delay: 0.0s; }
  .fade-up:nth-child(2) { animation-delay: 0.05s; }
  .fade-up:nth-child(3) { animation-delay: 0.1s; }
  .fade-up:nth-child(4) { animation-delay: 0.15s; }
  .fade-up:nth-child(5) { animation-delay: 0.2s; }
  .fade-up:nth-child(6) { animation-delay: 0.25s; }
  .fade-up:nth-child(7) { animation-delay: 0.3s; }
  .fade-up:nth-child(8) { animation-delay: 0.35s; }
</style>
</head>
<body>

<div class="orb orb-1"></div>
<div class="orb orb-2"></div>
<div class="orb orb-3"></div>

<div class="particles" id="particles"></div>

<div class="wrap">

  <div class="header fade-up">
    <div class="header-left">
      <div class="logo-box">
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
          <path d="M12 2L4 6.5V12C4 16.5 7.5 20.7 12 22C16.5 20.7 20 16.5 20 12V6.5L12 2Z" stroke="#f8789c" stroke-width="1.5" stroke-linejoin="round" fill="none"/>
          <path d="M9 12L11 14L15 10" stroke="#fcadc4" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
      </div>
      <div class="header-title">
        <h1>CN31 Token Server</h1>
        <div class="sub">Fresh Tokens &middot; TTL <span>14min</span> &middot; Auto-refresh 3s</div>
      </div>
    </div>
    <div class="header-right">
      <div class="mode-badge">Auto-Gen</div>
      <div class="status-badge">
        <div class="live-dot"></div>
        Online
      </div>
      <div class="time-badge" id="clock"></div>
    </div>
  </div>

  <div class="grid-4">
    <div class="card amber fade-up">
      <div class="card-icon">
        <svg width="18" height="18" viewBox="0 0 18 18" fill="none"><rect x="2" y="8" width="3" height="8" rx="1" fill="#f5a623" opacity="0.5"/><rect x="7.5" y="4" width="3" height="12" rx="1" fill="#f5a623" opacity="0.7"/><rect x="13" y="1" width="3" height="15" rx="1" fill="#f5a623"/></svg>
      </div>
      <div class="card-label">Pool Size</div>
      <div class="card-value"><span class="highlight" id="v-queue">0</span></div>
      <div class="card-sub"><span class="dot"></span> tokens ready</div>
    </div>
    <div class="card rose fade-up">
      <div class="card-icon">
        <svg width="18" height="18" viewBox="0 0 18 18" fill="none"><path d="M9 2v10M5 8l4 4 4-4" stroke="#f8789c" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/><path d="M3 14h12" stroke="#f8789c" stroke-width="1.8" stroke-linecap="round"/></svg>
      </div>
      <div class="card-label">Generated</div>
      <div class="card-value"><span class="highlight" id="v-generated">0</span></div>
      <div class="card-sub"><span class="dot"></span> all time</div>
    </div>
    <div class="card coral fade-up">
      <div class="card-icon">
        <svg width="18" height="18" viewBox="0 0 18 18" fill="none"><path d="M9 16V6M5 10l4-4 4 4" stroke="#fb7a5c" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/><path d="M3 4h12" stroke="#fb7a5c" stroke-width="1.8" stroke-linecap="round"/></svg>
      </div>
      <div class="card-label">Served</div>
      <div class="card-value"><span class="highlight" id="v-served">0</span></div>
      <div class="card-sub"><span class="dot"></span> dispatched</div>
    </div>
    <div class="card red fade-up">
      <div class="card-icon">
        <svg width="18" height="18" viewBox="0 0 18 18" fill="none"><circle cx="9" cy="9" r="6" stroke="#f87171" stroke-width="1.5"/><path d="M9 6v4M9 12.5v.5" stroke="#f87171" stroke-width="1.8" stroke-linecap="round"/></svg>
      </div>
      <div class="card-label">Expired</div>
      <div class="card-value"><span class="highlight" id="v-expired">0</span></div>
      <div class="card-sub"><span class="dot"></span> auto-cleaned</div>
    </div>
  </div>

  <div class="grid-4">
    <div class="card violet fade-up">
      <div class="card-icon">
        <svg width="18" height="18" viewBox="0 0 18 18" fill="none"><path d="M4 4l10 10M14 4L4 14" stroke="#c77dff" stroke-width="1.8" stroke-linecap="round"/></svg>
      </div>
      <div class="card-label">Duplicates</div>
      <div class="card-value"><span class="highlight" id="v-dupes">0</span></div>
      <div class="card-sub"><span class="dot"></span> rejected</div>
    </div>
    <div class="card rose fade-up">
      <div class="card-icon">
        <svg width="18" height="18" viewBox="0 0 18 18" fill="none"><path d="M2 13l3-4 3 2 4-6 4 3" stroke="#f8789c" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>
      </div>
      <div class="card-label">Rate</div>
      <div class="card-value"><span class="highlight" id="v-rate">0</span></div>
      <div class="card-sub"><span class="dot"></span> tok / min</div>
    </div>
    <div class="card gold fade-up">
      <div class="card-icon">
        <svg width="18" height="18" viewBox="0 0 18 18" fill="none"><path d="M9 2l2.2 4.5 5 .7-3.6 3.5.8 5L9 13.5 4.6 15.7l.8-5L1.8 7.2l5-.7L9 2z" stroke="#fbbf24" stroke-width="1.3" stroke-linejoin="round" fill="rgba(251,191,36,0.15)"/></svg>
      </div>
      <div class="card-label">Peak Pool</div>
      <div class="card-value"><span class="highlight" id="v-peak">0</span></div>
      <div class="card-sub"><span class="dot"></span> historical max</div>
    </div>
    <div class="card green fade-up">
      <div class="card-icon">
        <svg width="18" height="18" viewBox="0 0 18 18" fill="none"><path d="M3 9a6 6 0 1 1 12 0A6 6 0 0 1 3 9z" stroke="#10b981" stroke-width="1.5"/><path d="M9 6v3l2 1" stroke="#10b981" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>
      </div>
      <div class="card-label">Workers</div>
      <div class="card-value"><span class="highlight" id="v-workers">0</span></div>
      <div class="card-sub"><span class="dot"></span> active now</div>
    </div>
  </div>

  <div class="section fade-up">
    <div class="section-hdr">
      <div class="icon">
        <svg width="14" height="14" viewBox="0 0 14 14" fill="none"><rect x="1" y="3" width="12" height="8" rx="2" stroke="#f8789c" stroke-width="1.2"/><path d="M4 7h6" stroke="#f8789c" stroke-width="1.2" stroke-linecap="round"/></svg>
      </div>
      <h2>Pool Capacity</h2>
      <div class="line"></div>
    </div>

    <div class="bar-group">
      <div class="bar-header">
        <span class="bar-title">Current Pool</span>
        <span class="bar-value" id="bv-queue">0 / 0 peak</span>
      </div>
      <div class="bar-track">
        <div class="bar-fill gradient-rose" id="bar-queue" style="width: 0%"></div>
      </div>
    </div>

    <div class="bar-group">
      <div class="bar-header">
        <span class="bar-title">Served vs Generated</span>
        <span class="bar-value" id="bv-served">0 / 0</span>
      </div>
      <div class="bar-track">
        <div class="bar-fill gradient-amber" id="bar-served" style="width: 0%"></div>
      </div>
    </div>

    <div class="bar-group">
      <div class="bar-header">
        <span class="bar-title">Token TTL Lifecycle</span>
        <span class="bar-value">14 min window</span>
      </div>
      <div class="bar-track">
        <div class="bar-fill gradient-violet" style="width: 100%"></div>
      </div>
    </div>
  </div>

  <div class="grid-2">

    <div class="section fade-up">
      <div class="section-hdr">
        <div class="icon">
          <svg width="14" height="14" viewBox="0 0 14 14" fill="none"><circle cx="7" cy="7" r="5.5" stroke="#f8789c" stroke-width="1.2"/><path d="M7 4v3.5l2.5 1.5" stroke="#f8789c" stroke-width="1.2" stroke-linecap="round" stroke-linejoin="round"/></svg>
        </div>
        <h2>Runtime</h2>
        <div class="line"></div>
      </div>

      <div class="runtime-wrap">
        <div class="ring-container">
          <svg width="100" height="100" viewBox="0 0 100 100">
            <circle cx="50" cy="50" r="40" fill="none" stroke="rgba(248,120,156,0.06)" stroke-width="6"/>
            <circle id="ring-arc" cx="50" cy="50" r="40" fill="none" stroke="url(#rg)" stroke-width="6"
              stroke-dasharray="251" stroke-dashoffset="251" stroke-linecap="round"/>
            <defs>
              <linearGradient id="rg" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stop-color="#f8789c"/>
                <stop offset="50%" stop-color="#f5a623"/>
                <stop offset="100%" stop-color="#c77dff"/>
              </linearGradient>
            </defs>
          </svg>
          <div class="ring-center">
            <div class="ring-number" id="v-ring-uptime">0</div>
            <div class="ring-unit">minutes</div>
          </div>
        </div>

        <div class="stat-pills">
          <div class="stat-pill">
            <span class="stat-pill-label"><span class="indicator" style="background:var(--rose)"></span>Uptime</span>
            <span class="stat-pill-value c-rose" id="v-uptime-pill">0 min</span>
          </div>
          <div class="stat-pill">
            <span class="stat-pill-label"><span class="indicator" style="background:var(--amber)"></span>Tok/min</span>
            <span class="stat-pill-value c-amber" id="v-rate-pill">0</span>
          </div>
          <div class="stat-pill">
            <span class="stat-pill-label"><span class="indicator" style="background:var(--coral)"></span>TTL</span>
            <span class="stat-pill-value c-coral">14 min</span>
          </div>
          <div class="stat-pill">
            <span class="stat-pill-label"><span class="indicator" style="background:var(--violet)"></span>Peak</span>
            <span class="stat-pill-value c-violet" id="v-peak-pill">0</span>
          </div>
        </div>
      </div>
    </div>

    <div class="section fade-up">
      <div class="section-hdr">
        <div class="icon">
          <svg width="14" height="14" viewBox="0 0 14 14" fill="none"><path d="M2 4h10M2 7h10M2 10h7" stroke="#f8789c" stroke-width="1.2" stroke-linecap="round"/></svg>
        </div>
        <h2>API Endpoints</h2>
        <div class="line"></div>
      </div>

      <div class="endpoint-table">
        <div class="ep-row" onclick="window.open('/get-token','_blank')">
          <span class="ep-method get">GET</span>
          <span class="ep-path">/get-token</span>
          <span class="ep-desc">grab 1 fresh token</span>
        </div>
        <div class="ep-row" onclick="window.open('/stats','_blank')">
          <span class="ep-method get">GET</span>
          <span class="ep-path">/stats</span>
          <span class="ep-desc">pool statistics</span>
        </div>
        <div class="ep-row" onclick="window.open('/health','_blank')">
          <span class="ep-method get">GET</span>
          <span class="ep-path">/health</span>
          <span class="ep-desc">health check</span>
        </div>
      </div>
    </div>
  </div>

  <div class="token-actions fade-up">
    <div class="action-grid">
      <div class="action-card">
        <div class="action-badge single-badge">Single</div>
        <h3>Get Token</h3>
        <p>Get one fresh CN31 token. Returns only unused tokens under 14 minutes old.</p>
        <button class="fetch-btn btn-single" id="btn-single" onclick="fetchSingle()">Get Token</button>
      </div>
    </div>
  </div>

  <div class="result-section" id="result-section">
    <div class="result-box">
      <div class="result-header">
        <h4>Tokens</h4>
        <span class="result-count" id="result-count"></span>
      </div>
      <div id="result-container"></div>
    </div>
  </div>

  <div class="ts-strip fade-up">
    <div class="ts-card">
      <div class="ts-icon-wrap recv">
        <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
          <path d="M8 2v8M5 7l3 3 3-3" stroke="#f8789c" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
          <path d="M3 13h10" stroke="#f8789c" stroke-width="1.5" stroke-linecap="round"/>
        </svg>
      </div>
      <div class="ts-info">
        <div class="ts-label">Last Generated</div>
        <div class="ts-val" id="v-last-recv">---</div>
      </div>
    </div>
    <div class="ts-card">
      <div class="ts-icon-wrap serv">
        <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
          <path d="M8 14V6M5 9l3-3 3 3" stroke="#f5a623" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
          <path d="M3 3h10" stroke="#f5a623" stroke-width="1.5" stroke-linecap="round"/>
        </svg>
      </div>
      <div class="ts-info">
        <div class="ts-label">Last Served</div>
        <div class="ts-val" id="v-last-serv">---</div>
      </div>
    </div>
  </div>

  <div class="footer fade-up">
    <div class="footer-l">CN31 TOKEN SERVER &middot; FRESH TOKENS ONLY &middot; AUTO-REFRESH 3S</div>
    <div class="footer-r">TTL <span>14 MIN</span> &middot; AUTO-GEN</div>
  </div>
</div>

<script>
  function updateClock() {
    const now = new Date();
    const h = String(now.getHours()).padStart(2,'0');
    const m = String(now.getMinutes()).padStart(2,'0');
    const s = String(now.getSeconds()).padStart(2,'0');
    const el = document.getElementById('clock');
    if (el) el.textContent = h + ':' + m + ':' + s;
  }
  updateClock();
  setInterval(updateClock, 1000);

  function setTxt(id, val) {
    const el = document.getElementById(id);
    if (el && el.textContent !== String(val)) el.textContent = val;
  }

  let lastTokens = [];

  function doCopy(text, btn) {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(() => flashBtn(btn)).catch(() => { fallbackCopy(text); flashBtn(btn); });
    } else { fallbackCopy(text); flashBtn(btn); }
  }
  function fallbackCopy(text) {
    const ta = document.createElement('textarea'); ta.value = text;
    ta.style.position = 'fixed'; ta.style.left = '-9999px';
    document.body.appendChild(ta); ta.select(); document.execCommand('copy');
    document.body.removeChild(ta);
  }
  function flashBtn(btn) {
    const orig = btn.textContent; btn.textContent = 'Copied!'; btn.classList.add('copied');
    setTimeout(() => { btn.textContent = orig; btn.classList.remove('copied'); }, 1500);
  }
  function copyTokenByIndex(idx, btn) { if (lastTokens[idx]) doCopy(lastTokens[idx], btn); }
  function copyAllTokens(btn) {
    if (lastTokens.length > 0) { doCopy(lastTokens.join('\n'), btn); }
  }

  function showResult(tokens) {
    lastTokens = tokens;
    const section = document.getElementById('result-section');
    const container = document.getElementById('result-container');
    const count = document.getElementById('result-count');
    section.classList.add('visible');
    count.textContent = tokens.length + ' token' + (tokens.length > 1 ? 's' : '');
    let html = '';
    if (tokens.length > 1) {
      html += '<div style="padding:12px 20px;"><button class="copy-all-btn" onclick="copyAllTokens(this)">Copy All Tokens</button></div>';
    }
    tokens.forEach((token, i) => {
      const short = token.length > 80 ? token.substring(0, 80) + '...' : token;
      html += '<div class="token-item"><span class="token-index">#' + (i+1) + '</span><span class="token-value">' + short + '</span><button class="token-copy" onclick="copyTokenByIndex(' + i + ', this)">Copy</button></div>';
    });
    container.innerHTML = html;
  }

  function showError(msg) {
    const section = document.getElementById('result-section');
    const container = document.getElementById('result-container');
    const count = document.getElementById('result-count');
    section.classList.add('visible');
    count.textContent = 'Error';
    container.innerHTML = '<div class="error-box">' + msg + '</div>';
  }

  function setLoading(btnId, loading) {
    const btn = document.getElementById(btnId);
    btn.disabled = loading;
    if (loading) btn.classList.add('loading');
    else btn.classList.remove('loading');
  }

  async function fetchSingle() {
    setLoading('btn-single', true);
    try {
      const resp = await fetch('/get-token');
      const data = await resp.json();
      if (resp.ok && data.token) { showResult([data.token]); }
      else { showError(data.error || 'No tokens available'); }
    } catch (e) { showError('Connection error: ' + e.message); }
    setLoading('btn-single', false);
    refreshStats();
  }


  async function refreshStats() {
    try {
      const r = await fetch('/stats');
      const d = await r.json();
      const q = d.pool_size || 0;
      const pk = d.peak_queue || 0;
      const gen = d.tokens_generated || 0;
      const srv = d.tokens_served || 0;

      setTxt('v-queue', q);
      setTxt('v-generated', gen);
      setTxt('v-served', srv);
      setTxt('v-expired', d.tokens_expired || 0);
      setTxt('v-dupes', d.duplicates_rejected || 0);
      setTxt('v-workers', d.active_workers || 0);
      setTxt('v-peak', pk);

      // Calculate rate
      if (d.start_time) {
        const start = new Date(d.start_time);
        const now = new Date();
        const diffMin = Math.max((now - start) / 60000, 1);
        const rate = (gen / diffMin).toFixed(1);
        setTxt('v-rate', rate);
        setTxt('v-rate-pill', rate);

        const um = diffMin.toFixed(1);
        setTxt('v-ring-uptime', um);
        setTxt('v-uptime-pill', um + ' min');

        const ringOff = Math.round(251 - (251 * Math.min(diffMin / Math.max(diffMin + 60, 1), 1)));
        const arc = document.getElementById('ring-arc');
        if (arc) arc.setAttribute('stroke-dashoffset', ringOff);
      }

      setTxt('v-peak-pill', pk);

      const qPct = pk > 0 ? Math.min(Math.round(q / pk * 100), 100) : 0;
      const sPct = gen > 0 ? Math.min(Math.round(srv / gen * 100), 100) : 0;
      const bq = document.getElementById('bar-queue');
      const bs = document.getElementById('bar-served');
      if (bq) bq.style.width = qPct + '%';
      if (bs) bs.style.width = sPct + '%';
      setTxt('bv-queue', q + ' / ' + pk + ' peak');
      setTxt('bv-served', srv + ' / ' + gen);

      setTxt('v-last-recv', d.last_received || '---');
      setTxt('v-last-serv', d.last_served || '---');
    } catch(e) {}
  }

  refreshStats();
  setInterval(refreshStats, 3000);

  (function() {
    const container = document.getElementById('particles');
    if (!container) return;
    const colors = ['#f8789c','#f5a623','#c77dff','#fb7a5c'];
    for (let i = 0; i < 30; i++) {
      const p = document.createElement('div');
      p.className = 'particle';
      p.style.left = Math.random() * 100 + '%';
      p.style.width = p.style.height = (1 + Math.random() * 2) + 'px';
      p.style.background = colors[Math.floor(Math.random() * colors.length)];
      p.style.animationDuration = (8 + Math.random() * 15) + 's';
      p.style.animationDelay = (Math.random() * 10) + 's';
      container.appendChild(p);
    }
  })();
</script>

</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(FRONTEND_HTML)


@app.route("/get-token", methods=["GET"])
def get_token_endpoint():
    with stats_lock:
        stats["total_requests"] += 1
        stats["single_requests"] += 1

    token_entry = take_token()
    if token_entry:
        age_seconds = int(time.time() - token_entry["generated_at"])
        return jsonify({
            "status": "success",
            "token": token_entry["token"],
            "age_seconds": age_seconds,
            "expires_in_seconds": TOKEN_EXPIRY_SECONDS - age_seconds,
            "generated_at": datetime.fromtimestamp(token_entry["generated_at"], tz=timezone.utc).isoformat(),
        })

    return jsonify({
        "status": "error",
        "error": "No fresh tokens available. Workers are generating, try again shortly.",
        "pool_size": len(token_pool),
    }), 503



@app.route("/stats", methods=["GET"])
def stats_endpoint():
    prune_expired()
    with stats_lock:
        stats["pool_size"] = len(token_pool)
        return jsonify(stats)


@app.route("/save-token", methods=["POST"])
def save_token():
    data = request.json or {}
    token_value = data.get("token", "")
    source = data.get("source", "external")
    if not token_value:
        return jsonify({"error": "token required"}), 400
    entry = {
        "token": token_value,
        "generated_at": time.time(),
        "worker": source,
    }
    with pool_lock:
        already = any(t["token"] == token_value for t in token_pool)
        if already:
            return jsonify({"error": "duplicate"}), 409
        token_pool.append(entry)
    with stats_lock:
        stats["tokens_generated"] += 1
        stats["pool_size"] = len(token_pool)
        stats["last_received"] = datetime.now(timezone.utc).isoformat()
        if stats["pool_size"] > stats["peak_queue"]:
            stats["peak_queue"] = stats["pool_size"]
    return jsonify({"status": "saved", "pool_size": len(token_pool)}), 201

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "pool_size": len(token_pool)})


start_pool()

pruner_thread = threading.Thread(target=pruner_loop, daemon=True)
pruner_thread.start()

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=API_PORT,
        debug=False,
        threaded=True
    )