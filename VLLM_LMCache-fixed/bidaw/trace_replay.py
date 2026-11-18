#!/usr/bin/env python3
"""Length-driven synthetic replay of Bidaw's timestamped trace.

Prompts are exact, append-only token-ID prefixes, frozen by SHA-256 seeds.
Synthetic historical answers are filler, NOT the model's actual answers; this
is a cache/I/O microbenchmark, not a semantic conversation or full paper repro.
Generation uses ignore_eos and reports the actual completion-token usage.
Each user has a task sharing one epoch; --parallel limits active HTTP calls,
not active sessions. A dependent round is sent only after its predecessor.
Client queue/dependency delays are recorded separately from server latency.
Warm-up consumes an initial continuous interval of this SAME pass; it never
replays the measured future. All arms must use the same interval.
"""
from __future__ import annotations
import argparse
import asyncio
import hashlib
import json
import math
import random
import re
import time
from pathlib import Path
from typing import Any

DEMO_DIR = Path(__file__).resolve().parents[1]
MODEL_NAME = 'qwen2.5-1.5b-instruct'
METRIC_KEYS = (
    'lmcache_mp_num_finished_stores_total',
    'lmcache_mp_num_finished_retrieves_total',
    'lmcache_mp_lookup_hit_l1_tokens_total',
    'lmcache_mp_lookup_hit_l2_tokens_total',
    'lmcache_mp_lookup_requested_tokens_total',
    'lmcache_mp_lookup_hit_tokens_total',
)


def stable_seed(seed, uid):
    return int.from_bytes(hashlib.sha256(f'{seed}:{uid}'.encode()).digest()[:8], 'big')


def load_trace(trace_dir):
    files = sorted(Path(trace_dir).glob('total_traces_part*.txt'))
    if not files:
        files = sorted((Path(trace_dir).parent / 'trace_frozen').glob('total_traces_part*.txt'))
    if not files:
        raise ValueError(f'No trace files: {trace_dir}')
    users = {}
    for file in files:
        for line in file.read_text().splitlines():
            fields = line.split()
            if len(fields) != 5:
                continue
            uid, *nums = fields
            try:
                ts, qlen, rlen, rnd = map(int, nums)
            except ValueError:
                continue  # header
            if qlen < 0 or rlen < 1:
                raise ValueError(f'Invalid lengths: {file}: {line}')
            users.setdefault(uid, []).append(dict(ts=ts, qlen=qlen, rlen=rlen, rnd=rnd))
    for rounds in users.values():
        rounds.sort(key=lambda r: (r['ts'], r['rnd']))
    return users


def select_users(users, n_users, seed=0):
    ordered = sorted(users, key=lambda u: (users[u][0]['ts'], u))
    return ordered[:n_users] if n_users > 0 else ordered


class PromptBuilder:
    """Exact token IDs; no text decode/encode round trip or Python hash()."""
    def __init__(self, tokenizer, uid, seed=0):
        self.rng = random.Random(stable_seed(seed, uid))
        words = 'memory storage request cache tensor disk attention model user history'
        self.pool = tokenizer.encode(words, add_special_tokens=False)
        if not self.pool:
            raise ValueError('Empty token pool')
        self.tokens = []
        self.history_len = 0

    def build_prompt(self, query_tokens, round_idx, max_prompt_tokens):
        length = self.history_len + query_tokens
        if length < 1 or length > max_prompt_tokens:
            return [], length, True
        while len(self.tokens) < length:
            self.tokens.append(self.rng.choice(self.pool))
        return list(self.tokens[:length]), length, False

    def advance(self, qlen, rlen):
        self.history_len += qlen + rlen
        while len(self.tokens) < self.history_len:
            self.tokens.append(self.rng.choice(self.pool))


def build_schedule(users, picked, tokenizer, args):
    t0 = min(users[u][0]['ts'] for u in picked)
    schedules, digest = {}, hashlib.sha256()
    truncated = []
    for uid in picked:
        b = PromptBuilder(tokenizer, uid, args['seed'])
        rows = []
        for r in users[uid]:
            prompt, length, stop = b.build_prompt(r['qlen'], r['rnd'], args['max_prompt_tokens'])
            if stop or length + r['rlen'] > args['max_model_len']:
                truncated.append(dict(user=uid, round=r['rnd'], reason='context_limit'))
                break
            row = dict(r, prompt=prompt, due=(r['ts'] - t0) / args['speed'])
            rows.append(row)
            digest.update(json.dumps([uid, row], sort_keys=True, separators=(',', ':')).encode())
            b.advance(r['qlen'], r['rlen'])
        schedules[uid] = rows
    return schedules, digest.hexdigest(), truncated


def stream_ttft(base, prompt, salt, max_tokens, model):
    import requests
    payload = dict(model=model, prompt=prompt, max_tokens=max_tokens,
                   temperature=0.0, ignore_eos=True, stream=True,
                   stream_options={'include_usage': True}, cache_salt=salt)
    started, first, usage = time.perf_counter(), None, None
    chunks = []
    with requests.post(f'{base}/v1/completions', json=payload,
                       stream=True, timeout=(10, 600)) as resp:
        resp.raise_for_status()
        for raw in resp.iter_lines(decode_unicode=True):
            if not raw or not raw.startswith('data:'):
                continue
            data = raw[5:].strip()
            if data == '[DONE]':
                break
            item = json.loads(data)
            if 'error' in item:
                raise RuntimeError(str(item['error']))
            if item.get('usage'):
                usage = item['usage']
            for choice in item.get('choices', []):
                text = choice.get('text') or ''
                if text and first is None:
                    first = time.perf_counter()
                chunks.append(text)
    ended = time.perf_counter()
    if not usage or usage.get('completion_tokens') != max_tokens:
        raise RuntimeError(f'Expected {max_tokens} completion tokens, got usage={usage}')
    if usage.get('prompt_tokens') != len(prompt):
        raise RuntimeError(f'Prompt length mismatch: {len(prompt)} vs usage={usage}')
    return dict(text=''.join(chunks), ttft_s=(first or ended) - started,
                total_s=ended - started, completion_tokens=usage['completion_tokens'])


def report_answer(lmc, salt, answer_tokens):
    import requests
    response = requests.post(f'{lmc}/bidaw/answer',
        json=dict(cache_salt=salt, answer_tokens=answer_tokens), timeout=10)
    response.raise_for_status()
    return response.json().get('policies_updated', 0)


def metrics_snapshot(lmc):
    import requests
    response = requests.get(f'{lmc}/metrics', timeout=15)
    response.raise_for_status()
    totals = {}
    for line in response.text.splitlines():
        match = re.match(r'^(\w+)(?:\{[^}]*\})?\s+([0-9.eE+-]+)$', line)
        if match and match[1] in METRIC_KEYS:
            totals[match[1]] = totals.get(match[1], 0.0) + float(match[2])
    return totals


def percentile(values, p):
    return sorted(values)[min(len(values) - 1, math.ceil(len(values) * p) - 1)]


async def execute_schedule(schedules, args, sender=stream_ttft, reporter=report_answer,
                           snapshot=metrics_snapshot):
    # asyncio.to_thread uses the default executor; capacity covers HTTP + metrics.
    from concurrent.futures import ThreadPoolExecutor
    loop = asyncio.get_running_loop()
    executor = ThreadPoolExecutor(max_workers=args['parallel'] + 4)
    loop.set_default_executor(executor)
    sem = asyncio.Semaphore(args['parallel'])
    epoch = time.perf_counter()
    warmup = args['warmup_seconds']
    records = []

    async def boundary():
        # Stop admission at the boundary, let warm-up operations and answer
        # reports finish, then snapshot. Measured requests wait for this event.
        await asyncio.sleep(max(0, epoch + warmup - time.perf_counter()))
        for _ in range(args['parallel']):
            await sem.acquire()
        try:
            return {} if args.get('no_cache') else await asyncio.to_thread(snapshot, args['lmc'])
        finally:
            for _ in range(args['parallel']):
                sem.release()

    boundary_task = asyncio.create_task(boundary())

    async def replay_user(uid, rows):
        salt = f"{args['salt_prefix']}-{uid}"
        previous_failed = False
        for row in rows:
            phase = 'warmup' if row['due'] < warmup else 'measure'
            record = dict(user=salt, round=row['rnd'], ts=row['ts'],
                          qlen=row['qlen'], rlen=row['rlen'],
                          prompt_tokens=len(row['prompt']), phase=phase,
                          scheduled_s=row['due'])
            if previous_failed:
                record['error'] = 'previous_round_failed'
                records.append(record)
                continue
            await asyncio.sleep(max(0, epoch + row['due'] - time.perf_counter()))
            if phase == 'measure':
                await boundary_task
            # Warm-up requests that fall behind the boundary are skipped,
            # never contaminate metrics.  They do NOT break the chain: the
            # schedule builds prompts from deterministic filler tokens and
            # never back-fills model answers (FIXES.md #9/#96-97), so a
            # skipped round does not change any later round's prompt bytes.
            # The chain still breaks on real request failures below.
            elif time.perf_counter() >= epoch + warmup:
                record['error'] = 'warmup_overrun'
                records.append(record)
                continue
            async with sem:
                if phase == 'warmup' and time.perf_counter() >= epoch + warmup:
                    record['error'] = 'warmup_overrun'
                    records.append(record)
                    continue
                record['sent_s'] = time.perf_counter() - epoch
                record['scheduling_lag_ms'] = 1000 * (record['sent_s'] - row['due'])
                try:
                    result = await asyncio.to_thread(sender, args['vllm'], row['prompt'],
                        salt, row['rlen'], args['model'])
                    # Capture network completion before the reporting HTTP call.
                    record['completed_s'] = time.perf_counter() - epoch
                    record.update(ttft_ms=result['ttft_s'] * 1000,
                                  total_ms=result['total_s'] * 1000,
                                  offered_latency_ms=1000 * (record['completed_s'] - row['due']),
                                  actual_response_tokens=result['completion_tokens'])
                    if not args.get('no_cache'):
                        updated = await asyncio.to_thread(reporter, args['lmc'], salt,
                                                         result['completion_tokens'])
                        record['answer_policies_updated'] = updated
                        if args.get('require_answer_policy') and updated < 1:
                            raise RuntimeError('No Bidaw policy received the answer signal')
                except Exception as exc:
                    record['error'] = str(exc)
                    previous_failed = True
            records.append(record)

    tasks = [asyncio.create_task(replay_user(uid, rows)) for uid, rows in schedules.items()]
    try:
        await asyncio.gather(*tasks)
        before = await boundary_task
        after = {} if args.get('no_cache') else await asyncio.to_thread(snapshot, args['lmc'])
    finally:
        for task in tasks + [boundary_task]:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, boundary_task, return_exceptions=True)
        executor.shutdown(wait=True)
    return records, before, after, time.perf_counter() - epoch


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--trace-dir', default=str(DEMO_DIR / 'bidaw/trace_frozen'))
    ap.add_argument('--users', type=int, default=10)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--speed', type=float, default=60)
    ap.add_argument('--parallel', type=int, default=64)
    ap.add_argument('--max-prompt-tokens', type=int, default=5600)
    ap.add_argument('--max-model-len', type=int, default=6144)
    ap.add_argument('--warmup-seconds', type=float, default=0,
                    help='Initial replay-wall seconds excluded from measurement')
    ap.add_argument('--warmup-pass', action='store_true',
                    help='Deprecated; use --warmup-seconds instead')
    ap.add_argument('--vllm', default='http://127.0.0.1:8000')
    ap.add_argument('--lmc', default='http://127.0.0.1:8080')
    ap.add_argument('--model', default=MODEL_NAME)
    ap.add_argument('--model-dir', default=str(DEMO_DIR / 'models/Qwen2.5-1.5B-Instruct'))
    ap.add_argument('--salt-prefix', default='trace')
    ap.add_argument('--label', default='smoke')
    ap.add_argument('--out', default='')
    ap.add_argument('--no-cache', action='store_true')
    ap.add_argument('--require-answer-policy', action='store_true')
    ap.add_argument('--dry-run', action='store_true')
    args = vars(ap.parse_args())
    if args['warmup_pass']:
        ap.error('--warmup-pass reuses future data; replace with --warmup-seconds 30')
    if args['speed'] <= 0 or args['parallel'] < 1 or args['warmup_seconds'] < 0:
        ap.error('speed/parallel must be positive; warmup-seconds must be nonnegative')
    users = load_trace(args['trace_dir'])
    picked = select_users(users, args['users'], args['seed'])
    if not picked:
        ap.error('No users selected')
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args['model_dir'])
    schedules, digest, truncated = build_schedule(users, picked, tok, args)
    due_times = [r['due'] for rows in schedules.values() for r in rows]
    if not due_times or max(due_times) <= args['warmup_seconds']:
        ap.error('No measured interval after warm-up')
    # Use one fixed planned observation interval, including session drain.
    duration = max(due_times) - args['warmup_seconds']
    arrivals = sum(bool(rows) and rows[0]['due'] >= args['warmup_seconds']
                   for rows in schedules.values())
    rate = arrivals * 60 / duration
    print(f'[schedule] users={len(picked)} rounds={len(due_times)} '
          f'measured_new_users={arrivals} planned_arrival_rate={rate:.3f}/min '
          f'sha256={digest} truncated_sessions={len(truncated)}')
    if args['dry_run']:
        return 0
    import requests
    urls = [f"{args['vllm']}/v1/models"]
    if not args['no_cache']:
        urls.append(f"{args['lmc']}/healthcheck")
    for url in urls:
        requests.get(url, timeout=10).raise_for_status()
    records, before, after, wall = asyncio.run(execute_schedule(schedules, args))
    delta = {k: after.get(k, 0) - before.get(k, 0) for k in set(before) | set(after)}
    measured = [r for r in records if r['phase'] == 'measure' and 'error' not in r]
    errors = [r for r in records if 'error' in r]
    summary = dict(label=args['label'], args=args, content_mode='synthetic_token_prefix',
                   schedule_sha256=digest, planned_arrival_rate=rate,
                   measured_interval_s=duration, wall_s=wall,
                   truncated_sessions=truncated, metrics_before=before,
                   metrics_after=after, metrics_delta=delta, records=records)
    if measured:
        for field in ('total_ms', 'ttft_ms', 'offered_latency_ms', 'scheduling_lag_ms'):
            values = [r[field] for r in measured]
            print(f'{field}: mean={sum(values)/len(values):.2f} '
                  f'p99={percentile(values,.99):.2f}')
    print(f'[summary] measured_ok={len(measured)} errors={len(errors)} wall={wall:.2f}s')
    out = Path(args['out'] or DEMO_DIR / 'logs' / f"trace_replay_{args['label']}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    # Exit non-zero only for REAL failures.  'warmup_overrun' skips are
    # expected once a system saturates (the offered schedule outruns a
    # no-cache baseline); they are recorded but never measured, so they
    # must not fail the sweep's fail-fast pipeline.
    fatal = [r for r in records
             if r.get('error') not in (None, 'warmup_overrun')]
    return 1 if fatal or not measured else 0


if __name__ == '__main__':
    raise SystemExit(main())
