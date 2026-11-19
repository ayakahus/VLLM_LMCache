#!/usr/bin/env python3
"""Replay Bidaw's timestamped trace with generated answers in the history.

Query token IDs are generated from stable per-user seeds. Each round appends
the model's returned token IDs verbatim before the next query.
Generation uses ignore_eos and validates completion lengths against the trace.
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
from uuid import uuid4

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


def build_schedule(users, picked, tokenizer, args):
    words = 'memory storage request cache tensor disk attention model user history'
    pool = tokenizer.encode(words, add_special_tokens=False)
    if not pool:
        raise ValueError('Empty token pool')
    prefix_size = args.get('shared_prefix_tokens', 0)
    prefix_rng = random.Random(args['seed'])
    shared_prefix = [prefix_rng.choice(pool) for _ in range(prefix_size)]
    t0 = min(users[u][0]['ts'] for u in picked)
    schedules, digest = {}, hashlib.sha256()
    truncated = []
    for uid in picked:
        rng = random.Random(stable_seed(args['seed'], uid))
        history_len = 0
        rows = []
        for r in users[uid]:
            prefix = shared_prefix if not rows else []
            prompt_len = history_len + len(prefix) + r['qlen']
            if (not 1 <= prompt_len <= args['max_prompt_tokens']
                    or prompt_len + r['rlen'] > args['max_model_len']):
                truncated.append(dict(user=uid, round=r['rnd'], reason='context_limit'))
                break
            query_ids = prefix + [rng.choice(pool) for _ in range(r['qlen'])]
            row = dict(r, query_token_ids=query_ids, prompt_tokens=prompt_len,
                       due=(r['ts'] - t0) / args['speed'])
            rows.append(row)
            digest.update(json.dumps([uid, row], sort_keys=True, separators=(',', ':')).encode())
            history_len = prompt_len + r['rlen']
        schedules[uid] = rows
    return schedules, digest.hexdigest(), truncated


def stream_ttft(base, prompt, salt, max_tokens, model, *, round_id=None, history_tokens=None,
                session_id=None):
    import requests
    payload = dict(model=model, prompt=prompt, max_tokens=max_tokens,
                   temperature=0.0, ignore_eos=True, stream=True,
                   add_special_tokens=False, return_token_ids=True,
                   stream_options={'include_usage': True})
    if salt:
        payload['cache_salt'] = salt
    # LMCache forwards lmcache.* request metadata apart from cache identity.
    if history_tokens is not None:
        payload.setdefault('kv_transfer_params', {})['lmcache.bidaw_history_tokens'] = history_tokens
    if round_id is not None:
        payload.setdefault('kv_transfer_params', {})['lmcache.bidaw_round_id'] = round_id
    if session_id is not None:
        payload.setdefault('kv_transfer_params', {})['lmcache.bidaw_session_id'] = session_id
    started, first, usage = time.perf_counter(), None, None
    chunks = []
    completion_ids = []
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
                echoed_prompt = choice.get('prompt_token_ids')
                if echoed_prompt is not None and echoed_prompt != prompt:
                    raise RuntimeError('vLLM returned different prompt token IDs')
                delta_ids = choice.get('token_ids')
                if delta_ids is None:
                    if choice.get('text'):
                        raise RuntimeError('vLLM stream is missing completion token IDs')
                    continue
                if not isinstance(delta_ids, list) or any(
                    type(token_id) is not int or token_id < 0 for token_id in delta_ids
                ):
                    raise RuntimeError('vLLM returned invalid completion token IDs')
                if delta_ids and first is None:
                    first = time.perf_counter()
                completion_ids.extend(delta_ids)
                chunks.append(choice.get('text') or '')
    ended = time.perf_counter()
    if not usage or usage.get('completion_tokens') != max_tokens:
        raise RuntimeError(f'Expected {max_tokens} completion tokens, got usage={usage}')
    if usage.get('prompt_tokens') != len(prompt):
        raise RuntimeError(f'Prompt length mismatch: {len(prompt)} vs usage={usage}')
    if len(completion_ids) != max_tokens:
        raise RuntimeError(
            f'Expected {max_tokens} completion token IDs, received {len(completion_ids)}'
        )
    return dict(text=''.join(chunks), ttft_s=(first or ended) - started,
                total_s=ended - started, completion_tokens=usage['completion_tokens'],
                completion_token_ids=completion_ids)


def report_answer(lmc, salt, answer_tokens):
    import requests
    response = requests.post(f'{lmc}/bidaw/answer',
        json=dict(cache_salt=salt, answer_tokens=answer_tokens), timeout=10)
    response.raise_for_status()
    return response.json().get('policies_updated', 0)


def report_fate_round(lmc, session, round_index, phase, round_id, *, shared_cache=False):
    import requests
    response = requests.post(f'{lmc}/bidaw/fate/round',
        json=dict(cache_salt=session, round_index=round_index, phase=phase, round_id=round_id,
                  shared_cache=shared_cache),
        timeout=10)
    response.raise_for_status()


def report_shared_replay_end(lmc):
    import requests
    requests.post(f'{lmc}/bidaw/fate/shared-replay-end', timeout=10).raise_for_status()


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
        session = f"{args['salt_prefix']}-{uid}"
        shared_cache = args.get('shared_cache', False)
        salt = '' if shared_cache else session
        history = []
        previous_failed = False
        warmup_skipped = False
        track_fates = args.get('track_chunk_fates') and not args.get('no_cache')
        complete = bool(rows) and uid not in args.get('truncated_fate_users', ())
        for row in rows:
            phase = 'warmup' if row['due'] < warmup else 'measure'
            record = dict(user=session, round=row['rnd'], ts=row['ts'],
                          qlen=row['qlen'], rlen=row['rlen'],
                          prompt_tokens=row['prompt_tokens'], phase=phase,
                          scheduled_s=row['due'])
            if previous_failed:
                record['error'] = 'previous_round_failed'
                records.append(record)
                continue
            if warmup_skipped:
                record['skip_reason'] = 'missing_warmup_history'
                records.append(record)
                continue
            await asyncio.sleep(max(0, epoch + row['due'] - time.perf_counter()))
            if phase == 'measure':
                await boundary_task
            async with sem:
                if phase == 'warmup' and time.perf_counter() >= epoch + warmup:
                    # Without this reply, later rounds have no exact prefix.
                    warmup_skipped = True
                    record['skip_reason'] = 'warmup_overrun'
                    records.append(record)
                    continue
                try:
                    sender_options = {}
                    if not args.get('no_cache'):
                        sender_options['history_tokens'] = len(history)
                    if shared_cache:
                        sender_options['session_id'] = session
                    if track_fates:
                        round_id = uuid4().hex
                        sender_options['round_id'] = round_id
                        await asyncio.to_thread(report_fate_round, args['lmc'], session,
                                                row['rnd'], 'start', round_id,
                                                shared_cache=shared_cache)
                    prompt = history + row['query_token_ids']
                    record['sent_s'] = time.perf_counter() - epoch
                    record['scheduling_lag_ms'] = 1000 * (record['sent_s'] - row['due'])
                    result = await asyncio.to_thread(sender, args['vllm'], prompt,
                        salt, row['rlen'], args['model'], **sender_options)
                    # Capture network completion before the reporting HTTP call.
                    record['completed_s'] = time.perf_counter() - epoch
                    history = prompt + result['completion_token_ids']
                    record.update(ttft_ms=result['ttft_s'] * 1000,
                                  total_ms=result['total_s'] * 1000,
                                  offered_latency_ms=1000 * (record['completed_s'] - row['due']),
                                  actual_response_tokens=result['completion_tokens'])
                    if not args.get('no_cache'):
                        updated = await asyncio.to_thread(reporter, args['lmc'], session,
                                                         result['completion_tokens'])
                        record['answer_policies_updated'] = updated
                        if args.get('require_answer_policy') and updated < 1:
                            raise RuntimeError('No Bidaw policy received the answer signal')
                    if track_fates:
                        await asyncio.to_thread(report_fate_round, args['lmc'], session,
                                                row['rnd'], 'finish', round_id,
                                                shared_cache=shared_cache)
                        if complete and row is rows[-1]:
                            await asyncio.to_thread(report_fate_round, args['lmc'], session,
                                                    row['rnd'], 'session_end', round_id,
                                                    shared_cache=shared_cache)
                    record['prompt_sha256'] = hashlib.sha256(
                        json.dumps(prompt, separators=(',', ':')).encode()
                    ).hexdigest()
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
    ap.add_argument('--shared-cache', action='store_true',
                    help='Allow identical token prefixes to reuse KV across replay sessions')
    ap.add_argument('--shared-prefix-tokens', type=int,
                    help='Common synthetic prefix length (default: 256 with --shared-cache)')
    ap.add_argument('--label', default='smoke')
    ap.add_argument('--out', default='')
    ap.add_argument('--no-cache', action='store_true')
    ap.add_argument('--require-answer-policy', action='store_true')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--track-chunk-fates', action='store_true',
                    help='Send round markers to an LMCache server with BIDAW_L2_EVENT_LOG enabled')
    args = vars(ap.parse_args())
    if args['shared_prefix_tokens'] is None:
        args['shared_prefix_tokens'] = 256 if args['shared_cache'] else 0
    if args['shared_prefix_tokens'] < 0 or (args['shared_prefix_tokens'] and not args['shared_cache']):
        ap.error('--shared-prefix-tokens requires --shared-cache and a nonnegative length')
    if args['shared_cache'] and args['no_cache']:
        ap.error('--shared-cache requires LMCache')
    if args['track_chunk_fates'] and args['no_cache']:
        ap.error('--track-chunk-fates requires LMCache')
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
    if args['track_chunk_fates']:
        args['truncated_fate_users'] = [entry['user'] for entry in truncated]
    records, before, after, wall = asyncio.run(execute_schedule(schedules, args))
    delta = {k: after.get(k, 0) - before.get(k, 0) for k in set(before) | set(after)}
    executed_prompts = sorted(
        (r['user'], r['round'], r['prompt_sha256'])
        for r in records if 'prompt_sha256' in r
    )
    executed_digest = hashlib.sha256(
        json.dumps(executed_prompts, separators=(',', ':')).encode()
    ).hexdigest()
    measured = [r for r in records if r['phase'] == 'measure' and 'prompt_sha256' in r]
    errors = [r for r in records if 'error' in r]
    skipped = sum('skip_reason' in r for r in records)
    if args['shared_cache'] and args['track_chunk_fates'] and not errors:
        report_shared_replay_end(args['lmc'])
    summary = dict(label=args['label'], args=args, content_mode='generated_token_prefix',
                   schedule_sha256=digest, executed_prompt_sha256=executed_digest,
                   planned_arrival_rate=rate,
                   measured_interval_s=duration, wall_s=wall,
                   truncated_sessions=truncated, metrics_before=before,
                   metrics_after=after, metrics_delta=delta, records=records)
    if measured:
        for field in ('total_ms', 'ttft_ms', 'offered_latency_ms', 'scheduling_lag_ms'):
            values = [r[field] for r in measured]
            print(f'{field}: mean={sum(values)/len(values):.2f} '
                  f'p99={percentile(values,.99):.2f}')
    print(f'[summary] measured_ok={len(measured)} errors={len(errors)} '
          f'skipped={skipped} wall={wall:.2f}s')
    out = Path(args['out'] or DEMO_DIR / 'logs' / f"trace_replay_{args['label']}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if errors or not measured else 0


if __name__ == '__main__':
    raise SystemExit(main())
