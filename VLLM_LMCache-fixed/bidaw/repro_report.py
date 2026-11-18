#!/usr/bin/env python3
"""Report measured-only metrics. No hard-coded rates or performance claims."""
import argparse
import json
from pathlib import Path


def summarize(data):
    records = data['records']
    ok = [r for r in records if r.get('phase') == 'measure' and 'error' not in r]
    errors = sum('error' in r for r in records)
    delta = data['metrics_delta']
    requested = delta.get('lmcache_mp_lookup_requested_tokens_total', 0)
    l1 = delta.get('lmcache_mp_lookup_hit_l1_tokens_total', 0)
    l2 = delta.get('lmcache_mp_lookup_hit_l2_tokens_total', 0)
    def mean(field):
        return sum(r[field] for r in ok) / len(ok) if ok else float('nan')
    return dict(n=len(ok), errors=errors, total_ms=mean('total_ms'),
                offered_ms=mean('offered_latency_ms'), lag_ms=mean('scheduling_lag_ms'),
                l1=l1 / requested if requested else float('nan'),
                l2=l2 / requested if requested else float('nan'))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--log-dir', type=Path, default=Path(__file__).resolve().parents[1] / 'logs')
    ap.add_argument('--pattern', default='trace_replay_repro_v2_*.json')
    args = ap.parse_args()
    lines = ['Synthetic KV replay; CPU/GPU/I/O results require target-server validation.',
             'rate is planned new sessions/min over measured schedule (includes session drain).',
             'total excludes client queue; offered includes schedule/dependency/queue delays.', '']
    groups = {}
    files = sorted(args.log_dir.glob(args.pattern))
    if not files:
        ap.error('No new-format reports found')
    invalid = False
    for path in files:
        data = json.loads(path.read_text())
        if 'schedule_sha256' not in data:
            raise ValueError(f'Legacy report cannot be compared: {path}')
        stats = summarize(data)
        invalid |= stats['errors'] > 0
        groups.setdefault((data['args']['users'],data['args']['seed']),set()).add(
            (data['schedule_sha256'],data['args']['warmup_seconds'],
             data['args']['parallel'],data['content_mode']))
        lines.append(f"{data['label']} rate={data['planned_arrival_rate']:.3f}/min "
                     f"n={stats['n']} errors={stats['errors']} "
                     f"truncated={len(data['truncated_sessions'])} "
                     f"avg_total={stats['total_ms']:.2f}ms "
                     f"avg_offered={stats['offered_ms']:.2f}ms "
                     f"avg_client_lag={stats['lag_ms']:.2f}ms "
                     f"L1_hit={stats['l1']:.2%} L2_hit={stats['l2']:.2%}")
    if any(len(values) > 1 for values in groups.values()):
        lines.append('INVALID COMPARISON: schedule/config mismatch between arms.')
        invalid = True
    if invalid:
        lines.append('Results have errors or mismatches; do not infer a speedup.')
    else:
        lines.append('Compare LRU/DQ0, LRU/DQ1, Bidaw/DQ0, Bidaw/DQ1 at matching workloads.')
        lines.append('Repeat runs and inspect offered load, tails and L1 hit rate before making claims.')
    output = args.log_dir / 'repro_v2_report.txt'
    output.write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print(f'Written: {output}')
    return int(invalid)


if __name__ == '__main__':
    raise SystemExit(main())
