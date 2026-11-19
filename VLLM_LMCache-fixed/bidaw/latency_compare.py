#!/usr/bin/env python3
"""两臂延迟对比: LRU+DQ0(FCFS) vs Bidaw+DQ1, u70 同 schedule, 各 2 次重复。"""
import json
import statistics as st
from pathlib import Path

LOGS = Path("/home/fzz27/vllm_lmcache-fixed/VLLM_LMCache-fixed/logs")
ARMS = [
    ("lat3_lru_dq0_u70", "lat3b_lru_dq0_u70", "LRU + FCFS (DQ off)"),
    ("lat3_bidaw_dq1_u70", "lat3b_bidaw_dq1_u70", "Bidaw + dual-queue"),
]
FIELDS = ("total_ms", "ttft_ms", "offered_latency_ms", "scheduling_lag_ms")


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(len(xs) * q))]


def load(label):
    d = json.loads((LOGS / f"trace_replay_{label}.json").read_text())
    recs = [r for r in d["records"] if r.get("phase") == "measure"
            and "prompt_sha256" in r]
    delta = d["metrics_delta"]
    req = delta.get("lmcache_mp_lookup_requested_tokens_total", 0) or 0
    s = {"n": len(recs),
         "errors": sum("error" in r for r in d["records"]),
         "skipped": sum("skip_reason" in r for r in d["records"]),
         "sha": d["schedule_sha256"],
         "exec": d.get("executed_prompt_sha256"),
         "mode": d.get("content_mode")}
    for f in FIELDS:
        vals = [r[f] for r in recs if f in r]
        s[f] = dict(mean=st.mean(vals), p50=pct(vals, .5),
                    p90=pct(vals, .9), p99=pct(vals, .99))
    s["l1"] = delta.get("lmcache_mp_lookup_hit_l1_tokens_total", 0) / req if req else float("nan")
    s["l2"] = delta.get("lmcache_mp_lookup_hit_l2_tokens_total", 0) / req if req else float("nan")
    s["stores"] = delta.get("lmcache_mp_num_finished_stores_total", 0)
    s["retrieves"] = delta.get("lmcache_mp_num_finished_retrieves_total", 0)
    return s


lines = ["两臂延迟对比 — 固定回答历史 (--answer-history fixed)",
         "(paper trace frozen, u70, speed=20x, warmup=30s, L1=1GiB, L2 每臂冷启, 各 2 次)", ""]
all_stats = {}
all_loads = {}
for rep_labels, name in [(a[:2], a[2]) for a in ARMS]:
    reps = [load(lb) for lb in rep_labels]
    all_stats[name] = reps
    all_loads[name] = [json.loads((LOGS / f"trace_replay_{lb}.json").read_text())
                       for lb in rep_labels]
    shas = {r["sha"][:12] for r in reps}
    lines.append(f"== {name}  (schedule sha 一致: {len(shas) == 1})")
    for lb, r in zip(rep_labels, reps):
        lines.append(f"  [{lb.split('_')[0]}] n={r['n']} err={r['errors']} skip={r['skipped']} "
                     f"TTFT mean={r['ttft_ms']['mean']:.1f}/p50={r['ttft_ms']['p50']:.1f}/"
                     f"p99={r['ttft_ms']['p99']:.1f}  "
                     f"E2E mean={r['total_ms']['mean']:.1f}/p50={r['total_ms']['p50']:.1f}/"
                     f"p99={r['total_ms']['p99']:.1f}  "
                     f"L1={r['l1']:.1%} L2={r['l2']:.1%}")
    lines.append("")

execs = {d.get("executed_prompt_sha256")
         for loads in all_loads.values() for d in loads}
modes = {d.get("content_mode")
         for loads in all_loads.values() for d in loads}
lines.append(f"content_mode: {modes}")
lines.append(f"executed_prompt_sha256 跨全部4轮一致: {len(execs) == 1} "
             f"(共 {len(execs)} 个不同值)")
lines.append("")


def agg(name, field, agg_key):
    return st.mean(r[field][agg_key] for r in all_stats[name])


hdr = f"{'metric (2次重复均值)':<24} {'LRU + FCFS':>16} {'Bidaw + DQ':>16} {'Δ':>10}"
lines.append(hdr)
lines.append("-" * len(hdr))
for f, title in (("ttft_ms", "TTFT"), ("total_ms", "E2E total")):
    for agg_key, lab in (("mean", "mean"), ("p50", "p50"), ("p90", "p90"), ("p99", "p99")):
        a = agg(ARMS[0][2], f, agg_key)
        b = agg(ARMS[1][2], f, agg_key)
        lines.append(f"{title + ' ' + lab:<24} {a:>14.1f}ms {b:>14.1f}ms {b - a:>+9.1f}ms")
l1a = st.mean(r["l1"] for r in all_stats[ARMS[0][2]])
l1b = st.mean(r["l1"] for r in all_stats[ARMS[1][2]])
l2a = st.mean(r["l2"] for r in all_stats[ARMS[0][2]])
l2b = st.mean(r["l2"] for r in all_stats[ARMS[1][2]])
lines.append(f"{'L1 hit':<24} {l1a:>15.1%} {l1b:>15.1%} {l1b - l1a:>+9.1%}")
lines.append(f"{'L2 hit':<24} {l2a:>15.1%} {l2b:>15.1%} {l2b - l2a:>+9.1%}")

out = LOGS / "latency_compare.txt"
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

metrics = [("TTFT p50", "ttft_ms", "p50"), ("TTFT p99", "ttft_ms", "p99"),
           ("E2E p50", "total_ms", "p50"), ("E2E p99", "total_ms", "p99")]
x = range(len(metrics))
width = 0.36
fig, ax = plt.subplots(figsize=(8.4, 4.8))
a_vals = [agg(ARMS[0][2], m[1], m[2]) for m in metrics]
b_vals = [agg(ARMS[1][2], m[1], m[2]) for m in metrics]
ax.bar([i - width / 2 for i in x], a_vals, width, label=ARMS[0][2], color="#6aa9e0")
ax.bar([i + width / 2 for i in x], b_vals, width, label=ARMS[1][2], color="#e0b050")
for i, v in enumerate(a_vals):
    ax.text(i - width / 2, v, f"{v:.0f}", ha="center", va="bottom", fontsize=8)
for i, v in enumerate(b_vals):
    ax.text(i + width / 2, v, f"{v:.0f}", ha="center", va="bottom", fontsize=8)
ax.set_xticks(list(x))
ax.set_xticklabels([m[0] for m in metrics])
ax.set_ylabel("latency (ms)")
ax.set_title("Paper-trace replay latency: LRU+FCFS vs Bidaw+dual-queue "
             "(u70, 2 reps each)")
ax.legend()
fig.tight_layout()
fig.savefig(LOGS / "latency_compare.png", dpi=150)
print(f"\nWritten: {out} and {LOGS / 'latency_compare.png'}")

