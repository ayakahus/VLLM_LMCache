#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare the two frozen-trace policy arms and write a report file."""
import json
from pathlib import Path
import sys

if __name__ == "__main__" and len(sys.argv) > 1:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from bidaw.gandalf_report import main

    main()
    raise SystemExit

D = str(Path(__file__).resolve().parents[1] / "logs") + "/"
OUT = D + "trace70_comparison.txt"
ARMS = [
    ("trace_replay_clean70_bidaw", "Bidaw + DQ-off (FIFO)"),
    ("trace_replay_dq70_bidaw", "Bidaw + DQ-on (dual queue)"),
    ("trace_replay_clean70_lru", "LRU + DQ-off (FIFO)"),
    ("trace_replay_dq70_lru", "LRU + DQ-on (dual queue)"),
]
lines = []
for name, policy in ARMS:
    d = json.load(open(D + name + ".json"))
    # use metrics_delta (after-before): immune to warm-start pollution
    a = d.get("metrics_delta") or d["metrics_after"]
    req = a.get("lmcache_mp_lookup_requested_tokens_total", 0)
    hit = a.get("lmcache_mp_lookup_hit_tokens_total", 0)
    h1 = a.get("lmcache_mp_lookup_hit_l1_tokens_total", 0)
    h2 = a.get("lmcache_mp_lookup_hit_l2_tokens_total", 0)
    tt = sorted(r["ttft_ms"] for r in d["records"] if "ttft_ms" in r)
    n = len(tt)
    fresh = "(fresh-server)" if not d["metrics_before"] else "(warm-start!)"
    lines.append(f"=== {policy} ({name}) {fresh} ===")
    lines.append(f"  rounds_ok={n}  stores={a.get('lmcache_mp_num_finished_stores_total', 0):.0f}"
                 f"  retrieves={a.get('lmcache_mp_num_finished_retrieves_total', 0):.0f}"
                 f"  chunks_loaded={a.get('lmcache_mp_num_chunks_loaded_total', 0):.0f}")
    l1_rate, l2_rate = (h1 / req, h2 / req) if req else (0.0, 0.0)
    lines.append(f"  hit_rate={hit / req if req else 0:.4f}  (req={req:.0f})")
    lines.append(f"    L1_hit={h1:.0f} ({l1_rate:.1%})   L2_hit={h2:.0f} ({l2_rate:.1%})")
    lines.append(f"  TTFT p10={tt[int(n * 0.10)]:.0f}  p50={tt[int(n * 0.50)]:.0f}"
                 f"  p90={tt[int(n * 0.90)]:.0f}  p99={tt[int(n * 0.99)]:.0f}"
                 f"  max={tt[-1]:.0f} ms")
    # per-round p50 to see tail behavior by depth
    by_round = {}
    for r in d["records"]:
        if "ttft_ms" in r:
            by_round.setdefault(r["round"], []).append(r["ttft_ms"])
    spikes = 0
    for rnd, xs in by_round.items():
        xs.sort()
        if xs[len(xs) // 2] > 150:
            spikes += 1
    lines.append(f"  rounds_with_p50>150ms: {spikes}/{len(by_round)}")
    lines.append("")

with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("written", OUT)

# ---------------------------------------------------------------------------
# 附: 运行上下文
# ---------------------------------------------------------------------------
ctx = """
---------------------------------------------------------------
运行上下文 (2026-11-16, 官方 Bidaw FAST'26 trace 冻结快照):
  trace:   ShipengHu-777/Interactive-conversation-workload
           6/7 个 part 文件, 975,926 轮 / 43,710 用户
  采样:    按到达时间取最早 70 个用户, 1551 轮, 速度压缩 60x
  L1:      1GB DRAM (146 keys x 256-token chunk = 37,376 tokens)
  L2:      本地 SSD (l2_ssd/, fs adapter, inclusive 副本)
  并发工作集 ~13.3 万 token ≈ 3.6x L1 (论文 Phase3 口径 3~4x)
  DQ = Bidaw Phase 1 双队列调度 (READY 快速通道 + disk-HRRN 派发,
       BIDAW_DUAL_QUEUE=1/0), 2x2 因子: {Bidaw, LRU} x {DQ off, on}

结果解读 (2026-11-16 四臂实测):
  1) 命中率: 四臂全部 ~90.4% — 淘汰策略与调度在此压力下都不改变
     总命中量 (inclusive L2 兜底), 差异在命中所在的层级与排队时延。
  2) DQ 的机制效果 (从 /bidaw/status): 522 个 L1-only lookup 走
     READY 快速通道同步完成 (不再排在 L2 预取后面), 577 个 L2 请求
     进 PREPARING 队列, 8 次 disk-HRRN 重排, pending 等待峰值 2ms。
  3) 本机 L2 为本地 NVMe (远快于论文 SATA RAID-5 1.5GB/s), L2 读回
     代价小 -> DQ 的收益被硬件抹平; 论文的主要收益源 (I/O 感知调度
     与慢速 capacity 层的交互) 需要更慢 L2 才能显现。淘汰策略贡献
     (ablation 1.25x) 也依赖高压 (KV 总量 3~4x 且复用距离分布更散)。
  复现命令:
    ./reset_for_policy.sh Bidaw 1   # 策略 + DQ 开关, 含 vLLM 重启
    source env.sh
    python3 bidaw/trace_replay.py --trace-dir bidaw/trace_frozen \
        --users 70 --speed 60 --label <name>
    python3 bidaw/compare_arms.py
"""
with open(OUT, "a", encoding="utf-8") as f:
    f.write(ctx)

