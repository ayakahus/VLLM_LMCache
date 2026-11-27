# SPDX-License-Identifier: Apache-2.0
"""Validate and summarize the cache benchmark."""
import argparse
import json
from pathlib import Path
import re
from statistics import fmean

from bidaw.l2_fate_report import latest_run
from bidaw.trace_replay import percentile

ARMS = {
    "lru": ("write_through", False, False),
    "lru_writeback": ("lru_writeback", False, False),
    "bidaw": ("write_through", True, True),
    "gandalf": ("gandalf", False, False),
    "l1_only": ("l1_only", False, False),
}
TOKEN_SOURCE = re.compile(
    r'^vllm:prompt_tokens_by_source_total\{[^}]*\bsource="local_compute"[^}]*\}\s+(\S+)',
    re.MULTILINE,
)


def read_json(path):
    return json.loads(path.read_text())


def benchmark_arms(repeat):
    return tuple(arm for arm in ARMS if (repeat / arm).is_dir())


def format_ttft(stats):
    values = [(stats or {}).get(key) for key in ("mean", "p50", "p90", "p99")]
    return "/".join(f"{value:.2f}" if value is not None else "n/a" for value in values)


def hit_rate(value):
    return f"{value:.1%}" if value is not None else "n/a"


def gpu_prefill(path):
    values = TOKEN_SOURCE.findall(path.read_text())
    if not values:
        raise ValueError(f"Missing vLLM local_compute token counter in {path}")
    return sum(float(value) for value in values)


def summarize(path, expected):
    run = read_json(path / "replay.json")
    state = read_json(path / "status-after.json")
    tier = state["tier_policy"]
    mode, dual_queue, uses_bidaw = expected
    if (tier["mode"], state["scheduler"]["enabled"], bool(state["policies"])) != expected:
        raise ValueError(f"{path}: unexpected live arm configuration (expected {mode}, "
                         f"DQ={dual_queue}, Bidaw={uses_bidaw})")
    if state["padded_tail_store"] or run["args"]["shared_cache"]:
        raise ValueError(f"{path}: expected unpadded isolated sessions")
    if (run["args"]["warmup_seconds"] or run["args"]["answer_history"] != "fixed"
            or run["content_mode"] != "fixed_answer_token_prefix"):
        raise ValueError(f"{path}: expected fixed history and a single full-run measurement window")
    records = run["records"]
    if not records or any(r["phase"] != "measure" or "prompt_sha256" not in r
                          or "error" in r or "skip_reason" in r for r in records):
        raise ValueError(f"{path}: missing, failed, or skipped replay rounds")
    matches = (run["schedule_sha256"], run["executed_prompt_sha256"], len(records),
               tier["l1_capacity_bytes"],
               run["args"]["model"], run["args"]["parallel"])

    l1_only = mode == "l1_only"
    log = path / "l2_events.jsonl"
    marker, events = latest_run(log) if log.is_file() or not l1_only else ({}, [])
    if not marker and not l1_only:
        raise ValueError(f"{path}: missing event probe for SSD-backed arm")
    writes = [event for event in events if event["ev"] == "write"]
    reads = sum(event["ev"] == "load" for event in events)
    write_bytes = sum(event["bytes"] for event in writes)
    if (len(writes), write_bytes) != (tier.get("writes", 0), tier.get("write_bytes", 0)):
        raise ValueError(f"{path}: adapter counters do not match the FS event log at drain")
    if any(tier.get(key, 0) for key in ("failed_writes", "failed_migrations", "eviction_failures")):
        raise ValueError(f"{path}: failed SSD writes or migrations; inspect lmcache.log")
    delta = run["metrics_delta"]
    requested = delta.get("lmcache_mp_lookup_requested_tokens_total", 0)
    l1 = delta.get("lmcache_mp_lookup_hit_l1_tokens_total", 0)
    l2 = delta.get("lmcache_mp_lookup_hit_l2_tokens_total", 0)
    if l1_only and (tier["num_l2_adapters"] != 0 or tier["capacity_bytes"] != 0
                    or l2 or tier.get("reads", 0)
                    or any(e["ev"] in {"write", "load", "load_attempt", "l2_hit"}
                           for e in events)):
        raise ValueError(f"{path}: L2 activity observed in the L1-only arm")
    prefill = gpu_prefill(path / "vllm-metrics-after.txt") - gpu_prefill(path / "vllm-metrics-before.txt")
    if prefill < 0:
        raise ValueError(f"{path}: vLLM prompt token counter reset during replay")
    from bidaw.ttft_breakdown import histogram_deltas

    histograms = histogram_deltas(path)
    server_ttft = histograms["server_ttft"].summary() if "server_ttft" in histograms else None
    latency = {}
    for field in ("ttft_ms", "total_ms", "scheduling_lag_ms"):
        values = [r[field] for r in records]
        latency[field] = {"mean": fmean(values), **{
            f"p{p}": percentile(values, p / 100) for p in (50, 90, 95, 99)
        }}
    return matches, {
        "requests": len(records), "writes": len(writes), "write_bytes": write_bytes,
        "ssd_reads": reads, "l2_capacity_bytes": tier["capacity_bytes"],
        "bytes_per_1000": 1000 * write_bytes / len(records),
        "prefix_l1": l1 / requested if requested else None,
        "prefix_l2": 0.0 if l1_only else (l2 / requested if requested else None),
        "l1_hit_tokens": l1, "l2_hit_tokens": l2,
        "cache_miss_tokens": requested - l1 - l2, "gpu_prefill_tokens": int(prefill),
        "ssd_evictions": tier.get("ssd_evictions", 0),
        "l1_drops": tier.get("drops", 0),
        "migrations": tier.get("migrations", 0),
        "reused_ssd_copies": tier.get("reused_ssd_copies", 0),
        "blocked_migrations": tier.get("blocked_migrations", 0),
        "blocked_writes": tier.get("blocked_writes", 0),
        "truncated_sessions": len(run["truncated_sessions"]),
        "latency_ms": latency,
        "server_ttft_ms": server_ttft,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    results = {}
    reference = None
    l2_capacity = None
    expected_arms = None
    for repeat in sorted(args.run_dir.glob("repeat-*")):
        arms = benchmark_arms(repeat)
        if expected_arms is None:
            expected_arms = arms
        if arms != expected_arms:
            raise ValueError(f"{repeat}: arm set differs across repeats")
        for arm in arms:
            matches, result = summarize(repeat / arm, ARMS[arm])
            if reference is None:
                reference = matches
            if matches != reference:
                raise ValueError(f"{repeat.name}/{arm}: workload or L1 capacity differs")
            if arm != "l1_only":
                if l2_capacity is None:
                    l2_capacity = result["l2_capacity_bytes"]
                if result["l2_capacity_bytes"] != l2_capacity:
                    raise ValueError(f"{repeat.name}/{arm}: L2 capacity differs")
            results[f"{repeat.name}/{arm}"] = result
    if not results:
        parser.error("No complete repeats found")
    (args.run_dir / "comparison.json").write_text(json.dumps(results, indent=2) + "\n")
    print("arm                         n    writes    GiB    TTFT p95   E2E p95    GPU prefill")
    for name, r in results.items():
        latency = r["latency_ms"]
        print(f"{name:26} {r['requests']:5} {r['writes']:9} {r['write_bytes'] / 2**30:6.2f}"
              f" {latency['ttft_ms']['p95']:11.1f} {latency['total_ms']['p95']:9.1f}"
              f" {r['gpu_prefill_tokens']:14}")
    print()
    print("arm                         n    writes    GiB  reads     L1 hit  L2 hit  Client TTFT mean/p50/p90/p99 (ms)  Server TTFT mean/p50/p90/p99 (ms)")
    for name, r in results.items():
        print(f"{name:26} {r['requests']:5} {r['writes']:9} {r['write_bytes'] / 2**30:6.2f}"
              f" {r['ssd_reads']:6} {hit_rate(r['prefix_l1']):>8} {hit_rate(r['prefix_l2']):>7}"
              f"  {format_ttft(r['latency_ms']['ttft_ms']):34}  {format_ttft(r['server_ttft_ms'])}")


if __name__ == "__main__":
    main()
