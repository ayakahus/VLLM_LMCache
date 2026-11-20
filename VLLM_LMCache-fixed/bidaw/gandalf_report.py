# SPDX-License-Identifier: Apache-2.0
"""Validate and summarize the four-arm full-attention benchmark."""
import argparse
import json
from pathlib import Path
import re

from bidaw.l2_fate_report import latest_run
from bidaw.trace_replay import percentile

ARMS = {
    "lru": ("write_through", False, False),
    "lru_writeback": ("lru_writeback", False, False),
    "bidaw": ("write_through", True, True),
    "gandalf": ("gandalf", False, False),
}
TOKEN_SOURCE = re.compile(
    r'^vllm:prompt_tokens_by_source_total\{[^}]*\bsource="local_compute"[^}]*\}\s+(\S+)',
    re.MULTILINE,
)


def read_json(path):
    return json.loads(path.read_text())


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
               tier["l1_capacity_bytes"], tier["capacity_bytes"],
               run["args"]["model"], run["args"]["parallel"])

    _, events = latest_run(path / "l2_events.jsonl")
    writes = [event for event in events if event["ev"] == "write"]
    write_bytes = sum(event["bytes"] for event in writes)
    if (len(writes), write_bytes) != (tier.get("writes", 0), tier.get("write_bytes", 0)):
        raise ValueError(f"{path}: adapter counters do not match the FS event log at drain")
    if any(tier.get(key, 0) for key in ("failed_writes", "failed_migrations", "eviction_failures")):
        raise ValueError(f"{path}: failed SSD writes or migrations; inspect lmcache.log")
    delta = run["metrics_delta"]
    requested = delta.get("lmcache_mp_lookup_requested_tokens_total", 0)
    l1 = delta.get("lmcache_mp_lookup_hit_l1_tokens_total", 0)
    l2 = delta.get("lmcache_mp_lookup_hit_l2_tokens_total", 0)
    prefill = gpu_prefill(path / "vllm-metrics-after.txt") - gpu_prefill(path / "vllm-metrics-before.txt")
    if prefill < 0:
        raise ValueError(f"{path}: vLLM prompt token counter reset during replay")
    return matches, {
        "requests": len(records), "writes": len(writes), "write_bytes": write_bytes,
        "bytes_per_1000": 1000 * write_bytes / len(records),
        "prefix_l1": l1 / requested if requested else None,
        "prefix_l2": l2 / requested if requested else None,
        "cache_miss_tokens": requested - l1 - l2, "gpu_prefill_tokens": int(prefill),
        "ssd_evictions": tier.get("ssd_evictions", 0),
        "l1_drops": tier.get("drops", 0),
        "migrations": tier.get("migrations", 0),
        "reused_ssd_copies": tier.get("reused_ssd_copies", 0),
        "blocked_migrations": tier.get("blocked_migrations", 0),
        "blocked_writes": tier.get("blocked_writes", 0),
        "truncated_sessions": len(run["truncated_sessions"]),
        "latency_ms": {field: {f"p{p}": percentile([r[field] for r in records], p / 100)
                                for p in (50, 95, 99)}
                       for field in ("ttft_ms", "total_ms", "scheduling_lag_ms")},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    results = {}
    reference = None
    for repeat in sorted(args.run_dir.glob("repeat-*")):
        for arm, expected in ARMS.items():
            matches, result = summarize(repeat / arm, expected)
            if reference is None:
                reference = matches
            if matches != reference:
                raise ValueError(f"{repeat.name}/{arm}: workload or L2 capacity differs")
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


if __name__ == "__main__":
    main()
