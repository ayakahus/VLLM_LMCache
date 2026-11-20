import json
import sys
from pathlib import Path

sys.path.insert(0, "/home/fzz27/vllm_lmcache-fixed/VLLM_LMCache-fixed")
from bidaw.trace_replay import percentile

ROOT = Path("/home/fzz27/vllm_lmcache-fixed/VLLM_LMCache-fixed/logs/gandalf_benchmarks/run.ofRUSFwC")
ARMS = [("lru", "LRU baseline"), ("lru_writeback", "LRU write-back"),
        ("bidaw", "Bidaw"), ("gandalf", "Gandalf")]


def stats(values):
    return {k: percentile(values, p) for k, p in (("p50", .5), ("p95", .95), ("p99", .99))}


rows = {}
for arm, label in ARMS:
    d = json.load(open(ROOT / f"repeat-1/{arm}/replay.json"))
    recs = [r for r in d["records"] if r.get("phase") == "measure" and "ttft_ms" in r]
    mismatch = sum(1 for r in recs if r.get("actual_response_tokens") != r.get("rlen"))
    bad = sum(1 for r in recs if r["ttft_ms"] > r["total_ms"] + 1e-6)
    ttft = [r["ttft_ms"] for r in recs]
    total = [r["total_ms"] for r in recs]
    t2t = []
    for r in recs:
        n = r.get("actual_response_tokens") or r["rlen"]
        t2t.append((r["total_ms"] - r["ttft_ms"]) / max(1, n - 1))
    rows[arm] = dict(
        label=label, n=len(recs), token_length_mismatch=mismatch, ttft_gt_total=bad,
        ttft_mean=sum(ttft) / len(ttft), ttft=stats(ttft),
        t2t_mean=sum(t2t) / len(t2t), t2t=stats(t2t),
        total_mean=sum(total) / len(total), total=stats(total),
    )

lines = ["四臂延迟指标 (measured phase; T2T = (total-ttft)/(actual_tokens-1))", "",
         "| Arm | n | TTFT mean | TTFT p99 | T2T mean | T2T p99 | E2E mean | E2E p99 |",
         "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
for arm, _ in ARMS:
    r = rows[arm]
    lines.append("| {label} | {n} | {tm:.2f} | {tp:.2f} | {t2m:.2f} | {t2p:.2f} | {em:.2f} | {ep:.2f} |".format(
        label=r["label"], n=r["n"], tm=r["ttft_mean"], tp=r["ttft"]["p99"],
        t2m=r["t2t_mean"], t2p=r["t2t"]["p99"], em=r["total_mean"], ep=r["total"]["p99"]))
lines += ["", "| Arm | TTFT p50 | TTFT p95 | T2T p50 | T2T p95 | E2E p50 | E2E p95 |",
          "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
for arm, _ in ARMS:
    r = rows[arm]
    lines.append("| {label} | {a:.2f} | {b:.2f} | {c:.2f} | {d:.2f} | {e:.2f} | {f:.2f} |".format(
        label=r["label"], a=r["ttft"]["p50"], b=r["ttft"]["p95"],
        c=r["t2t"]["p50"], d=r["t2t"]["p95"], e=r["total"]["p50"], f=r["total"]["p95"]))
report = "\n".join(lines) + "\n"
(ROOT / "latency_ttft_t2t.txt").write_text(report, encoding="utf-8")
(ROOT / "latency_ttft_t2t.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
print(report)
for arm, _ in ARMS:
    r = rows[arm]
    print("[check] {a}: n={n} actual_vs_rlen_mismatch={m} ttft_gt_total={b}".format(
        a=arm, n=r["n"], m=r["token_length_mismatch"], b=r["ttft_gt_total"]))
