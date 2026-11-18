#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Classify SSD-written chunks by lifecycle events (Bidaw, paper trace).

Categories over unique physically-written chunks:
  C  effective  : successfully loaded back from SSD >= 1 time
                  (backend-confirmed load events, not atime heuristics)
  A  retained   : never loaded back, but a LATER lookup submission of the
                  same salt contained this chunk and it was still
                  L1-resident at observation end (DRAM served the reuse)
  B  dead write : never loaded back and no later lookup contained it, with
                  the write outside the end-of-observation horizon
  U  unresolved : written inside the end-of-observation horizon (next use
                  unknown), or lookup-contained-but-not-resident anomaly
  D  duplicates : reported separately at event level --
                  skip events (duplicate submission, no physical write)
                  and physical rewrites (only possible after a delete)
"""
import json
from collections import defaultdict
from pathlib import Path

FIX = Path(__file__).resolve().parents[1]
EV_LOG = FIX / "logs" / "l2_events.jsonl"
METRICS_TXT = FIX / "logs" / "l2class2_metrics.txt"
STATUS_JSON = FIX / "logs" / "l2class2_status.json"
OUT_PNG = FIX / "logs" / "ssd_chunk_fate_pie.png"
OUT_TXT = FIX / "logs" / "ssd_chunk_fate.txt"
HORIZON = 30.0  # end-of-observation horizon (s), == warmup window

# ---- load lifecycle events ----
events = []
with open(EV_LOG) as f:
    for line in f:
        line = line.strip()
        if line:
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                pass

per_key = defaultdict(lambda: {"write": 0, "skip": 0, "load": 0,
                               "delete": 0, "first_w": None, "last_w": None})
key_submits = defaultdict(list)
t_end = 0.0
for e in events:
    t_end = max(t_end, e["t"])
    k = (e["salt"], e["hash"])
    if e["ev"] == "submit":
        key_submits[k].append(e["t"])
        continue
    p = per_key[k]
    p[e["ev"]] = p.get(e["ev"], 0) + 1
    if e["ev"] == "write":
        if p["first_w"] is None:
            p["first_w"] = e["t"]
        p["last_w"] = e["t"]

written = {k: p for k, p in per_key.items() if p["write"] >= 1}


# ---- resident-at-end identities (policy status) ----
status = json.load(open(STATUS_JSON))
pol = status["policies"][0]
resident = {(e["salt"], e["hash"]) for e in pol.get("resident_identities", [])}

# ---- classify (event-driven) ----
cls = {"C": 0, "A": 0, "B": 0, "U": 0}
bytes_cls = {"C": 0, "A": 0, "B": 0, "U": 0}
anomalies = []
skip_events = rewrite_events = load_events = delete_events = 0
CHUNK_BYTES = 7340032  # 256 tokens x 28,672 B (Qwen2.5-1.5B bf16)
for k, p in written.items():
    loads = p["load"]
    writes = p["write"]
    skip_events += p["skip"]
    load_events += loads
    delete_events += p["delete"]
    if writes >= 2:
        rewrite_events += writes - 1
    later_lookup = any(t > p["last_w"] + 0.5 for t in key_submits.get(k, []))
    if loads >= 1:
        kk = "C"      # backend-confirmed reuse from SSD
    elif later_lookup:
        # A later lookup contained this chunk and no SSD load followed:
        # the only way the lookup could serve it is an L1 hit, so DRAM
        # (not SSD) served every later access.  End-of-run residency is
        # irrelevant (the chunk may be evicted after its last access).
        kk = "A"
    elif p["last_w"] >= t_end - HORIZON:
        kk = "U"                                  # next use not observed yet
    else:
        kk = "B"                                  # session ended, never used
    cls[kk] += 1
    bytes_cls[kk] += CHUNK_BYTES
n_unique = len(written)

# ---- metrics cross-check ----
store_events = hit_l2_tokens = req_tokens = 0
for line in open(METRICS_TXT):
    for key, var in (("l2_store_completed_objects_chunks_total", "se"),
                     ("lookup_hit_l2_tokens_total", "hl2"),
                     ("lookup_requested_tokens_total", "req")):
        if line.startswith("lmcache_mp_" + key):
            try:
                v = float(line.rsplit(" ", 1)[1])
            except ValueError:
                continue
            if var == "se":
                store_events += v
            elif var == "hl2":
                hit_l2_tokens += v
            else:
                req_tokens += v


# ---- report + pie ----
lines = [f"unique physically-written chunks: {n_unique}",
         f"store key-events (completed, incl. skipped): {int(store_events)}",
         f"skip events (logical dup submission, no physical write): {skip_events}",
         f"physical rewrite events: {rewrite_events}",
         f"backend load events (SSD reads): {load_events}",
         f"delete events: {delete_events}", ""]
for kk in ("C", "A", "B", "U"):
    lines.append(f"{kk}: {cls[kk]:4d} chunks "
                 f"({100*cls[kk]/max(1, n_unique):.1f}% of unique) "
                 f"{bytes_cls[kk]/2**30:6.2f} GiB")
lines.append("")
lines.append(f"requests served from SSD: {hit_l2_tokens/req_tokens:.1%} "
             f"({int(hit_l2_tokens)}/{int(req_tokens)} tokens)")
if anomalies:
    lines.append(f"anomalies: {len(anomalies)} e.g. {anomalies[:3]}")

json.dump({"classes": cls, "bytes": bytes_cls, "unique": n_unique,
           "skip_events": skip_events, "rewrite_events": rewrite_events,
           "load_events": load_events, "delete_events": delete_events,
           "anomalies": anomalies[:20]},
          open(OUT_TXT.with_suffix(".json"), "w"), indent=1)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sizes = [cls["C"], cls["A"], cls["B"], cls["U"]]
labels = [
    f"C effective (read back)\n{cls['C']} chunks",
    f"A DRAM-retained redundancy\n{cls['A']} chunks",
    f"B dead write (no later use)\n{cls['B']} chunks",
    f"U unresolved (horizon)\n{cls['U']} chunks",
]
colors = ["#6aa9e0", "#8fd18f", "#e0b050", "#b9b9b9"]
fig, ax = plt.subplots(figsize=(8.8, 7.2))
ax.pie(sizes, labels=labels, colors=colors, startangle=90,
       autopct=lambda p: f"{p:.1f}%", pctdistance=0.72,
       wedgeprops=dict(edgecolor="white", linewidth=1.5),
       textprops=dict(fontsize=11))
ax.set_title(
    "Bidaw SSD chunk-fate classification (official trace, u70=25.6 users/min)\n"
    f"unique written chunks {n_unique} | logical-dup skips {skip_events} | "
    f"SSD served {hit_l2_tokens/req_tokens:.1%} of requested tokens",
    fontsize=12)
ax.axis("equal")
fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")

with open(OUT_TXT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print("written:", OUT_PNG, "and", OUT_TXT)
