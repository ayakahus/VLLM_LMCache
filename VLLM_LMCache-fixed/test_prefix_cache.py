#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证 vLLM + LMCache 的 KV Cache 复用:

  第 1 步 (COLD): 发送一个超长公共前缀 + 问题 A 的请求
          -> vLLM prefill 后把 KV 写入 (STORE) 独立 LMCache Server
  第 2 步     : 轮询 LMCache HTTP API /cache/objects, 确认 KV 已入库
  第 3 步 (WARM): 发送 相同前缀 + 问题 B 的请求
          -> LMCache LOOKUP 命中, 从独立 Server 拉回 KV, 跳过大部分 prefill
  第 4 步 (WARM2): 再次发送与第 1 步完全相同的请求, 复测一次
  第 5 步     : 汇总 TTFT 对比 + LMCache /metrics 指标

由于 vLLM 启动时加了 --disable-prefix-caching, GPU 侧不缓存任何前缀,
所以 WARM 阶段的加速只能来自 LMCacheMPConnector 从独立 LMCache Server
检索回来的 KV Cache, 结论严格成立。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any
from uuid import uuid4

import requests

# ---------------------------------------------------------------------------
# 可配置项
# ---------------------------------------------------------------------------
DEFAULT_VLLM = "http://127.0.0.1:8000"
DEFAULT_LMC = "http://127.0.0.1:8080"
MODEL = "qwen2.5-1.5b-instruct"

# 公共长前缀: 由固定文本重复拼接 + 本次运行唯一 ID, ~1900 token,
# 保证"同一运行内的多个请求"共享前缀, 而"多次运行之间"前缀互不相同
# (否则上一次运行的缓存会污染本次的 COLD 基线)。
RUN_ID = uuid4().hex[:8]
_PREFIX_SENTENCE = (
    f"[run-{RUN_ID}] "
    "The LMCache system stores the key-value cache of large language models "
    "so that identical prompt prefixes can be reused across requests and "
    "across engine restarts, which greatly reduces time-to-first-token. "
)
LONG_PREFIX = _PREFIX_SENTENCE * 60  # ~4800 token: prefill 越长, LMCache 复用的 TTFT 收益越明显


def build_prompt(question: str) -> str:
    return f"{LONG_PREFIX}\n\nQuestion: {question}\nAnswer:"


def stream_completion(base_url: str, prompt: str, max_tokens: int = 32, *, model=MODEL) -> tuple[str, float]:
    """流式请求 /v1/completions, 返回 (文本, TTFT 秒)。"""
    url = f"{base_url}/v1/completions"
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }
    t0 = time.perf_counter()
    ttft: float | None = None
    chunks: list[str] = []
    with requests.post(url, json=payload, stream=True, timeout=600) as resp:
        resp.raise_for_status()
        for raw in resp.iter_lines(decode_unicode=True):
            if not raw or not raw.startswith("data:"):
                continue
            data = raw[len("data:"):].strip()
            if data == "[DONE]":
                break
            obj = json.loads(data)
            if obj.get("choices"):
                text = obj["choices"][0].get("text") or ""
                if text and ttft is None:
                    ttft = time.perf_counter() - t0
                chunks.append(text)
    if ttft is None:
        ttft = time.perf_counter() - t0
    return "".join(chunks).strip(), ttft


def cache_objects(lmc: str) -> Any:
    try:
        r = requests.get(f"{lmc}/cache/objects", timeout=10)
        return r.json()
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


def count_objects(objs: Any) -> int:
    """尽量稳健地统计 /cache/objects 返回中的对象数量。"""
    if not isinstance(objs, dict) or objs.get("error"):
        return -1
    for key in ("objects", "keys", "data", "result"):
        v = objs.get(key)
        if isinstance(v, list):
            return len(v)
        if isinstance(v, dict):
            return len(v)
    if isinstance(objs, list):
        return len(objs)
    return -1


def finished_stores(lmc: str) -> float:
    """从 /metrics 读取已完成的 STORE 总数 (L1/任意层级, 比 /cache/objects 更可靠)。"""
    try:
        text = metrics_snapshot(lmc)
        total = 0.0
        for line in text.splitlines():
            if line.startswith("lmcache_mp_num_finished_stores_total"):
                try:
                    total += float(line.rsplit(" ", 1)[1])
                except (ValueError, IndexError):
                    pass
        return total
    except Exception:  # noqa: BLE001
        return -1.0


def wait_for_store(lmc: str, baseline: int, timeout: float = 90.0) -> int:
    """等待 LMCache 出现新增缓存对象 (STORE 是请求结束后异步执行的)。

    优先用 /metrics 的 finished_stores 计数; /cache/objects 仅作展示。
    """
    print(f"\n[等待 STORE] 轮询 {lmc}/metrics (最长 {timeout:.0f}s) ...")
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        cur = finished_stores(lmc)
        if cur != last:
            print(f"    finished_stores_total = {cur}")
            last = cur
        if cur >= 0 and baseline >= 0 and cur > baseline:
            return int(cur)
        if cur < 0 and baseline < 0:
            print("    [警告] /metrics 不可用, 跳过入库确认")
            return -1
        time.sleep(2)
    return int(finished_stores(lmc))


def metrics_snapshot(lmc: str) -> str:
    try:
        r = requests.get(f"{lmc}/metrics", timeout=10)
        return r.text
    except Exception as exc:  # noqa: BLE001
        return f"# metrics unavailable: {exc}"


def pick_metrics(text: str) -> list[str]:
    """从 Prometheus 文本中挑出与命中/流量相关的行。"""
    keys = ("hit", "retrieve", "store", "lookup", "token")
    out = []
    for line in text.splitlines():
        low = line.lower()
        if line and not line.startswith("#") and any(k in low for k in keys):
            out.append(line)
    return out


def wait_for_evidence(path, condition, label, timeout=90):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.is_file():
            with path.open(encoding="utf-8") as stream:
                events = [json.loads(line) for line in stream if line.strip() and line.endswith("\n")]
            result = condition(events)
            if result:
                return result
        time.sleep(0.25)
    raise RuntimeError(f"Timed out waiting for {label} in {path}")


def smoke_objects(events, kind, salt, turn, model, group_ids):
    """Address each object by rank and hash so unrelated group matches cannot pass."""
    objects = {group_id: set() for group_id in group_ids}
    for event in events:
        if (event["event"] != kind or event["salt"] != salt
                or event["turn"] != turn or event["model"] != model):
            continue
        if kind == "retrieve" and not event["complete"]:
            continue
        for group_id in group_ids:
            objects[group_id].update((event["rank"], key) for key in event["groups"].get(group_id, []))
    return objects


def reset_local_prefix_cache(vllm):
    deadline = time.monotonic() + 30
    url = f"{vllm}/reset_prefix_cache"
    while time.monotonic() < deadline:
        response = requests.post(url, params={"reset_external": "false", "reset_running_requests": "false"},
                                 timeout=5)
        response.raise_for_status()
        if response.json().get("success") is True:
            return
        time.sleep(0.25)
    raise RuntimeError("Could not clear vLLM's local prefix cache after round 1")


def hybrid_smoke(args):
    from bidaw.hybrid_smoke_probe import SMOKE_SALT_PREFIX
    from bidaw.trace_replay import stream_ttft
    from transformers import AutoTokenizer

    def latest_layout(events):
        return next((event for event in reversed(events) if event["event"] == "layout"), None)

    for endpoint in (f"{args.vllm}/v1/models", f"{args.lmcache}/healthcheck"):
        requests.get(endpoint, timeout=10).raise_for_status()
    layout = wait_for_evidence(args.hybrid_evidence, latest_layout, "registered layout")
    print(json.dumps(layout, indent=2), flush=True)
    kinds = layout["group_kinds"]
    windows = layout["num_chunks_in_sw"]
    group_ids = {
        str(index) for index, (kind, window) in enumerate(zip(kinds, windows, strict=True))
        if kind == "recurrent" or (kind == "attention" and window < 0)
    }
    if (layout["world_size"] != 1
            or not any(kind == "recurrent" and window == 1 for kind, window in zip(kinds, windows, strict=True))
            or not any(kind == "attention" and window < 0 for kind, window in zip(kinds, windows, strict=True))):
        raise RuntimeError("Expected one rank with full-attention and one-chunk recurrent groups")

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    encode = lambda text: tokenizer.encode(text, add_special_tokens=False)
    salt = SMOKE_SALT_PREFIX + uuid4().hex
    turn1, turn2 = f"{salt}-1", f"{salt}-2"
    target = 3 * layout["chunk_size"]
    prefix = encode(LONG_PREFIX)
    if not prefix:
        raise RuntimeError("Tokenizer returned an empty test prefix")
    prefix = (prefix * ((target + len(prefix) - 1) // len(prefix)))[:target]
    prompt1 = prefix + encode("\nQuestion: What does LMCache store?\nAnswer:")
    result1 = stream_ttft(args.vllm, prompt1, salt, args.max_tokens, args.model, round_id=turn1)

    def stored_groups(events):
        groups = smoke_objects(events, "store", salt, turn1, layout["model"], group_ids)
        return groups if all(groups.values()) else None

    stored = wait_for_evidence(args.hybrid_evidence, stored_groups, "round 1 STORE for every group")
    print("Round 1 STORE counts:", {g: len(keys) for g, keys in sorted(stored.items())}, flush=True)
    reset_local_prefix_cache(args.vllm)

    prompt2 = prompt1 + result1["completion_token_ids"] + encode("\nQuestion: Why reuse it?\nAnswer:")
    result2 = stream_ttft(args.vllm, prompt2, salt, args.max_tokens, args.model, round_id=turn2)

    def reused_groups(events):
        # More STORE callbacks from the first round may have arrived after its first group commit.
        expected = smoke_objects(events, "store", salt, turn1, layout["model"], group_ids)
        retrieved = smoke_objects(events, "retrieve", salt, turn2, layout["model"], group_ids)
        matched = {g: expected[g] & retrieved[g] for g in group_ids}
        return matched if all(matched.values()) else None

    matched = wait_for_evidence(args.hybrid_evidence, reused_groups, "round 2 RETRIEVE of round 1 objects")
    print("Round 2 RETRIEVE counts:", {g: len(keys) for g, keys in sorted(matched.items())})
    print(f"PASS: all groups reused across turns; TTFT {result1['ttft_s'] * 1000:.1f} ms / "
          f"{result2['ttft_s'] * 1000:.1f} ms")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="vLLM+LMCache KV 复用验证")
    ap.add_argument("--vllm", default=DEFAULT_VLLM)
    ap.add_argument("--lmcache", default=DEFAULT_LMC)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--model-dir", type=Path, help="Local tokenizer/model for the two-turn smoke")
    ap.add_argument("--hybrid-evidence", type=Path, help="JSONL from bidaw.hybrid_smoke_probe")
    args = ap.parse_args()

    if args.hybrid_evidence:
        if not args.model_dir:
            ap.error("--hybrid-evidence requires --model-dir")
        try:
            return hybrid_smoke(args)
        except (OSError, ValueError, RuntimeError, requests.RequestException) as exc:
            print(f"FAIL: {exc}", file=sys.stderr)
            return 2

    print("=" * 78)
    print("vLLM + LMCache 独立 Server — KV Cache 复用验证")
    print("=" * 78)
    print(f"vLLM API   : {args.vllm}")
    print(f"LMCache API: {args.lmcache}")
    print(f"公共前缀长度: {len(LONG_PREFIX)} 字符")

    for name, url in (("vLLM", f"{args.vllm}/v1/models"), ("LMCache", f"{args.lmcache}/healthcheck")):
        try:
            r = requests.get(url, timeout=10)
            print(f"[健康检查] {name}: HTTP {r.status_code}")
        except Exception as exc:  # noqa: BLE001
            print(f"[健康检查] {name}: 失败 ({exc}) — 请确认对应服务已启动")
            return 1

    # ------------------------------------------------------------------
    phase1_prompt = build_prompt("What does LMCache store, in one short sentence?")
    baseline = finished_stores(args.lmcache)
    print(f"\n[基线] LMCache 已完成 STORE 次数: {baseline}")

    print("\n--- 第 1 步 / COLD: 首次请求 (完整 prefill, 并把 KV 存入 LMCache) ---")
    text1, ttft1 = stream_completion(args.vllm, phase1_prompt, args.max_tokens, model=args.model)
    print(f"TTFT = {ttft1 * 1000:8.1f} ms   输出: {text1[:80]!r}")

    stored = wait_for_store(args.lmcache, baseline)
    print(f"[STORE 确认] LMCache 完成 STORE 次数: {baseline} -> {stored}")

    # ------------------------------------------------------------------
    print("\n--- 第 2 步 / WARM: 相同前缀 + 不同问题 (KV 应从 LMCache 检索回来) ---")
    warm_prompt = build_prompt("List the two main benefits of caching KV, in a few words.")
    text2, ttft2 = stream_completion(args.vllm, warm_prompt, args.max_tokens, model=args.model)
    print(f"TTFT = {ttft2 * 1000:8.1f} ms   输出: {text2[:80]!r}")

    # ------------------------------------------------------------------
    print("\n--- 第 3 步 / WARM2: 与首次完全相同的请求, 再测一次 ---")
    text3, ttft3 = stream_completion(args.vllm, phase1_prompt, args.max_tokens, model=args.model)
    print(f"TTFT = {ttft3 * 1000:8.1f} ms   输出: {text3[:80]!r}")

    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("结果汇总")
    print("=" * 78)
    cold_ms, warm_ms, warm2_ms = ttft1 * 1000, ttft2 * 1000, ttft3 * 1000
    print(f"COLD  TTFT (首次, 全量 prefill)   : {cold_ms:9.1f} ms")
    print(f"WARM  TTFT (同前缀+新问题)        : {warm_ms:9.1f} ms")
    print(f"WARM2 TTFT (与首次完全相同的请求) : {warm2_ms:9.1f} ms")
    if cold_ms > 0:
        print(f"WARM  加速比 (COLD/WARM)          : {cold_ms / max(warm_ms, 0.001):9.2f} x")
        print(f"WARM2 加速比 (COLD/WARM2)         : {cold_ms / max(warm2_ms, 0.001):9.2f} x")

    print("\n[LMCache /cache/objects 摘要] (该接口只统计 L2 适配器对象;")
    print("  L1(DRAM) 的命中请看上方 metrics 的 lookup_hit_l1_tokens_total)")
    objs = cache_objects(args.lmcache)
    s = json.dumps(objs, ensure_ascii=False)
    print(s[:2000] + ("..." if len(s) > 2000 else ""))

    print("\n[LMCache /metrics 相关指标]")
    rel = pick_metrics(metrics_snapshot(args.lmcache))
    for line in rel[:40]:
        print("   ", line)
    if not rel:
        print("    (无相关指标)")

    print("\n[结论判定]")
    ok = warm2_ms < cold_ms * 0.6 and stored != 0
    if ok:
        print("PASS: KV Cache 复用成功 — 第二次相同前缀请求 TTFT 显著下降,")
        print("且 vLLM 已关闭自带 prefix caching, 复用只能来自 LMCache Server。")
    else:
        print("FAIL: 未能确认复用 (TTFT 未明显下降或缓存为空)。")
        print("排查建议: 查看 logs/lmcache_server.log 与 logs/vllm.log 中的")
        print("LMCacheMPConnector STORE/LOOKUP 日志, 确认两个服务的端口配置。")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
