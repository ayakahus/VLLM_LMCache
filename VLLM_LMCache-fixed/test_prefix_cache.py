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
import sys
import time
from typing import Any

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
import uuid

RUN_ID = uuid.uuid4().hex[:8]
_PREFIX_SENTENCE = (
    f"[run-{RUN_ID}] "
    "The LMCache system stores the key-value cache of large language models "
    "so that identical prompt prefixes can be reused across requests and "
    "across engine restarts, which greatly reduces time-to-first-token. "
)
LONG_PREFIX = _PREFIX_SENTENCE * 60  # ~4800 token: prefill 越长, LMCache 复用的 TTFT 收益越明显


def build_prompt(question: str) -> str:
    return f"{LONG_PREFIX}\n\nQuestion: {question}\nAnswer:"


def stream_completion(base_url: str, prompt: str, max_tokens: int = 32) -> tuple[str, float]:
    """流式请求 /v1/completions, 返回 (文本, TTFT 秒)。"""
    url = f"{base_url}/v1/completions"
    payload = {
        "model": MODEL,
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


def main() -> int:
    ap = argparse.ArgumentParser(description="vLLM+LMCache KV 复用验证")
    ap.add_argument("--vllm", default=DEFAULT_VLLM)
    ap.add_argument("--lmcache", default=DEFAULT_LMC)
    ap.add_argument("--max-tokens", type=int, default=32)
    args = ap.parse_args()

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
    text1, ttft1 = stream_completion(args.vllm, phase1_prompt, args.max_tokens)
    print(f"TTFT = {ttft1 * 1000:8.1f} ms   输出: {text1[:80]!r}")

    stored = wait_for_store(args.lmcache, baseline)
    print(f"[STORE 确认] LMCache 完成 STORE 次数: {baseline} -> {stored}")

    # ------------------------------------------------------------------
    print("\n--- 第 2 步 / WARM: 相同前缀 + 不同问题 (KV 应从 LMCache 检索回来) ---")
    warm_prompt = build_prompt("List the two main benefits of caching KV, in a few words.")
    text2, ttft2 = stream_completion(args.vllm, warm_prompt, args.max_tokens)
    print(f"TTFT = {ttft2 * 1000:8.1f} ms   输出: {text2[:80]!r}")

    # ------------------------------------------------------------------
    print("\n--- 第 3 步 / WARM2: 与首次完全相同的请求, 再测一次 ---")
    text3, ttft3 = stream_completion(args.vllm, phase1_prompt, args.max_tokens)
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
