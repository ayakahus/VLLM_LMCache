#!/usr/bin/env bash
# =====================================================================
# 启动独立的 LMCache Server (multiprocess 模式, ZMQ 传输)
# - 请求端口: 5555 (vLLM 的 LMCacheMPConnector 通过 tcp://localhost:5555 连接)
# - HTTP 管理端口: 8080 (GET /cache/objects, GET /metrics, GET /bidaw/status ...)
# - L1 缓存: DRAM, 默认 8GB (L1_GB 可覆盖)
# - L2 缓存: 本地文件系统(SSD), 默认 <demo>/l2_ssd/padded-tail
# - 从 L2 读回的 chunk 留在 L1，直到被淘汰
# - 淘汰策略: EVICTION_POLICY 环境变量, 默认 Bidaw
#     Bidaw  -> 经 bidaw.lmcache_server_entry 注入的策略(上一轮回答长度感知)
#     LRU/ARC/IsolatedLRU -> LMCache 原生策略(对照组)
# =====================================================================
set -euo pipefail

DEMO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${DEMO_DIR}/lmcache_geometry.sh"
# shellcheck disable=SC1091
source "${DEMO_DIR}/env.sh"

export PYTHONPATH="${DEMO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
# Qwen2.5-1.5B BF16: 28,672 KV bytes/token x 256 tokens.
export BIDAW_CHUNK_BYTES="${BIDAW_CHUNK_BYTES:-7340032}"

L1_GB="${L1_GB:-8}"
export BIDAW_L1_CAPACITY_KEYS="${BIDAW_L1_CAPACITY_KEYS:-$(python3 -c \
    'import sys; print(int(float(sys.argv[1]) * (1 << 30)) // int(sys.argv[2]))' \
    "$L1_GB" "$BIDAW_CHUNK_BYTES")}"
export BIDAW_EVICTION_WATERMARK="${BIDAW_EVICTION_WATERMARK:-0.95}"
EVICTION_POLICY="${EVICTION_POLICY:-Bidaw}"
mkdir -p "${DEMO_DIR}/logs" "$LMCACHE_L2_PATH"

export LMCACHE_LOG_LEVEL="${LMCACHE_LOG_LEVEL:-INFO}"
printf '[lmcache] chunk=256 tokens, padded_tail=%s, bidaw_capacity=%s keys, L2=%s\n' \
    "$BIDAW_PAD_TAIL" "$BIDAW_L1_CAPACITY_KEYS" "$LMCACHE_L2_PATH"

# All arms use the same patches/watermark. Policy and DQ are independent.
exec python3 -m bidaw.lmcache_server_entry server \
    --host 0.0.0.0 --port 5555 \
    --http-host 0.0.0.0 --http-port 8080 \
    --chunk-size 256 \
    --l1-size-gb "$L1_GB" \
    --eviction-policy "$EVICTION_POLICY" \
    --l2-prefetch-policy retain \
    --l2-adapter "{\"type\":\"fs\",\"base_path\":\"$LMCACHE_L2_PATH\"}"
