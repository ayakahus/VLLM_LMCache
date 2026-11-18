#!/usr/bin/env bash
# 停止 LMCache Server 与 vLLM
pkill -f 'lmcache server' 2>/dev/null && echo 'stopped: lmcache server' || echo 'lmcache server: not running'
pkill -f 'bidaw.lmcache_server_entry' 2>/dev/null && echo 'stopped: bidaw server entry' || true
pkill -f 'vllm serve' 2>/dev/null && echo 'stopped: vllm serve' || echo 'vllm serve: not running'
pkill -f 'VLLM::EngineCore' 2>/dev/null || true
pkill -f 'from vllm' 2>/dev/null || true
exit 0