#!/usr/bin/env bash
# =====================================================================
# 启动 vLLM, 通过 LMCacheMPConnector 连接独立的 LMCache Server
#
# 关键配置:
#   --kv-transfer-config '{
#       "kv_connector": "LMCacheMPConnector",
#       "kv_role": "kv_both",              # 既生产(存)也消费(取) KV
#       "kv_connector_extra_config": {
#           "lmcache.mp.server_urls": "tcp://localhost:5555"
#       }
#   }'
#
# 说明:
#   * --disable-prefix-caching 关闭 vLLM 自带的 GPU 侧前缀缓存,
#     这样第二次请求的加速只能来自 LMCache(独立 Server), 验证才严格。
#   * 本机 GPU 正被其他任务占用, 用 util=0.30 只申请约 9.8GB。
#   * CUDA_VISIBLE_DEVICES=1 使用空闲显存稍多的一张卡。
# =====================================================================
set -euo pipefail

DEMO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${DEMO_DIR}/env.sh"

MODEL_DIR="${DEMO_DIR}/models/Qwen2.5-1.5B-Instruct"
PORT="${VLLM_PORT:-8000}"

mkdir -p "${DEMO_DIR}/logs"

# 论文基线: vLLM 不挂任何 KV 缓存 (全部历史重算)。设 VLLM_WITHOUT_LMCACHE=1 启用。
# 注意: JSON 不能含空格 (无引号展开会被 word-split)。
KV_ARGS=''
if [ "${VLLM_WITHOUT_LMCACHE:-0}" != "1" ]; then
  KV_ARGS="--kv-transfer-config {\"kv_connector\":\"LMCacheMPConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"lmcache.mp.server_urls\":\"tcp://localhost:5555\",\"lmcache.mp.save_decode_cache\":true}}"
fi

echo "[start_vllm] serving ${MODEL_DIR} on port ${PORT} (GPU=${CUDA_VISIBLE_DEVICES:-1} lmcache=$([ -n "$KV_ARGS" ] && echo on || echo OFF))"

exec vllm serve "${MODEL_DIR}" \
    --served-model-name qwen2.5-1.5b-instruct \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --dtype bfloat16 \
    --max-model-len 6144 \
    --gpu-memory-utilization 0.30 \
    --disable-prefix-caching \
    ${KV_ARGS}