#!/usr/bin/env bash
# 步骤 1 冒烟测试: 确认 vLLM 已加载模型并能正常推理
set -euo pipefail
PORT="${VLLM_PORT:-8000}"
echo "[1] 模型列表:"
curl -s "http://127.0.0.1:${PORT}/v1/models" | python3 -m json.tool
echo "[2] 简单推理:"
curl -s "http://127.0.0.1:${PORT}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{
        "model": "qwen2.5-1.5b-instruct",
        "messages": [{"role": "user", "content": "用一句话介绍 LMCache 是什么。"}],
        "max_tokens": 64,
        "temperature": 0
      }' | python3 -c "import json,sys; r=json.load(sys.stdin); print(r['choices'][0]['message']['content'])"
echo "OK: vLLM 推理正常"