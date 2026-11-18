#!/bin/bash
# 重启 vLLM 并等待就绪 (Bidaw 验证用)
DEMO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
pkill -f 'vllm serve' 2>/dev/null
pkill -f 'VLLM::EngineCore' 2>/dev/null
pkill -f 'EngineCore' 2>/dev/null
pkill -f 'from vllm' 2>/dev/null
sleep 10
cd "${DEMO_DIR}"
setsid nohup "${DEMO_DIR}/start_vllm.sh" \
    > "${DEMO_DIR}/logs/vllm_bidaw.log" 2>&1 < /dev/null &
disown
for i in $(seq 1 40); do
    if grep -aq 'Application startup complete' \
        "${DEMO_DIR}/logs/vllm_bidaw.log" 2>/dev/null; then
        echo VLLM_STARTED
        exit 0
    fi
    if grep -aq 'EngineCore failed to start' \
        "${DEMO_DIR}/logs/vllm_bidaw.log" 2>/dev/null; then
        echo VLLM_FAILED
        exit 1
    fi
    sleep 10
done
echo VLLM_TIMEOUT
exit 1