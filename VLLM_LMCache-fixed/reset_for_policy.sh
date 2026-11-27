#!/usr/bin/env bash
# 参数化重置: ./reset_for_policy.sh <Bidaw|LRU|ARC|l1_only> [dq=1|0]
#   第 2 参数控制 Bidaw 双队列调度 (BIDAW_DUAL_QUEUE, 默认 1=开)
# stop -> 清理当前尾块模式的 L2 -> 起 Server(指定策略, L1=1GiB) -> 重启 vLLM
set -e
POLICY="${1:-${POLICY:-Bidaw}}"
DQ="${2:-1}"
DEMO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${DEMO_DIR}/lmcache_geometry.sh"
cd "${DEMO_DIR}"
./stop_all.sh
sleep 3
if [[ "$POLICY" == l1_only || "${DISABLE_L2:-0}" == 1 ]]; then
    DQ=0
else
    rm -f "${LMCACHE_L2_PATH}"/*.data
fi
mkdir -p logs
EVICTION_POLICY="$POLICY" BIDAW_DUAL_QUEUE="$DQ" L1_GB="${L1_GB:-1}" \
    setsid nohup ./start_lmcache_server.sh \
    > "logs/lmcache_server_${POLICY}_dq${DQ}.log" 2>&1 < /dev/null &
disown
sleep 8
for i in $(seq 1 20); do
    curl -sf "http://127.0.0.1:${BIDAW_LMCACHE_HTTP_PORT:-8080}/healthcheck" >/dev/null && break
    sleep 2
done
curl -fsS "http://127.0.0.1:${BIDAW_LMCACHE_HTTP_PORT:-8080}/healthcheck" >/dev/null
echo "server[$POLICY] healthy"
./restart_vllm.sh
