#!/usr/bin/env bash
# Isolated two-turn Qwen3.5 STORE/RETRIEVE verification on a single GPU.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

model_dir=${1:-"$PWD/models/Qwen3.5-0.8B"}
model=qwen3.5-0.8b-smoke
chunk=${HYBRID_CHUNK_SIZE:-544}
gpu_memory=${HYBRID_GPU_MEMORY_UTILIZATION:-0.5}
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

source ./env.sh
python3 - "$model_dir" <<'PY'
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import socket
import sys

for package, expected in (("vllm", "0.33.0"), ("lmcache", "0.5.7.post1")):
    try:
        installed = version(package)
    except PackageNotFoundError:
        installed = "missing"
    if installed != expected:
        sys.exit(f"{package} {expected} is required (found {installed}).")

import torch

if torch.cuda.device_count() != 1:
    sys.exit("Select one visible NVIDIA GPU using CUDA_VISIBLE_DEVICES.")
config = Path(sys.argv[1]) / "config.json"
if not config.is_file():
    sys.exit(f"Local Qwen3.5 model is required: {config}")
for port in (5555, 8080, 8000):
    with socket.socket() as sock:
        try:
            sock.bind(("127.0.0.1", port))
        except OSError as exc:
            sys.exit(f"Local port {port} must be free: {exc}")
PY

mkdir -p logs/hybrid_smoke
run=$(mktemp -d "$PWD/logs/hybrid_smoke/run.XXXXXXXX")
printf 'Smoke artifacts: %s\n' "$run"
lmcache_pid=
vllm_pid=
stop_group() {
    local pid="$1" i
    [[ -n "$pid" ]] || return 0
    kill -TERM -- "-$pid" 2>/dev/null || true
    for ((i=0; i<20; i++)); do
        kill -0 -- "-$pid" 2>/dev/null || break
        sleep 0.5
    done
    kill -KILL -- "-$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
}
cleanup() {
    stop_group "$vllm_pid"
    stop_group "$lmcache_pid"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

wait_ready() {
    local url="$1" pid="$2" log="$3" i
    for ((i=0; i<150; i++)); do
        if ! kill -0 "$pid" 2>/dev/null; then
            tail -60 "$log" >&2
            echo "Service exited. See $log" >&2
            return 1
        fi
        if curl -fs --max-time 2 "$url" >/dev/null; then
            return 0
        fi
        sleep 2
    done
    tail -60 "$log" >&2
    echo "Timed out waiting for $url. See $log" >&2
    return 1
}

setsid python3 -m bidaw.hybrid_smoke_probe "$run/evidence.jsonl" server \
    --host 127.0.0.1 --port 5555 --http-host 127.0.0.1 --http-port 8080 \
    --chunk-size "$chunk" --separate-object-groups --l1-size-gb 2 --eviction-policy LRU \
    >"$run/lmcache.log" 2>&1 &
lmcache_pid=$!
wait_ready http://127.0.0.1:8080/healthcheck "$lmcache_pid" "$run/lmcache.log"

connector='{"kv_connector":"LMCacheMPConnector","kv_role":"kv_both","kv_connector_extra_config":{"lmcache.mp.server_urls":"tcp://127.0.0.1:5555","lmcache.mp.mp_transfer_mode":"lmcache_driven"}}'
VLLM_SERVER_DEV_MODE=1 setsid vllm serve "$model_dir" --served-model-name "$model" \
    --host 127.0.0.1 --port 8000 --language-model-only --dtype bfloat16 \
    --max-model-len 4096 --max-num-seqs 2 --gpu-memory-utilization "$gpu_memory" \
    --enable-prefix-caching --mamba-cache-mode align --kv-transfer-config "$connector" \
    >"$run/vllm.log" 2>&1 &
vllm_pid=$!
wait_ready http://127.0.0.1:8000/v1/models "$vllm_pid" "$run/vllm.log"

python3 -u test_prefix_cache.py --model "$model" --model-dir "$model_dir" \
    --hybrid-evidence "$run/evidence.jsonl" | tee "$run/smoke.log"
