#!/usr/bin/env bash
# Run cold-start arms on a dedicated, single-GPU experiment host.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
users=${GANDALF_USERS:-70}
speed=${GANDALF_SPEED:-20}
parallel=${GANDALF_PARALLEL:-64}
repeats=${GANDALF_REPEATS:-1}
l1_gb=${GANDALF_L1_GB:-1}
l2_bytes=${GANDALF_L2_BYTES:-2147483648}
seed=${GANDALF_SEED:-7}
cache_root=${GANDALF_CACHE_ROOT:-"$PWD/l2_ssd/gandalf_benchmarks"}
run_parent=${GANDALF_RUN_ROOT:-"$PWD/logs/gandalf_benchmarks"}
trace_dir=${GANDALF_TRACE_DIR:-"$PWD/bidaw/trace_frozen"}
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export BIDAW_PAD_TAIL=0
export BIDAW_EVICTION_WATERMARK="${BIDAW_EVICTION_WATERMARK:-0.95}"
export VLLM_WITHOUT_LMCACHE=0
unset BIDAW_FORCE_POLICY
unset POLICY
export GANDALF_PARALLEL="$parallel"
export GANDALF_HYBRID="${GANDALF_HYBRID:-0}"
model_dir=models/Qwen2.5-1.5B-Instruct
replay_model_args=()
if [[ "$GANDALF_HYBRID" == 1 ]]; then
    model_dir=models/Qwen3.5-0.8B
    replay_model_args=(--model qwen3.5-0.8b --model-dir "$model_dir"
                       --max-model-len 4096 --max-prompt-tokens 3800)
fi

read -r -a arms <<< "${GANDALF_ARMS:-lru lru_writeback bidaw gandalf l1_only}"
declare -A selected=()
with_l2=0
for arm in "${arms[@]}"; do
    case "$arm" in
        lru|lru_writeback|bidaw|gandalf) with_l2=1 ;;
        l1_only) ;;
        *) echo "Unknown GANDALF_ARMS entry: $arm" >&2; exit 2 ;;
    esac
    if [[ -v "selected[$arm]" ]]; then
        echo "Duplicate GANDALF_ARMS entry: $arm" >&2
        exit 2
    fi
    selected[$arm]=1
done
if [[ "${1:-}" == "--plan" ]]; then
    printf 'users=%s speed=%s parallel=%s repeats=%s L1_GiB=%s L2_bytes=%s seed=%s\n' \
        "$users" "$speed" "$parallel" "$repeats" "$l1_gb" "$l2_bytes" "$seed"
    printf 'arms=%s trace_dir=%s\n' "${arms[*]}" "$trace_dir"
    exit 0
fi
if [[ $# != 0 || ${#arms[@]} == 0 || ! "$repeats" =~ ^[1-9][0-9]*$ ]]; then
    echo "Usage: $0 [--plan] (GANDALF_REPEATS must be a positive integer)" >&2
    exit 2
fi

source ./env.sh
python3 - "$users" "$speed" "$parallel" "$l1_gb" "$l2_bytes" "$model_dir" "$with_l2" <<'PY'
import math
import socket
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import sys

users, speed, parallel, dram, ssd, model, with_l2 = sys.argv[1:]
if (min(int(users), int(parallel)) <= 0 or min(float(speed), float(dram)) <= 0
        or not all(math.isfinite(float(x)) for x in (speed, dram))
        or (with_l2 == "1" and int(ssd) < 7340032)):
    sys.exit("Invalid workload or capacity; L2 must fit at least one full KV chunk.")
try:
    pinned = version("lmcache") == "0.5.7.post1" and version("vllm") == "0.33.0"
except PackageNotFoundError:
    pinned = False
if not pinned:
    sys.exit("Install the repository's pinned LMCache and vLLM versions.")

import torch

if torch.cuda.device_count() != 1:
    sys.exit("Select one available GPU using CUDA_VISIBLE_DEVICES.")
if not (Path(model) / "config.json").is_file():
    sys.exit(f"The local model is required: {model}")
for port in (5555, 8080, 8000):
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(("0.0.0.0", port))
        except OSError as exc:
            sys.exit(f"Port {port} must be available before starting the benchmark: {exc}")
PY

mkdir -p "$run_parent"
run_root=$(mktemp -d "${run_parent}/run.XXXXXXXX")
cache_run=
if [[ "$with_l2" == 1 ]]; then
    mkdir -p "$cache_root"
    cache_run=$(mktemp -d "${cache_root}/run.XXXXXXXX")
fi
printf 'Reports: %s\nL2 directories: %s\n' "$run_root" "${cache_run:-disabled}"
lmcache_pid=
vllm_pid=
stop_process() {
    local pid="$1" i
    [[ -n "$pid" ]] || return 0
    kill -TERM -- "-$pid" 2>/dev/null || true
    for ((i=0; i<60; i++)); do
        kill -0 -- "-$pid" 2>/dev/null || break
        sleep 0.5
    done
    kill -KILL -- "-$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
}
cleanup() {
    stop_process "$vllm_pid"
    stop_process "$lmcache_pid"
    vllm_pid=
    lmcache_pid=
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

wait_ready() {
    local url="$1" pid="$2" log="$3" i
    for ((i=0; i<180; i++)); do
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "Service exited. See $log" >&2
            return 1
        fi
        if curl -fs --max-time 2 "$url" >/dev/null; then
            return 0
        fi
        sleep 2
    done
    echo "Startup timeout. See $log" >&2
    return 1
}

wait_for_writes() {
    python3 - "$1" <<'PY'
import json
from pathlib import Path
import re
import sys
import time
from urllib.request import urlopen

out = Path(sys.argv[1])
last = None
quiet_since = time.monotonic()
deadline = quiet_since + 120
while time.monotonic() < deadline:
    with urlopen("http://127.0.0.1:8080/bidaw/status", timeout=5) as response:
        state = json.load(response)
    with urlopen("http://127.0.0.1:8080/metrics", timeout=5) as response:
        metrics = response.read().decode()
    staging = re.findall(r"^lmcache_mp_l1_staging_bytes(?:\{[^}]*\})?\s+(\S+)", metrics, re.M)
    tier = state["tier_policy"]
    counters = tuple(tier.get(k, 0) for k in (
        "writes", "migrations", "reused_ssd_copies", "drops", "failed_migrations",
        "failed_writes", "blocked_migrations", "eviction_failures",
    ))
    busy = (any(tier.get(k, 0) for k in ("pending_keys", "in_flight_stores", "active_evictions"))
            or any(float(value) for value in staging))
    if busy or last != counters:
        quiet_since = time.monotonic()
        last = counters
    if time.monotonic() - quiet_since >= 3:
        out.write_text(json.dumps(state, indent=2) + "\n")
        break
    time.sleep(0.25)
else:
    sys.exit("Timed out waiting for LMCache stores and migrations to drain.")
PY
}

run_arm() {
    local repeat="$1" arm="$2"
    local policy=LRU dq=0 mode=write_through disable_l2=0
    local out="$run_root/$repeat/$arm" cache=
    local extra=()
    case "$arm" in
        lru) ;;
        lru_writeback) mode=lru_writeback ;;
        bidaw) policy=Bidaw; dq=1; extra+=(--require-answer-policy) ;;
        gandalf) mode=gandalf ;;
        l1_only) disable_l2=1; mode=l1_only ;;
    esac
    mkdir -p "$out"
    if [[ "$disable_l2" == 0 ]]; then
        cache="$cache_run/$repeat/$arm"
        mkdir -p "$cache"
    fi
    printf '\n%s/%s\n' "$repeat" "$arm"
    EVICTION_POLICY="$policy" BIDAW_DUAL_QUEUE="$dq" L1_GB="$l1_gb" DISABLE_L2="$disable_l2" \
        GANDALF_TIER_MODE="$mode" GANDALF_L2_BYTES="$l2_bytes" BIDAW_L2_DIR="$cache" \
        BIDAW_L2_EVENT_LOG="$out/l2_events.jsonl" \
        setsid bash ./start_lmcache_server.sh >"$out/lmcache.log" 2>&1 &
    lmcache_pid=$!
    wait_ready http://127.0.0.1:8080/healthcheck "$lmcache_pid" "$out/lmcache.log"
    curl -fsS http://127.0.0.1:8080/bidaw/status >"$out/status-before.json"

    BIDAW_VLLM_TIMING_LOG="$out/vllm_timing.jsonl" setsid bash ./start_vllm.sh >"$out/vllm.log" 2>&1 &
    vllm_pid=$!
    wait_ready http://127.0.0.1:8000/v1/models "$vllm_pid" "$out/vllm.log"
    curl -fsS http://127.0.0.1:8000/metrics >"$out/vllm-metrics-before.txt"

    python3 -m bidaw.trace_replay --trace-dir "$trace_dir" --users "$users" --speed "$speed" --parallel "$parallel" \
        --seed "$seed" --warmup-seconds 0 --answer-history fixed --track-chunk-fates \
        --label "$arm" --out "$out/replay.json" "${replay_model_args[@]}" "${extra[@]}" | tee "$out/replay.log"

    wait_for_writes "$out/status-after.json"
    curl -fsS http://127.0.0.1:8000/metrics >"$out/vllm-metrics-after.txt"
    cleanup
}

for ((repeat=0; repeat<repeats; repeat++)); do
    for ((slot=0; slot<${#arms[@]}; slot++)); do
        run_arm "repeat-$((repeat + 1))" "${arms[(slot + repeat) % ${#arms[@]}]}"
    done
done

python3 -m bidaw.gandalf_report "$run_root" | tee "$run_root/comparison.txt"
