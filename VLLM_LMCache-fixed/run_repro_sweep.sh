#!/usr/bin/env bash
# Target-server experiment. Resets services and clears this demo's L2 .data files.
# All arms use the same warm-up and query schedule. The report compares actual prompts.
set -euo pipefail
DEMO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DEMO_DIR"
source env.sh
mkdir -p logs/repro
WARMUP_SECONDS="${WARMUP_SECONDS:-30}"
REPLAY_PARALLEL="${REPLAY_PARALLEL:-64}"
ANSWER_HISTORY="${ANSWER_HISTORY:-generated}"
case "$ANSWER_HISTORY" in
    generated) label_prefix=repro_v2 ;;
    fixed) label_prefix=repro_v2fixed ;;
    *) echo "ANSWER_HISTORY must be generated or fixed" >&2; exit 2 ;;
esac
MODE="${1:-all}"  # all | baseline | cache

replay() {
    local label="$1" users="$2"
    shift 2
    python3 bidaw/trace_replay.py --trace-dir bidaw/trace_frozen \
        --users "$users" --speed 60 --parallel "$REPLAY_PARALLEL" \
        --warmup-seconds "$WARMUP_SECONDS" --answer-history "$ANSWER_HISTORY" --label "$label" "$@" \
        2>&1 | tee "logs/repro/${label}.log"
}

if [[ "$MODE" == all || "$MODE" == baseline ]]; then
    for U in 35 70 140; do
        ./stop_all.sh
        # Restart independently at every point; fail if readiness fails.
        VLLM_WITHOUT_LMCACHE=1 ./restart_vllm.sh
        replay "${label_prefix}_vllm_u${U}" "$U" --no-cache
    done
fi

if [[ "$MODE" == all || "$MODE" == cache ]]; then
    for POLICY in LRU Bidaw; do
        for DQ in 0 1; do
            for U in 35 70 140; do
                ./reset_for_policy.sh "$POLICY" "$DQ" \
                    > "logs/repro/reset_${POLICY}_dq${DQ}_u${U}.log" 2>&1
                # Store arm metadata; actual HTTP status/config remains auditable.
                curl -fsS http://127.0.0.1:8080/bidaw/status \
                    > "logs/repro/status_${POLICY}_dq${DQ}_u${U}.json"
                EXTRA=()
                if [[ "$POLICY" == Bidaw ]]; then EXTRA+=(--require-answer-policy); fi
                replay "${label_prefix}_${POLICY}_dq${DQ}_u${U}" "$U" "${EXTRA[@]}"
            done
        done
    done
fi
if [[ "$MODE" != all && "$MODE" != baseline && "$MODE" != cache ]]; then
    echo "Usage: $0 [all|baseline|cache]" >&2
    exit 2
fi
python3 bidaw/repro_report.py --pattern "trace_replay_${label_prefix}_*.json"
