#!/usr/bin/env bash
# 一键环境搭建 (新机器/新 clone):
#   1. 创建 venv 并安装锁定版本依赖 (清华源)
#   2. 下载 Qwen2.5-1.5B-Instruct 模型 (HF 镜像)
#   3. 下载 Bidaw FAST'26 官方轨迹 (可选; 仓库已含冻结快照可跳过)
# 用法: bash bootstrap.sh [--skip-model] [--skip-trace]
set -euo pipefail
DEMO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${DEMO_DIR}"

PY="${PYTHON:-python3}"
echo "[1/3] 创建 venv: ${DEMO_DIR}/venv (${PY})"
if [ ! -x venv/bin/python3 ]; then
    "${PY}" -m venv venv
fi
# shellcheck disable=SC1091
source venv/bin/activate
pip install -U pip
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

if [ "${1:-}" != "--skip-model" ] && [ "${2:-}" != "--skip-model" ]; then
    echo "[2/3] 下载模型 Qwen2.5-1.5B-Instruct (~2.9GB, HF 镜像)"
    if [ ! -f models/Qwen2.5-1.5B-Instruct/model.safetensors ]; then
        mkdir -p models
        hf download Qwen/Qwen2.5-1.5B-Instruct \
            --local-dir models/Qwen2.5-1.5B-Instruct \
            || echo "!! 模型下载失败: 检查网络/代理后重跑 (README 第 0 步)"
    else
        echo "    已存在, 跳过"
    fi
fi

if [ "${1:-}" != "--skip-trace" ] && [ "${2:-}" != "--skip-trace" ]; then
    echo "[3/3] 下载 Bidaw 官方轨迹 (7 个 part, ~22MB)"
    mkdir -p bidaw/trace
    for i in 1 2 3 4 5 6 7; do
        f="bidaw/trace/total_traces_part${i}.txt"
        if [ ! -s "$f" ]; then
            curl -sL --retry 3 -o "$f" \
                "https://raw.githubusercontent.com/ShipengHu-777/Interactive-conversation-workload/main/total_workload/total_traces_part${i}.txt" \
                || echo "!! part${i} 下载失败 (可跳过: 实验用 trace_frozen 已随仓库)"
        fi
    done
    echo "    完成 (部分失败不影响: trace_frozen 快照已随仓库)"
fi

echo
echo "bootstrap 完成。下一步:"
echo "  source env.sh && ./test_basic_inference.sh   # 先起服务: ./start_vllm.sh ..."