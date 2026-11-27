# 公共环境: source 本文件后再运行其他脚本
# 路径解析: repo 可位于任意目录; venv 按以下顺序探测
#   1) $BIDAW_VENV 环境变量
#   2) repo 内 ./venv          (bootstrap.sh 创建)
#   3) ../venvs/vllm-lmcache   (合并目录布局: venvs 与 repo 相邻)
#   4) ~/venvs/vllm-lmcache    (本机历史位置)
export VLLM_PORT="${VLLM_PORT:-8000}"
export BIDAW_LMCACHE_PORT="${BIDAW_LMCACHE_PORT:-5555}"
export BIDAW_LMCACHE_HTTP_PORT="${BIDAW_LMCACHE_HTTP_PORT:-8080}"
export HF_ENDPOINT=https://hf-mirror.com
export LMCACHE_LOG_LEVEL="${LMCACHE_LOG_LEVEL:-INFO}"

_DEMO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -z "${BIDAW_VENV:-}" ]; then
    for _cand in "${_DEMO_DIR}/venv" "${_DEMO_DIR}/../venvs/vllm-lmcache" \
                 "${HOME}/venvs/vllm-lmcache"; do
        if [ -x "${_cand}/bin/python3" ]; then
            BIDAW_VENV="${_cand}"
            break
        fi
    done
fi
BIDAW_VENV="${BIDAW_VENV:-${_DEMO_DIR}/venv}"
export BIDAW_VENV
# shellcheck disable=SC1091
source "${BIDAW_VENV}/bin/activate"
echo "[env] venv=${BIDAW_VENV}  vllm=$(python3 -c 'import vllm;print(vllm.__version__)' 2>/dev/null)  lmcache=$(python3 -c 'import lmcache;print(getattr(lmcache,"__version__","installed"))' 2>/dev/null)"