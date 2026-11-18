> Bidaw 修订版以 [FIXES.md](FIXES.md) 为准；下方原始部署/前缀复用说明保留作参考，历史实测不代表修复版验证。

# vLLM + LMCache(独立 Server) 部署与 KV Cache 复用验证

本目录包含在单机 (2x RTX 4080 32GB, CUDA 13.0 驱动, Python 3.12) 上搭建
**vLLM 0.33.0 + LMCache 0.5.7.post1** 架构的完整脚本:

```
                    ┌──────────────────────────────┐
  test_prefix_      │  vLLM (vllm serve, :8000)    │
  cache.py  ──────► │  kv_transfer_config:         │
                    │   LMCacheMPConnector         │
                    │   kv_role: kv_both           │
                    └──────────┬───────────────────┘
                        ZMQ tcp│:5555  (STORE / LOOKUP / RETRIEVE)
                    ┌──────────▼───────────────────┐
                    │  独立 LMCache Server          │
                    │  (lmcache server, DRAM L1)    │
                    │  HTTP 管理 API :8080          │
                    └──────────────────────────────┘
```

## 目录内容

| 文件 | 作用 |
|---|---|
| `env.sh` | 激活 venv 与公共环境变量 |
| `start_lmcache_server.sh` | 启动独立 LMCache Server (ZMQ :5555 / HTTP :8080, L1=8GB DRAM) |
| `start_vllm.sh` | 启动 vLLM, 通过 `LMCacheMPConnector` 连接 LMCache |
| `test_basic_inference.sh` | 步骤 1 冒烟: 验证 vLLM 能正常加载模型并推理 |
| `test_prefix_cache.py` | 步骤 4 验证: 连续发送相同前缀请求, 验证 KV Cache 复用 |
| `stop_all.sh` | 停止两个服务 |
| `logs/` | 各服务日志 |
| `models/` | 下载的模型 (Qwen2.5-1.5B-Instruct) |

## 第 0 步: 安装 (一次性)

本机网络说明: pypi.org / huggingface.co 直连不稳定, 实测可用通道:
* pip → 清华源 `https://pypi.tuna.tsinghua.edu.cn/simple`;
* 模型下载 → 本机代理 `http://127.0.0.1:17890`(clash 类端口) + 官方 HF;
  若代理不可用, 退化为 `HF_ENDPOINT=https://hf-mirror.com` 直连重试。

```bash
python3 -m venv /home/fzz27/venvs/vllm-lmcache
/home/fzz27/venvs/vllm-lmcache/bin/pip install vllm==0.33.0 lmcache==0.5.7.post1 \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

# 下载模型 (走本机代理; 仅在下载命令上挂代理, 不写入全局环境)
export HTTPS_PROXY=http://127.0.0.1:17890 HTTP_PROXY=http://127.0.0.1:17890
/home/fzz27/venvs/vllm-lmcache/bin/hf download Qwen/Qwen2.5-1.5B-Instruct \
    --local-dir /home/fzz27/vllm-lmcache-demo/models/Qwen2.5-1.5B-Instruct
unset HTTPS_PROXY HTTP_PROXY
```

> 版本说明: LMCache 0.5.x 是 2026/04 发布的全新多进程 (MP) 架构,
> 它不再把 vLLM 声明为 pip 依赖, 由用户自行安装并配对;
> vLLM 侧通过 `LMCacheMPConnector` 与独立 Server 通信。

## 第 1 步: 安装 vLLM 并确认能加载模型正常推理

```bash
cd /home/fzz27/vllm-lmcache-demo
source env.sh
./start_vllm.sh          # 前台运行; 或: nohup ./start_vllm.sh > logs/vllm.log 2>&1 &
tail -f logs/vllm.log    # 等待出现 "Application startup complete"

./test_basic_inference.sh   # 发一条请求验证推理
```

## 第 2 步: 安装 LMCache 并启动独立 LMCache Server

另开一个终端:

```bash
cd /home/fzz27/vllm-lmcache-demo
source env.sh
./start_lmcache_server.sh &     # 或 nohup ./start_lmcache_server.sh > logs/lmcache_server.log 2>&1 &

curl -s http://127.0.0.1:8080/healthcheck; echo    # 应返回 {"status":"healthy"}
curl -s http://127.0.0.1:8080/status | head -c 500; echo
```

## 第 3 步: vLLM 通过 LMCacheMPConnector 连接 LMCache

`start_vllm.sh` 中的关键配置 (已写好):

```bash
vllm serve models/Qwen2.5-1.5B-Instruct \
  --disable-prefix-caching \
  --kv-transfer-config '{
      "kv_connector": "LMCacheMPConnector",
      "kv_role": "kv_both",
      "kv_connector_extra_config": {
          "lmcache.mp.server_urls": "tcp://localhost:5555"
      }
  }'
```

* `LMCacheMPConnector`: LMCache 多进程架构的 vLLM KV 连接器,
  由调度器/工作进程经 ZMQ (`tcp://`) 或 gRPC (`grpc://`) 访问独立 Server;
* `kv_role: kv_both`: vLLM 既作为 KV 生产者 (STORE) 也作为消费者 (LOOKUP/RETRIEVE);
* `lmcache.mp.server_urls`: 独立 Server 地址列表 (可多机, 逗号分隔);
* 其他可选: `lmcache.mp.save_decode_cache`(默认只存 prompt 块),
  `lmcache.mp.eager_prefetch`, `lmcache.mp.autostart` 等;
* `--disable-prefix-caching`: 关闭 vLLM 自带 GPU 前缀缓存,
  使"第二次请求变快"只能来自 LMCache, 验证结论才严格。

启动后可在 `logs/vllm.log` 中看到 `Resolved LMCache MP geometry ...` 日志,
表示连接器已连上独立 Server。

## 第 4 步: 连续发送相同前缀请求, 验证 KV Cache 复用

```bash
cd /home/fzz27/vllm-lmcache-demo
source env.sh
python3 test_prefix_cache.py
```

脚本流程:
1. **COLD**: 发送 ~4800 token 公共前缀 + 问题 A (全量 prefill, 结束后异步 STORE 到 LMCache);
2. 轮询 `GET :8080/metrics` 的 `finished_stores_total` 直到 STORE 完成 (硬证据);
3. **WARM**: 相同前缀 + 问题 B → LMCache LOOKUP 命中 → RETRIEVE 回 KV → TTFT 大幅下降;
4. **WARM2**: 重放与第 1 步完全相同的请求, 复测加速;
5. 打印 TTFT 对比、`/metrics` 命中指标, 并给出 PASS/FAIL 判定。
   (每次运行的前缀带随机 run-id, 避免上一次运行的缓存污染 COLD 基线。)

**本机实测结果 (2026-11-13, RTX 4080 32GB, Qwen2.5-1.5B-Instruct, ~4800 token 前缀):**

```
COLD  TTFT (首次, 全量 prefill)   :   170.0 ms
WARM  TTFT (同前缀+新问题)        :    94.8 ms     <- 1.79x
WARM2 TTFT (与首次完全相同的请求) :    96.7 ms     <- 1.76x
```

同时 LMCache Server `/metrics` 给出硬证据 (复用确实来自独立 Server):

```
lmcache_mp_num_finished_stores_total     3 -> 5     # KV 已存入独立 Server
lmcache_mp_num_finished_retrieves_total  ...        # 每次 WARM 请求 +1 (CPU→GPU 检索)
lmcache_mp_lookup_hit_l1_tokens_total    6144+      # 命中的 token 全部来自 Server 的 L1(DRAM)
```

**PASS 判定标准**: WARM/WARM2 的 TTFT < COLD 的 60%, 且 metrics 中 STORE 计数增长。
(注意: 本机 GPU 同时跑着其他任务, TTFT 绝对值有噪声; 前缀越长收益越大——
若把前缀从 ~2000 token 加到 ~5000 token, 加速比会从 1.2x 提升到 1.8x。)

### (可选) 跨重启复用 — 独立 Server 的最强证明 (本机已实测)

1. `python3 test_prefix_cache.py` 跑完一次 (KV 已在独立 Server 中);
2. 重启 vLLM: `./stop_all.sh && ./start_vllm.sh > logs/vllm2.log 2>&1 &`
   (GPU 显存中的 KV 全部消失, 独立 Server 中的 KV 仍在);
3. 直接重放 WARM 请求, 观察 `/metrics`:

   ```
   finished_retrieves_total: 2.0 -> 3.0     # 新实例确实从 Server 拉取了 KV
   lookup_hit_tokens_total : 1536.0 -> 2304.0
   ```
   新实例的**首次**请求即命中 (TTFT 包含进程预热噪声, 约几百 ms,
   对比同一新实例上全新前缀的 COLD 值即可看出收益) → KV 跨进程/跨重启复用成功。

## 手工验证命令 (速查)

```bash
# LMCache 侧: 缓存对象与指标
curl -s http://127.0.0.1:8080/cache/objects | python3 -m json.tool | head
curl -s http://127.0.0.1:8080/metrics | grep -Ei 'hit|lookup|retrieve|store'

# 直接发一条相同前缀请求 (两次对比 TTFT)
curl -s http://127.0.0.1:8000/v1/completions -H 'Content-Type: application/json' -d '{
  "model": "qwen2.5-1.5b-instruct",
  "prompt": "The LMCache system stores the key-value cache...<长前缀>...Question: hi\nAnswer:",
  "max_tokens": 16 }'
```

## Bidaw 修订版

详见 [FIXES.md](FIXES.md)：包含应用方式、CPU 回归测试、GPU 单臂验证和四臂扫描命令。

已修复 ghost 重建、WRD 统计、访问生命周期混淆、统一时钟、预热数据泄漏、
各臂水位不一致、随机负载不一致和 DQ 时间戳问题。旧实验结论已撤下，需修复后重测。
重放器是长度驱动的合成缓存微基准；当前不宣称完整复现论文或保证性能提升。

```bash
python3 -m unittest discover -s tests -v
# 仅在 GPU 服务器运行；会重启服务并清除本 demo 的 L2 .data 缓存
WARMUP_SECONDS=30 bash run_repro_sweep.sh
```
