# Bidaw 修订版使用说明

基于远程 master 提交 `675dee31510175b633237e407f41d835e03cf88e`。
目标版本：vLLM 0.33.0、LMCache 0.5.7.post1。

## 应用补丁

将下载的 `bidaw-fixes.patch` 放到现有仓库外部，在仓库内执行：

```bash
cd /home/fzz27/vllm-lmcache-demo
git status --short
git apply --check /实际路径/bidaw-fixes.patch
git apply /实际路径/bidaw-fixes.patch
python3 -m unittest discover -s tests -v
```

`git apply --check` 不修改文件；如本地文件与基准不同，先查看差异，勿强制覆盖。
也提供完整源码 ZIP，可解压到新目录使用；模型、虚拟环境、运行日志和 SSD 缓存未包含。
现有环境推荐用补丁，保留原模型与虚拟环境路径。

## 修改内容

1. ghost-cache 的 math 导入；后台异常改为 WARNING，status 暴露错误信息。
2. Fenwick 树计算精确的不同对象重用距离，在线和 ghost 使用同一实现。
   当前单位为等大小 chunk 个数。历史元数据有界，遗忘后重访是删失样本，
   不伪造短距离；内存压缩保持最后访问顺序。
3. 完整对象 identity 包含 model/rank/group/salt/hash，避免跨用户/模型混淆。
4. 只从逻辑 prefetch 提交收集访问信号，包括 L1/L2 未命中；创建、回载、touch
   不训练预测器；同一 spec 内去重。L1 淘汰只删驻留状态，不删访问历史。
5. 回答长度未知时不学习零长度下界；滚动 IQR 过滤异常样本。
   非论文的宽限期默认关闭，`BIDAW_GRACE_SECONDS` 可显式开启做独立实验。
6. 所有缓存策略共用包装入口、水位和 HTTP API，DQ 可独立开关。
   Bidaw 仅替换 L1 策略；L2 保留原策略。chunk 容量按默认模型几何估算。
7. DQ 时间戳为实例字段，在入队/唤醒前原子记录；快速通道不写时间戳。
   派发时删除时间戳，避免完成路径重插入导致的泄漏。
8. 重放改为共享 epoch 的异步会话；并发上限只限制 HTTP，不占满整场会话。
   保留轮间依赖，单轮失败停止后续依赖请求；记录计划/实际发送/完成时间。
9. SHA-256 种子，直接发送 token IDs；跨进程一致且严格保持前缀。
   输出 ignore_eos；用 SSE usage 校验实际输出长度，并上报实际长度。
10. 同一条轨迹前段预热，后段测量，绝不重放未来数据。
    边界暂停准入并等待在途请求/回答上报结束，再取 metrics_before。
11. 报告仅使用测量窗口数据和指标差值，单列 L1/L2 命中率、客户端延迟、错误、
    截断会话数；自动检查各臂轨迹 digest/预热/并发参数一致性，不写死加速结论。
12. 扫描脚本改为四臂消融 + 无缓存基线，所有点使用相同预热配置；失败立即退出。

## 在目标 GPU 服务器验证

首先运行 CPU 回归测试（只依赖 Python 标准库）：

```bash
python3 -m unittest discover -s tests -v
```

再使用现有虚拟环境启动单臂。以下 reset 命令会停止本 demo 服务并删除
`l2_ssd/*.data`，用于建立干净的实验状态：

```bash
source env.sh
./reset_for_policy.sh Bidaw 1
curl -fsS http://127.0.0.1:8080/bidaw/status

# 先验证输入构建；不发送请求
python3 bidaw/trace_replay.py --users 35 --speed 60 \
  --warmup-seconds 30 --dry-run

# 单臂实验，要求回答确实到达至少一个策略实例
python3 bidaw/trace_replay.py --users 35 --speed 60 \
  --parallel 64 --warmup-seconds 30 --label smoke_fixed \
  --require-answer-policy
curl -fsS http://127.0.0.1:8080/bidaw/status
```

检查：logical_requests 持续增长；满足样本阈值后 ghost 非空且 ghost_errors=0；
实际输出 token 数等于设定长度；JSON 中 errors=0；客户端 scheduling_lag 未明显
主导结果。如果所有用户都在预热阶段到达，测量期新用户到达率可能为零，这反映
所选轨迹，不应改成固定数字。扩大采样时间窗/改选用户后重新构建负载。

完整扫描（包含服务重启、清除 demo 的 L2 缓存，耗时较长）：

```bash
WARMUP_SECONDS=30 REPLAY_PARALLEL=64 bash run_repro_sweep.sh
python3 bidaw/repro_report.py
```

结果写入 `logs/trace_replay_repro_v2_*.json` 和 `logs/repro_v2_report.txt`。
旧报告脚本 `compare_arms.py` 仅保留用于历史结果，不用于新版复现实验。

## 验证范围与剩余限制

- 本次在无 GPU 环境通过 CPU 回归、Python 编译、Shell 语法及补丁可应用检查。
  测试中 LMCache 接口使用替身；这不等于已通过真实 CUDA/ZMQ/HTTP 集成测试。
  必须在原服务器上完成上述单臂验证，再开展性能结论比较。
- DQ 调整的是 LMCache 预取准入；未修改 vLLM GPU scheduler，不能宣称已完整复现
  论文的 GPU 就绪队列顺序。应进一步采集 vLLM 的请求准入和 GPU 开始计算时间。
- 负载是长度驱动的合成 token 前缀：历史回答使用固定填充 token，不回填模型实际
  输出。这保证各臂输入完全相同，适合缓存/I/O 微基准，不代表真实语义多轮对话。
- 当前只适用于同模型、同大小 BF16 chunk 的实验。默认 Qwen2.5-1.5B、256 token
  chunk 为 7,340,032 bytes；切换模型/dtype/chunk/并行度时，显式修改
  BIDAW_CHUNK_BYTES 或 BIDAW_L1_CAPACITY_KEYS。不同大小对象需要字节加权扩展。
- ghost 采用有限历史窗口，边缘处未来信息不完整；极端区间阈值仍是
  BIDAW_EXTREME_MULT×capacity 的可配置启发式，不是论文阈值的严格复现。
  未观测到的潜力桶使用先验值，应结合 ghost 状态解释结果。
- 下界是观测统计，不是对所有用户都成立的数学保证；有限样本和行为变化会影响预测。
  在线事件来自 prefetch 提交，若未来连接器对同一轮重复提交，需要加入稳定 round ID
  做跨 spec 去重。本配置关闭 vLLM 前缀缓存，不评估 GPU 命中前缀漏采的情况。
- 未实现 SSD 满容量管理的新算法；沿用 LMCache 原有 L2 配置。
- 不承诺加速比。修复前 README 的历史结果不能作为修复版的性能证据。
