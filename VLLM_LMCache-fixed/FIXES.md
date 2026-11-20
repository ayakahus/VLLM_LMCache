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
2. 论文 §3.3.1 的 WRD 按用户轮次记录，使用其他不同用户最近一次访问的历史 KV
   字节数之和。Fenwick 树与按字节限额的 ghost 共用同一距离定义。有限历史中
   被遗忘的用户再次访问时，跳过该次距离样本。
3. 无 salt 的共享前缀按 model/rank/group/hash 去重并计入全局压力，淘汰时沿用
   共享块的候选顺序。
4. 服务端在 MP LOOKUP 入口观察轮次，包括没有完整 chunk 时的查找。回放提供
   实际历史 token 数，普通请求缺失此元数据时用提示词总 token 数估算。创建、
   回载和 touch 只更新驻留状态。
5. 按确切的上一轮回答长度维护滚动 IQR 下界，未知答长或样本不足时跳过下界约束。
   式 2 仅清除下界所在桶之前的概率，再对剩余桶归一化。
   非论文的宽限期默认关闭，`BIDAW_GRACE_SECONDS` 可显式开启做独立实验。
6. 所有缓存策略共用包装入口、水位和 HTTP API，DQ 可独立开关。
   Bidaw 仅替换 L1 策略；L2 保留原策略。chunk 容量按默认模型几何估算。
7. 双队列开启时，默认尾块 connector 在请求进入 vLLM 等待队列时启动 LOOKUP。
   已在 L1 的块在等待预取期间持有读锁；需要 SSD 时，按进入 LMCache 的等待时长
   和已预留的实际 L2 读取字节数计算 disk-HRRN，在同一调度周期优先派发高分请求。
   SSD 派发沿用 ABCU 的 preparing 事件关联。L2 读回后由 retain 策略保留在 L1。
8. 重放改为共享 epoch 的异步会话；并发上限只限制 HTTP，不占满整场会话。
   保留轮间依赖，单轮失败停止后续依赖请求；记录计划/实际发送/完成时间。
9. 查询 token IDs 由稳定的 SHA-256 种子生成。通过 vLLM 的 `return_token_ids`
   收集流式回答的实际 token IDs，校验 SSE usage 和 token 数，再原样接入下一轮前缀。
   被跳过的预热轮没有可用回答，同一会话的依赖轮也会跳过。
10. 同一条轨迹前段预热，后段测量，绝不重放未来数据。
    边界暂停准入并等待在途请求/回答上报结束，再取 metrics_before。
11. 报告使用测量窗口数据和指标差值，单列 L1/L2 命中率、客户端延迟、错误、
    跳过轮次及截断会话数。横向对比同时核对计划轨迹、预热及并发参数、实际发送的
    prompt 指纹。模型生成分歧或预热跳过造成的输入差异会被标记。
12. 扫描脚本改为四臂消融 + 无缓存基线，所有点使用相同预热配置；失败立即退出。

## 在目标 GPU 服务器验证

首先运行 CPU 回归测试（只依赖 Python 标准库）：

```bash
python3 -m unittest discover -s tests -v
```

再使用现有虚拟环境启动单臂。以下 reset 命令会停止本 demo 服务并清理当前尾块模式
对应的 SSD 缓存目录，用于建立干净的实验状态：

```bash
source env.sh
./reset_for_policy.sh Bidaw 1
curl -fsS http://127.0.0.1:8080/bidaw/status

# 验证查询 token 和上下文长度规划
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
固定历史的 sweep 使用 `ANSWER_HISTORY=fixed bash run_repro_sweep.sh`，其回放 JSON
使用 `trace_replay_repro_v2fixed_*.json`。旧报告脚本 `compare_arms.py` 仅保留用于历史结果。

### 选择回放的回答历史

- `--answer-history generated`（默认）把模型本轮实际生成的 token 放入后续轮次的历史。
- `--answer-history fixed` 在回放前用相同的 seed、会话 ID 和轨迹回答长度生成固定的合成
  回答 token。每轮仍实际调用模型生成输出并测量延迟，后续 prompt 使用预先固定的 token。
  两组使用相同 trace、seed、tokenizer 和回放参数，并核对实际执行的 prompt 指纹。

例如：

```bash
python3 bidaw/trace_replay.py --users 70 --speed 20 --answer-history fixed --label lru_fixed
```

固定历史下，当模型实际生成的回答与合成历史不同时，下一轮无法直接复用不同 token 所对应的
decode KV。比较命中率和 SSD 写入时，所有实验组应使用同一种回答历史模式。

## 验证范围与剩余限制

- 本次在无 GPU 环境通过 CPU 回归、Python 编译、Shell 语法及补丁可应用检查。
  测试中 LMCache 接口使用替身；这不等于已通过真实 CUDA/ZMQ/HTTP 集成测试。
  必须在原服务器上完成上述单臂验证，再开展性能结论比较。
- 双队列开启时，vLLM 合并扫描两个等待队列的队首，按请求原始到达时间尝试
  GPU 准入。异步 KV 加载完成或请求被跳过后再入队，仍按原时间参与排序。
  显存分配失败的请求留待下一轮，引擎在当前轮继续尝试后续请求。
  GPU 沿用 16-token 物理页。回放通过 HTTP 接口上报上一轮实际回答长度。
- 负载的时间和长度来自轨迹，查询内容为合成 token，历史回答可选实际生成或固定合成。
  该设置可用于验证 decode KV 的跨轮复用。实际命中取决于连接器已保存的完整 KV 块。
- 当前只适用于同模型、同大小 BF16 chunk 的实验。Qwen2.5-1.5B 的 256-token
  chunk 占 7,340,032 bytes，补齐的尾块占用相同容量。其他模型或并行度需要验证布局，
  并显式指定 `BIDAW_CHUNK_BYTES` 或 `BIDAW_L1_CAPACITY_KEYS`。
- ghost 采用有限历史窗口，按下一次访问最远的会话逐个释放容量。它近似评估会话
  驻留概率与变长会话的容量占用。极端区间阈值由 `BIDAW_EXTREME_MULT` 控制，
  缺少样本的潜力桶使用先验值。
- 未实现 SSD 满容量管理的新算法；沿用 LMCache 原有 L2 配置。
- 不承诺加速比。修复前 README 的历史结果不能作为修复版的性能证据。

## 256-token 尾块填充落盘

默认 `BIDAW_PAD_TAIL=1`。LMCache 保持 256 tokens/chunk，vLLM 的物理页为 16 tokens。
请求正常结束时，将已计算的非空尾部补零到 256 tokens，使用独立的填充键提交一个
STORE，随后通过常规 L1 → L2 路径落盘。600 个已计算 tokens 对应两个完整 chunk
和一个包含 88 个有效 tokens 的尾块。恰好对齐时没有额外尾块。

此模式针对本文的单组全注意力 CUDA 模型和 LMCache 驱动的传输。尾部填充只进入存储，
普通推理输入保持原样。常规 MP 查找只匹配完整的真实前缀，补齐尾块通过事件探针单独
计数，实际跨轮复用仍来自已匹配的完整块。尾块传输需要等待当前 CUDA 拷贝完成后清零，
因此会产生额外的传输和存储开销。

- 默认 L2 目录为 `l2_ssd/padded-tail/`。
- 对 vLLM 和 LMCache 同时设置 `BIDAW_PAD_TAIL=0` 可恢复只保存完整块的配置，
  使用原 `l2_ssd/` 目录。切换时一起重启两个服务。
- 启用 L2 事件探针后，报告中的 `padded_tail_writes` 是确认成功落盘的尾块数，
  已包含在 `physical_writes` 中。

## SSD 写入分类（A / B / C / U）

从本目录用以下命令启动 **Bidaw + DQ** 的 LMCache 服务。启动 vLLM 并确认两个服务健康后，再运行回放和报告：

```bash
BIDAW_L2_EVENT_LOG="$PWD/logs/l2_events.jsonl" EVICTION_POLICY=Bidaw BIDAW_DUAL_QUEUE=1 bash start_lmcache_server.sh
```

```bash
source env.sh
python3 bidaw/trace_replay.py --users 70 --track-chunk-fates --require-answer-policy
python3 bidaw/l2_fate_report.py
```

报告读取最近一次 LMCache 启动产生的第 4 版日志，按成功的物理写入计数。回放为每轮
生成独立的 ID，通过 LMCache 请求元数据关联写入、查找和轮次事件。每次实验使用独立的
服务运行和会话 salt，统计包含预热。第 3 版日志可查看写入和读回次数，缺失轮次 ID
的写入归 U。带标记的回放会增加 HTTP 和日志 I/O 开销。

- A 要求至少一次后续轮次查询该块，写入存续期间所有已知的后续查询均命中 L1，且无 SSD 读取尝试。通过会话结束或该次写入结束确定观测范围。
- B 要求显式的完整会话结束标记。SSD 写入后，后续轮次从未查询同一对象键，也无轮次不明的查询或 SSD 读取尝试。SSD 副本仍存在的写入也可归 B。填充尾块按自身对象键匹配。
- C 要求该写入之后最近一次 L1 移除由 Bidaw 淘汰造成，且淘汰早于后续轮次的缓存查询提交，
  可以发生在该轮开始之后。同一次预取须经双队列等待路径派发，FS 确认读回且预取结果
  在该轮结束前计为 L2 命中。独立填充尾块计入物理写入
  分母，普通回放中归类时不会计为 C。
- U 包括证据不完整的写入。因上下文上限截断、预热跳过或请求失败，回放不会为该会话发送完整结束标记。

报告列出探针版本和运行 ID。读回强度使用可匹配到本次写入生命周期的成功读回次数，
并单列其中涉及的不同「预取请求、写入」组合及满足 C 的组合。后端成功读回是 FS 层
观测，实际设备 I/O 还受操作系统页缓存和 FS 打开方式影响。
探针仅在 FS 读取任务中记录成功读回。旧日志中同时缺少 I/O 任务 ID 和预取请求 ID 的
`load` 记录来自淘汰顺序刷新，报告会单独计数，并从读回统计和 A/B 判定中排除。

默认 FS L2 配置不提供基于容量的全局淘汰。B 类衡量观测窗口内未被请求复用的写入。

### 跨对话共享回放

启用 `--shared-cache` 后，各对话使用 LMCache 的无 salt 缓存空间，相同 token 前缀可以复用 KV。
回放保留独立的对话 ID、轮次和回答历史。默认给每个会话首轮加入同一段 256-token 合成前缀，
后续轮次通过会话历史保留前缀。用 `--shared-prefix-tokens 0` 可保留原始回放的 token 生成方式。
前缀计入上下文限制。无 salt 的对象使用 Bidaw 现有的共享对象淘汰分支，按对象最近使用时间排序。
做策略对比时，各组使用相同的共享模式和前缀长度。

用启用了 `BIDAW_L2_EVENT_LOG` 的独立 LMCache 实例，每次启动只跑一次带标记的共享回放：

```bash
python3 bidaw/trace_replay.py --users 35 --shared-cache --track-chunk-fates --label shared
python3 bidaw/l2_fate_report.py
```

共享模式下，B 类等到全局回放结束标记，再检查写入后是否有任何对话查询或读取同一对象。
该标记表示本次回放的观察窗口结束。报告的 B 类仅覆盖这个窗口，回放跳过或截断的轮次没有产生查询证据。

## Gandalf 第一阶段与四组基准

第一阶段使用 Qwen2.5-1.5B 的单 rank、单全注意力 KV group。新 KV 先放入
DRAM。Gandalf 在 L1 达到淘汰水位时，分别找归一化丢弃成本最低的叶子
（式 7）和归一化迁移成本最低的根侧块（式 8），再跨对象比较。仅当 `α × m_j < h_d`
时优先迁移，等成本时选丢弃。如果只有一种动作可行，则使用该动作。已有 SSD 副本的
L1 叶子也按其复用分数参与选择，选中后释放 DRAM 副本即可。

迁移从 SSD 已有前缀的边界向后写。SSD 满时按叶子规划足够的实际置换空间，
式 (8) 的评分取该计划中最低的归一化 SSD 置换成本。同一对象在 L1 还有副本时，
移除它的 SSD 叶子不计重算损失。迁移失败时保留 L1 对象。事件日志中的
`drop_score` 和 `move_score` 记录两个候选的归一化成本，`replacement_cost` 记录
实际计划中所有 SSD 牺牲对象的未归一化重算成本之和。

在线预测只使用已经观察到的请求。首版用以请求数计的有限回访窗口和经验概率估计，
并按对象大小和空闲请求数归一化成本。候选集使用直接扫描。磨损参数请按实际硬件校准。
论文的混合注意力状态放置和惰性候选堆留待下一阶段。

| 组 | L1 淘汰与 L2 写入 | 调度 |
| --- | --- | --- |
| `lru` | LRU，L1 写完后异步复制 | FCFS |
| `lru_writeback` | LRU，L1 淘汰时写回 | FCFS |
| `bidaw` | Bidaw，L1 写完后异步复制 | 开启双队列 |
| `gandalf` | Gandalf，L1 有压力时选择性写回 | FCFS |

用仓库固定版本的 vLLM 0.33.0、LMCache 0.5.7.post1、已下载的模型和一张可见的
NVIDIA GPU。服务使用 5555、8080、8000，运行前保持这三个端口空闲。用本地 NVMe
路径作为 `GANDALF_CACHE_ROOT`。脚本为每组创建全新的 L2 目录，保留原始结果和缓存文件。

```bash
bash run_gandalf_benchmark.sh --plan
CUDA_VISIBLE_DEVICES=0 GANDALF_CACHE_ROOT=/mnt/nvme/gandalf \
  GANDALF_USERS=35 bash run_gandalf_benchmark.sh
CUDA_VISIBLE_DEVICES=0 GANDALF_CACHE_ROOT=/mnt/nvme/gandalf \
  GANDALF_REPEATS=3 bash run_gandalf_benchmark.sh
```

默认 70 个会话、20 倍时间压缩、并发 64、1 GiB L1、2 GiB L2。可用
`GANDALF_USERS`、`GANDALF_SPEED`、`GANDALF_PARALLEL`、`GANDALF_L1_GB`、
`GANDALF_L2_BYTES`、`GANDALF_SEED` 和 `GANDALF_REPEATS` 调整。各组使用同一
固定回答历史，关闭填充尾块和跨会话共享，从首条回放请求开始计量。重复运行会轮换组顺序。

门控默认 `GANDALF_ALPHA=1`、`GANDALF_RETURN_HORIZON=128`（请求数）、
`GANDALF_RECOMPUTE_COST=0.00002`（每个重算 chunk 的示例价格）、
`GANDALF_SSD_PRICE=100` 和 `GANDALF_SSD_TBW_BYTES=6e14`。
通过 `GANDALF_TARGET_DAYS` 启用随累计写入调节的寿命目标，默认 0 时仅使用静态写入价格。

输出在 `logs/gandalf_benchmarks/run.*/` 下。报告要求四组实际 prompt 指纹、请求数和
容量一致，交叉核对 FS 成功写入事件与适配器计数。每组保留服务状态、回放 JSON、vLLM
指标快照和 L2 事件。比较报告包含 TTFT 和端到端 P50/P95/P99、FS 写入次数和字节、
每千请求写入量、L1/L2 连续前缀命中、LMCache 未命中 token 和 vLLM 本地 prefill token。
本地 prefill 包括新输入与历史重算；FS 写入统计反映主机侧 KV 写入量。

### 四组 SSD 写入分类

实验结束后，用四组的原始 L2 事件生成统一口径的 A/B/C/U 分布：

```bash
python3 -m bidaw.gandalf_fate_report logs/gandalf_benchmarks/run.XXXXXXXX
```

在实验目录内生成 `ssd_write_fates.txt`、`ssd_write_fates.json` 和一张
`ssd_write_fates.png`。汇总图标出四组的淘汰策略、调度和写入方式，并以写入次数为分母，
显示每组各类的占比。多次重复时，先按组累加写入次数，再计算占比。文本和 JSON 保留各次的
写入次数、字节数及占比。也可将单个 `l2_events.jsonl` 文件作为参数，独立生成文本和 JSON。
C 要求该次物理写入实际发生过 I/O 读取尝试和成功读回，随后同一请求对同一对象记录
L2 命中。C 的判定独立于淘汰策略和调度路径。A、B 仍使用前文的回放证据，
其余写入归 U。原有 `bidaw.l2_fate_report` 继续使用 Bidaw 专用判据。
