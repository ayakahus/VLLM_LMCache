> 此文件为原始设计及历史记录，现状与使用方式以 [FIXES.md](FIXES.md) 为准。历史性能结论需在修复后重新验证。

# Bidaw 在 vLLM 0.33.0 + LMCache 0.5.7.post1 上的实现设计

> 论文: *Bidaw: Enhancing Key-Value Caching for Interactive LLM Serving via
> Bidirectional Computation–Storage Awareness*, FAST'26 (清华).
> 目标: 在已有的 vLLM + 独立 LMCache Server 架构上复现其两个核心机制
> (I/O 感知双队列调度 + 基于上一轮回答长度的淘汰) 并做对照实验。

## 0. 论文机制 → 本架构映射总览

| 论文概念 | 本架构对应物 | 现状 |
|---|---|---|
| 性能层(内存) | L1: LMCache Server DRAM 池 | ✅ 已有(8GB) |
| 容量层(SSD) | L2: `fs` 适配器 | ❌ 待启用(Phase 0) |
| GPU 显存 | L0: vLLM paged buffer | ✅ 引擎自有 |
| 请求 KV 位置/大小 | LOOKUP 应答命中块数(需扩展 L1/L2 分拆) | 🔧 小改协议 |
| 双队列 + disk-HRRN | vLLM waiting 队列准入闸门 + 预取排序 | 🔧 新代码 |
| L2→内存预取 | `POST /cache/prefetches`(token_ids/cache_salt/L2→L1) | ✅ 现成 |
| "尚未就绪"闸门 | `get_num_new_matched_tokens()` 返回 `None` | ✅ 协议现成 |
| 多用户隔离 | `cache_salt` → ObjectKey 全链路透传 | ✅ 现成 |
| 回答长度上报 | STORE op 的 `request_configs` | 🔧 小改 |
| 新淘汰算法 | `EvictionPolicy` ABC + factory 分支 | 🔧 核心新代码 |
| inclusive 缓存 | default store 策略: L1 保留 + L2 副本 | ✅ 默认即是 |
| 存储高效张量 | Phase 4: GQA 上退化为普通 KV, 不实施 | — |
| 多轮历史 KV | `lmcache.mp.save_decode_cache=true` | 🔧 配置项 |

## 1. 已核实的系统接口(设计依据, 来自已安装源码)

```
LMCacheMPRequestState: PREFETCHING → WAITING_FOR_LOAD → READY → BYPASS_LMCACHE
get_num_new_matched_tokens(request, num_computed_tokens) -> (int|None, bool)
    None  = "connector 还需时间判定, 调度器稍后再问"  ← preparing 队列闸门
    (0,False)=无外部 KV; (n,True)=n 个 token 异步加载
POST /cache/prefetches  body: {model_name, world_size, token_ids,
                               cache_salt, source_tier=L2, target_tier=L1}
    → 202 + request_id;  GET /cache/prefetches/{id} 轮询
EvictionPolicy ABC: on_keys_created/touched/removed +
    get_eviction_actions(expected_ratio, key_eligible_filter, cache_salt)
    -> list[EvictionAction]; EvictionAction(destination: DISCARD|L2_CACHE, keys)
    support_isolation=True 时按 cache_salt 隔离(IsolatedLRU 先例)
ObjectKey: {chunk_hash, kv_rank, model_name, object_group_id, cache_salt}
    + extra_attributes: dict("请求级元数据, 上游可 enrich")
request_configs: dict, 从 sampling_params 提取, 随 STORE op 透传到 Server
```

## 2. Phase 0 — 两级存储就位(无新代码, 纯配置)

改动: ① `start_lmcache_server.sh` 追加 `--l2-adapter '{"type":"fs","base_path":"<demo>/l2_ssd"}'` 并建目录; ② `start_vllm.sh` 的 extra config 增加 `"lmcache.mp.save_decode_cache": true`(多轮复用前提); ③ 负载请求体带 `cache_salt: "<user_id>"`。
注意: Server 重启清空 L1, L2 空目录需重跑负载填充。
验收: `:8080/config` 的 adapters 非空; `lookup_hit_l2_tokens_total` 开始增长; `l2_ssd/` 出现数据文件。

## 3. Phase 1 — I/O 感知双队列调度(论文 §3.2)

### 3.1 核心思路
```
新请求到达 → connector LOOKUP → Server 返回 (l1_hit_chunks, l2_hit_chunks)
   ├─ l2_hit==0(全在 L1 或未命中)   → READY  → 正常上报匹配数, vLLM 照常调度
   └─ l2_hit>0(部分/全部在 SSD)     → PREPARING
         → 上报 None(闸门: vLLM 留在 waiting, 稍后再问)
         → 后台线程按 disk-HRRN 向 /cache/prefetches 下发 L2→L1 预取
         → 预取完成 → 置 READY, 恢复上报(按原到达位次竞争调度)
```

### 3.2 disk-HRRN
lookup 完成后按实际 L2 读盘计划的 KV 字节数和请求等待时长计算 HRRN 优先级。
Preparing Queue 等待期间只保留计划所需的读锁，取得名额后才预留 L1 写缓冲并读盘。
在途预取请求上限为 `min(4, BIDAW_L2_READ_CONCURRENCY)`，默认 4。

### 3.3 文件改动清单
| 文件 | 类型 | 内容 |
|---|---|---|
| `bidaw/patch_lookup_tier.py` | 新增 | 协议扩展: lookup 应答 `hit_chunks:int` → `(l1_hit, l2_hit)`; client `mp_lookup_state.py` 同步解析(服务端已有 hit_l1/l2 区分, 只改 wire 字段) |
| `bidaw/bidaw_connector.py` | 新增 | `BidawLMCacheMPConnector(LMCacheMPConnector)`: 闸门 + PreparingTable + disk-HRRN 预取线程 + 完成轮询/提升 |
| `bidaw/register.py` | 新增 | monkey-patch vLLM factory 中 LMCacheMPConnector 解析指向子类(启动最早期) |
| `start_vllm.sh` | 修改 | PYTHONPATH 加 bidaw; 入口改 `python -m bidaw.vllm_entry serve ...` |

### 3.4 公平性
READY 队列 FCFS(装不下跳过); PREPARING 提升后按原始到达序插回(记 arrival_seq);
调度开销目标 <1ms/次(论文 0.62ms), PreparingTable 用 heapq。

### 3.5 验收
3+ 用户并发、部分历史在 SSD: `/cache/prefetches/{id}` 被 disk-HRRN 顺序消费;
ready 请求 TTFT 不被 SSD 大 KV 请求拖长(排队时间 CDF 对比 FCFS)。

## 4. Phase 2 — 基于上一轮回答长度的淘汰(论文 §3.3)

### 4.1 数据流
```
生成结束 → connector 在 STORE op 的 request_configs 附 "answer_tokens": N
        → Server store handler 透传 → 通知淘汰策略 on_answer(salt, N)
淘汰触发(水位 0.8) → EvictionController 调 get_eviction_actions(...)
        → BidawPolicy 对每条候选 key 计算 hit potential, 升序淘汰
```

### 4.2 BidawPolicy 状态(Server 进程内)
- per_salt: {最近 answer_tokens, WRD(加权复用距离)直方图, 访问序}
- ghost cache: 后台线程对最近 W 条 LOOKUP trace 跑 Belady, 分桶统计最优命中率:
  small 桶(WRD < L1 容量) hit=1.0; extreme 桶 hit=0.0; 中间 m 个 promising 桶插值
- LB 曲线: 回答长度每 32 tokens 一档，每档至少 8 个样本时取 IQR 去离群后的 WRD 最小值。
  在档位中心拟合不超过各档最小值的单调下界，档间线性插值，区间外沿用最近端点。

### 4.3 淘汰决策(论文式 2)
```
候选 key k(属 salt s, 上次访问至今被其他 key 占用 W(s,k) 字节):
  有效下界 = max(当前累计 WRD W(s,k), 上一轮回答长度预测的 LB(s))
  bucket 概率 = 用户 s 的 WRD 直方图，移除有效下界所在桶之前的桶后归一化
  potential(k) = prob_small×1.0 + Σ prob_promising(i)×hit_promising(i) + prob_extreme×0.0
淘汰 potential 最低的 expected_ratio 比例; destination=L2_CACHE(inclusive)
```
WRD 按逻辑请求更新。淘汰时用 Fenwick 树查询会话上次访问之后各身份的最近权重，
查询不改变访问顺序，时间复杂度为 O(log N)。

### 4.4 文件改动清单
| 文件 | 类型 | 内容 |
|---|---|---|
| `bidaw/eviction_policy/bidaw_policy.py` | 新增 | BidawEvictionPolicy(EvictionPolicy): 状态机+决策(~400 行) |
| `bidaw/lmcache_server_entry.py` | 新增 | 包装入口: 先 monkey-patch `CreateEvictionPolicy`(加 "Bidaw" 分支), 再调 `lmcache.cli.main.main()`; `start_lmcache_server.sh` 改用它并 `--eviction-policy Bidaw` |
| connector(bidaw_connector.py) | 修改 | store 路径写入 `answer_tokens = len(all_token_ids)-num_prompt_tokens` |
| Server store handler | 微改 | 读 op.request_configs["answer_tokens"] → `policy.on_answer(salt,n)`(EvictionPolicy 加默认 no-op 方法, 向后兼容) |

### 4.5 验收
- `--eviction-policy LRU / ARC / Bidaw` 三组对照, 记录 L1 命中率
  (`lookup_hit_l1_tokens/lookup_requested_tokens`); 预期高压下 Bidaw 显著更优。
- 行为抽查: 同 salt 连续两轮, 第二轮回答更长 → LB 抬升 → 该 salt 旧块 potential
  下降 → 更早被淘汰(符合 §3.3.1)。

## 5. Phase 3 — 多用户交互负载与对照实验

### 5.1 负载发生器 `bidaw/workload_sim.py`
- N 个虚拟用户(user_id → cache_salt), 每人一条多轮对话链(共享长文+追问),
  Poisson 到达(λ 可调), 轮间隔与上一轮回答长度正相关(§3.3.1 的物理来源);
- 压力标定: Qwen2.5-1.5B 每 token KV≈56KB; 8GB L1≈14.3 万 token;
  目标 KV 总量 ≈ 3~4×L1 → 例如 40 用户 × 平均 1.4 万 token 历史。

### 5.2 对照组与指标
| 组 | 调度 | 淘汰 |
|---|---|---|
| A(基线) | FCFS | LRU |
| B | Bidaw 双队列 | LRU |
| C | FCFS | Bidaw |
| D(完整) | Bidaw 双队列 | Bidaw |

指标: ① L1 命中率 ② TTFT p50/p99(客户端打点) ③ 排队时间 CDF ④ 固定延迟下最大
到达率(吞吐)。画图对齐论文 Figure 18/19 口径。

## 6. Phase 4(可选, 建议不做) — 存储高效张量

论文 tensor6 依赖 OPT 的 MHA: h(5120) < K+V(10240), 缓存 h 省一半空间。
Qwen2.5-1.5B 是 GQA: 每 token KV=1KB, layernorm 后 h=3KB——比 KV 还大 3 倍,
缓存 h 反而更亏。GQA 上该优化通常退化为直接缓存 KV → 跳过, 理由记录在案。

## 7. 风险与回退
| 风险 | 缓解 |
|---|---|
| lookup 协议扩展引发多 Server 不一致 | 单 Server 先行; 扩展字段向后兼容 |
| 闸门误判致请求卡 waiting | PREPARING 120s 超时, 超时降级 READY(接受 L2 直读) |
| Bidaw 淘汰低压力下不如 LRU | 保留 `--eviction-policy` 开关, 随时回退 |
| save_decode_cache 增大写入量 | 仅多轮实验开启; store 吞吐实测 5.6GB/s 余量足 |
| monkey-patch 脆弱 | 只挂 factory/入口两个稳定点; 失败即抛错退出 |

## 8. 预计工作量
Phase 0: 0.5h | Phase 1: 1~2 天 | Phase 2: 2~3 天 | Phase 3: 1 天 | 合计 ~1 周

---

## 9. 实现状态(2026-11-14)

### 已交付(Phase 0 + Phase 2 + Phase 3 冒烟)
| 组件 | 文件 | 状态 |
|---|---|---|
| BidawEvictionPolicy(WRD 记账/LB 曲线/ghost Belady/式2 决策/整用户淘汰/宽限期) | `bidaw/eviction_policy/bidaw_policy.py` (416 行) | ✅ 实测淘汰触发 |
| 工厂注入(CreateEvictionPolicy 双命名空间 + BIDAW_FORCE_POLICY) | `bidaw/server_patch.py` | ✅ |
| HTTP 注入(POST /bidaw/answer, GET /bidaw/status) | `bidaw/api.py` + server_patch | ✅ 实测 8 用户回答长度入库 |
| Server 包装入口 | `bidaw/lmcache_server_entry.py` | ✅ |
| 两级存储(L2=fs SSD) + save_decode_cache | 启动脚本 | ✅ FSL2AdapterConfig 生效 |
| 多用户多轮负载(10 用户×8 轮, cache_salt, 回答上报, 轮间隔∝回答长度) | `bidaw/multi_turn_sim.py` | ✅ |
| vLLM 重启脚本(处理 VLLM::EngineCore 改名进程) | `restart_vllm.sh` | ✅ |

### 实测结果(L1=1GB, 10 用户×8 轮, KV 总量≈1.4GB>0.8GB 水位)
```
              LRU          Bidaw
L1 命中率     88.2%        88.5%
rounds1-7    105~156ms    105~119ms
TTFT p50     (周期性尖刺:   (稳定, 尾部 max
              3/7 轮 >150)  113~182ms vs 164~216ms)
淘汰行为      逐条 LRU     整用户淘汰 25 keys/次,
                          potential 日志可见
```
结论: 本负载的复用距离普遍 < L1 容量, 两策略命中率相当(理论预期一致);
Bidaw 的差异化收益在 TTFT 稳定性(用户粒度决策 + 活跃用户宽限期消除 LRU
的周期性尖刺)。要复现论文的命中率差距需更高压力(KV 总量 3~4×L1, 即
~40 用户或 4×历史长度), 见 Phase 3 完整版。

### 已知偏差(与论文实现)
1. 回答长度经 HTTP(/bidaw/answer) 由客户端上报, 而非 connector 的
   request_configs(避免 Phase 1 前的深度改造; 时序等价: 都在下一轮访问前到达)
2. WRD 以"访问的 key 数"为单位(均匀 chunk 下与字节等价)
3. ghost cache 在 trace 事件 ≥ 512(可调)后启动; 小负载下 potential 全为
   1.0(small 桶), 决策退化为 recency——高压负载下才会分化
4. L1 淘汰 destination=DISCARD(L2 副本由 store 策略写入, inclusive 语义不变)
5. Phase 1(双队列调度)——**已于 2026-11-16 实现**(`bidaw/dual_queue_scheduler.py`,
   BIDAW_DUAL_QUEUE 开关): 原生栈已自带"lookup 触发 L2→L1 预取 + 未完成
   返回 None 闸门", 本实现补齐 READY 快速通道(L1-only lookup 不排队)与
   disk-HRRN 派发(防队头阻塞+防饿死); 四臂验收见 logs/trace70_comparison.txt

### 运维备忘
- 换策略: `EVICTION_POLICY=LRU|ARC|Bidaw ./start_lmcache_server.sh`
  (切换策略需同时重启 vLLM: Server 重启会使 connector 会话失效,
  表现为所有 STORE 失败/BYPASS, 见 restart_vllm.sh)
- vLLM EngineCore 进程改名 `VLLM::EngineCore`, pkill 需匹配该名
- 校准: `BIDAW_L1_CAPACITY_KEYS ≈ L1_GB×1024/7.34` (本模型 256-token chunk)
- 运行时观测: `curl :8080/bidaw/status` (LB 曲线/ghost 桶命中率/最近回答)
