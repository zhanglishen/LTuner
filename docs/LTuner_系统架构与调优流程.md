# LTuner：基于 LLM 自省式反馈的数据库参数自动调优系统

## 一、系统概述

LTuner 是一个面向 PostgreSQL 的数据库参数自动调优系统，采用**大语言模型（LLM）驱动的自省式反馈机制**替代传统的贝叶斯优化方法（如 SMAC3、Bayesian Optimization）。系统的核心创新在于将 LLM 的自然语言推理能力与数据库调优领域知识相结合，通过"观察→反思→调整"的闭环迭代，使 LLM 能够像资深 DBA 一样理解性能变化的因果关系并做出调优决策。

### 核心设计理念

| 设计维度 | 传统方法（GPTuner/OtterTune） | LTuner 方法 |
|---------|---------------------------|-----------|
| 搜索策略 | 贝叶斯优化/强化学习（黑盒） | LLM 自省式反馈（白盒因果推理） |
| 参数选择 | 手动指定/简单统计 | MoE 多专家动态评估 |
| 值域约束 | 硬编码范围 | SPC 三智能体协作剪枝 |
| 参数关联 | 忽略或简单规则 | 因果知识图谱追溯 |
| 反馈信号 | 数值梯度/代理模型 | 文本梯度（自然语言因果分析） |
| 负载理解 | 聚合 TPS/延迟 | 语义化负载分析（逐查询+执行计划+系统诊断） |

### 系统入口

```
python run_ltuner.py postgres tpch 600 -max_iter 20 -top_k 25 -scenario auto
```

命令行参数：`db`（数据库类型）、`test`（基准测试名称）、`timeout`（单次 benchmark 超时秒数）、`-max_iter`（最大迭代次数）、`-top_k`（MoE 筛选参数数）、`-threshold`（收敛阈值）、`-scenario`（OLTP/OLAP/HYBRID/auto）。

---

## 二、系统架构

LTuner 采用**五阶段流水线架构**，由主编排器（`LTunerOrchestrator`）串联五个核心模块：

```
┌─────────────────────────────────────────────────────────────────┐
│                    LTuner Orchestrator（主编排器）                 │
│                                                                   │
│  Step 1         Step 2         Step 3           Step 4    Step 5 │
│ ┌──────────┐  ┌──────────┐  ┌─────────────┐  ┌────────┐ ┌─────┐│
│ │环境感知与 │→│MoE 多专家│→│因果图谱分析 +│→│自省反馈│→│结果 ││
│ │特征翻译   │  │参数筛选  │  │SPC 值域剪枝  │  │迭代优化│  │输出 ││
│ └──────────┘  └──────────┘  └─────────────┘  └────────┘ └─────┘│
│       ↓              ↓              ↓               ↓            │
│  env_context   target_knobs   causal_context   best_config      │
│                               value_ranges                       │
└─────────────────────────────────────────────────────────────────┘
                              ↕
              ┌──────────────────────────────┐
              │   PostgreSQL DBMS 抽象层       │
              │   (连接管理/配置设置/恢复机制)  │
              └──────────────────────────────┘
                              ↕
              ┌──────────────────────────────┐
              │   Benchbase 基准执行器         │
              │   (TPC-C/TPC-H/Twitter 等)    │
              └──────────────────────────────┘
```

### 模块清单

| 模块 | 源文件 | 核心类 | 职责 |
|------|--------|--------|------|
| 主编排器 | `ltuner_orchestrator.py` | `LTunerOrchestrator` | 串联五步工作流，协调所有子模块 |
| 特征翻译器 | `feature_translator.py` | `FeatureTranslator` | 将硬件配置和运行指标翻译为 LLM 可理解的自然语言 |
| MoE 多专家系统 | `moe_experts.py` | `MoEManager` + 7 个 Expert | 多专家融合评估，从数百个候选中筛选 Top-K 参数 |
| 因果知识图谱 | `causal_graph.py` | `CausalKnowledgeGraph` | 建立参数-指标因果关系图，支持瓶颈溯源和协同分析 |
| SPC 值域剪枝 | `value_pruner.py` | `ValuePruner`(Surveyor+Proposer+Corrector) | 三智能体协作确定安全值域 |
| 自省反馈引擎 | `reflective_engine.py` | `ReflectiveEngine` | 核心优化循环：LLM 自省→文本梯度→配置生成 |
| 语义分析器 | `workload_semantic_analyzer.py` | `WorkloadSemanticAnalyzer` | 逐查询延迟变化、执行计划、系统诊断的语义化翻译 |
| 性能监控器 | `postgres_monitor.py` | `PostgreSQLMonitor` | 实时采集 PostgreSQL 性能指标（TPS/缓存/IO/锁等） |
| 基准执行器 | `workload_runner.py` | `BenchbaseRunner` | 运行 Benchbase 基准测试，解析 TPS、延迟、逐查询结果 |
| DBMS 抽象层 | `postgres.py` | `PgDBMS` | 数据库连接、参数设置、重启恢复 |
| LADO 知识库 | `knowledge_base.py` | `KnowledgeBase` + `KnowledgeUnit` | 结构化参数知识的三维增强检索 |

---

## 三、各模块详细设计

### 3.1 特征翻译器（Feature Translator）

**目的**：将数据库的原始数值指标转化为 LLM 可理解的语义描述，使大模型能够基于自然语言进行逻辑推理。

**设计原理**：传统调优系统直接将数值指标（如 `buffer_hit_ratio=87.3`）输入模型，但 LLM 更擅长理解语义化描述（如"缓冲区命中率为 87.3%，略低于推荐值。存在一定的缓存缺页现象，可适当增大 shared_buffers"）。

**核心方法**：

| 方法 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `translate_static_features()` | 无（自动采集硬件） | 硬件环境描述 | CPU 核心数、内存大小、存储类型、PG 版本、资源预算参考 |
| `translate_dynamic_metrics()` | 性能指标字典 | 运行状态分析 | 聚合 6 个子翻译方法的结果 |
| `translate_performance_delta()` | 新旧指标+配置变更 | 性能对比分析 | 用于自省反馈环节的因果分析 |

**6 个子翻译维度**：
1. **事务性能**（`_translate_transaction_metrics`）：TPS 水平分级（空闲/低/中/高/极高）+ 回滚率警告
2. **缓存命中率**（`_translate_cache_metrics`）：四级阈值判断（严重<70% / 偏低<85% / 良好<95% / 优秀≥95%）
3. **I/O 状态**（`_translate_io_metrics`）：物理读取比例分析 + 写入活动诊断
4. **连接状态**（`_translate_connection_metrics`）：连接数评估和并发诊断
5. **锁状态**（`_translate_lock_metrics`）：排他锁数量和竞争程度
6. **慢查询**（`_translate_slow_queries`）：慢查询列表和调优建议

### 3.2 MoE 多专家系统（Mixture of Experts）

**目的**：从 PostgreSQL 数百个候选参数中，通过多领域专家融合评估，筛选出对当前负载场景最关键的 Top-K 个参数。

**设计原理**：不同负载场景（OLTP/OLAP/HYBRID）对参数的敏感度差异巨大。例如 OLTP 场景下并发连接参数最关键，而 OLAP 场景下并行查询参数优先级更高。MoE 架构允许每个专家独立评估参数，再由管理器根据场景动态调整权重进行融合。

**7 个专家评估器**：

| 专家 | 关注领域 | 核心参数示例 | 动态调整条件 |
|------|---------|------------|------------|
| MemoryExpert | 内存管理 | shared_buffers(95), effective_cache_size(90), work_mem(85) | 缓存命中率<85%时+15分 |
| IOExpert | I/O 调度 | max_wal_size(85), checkpoint_completion_target(80) | 物理读取率>20%时+15分 |
| QueryExpert | 查询执行 | max_parallel_workers_per_gather(85), random_page_cost(70) | OLAP场景+15分；有慢查询+10分 |
| ConcurrencyExpert | 并发事务 | max_connections(90), idle_in_transaction_session_timeout(65) | 连接数>80时+10分；锁>50时+15分 |
| BackgroundProcessExpert | 后台进程 | autovacuum_vacuum_cost_delay(70), bgwriter_delay(60) | — |
| HardwareAdaptExpert | 硬件适配 | — | 大内存(≥16GB)时内存参数+60分；多核(≥8)时并行参数+55分 |
| SafetyAuditExpert | 安全审计 | max_connections(85), shared_buffers(80), work_mem(70) | postmaster参数+10分 |

**评估流程**：
```
对每个候选参数 knob:
    weighted_score = 0
    for expert in experts:
        score = expert.evaluate(knob, knob_info, workload_context, metrics)  # 0-100
        weighted_score += score × expert.weight
    knob_scores.append((knob, weighted_score))

knob_scores.sort(descending)
return knob_scores[:top_k]
```

**场景权重分配**（权重和 = 1.0）：

| 专家 | OLTP | OLAP | HYBRID |
|------|------|------|--------|
| 内存管理 | 0.20 | 0.15 | 0.18 |
| I/O 调度 | 0.12 | 0.15 | 0.14 |
| 查询执行 | 0.08 | **0.25** | 0.15 |
| 并发事务 | **0.25** | 0.05 | 0.15 |
| 后台进程 | 0.08 | 0.08 | 0.08 |
| 硬件适配 | 0.12 | 0.12 | 0.12 |
| 安全审计 | 0.15 | 0.20 | 0.18 |

### 3.3 因果知识图谱（Causal Knowledge Graph）

**目的**：建立参数(Knob) ↔ 性能指标(Metric) ↔ 参数(Knob) 的三元因果关系图，支持从性能瓶颈反向追溯关键参数，以及参数间的协同/冲突分析。

**设计原理**：数据库参数之间存在复杂的因果链。例如 `shared_buffers↑ → buffer_hit_ratio↑ → disk_read_rate↓`。通过显式建模这些因果关系，LTuner 能够在检测到瓶颈时（如磁盘读取率高），自动追溯到应调整的参数（如 shared_buffers）。

**图谱结构**：
- **参数节点**（18 个）：shared_buffers, effective_cache_size, work_mem, maintenance_work_mem, max_wal_size, wal_buffers, checkpoint_completion_target, min_wal_size, max_connections, max_parallel_workers_per_gather, max_parallel_workers, max_worker_processes, random_page_cost, effective_io_concurrency, default_statistics_target, autovacuum_vacuum_cost_delay 等
- **性能指标节点**（13 个）：buffer_hit_ratio, disk_read_rate, checkpoint_frequency, wal_write_latency, query_sort_spill, lock_contention, connection_wait, parallel_efficiency, vacuum_efficiency, seq_scan_ratio, temp_file_usage, memory_pressure, transaction_throughput
- **因果边类型**：causes（因果）、triggers（触发）、constrains（约束）、conflicts（冲突）
- **边权重**：0-1 浮点数，表示因果强度

**核心因果链**（示例）：

```
内存子系统:
  shared_buffers --[causes 0.95]--> buffer_hit_ratio --[triggers 0.90]--> disk_read_rate
  work_mem --[causes 0.90]--> query_sort_spill --[triggers 0.85]--> temp_file_usage
  
WAL子系统:
  max_wal_size --[constrains 0.90]--> checkpoint_frequency --[triggers 0.60]--> disk_read_rate
  
并行查询:
  max_parallel_workers_per_gather --[causes 0.85]--> parallel_efficiency
  max_parallel_workers_per_gather --[constrains 0.95]--> max_parallel_workers --[constrains 0.95]--> max_worker_processes

协同关系:
  shared_buffers --[causes 0.85]--> max_wal_size  (增大缓冲池需同步增大WAL)
  
冲突关系:
  shared_buffers ⚡ max_connections  (两者均消耗内存，需平衡)
```

**关键算法**：

1. **瓶颈溯源**（`trace_bottleneck`）：从瓶颈指标节点出发，沿反向边 DFS 遍历，累计因果权重，返回按权重排序的关键参数列表。
2. **协同参数发现**（`get_synergy_group`）：正向查找（直接影响的参数）+ 反向查找（影响该参数的参数）+ 间接关联（共同影响同一指标的参数，权重衰减 0.7）。
3. **冲突检测**（`get_conflict_warnings`）：检查参数集合中预定义的冲突对。

### 3.4 SPC 值域剪枝（Surveyor-Proposer-Corrector）

**目的**：为筛选出的 Top-K 参数确定安全、合理的值域范围和推荐值，防止 LLM 生成导致系统崩溃的危险配置。

**设计原理**：采用三智能体协作模式，分离"信息收集"、"方案生成"和"安全校验"三个关切点，提升值域剪枝的可解释性和可靠性。

**三阶段流程**：

```
Phase 1: Surveyor（勘测员）
  ├── 从 pg_settings 获取物理上下限（physical_min/max）
  ├── 从 LADO 知识库获取约束规则
  └── 输出: ValueRange（含物理范围+建议范围）

Phase 2: Proposer（提案者）
  ├── 根据硬件信息（RAM/CPU）将相对值转为绝对值
  ├── 根据场景（OLTP/OLAP/HYBRID）选择推荐公式
  └── 输出: ValueRange（含推荐值+建议范围）

Phase 3: Corrector（校验者）
  ├── Rule 1: 内存总预算检查（shared_buffers + work_mem × max_connections + maintenance_work_mem ≤ RAM × 80%）
  ├── Rule 2: 并行层级检查（max_worker_processes ≥ max_parallel_workers ≥ max_parallel_workers_per_gather）
  ├── Rule 3: WAL 一致性检查（min_wal_size < max_wal_size）
  └── 输出: 修正后的 ValueRange + 安全备注
```

**Proposer 推荐公式类型**：

| 公式类型 | 示例参数 | 计算方式 |
|---------|---------|---------|
| `ram_mb * ratio` | shared_buffers, effective_cache_size | 内存百分比（如 RAM × 25%） |
| `fixed_mb` | work_mem, max_wal_size | 固定 MB 值（按场景不同） |
| `fixed` | checkpoint_completion_target, random_page_cost | 固定数值 |
| `cpu_based` | max_parallel_workers_per_gather | CPU 核心数 × 比例 |

### 3.5 自省反馈引擎（Reflective Engine）

**目的**：LTuner 的核心优化模块，实现 LLM 驱动的迭代调优循环，通过"文本梯度"替代数学梯度进行优化方向决策。

**核心概念——文本梯度（Text Gradient）**：
- 传统优化使用数学梯度 ∇f(x) 指导参数更新方向
- LTuner 使用 LLM 生成的自然语言梯度（如"shared_buffers 当前值足够大，继续增大收益递减。work_mem 过小导致 Q5 排序溢出，建议从 64MB 增大到 128MB"）
- 文本梯度包含因果解释，使调优过程可解释

**优化循环**（`optimize()` 方法）：

```
Step 0: 基准测量
  - 重置所有参数到默认值
  - 运行 benchmark 获取 baseline_performance

Step 1: 生成初始配置
  - LLM 根据环境上下文、因果关系和值域约束生成初始配置
  - 随机选择探索方向种子（5 个方向之一），增加不同实验的起点多样性

for iteration in range(1, max_iterations+1):
    
    Step 2: 应用配置并运行 benchmark
      - ALTER SYSTEM SET + pg_ctl restart
      - 执行 benchbase 获取 TPS/Latency
      - 采集逐查询延迟（TPC-H 的 22 条查询）
    
    Step 3: 计算性能变化
      - delta_performance = (current - previous) / |previous| × 100%
      - 更新历史最优配置
    
    回滚策略检查:
      - 若 TPS 跌破基线 50%: 严重崩溃，立即回滚到历史最优配置
    
    Step 4: LLM 自省分析
      - 生成查询级性能变化描述（语义分析器）
      - 生成系统诊断信息（等待事件/临时文件/扫描模式）
      - 传入完整历史轨迹（含每轮反思与梯度）
      - LLM 输出: reflection（深度分析）+ gradient（调整方向建议）
    
    Step 5: 收敛检查
      - 连续 max_iterations/3 轮无改善 → 触发收敛保护
      - 性能变化 < threshold → 收敛终止
    
    Step 6: 生成下一轮配置
      - 微调模式: 步长 ≤ 20%
      - 探索模式: 步长 30-50%，每 4 轮触发一次
      - 配置多样性保护: 近 5 轮不变的参数强制调整
```

**四大调优策略**：

| 策略 | 触发条件 | 行为 |
|------|---------|------|
| **动态温度调度** | 按迭代进度自动切换 | 前30%=0.2(精准)→中30%=0.7(探索)→中后期=0.4(验证)→最后20%=0.1(收敛) |
| **主动探索** | 每 4 轮触发 | 扩大调整步长(30-50%)，探索未变动的参数维度 |
| **收敛保护** | 连续 max_iter/3 轮无改善 | 触发最后一次大范围探索，探索后仍无突破则终止 |
| **配置多样性保护** | 近 5 轮某参数值不变 | 强制 LLM 调整该参数 |

**LLM Prompt 结构**（self_reflect）：

```
System: "你是一位资深的 PostgreSQL DBA 和性能分析专家..."
User:
  ## 当前状态（迭代轮次、基线TPS、当前TPS、历史最优）
  ## 当前配置（所有参数当前值）
  ## 完整历史调优轨迹（每轮: TPS + 反思文本 + 文本梯度）
  ## 因果图谱上下文
  ## 查询级性能变化（Q5: 1250ms→600ms, 改善52%）
  ## 系统诊断信息（等待事件分布、临时文件统计、表扫描模式）
  ## 分析任务（5 项分析要求）
→ 输出: {"reflection": "...", "gradient": "..."}
```

### 3.6 工作负载语义分析器（Workload Semantic Analyzer）

**目的**：将原始数据库指标和执行计划数据转换为 LLM 可理解的自然语言诊断文本，使 LLM 能精确定位参数变化对具体查询的影响。

**四类语义化信息**：

#### 1. 逐查询延迟变化分析（`analyze_query_deltas`）
- 解析 benchbase 生成的 `*.results.Q{N}.csv` 文件
- 对比两轮 benchmark 各查询延迟变化
- 只报告变化超过阈值（默认 5%）的查询
- 附带 TPC-H 查询语义描述（如 "Q5: 本地供应商收入(六表JOIN)"）

输出示例：
```
## 查询级性能变化（本轮 vs 上轮）
### 改善的查询:
  Q5: 1250ms -> 600ms (改善 52.0%)  [本地供应商收入(六表JOIN)]
### 退化的查询（需关注）:
  Q12: 1600ms -> 1800ms (退化 12.5%)  [运输模式(二表JOIN+CASE)]
总延迟: 34650ms -> 34200ms (改善 1.3%)
```

#### 2. EXPLAIN 执行计划诊断（`analyze_explain_plans`）
- 对 22 条 TPC-H 查询执行 `EXPLAIN (FORMAT JSON)`
- 递归提取关键特征：根节点类型、并行度、排序溢出（external merge）、Hash 批次溢出、行数估计偏差
- 用于首轮诊断和异常检测

#### 3. 系统级诊断（`analyze_system_diagnostics`）
- **等待事件分布**：从 `pg_stat_activity` 采集，映射到调优建议（如 `IO:DataFileRead → 增大 shared_buffers/effective_cache_size`）
- **临时文件统计**：从 `pg_stat_database` 采集 `temp_files/temp_bytes`，诊断 `work_mem` 是否不足
- **表扫描模式**：从 `pg_stat_user_tables` 采集顺序扫描 vs 索引扫描比例，判断 `effective_cache_size` 和 `random_page_cost` 是否合理

#### 4. 等待事件语义解释映射

| 等待事件 | 语义解释 | 调优建议 |
|---------|---------|---------|
| IO:DataFileRead | 磁盘数据读取瓶颈 | 增大 shared_buffers/effective_cache_size |
| IO:BufFileRead | 临时文件读取(排序/Hash溢出) | 增大 work_mem |
| IO:WALSync | WAL 同步等待 | 考虑 synchronous_commit=off 或增大 wal_buffers |
| LWLock:BufferMapping | shared_buffers 内部竞争 | 可能过大或并发过高 |
| Lock:transactionid | 事务锁等待 | 长事务阻塞 |

### 3.7 性能监控器（PostgreSQL Monitor）

**目的**：实时采集 PostgreSQL 的全方位性能指标，为特征翻译器和 MoE 专家提供数据支撑。

**采集指标**（`collect_comprehensive_metrics` 方法，7 类）：

| 类别 | 数据源 | 采集内容 |
|------|--------|---------|
| 事务统计 | pg_stat_database | commit_tps, rollback_tps, total_tps, qps |
| 连接统计 | pg_stat_activity | active, idle, total 连接数 |
| 缓存命中率 | pg_stat_database + pg_statio_user_indexes | buffer_hit_ratio, index_hit_ratio |
| I/O 统计 | pg_stat_database | blks_read/hit_per_sec, rows_modified_per_sec |
| 慢查询 | pg_stat_statements | top-N 慢查询(需扩展) |
| 表统计 | pg_stat_user_tables | seq_scan, idx_scan, modifications |
| 锁统计 | pg_locks | 各类锁数量 |

**新增诊断方法**：
- `get_wait_events()`: 等待事件分布
- `get_temp_file_stats()`: 临时文件统计
- `get_table_scan_patterns()`: 顺序 vs 索引扫描比率

### 3.8 LADO 知识库

**目的**：提供结构化的参数调优知识，支持三维增强检索（定义、症状、调优视角）。

**KnowledgeUnit 数据结构**：
```python
@dataclass
class KnowledgeUnit:
    name: str              # 参数名
    category: str          # 类别
    context: str           # user/sighup/postmaster
    description: str       # 描述
    symptoms: List[str]    # 可解决的性能症状
    dependencies: List[str] # 依赖参数
    constraints: Dict      # 约束规则 {hard_rule, min, max}
    tuning_tips: str       # 调优建议
    default_formula: str   # 默认计算公式
    related_knobs: List[str] # 相关参数（资源竞争）
```

**四视角嵌入文本**：
1. 定义视角（Definition View）: 参数名+类别+描述
2. 症状视角（Symptom View）: 可解决的性能问题
3. 调优视角（Tuning View）: 调优建议+默认公式
4. 依赖视角（Dependency View）: 依赖和相关参数

---

## 四、完整调优流程详解

### Step 1：环境感知与特征翻译

```
输入: PostgreSQL 连接
输出: env_context (自然语言环境描述)

1. FeatureTranslator.translate_static_features()
   → "CPU: 4 物理核心, 内存: 7.8 GB, PostgreSQL 14.x"
   → "shared_buffers 建议上限: 2000 MB (RAM 的 25%)"

2. PostgreSQLMonitor.collect_comprehensive_metrics()
   → 7 类指标字典

3. FeatureTranslator.translate_dynamic_metrics(metrics)
   → "事务负载较轻...缓冲区命中率 99.8%...物理磁盘读取占比 0.1%..."

4. WorkloadSemanticAnalyzer.generate_initial_diagnostic()
   → EXPLAIN 执行计划诊断 + 系统诊断（等待事件/临时文件/扫描模式）
```

### Step 2：MoE 多专家参数筛选

```
输入: 全部候选参数 (~300个), env_context, 场景类型
输出: target_knobs (Top-K 参数列表)

1. MoEManager.assign_weights(scenario)
   → 根据 OLTP/OLAP/HYBRID 分配 7 个专家权重

2. 对每个候选参数:
   → 7 个专家独立评估 (score 0-100)
   → 加权求和得到综合分
   
3. 按综合分降序排列，取 Top-K
   → 过滤掉 string 类型参数
```

### Step 3：因果图谱分析 + SPC 值域剪枝

```
输入: target_knobs
输出: causal_context, value_ranges

1. 瓶颈检测
   → 检查缓存命中率、物理IO比、锁数量、慢查询数

2. CausalKnowledgeGraph.generate_causal_context()
   → 瓶颈追溯 + 参数协同关系 + 冲突警告

3. ValuePruner.prune()
   → Phase 1: Surveyor 勘测物理范围
   → Phase 2: Proposer 计算推荐值
   → Phase 3: Corrector 安全校验

4. 协同参数扩展
   → 检查因果图谱中关联但未被选中的参数
   → 追加到 target_knobs 并做值域剪枝
```

### Step 4：自省反馈迭代优化

```
输入: target_knobs, value_ranges, causal_context, env_context
输出: best_config, best_performance

0. get_baseline()  → 默认配置基准性能

1. generate_initial_config()  → LLM 生成初始配置
   (含随机探索种子，5 个方向之一)

for iter 1..max_iterations:
  2. apply_config() → run_benchmark()
     → 获取 TPS, Latency
     → 采集逐查询延迟（22条TPC-H查询）

  3. 计算 delta_performance
     → 更新 best_config / best_performance
     → 回滚策略检查（跌破基线50%立即回滚）

  4. self_reflect(全量历史轨迹 + 查询级变化 + 系统诊断)
     → LLM 分析因果关系
     → 输出: reflection + text_gradient

  5. 收敛检查
     → 连续无改善触发收敛保护
     → 变化幅度<阈值则终止

  6. generate_next_config(text_gradient)
     → 微调模式(步长≤20%) 或 探索模式(步长30-50%)
     → 配置多样性保护（检测停滞参数）
```

### Step 5：结果输出

```
输出:
  - ltuner_result.json (优化结果，含完整历史)
  - ltuner_workflow_report.json (完整工作流报告)
  - 终端摘要 (性能提升百分比、最佳配置)
```

---

## 五、关键设计模式与技术创新

### 5.1 文本梯度优化（Text Gradient Optimization）

传统数据库调优使用数学代理模型（如高斯过程）近似目标函数并计算梯度方向。LTuner 创新性地使用 LLM 的自然语言推理能力替代数学梯度：

- **输入**：性能变化事实 + 配置变更历史 + 因果知识
- **输出**：自然语言描述的调整方向（"基于轮次5的配置出发，增大 work_mem 从 64MB 到 128MB 以解决 Q5 的 Hash Join 溢出"）
- **优势**：
  1. 包含因果解释，调优过程完全可解释
  2. 能利用 LLM 的数据库领域知识
  3. 无需大量历史数据训练代理模型
  4. 能处理参数间的复杂交互关系

### 5.2 多智能体协作范式

LTuner 在多个环节采用多智能体协作：
- **MoE 多专家系统**：7 个领域专家独立评估 → 管理器加权融合
- **SPC 三智能体**：勘测 → 提案 → 校验，分离关切点
- **自省反馈循环**：LLM 扮演"分析师"（self_reflect）和"工程师"（generate_next_config）两个角色

### 5.3 语义化负载感知

LTuner 不仅使用聚合的 TPS/延迟作为反馈信号，还引入了四类语义化信息：
1. **逐查询延迟变化**：精确定位哪条查询改善/退化
2. **执行计划特征**：Hash 溢出、排序溢出、并行度等
3. **等待事件分布**：定位瓶颈类型（IO/Lock/CPU）
4. **临时文件+扫描模式**：诊断 work_mem 和 effective_cache_size

这使 LLM 能建立"参数变化 → 查询行为变化"的精确因果映射。

### 5.4 动态探索-开发平衡

通过四层策略实现探索-开发权衡：
- **温度调度**：控制 LLM 输出的随机性
- **主动探索轮次**：周期性扩大搜索范围
- **收敛保护**：防止过早终止
- **配置多样性保护**：防止参数陷入局部最优

---

## 六、数据流总览

```
硬件信息 ──→ FeatureTranslator ──→ env_context
pg_stat_* ──→ PostgreSQLMonitor ──→ metrics ──→ FeatureTranslator
                                        │
                                        ├──→ MoE 7专家评估 ──→ target_knobs (Top-K)
                                        │
                                        └──→ 瓶颈检测 ──→ CausalGraph.trace_bottleneck()
                                                                    │
                                                            causal_context
                                                                    │
pg_settings ──→ Surveyor ──→ Proposer ──→ Corrector ──→ value_ranges
                                                                    │
                                            ┌───────────────────────┘
                                            │
env_context + causal_context + value_ranges + target_knobs
                          │
                          ↓
                  ReflectiveEngine.optimize()
                          │
          ┌───────────────┼───────────────┐
          ↓               ↓               ↓
    LLM:初始配置    LLM:自省分析     LLM:生成新配置
          │           ↑       │           │
          ↓           │       ↓           ↓
    Benchbase ──→ TPS/延迟  文本梯度 ──→ 新配置
          │                               │
          ↓                               │
    逐查询CSV ──→ WorkloadSemanticAnalyzer │
    pg_stat_* ──→ 系统诊断                 │
    EXPLAIN   ──→ 执行计划分析             │
          │                               │
          └─── 语义化上下文 ──→ LLM Prompt ─┘
```

---

## 七、与 GPTuner 原系统的对比

| 维度 | GPTuner (原系统) | LTuner (改造后) |
|------|-----------------|----------------|
| 搜索策略 | SMAC3 贝叶斯优化 | LLM 自省式反馈 + 文本梯度 |
| 参数选择 | LADO 知识库手动指定/全量 | MoE 7 专家动态评估 Top-K |
| 值域约束 | 知识库硬编码 | SPC 三智能体(勘测→提案→校验) |
| 参数关联 | 忽略 | 因果知识图谱(18节点+13指标+因果边) |
| 反馈信号 | 数值(TPS/延迟) | 文本梯度 + 语义化诊断 |
| LLM 角色 | 辅助知识检索 | 核心决策引擎(分析师+工程师) |
| 负载理解 | 聚合指标 | 逐查询延迟+EXPLAIN+等待事件+扫描模式 |
| 稳定性 | 无回滚机制 | 回滚策略+崩溃恢复+收敛保护 |
| 可解释性 | 黑盒 | 全流程自然语言解释 |
