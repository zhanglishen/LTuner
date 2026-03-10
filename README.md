<img align='right' src="/assets/gptuner.png" alt="GPTuner logo" width="130">

# LTuner: 基于大语言模型自省反馈的轻量级数据库参数调优系统

> 本项目在 [GPTuner (VLDB 2024)](https://dl.acm.org/doi/abs/10.14778/3659437.3659449) 基础上进行了系统性改造，提出 **LTuner** —— 一种基于 LLM 自省式反馈（Self-Reflective Feedback）的数据库参数调优方法，替代原有贝叶斯优化框架，实现更高效、更安全的自动化调优。

### 核心改进

| 维度 | GPTuner (原始) | LTuner (本项目) |
|------|---------------|-----------------|
| **优化引擎** | SMAC3 贝叶斯优化（Coarse-to-Fine BO） | LLM 自省式反馈 + 文本梯度驱动 |
| **参数筛选** | 静态旋钮选择 | MoE 多智能体专家动态评估 |
| **值域优化** | GPT 单次生成 | Surveyor-Proposer-Corrector 三智能体协同剪枝 |
| **安全机制** | 崩溃后恢复 | 规则引擎预校验 + 单位自动修正 |
| **知识利用** | Prompt Ensemble 结构化 | 因果知识图谱 + RAG 语义检索 |
| **LLM 依赖** | GPT-4 (高成本) | 通义千问 qwen-plus (低成本) |

### 实验结果摘要 (TPC-H, PostgreSQL 14)

| 指标 | GPTuner (BO) | LTuner (Self-Reflective) |
|------|-------------|-------------------------|
| **延迟改善** | 2.8% | **12.8%** |
| **最佳延迟** | 68,086 μs | **61,065 μs** |
| **所需迭代** | 70 轮 | **20 轮** (减少 71.4%) |
| **调优耗时** | 14 分钟 | **10 分钟** (减少 27.7%) |
| **配置崩溃** | 大量 (74%+) | **0 次** |

---

**原始 GPTuner 相关信息：**
- GPTuner 获得 SIGMOD Research Highlight Award 2024
- 原始论文：[VLDB 2024](https://dl.acm.org/doi/abs/10.14778/3659437.3659449) | [SIGMOD 2024 Demo](https://dl.acm.org/doi/10.1145/3626246.3654739) | [SIGMOD Record](https://doi.org/10.1145/3733620.3733641)
- 视频演示：[YouTube](https://youtu.be/Hz5Zck-9TlA)

## Table of Contents
- [LTuner System Overview](#ltuner-system-overview)
- [GPTuner System Overview](#gptuner-system-overview)
- [Quick Start](#quick-start)
- [LTuner vs GPTuner Comparison](#ltuner-vs-gptuner-comparison)
- [Demo Guidance](#demo-usage-guide)
- [Code Structure](#code-structure)
- [Roadmap](#roadmap)
- [Citation](#citation)

## LTuner System Overview

LTuner 采用 **LLM 自省式反馈（Self-Reflective Feedback）** 替代传统贝叶斯优化，通过让大语言模型分析每轮调优的性能变化，生成"文本梯度"指导下一轮参数调整，实现高效收敛。

### 系统架构

```
LTuner 调优工作流
┌─────────────────────────────────────────────────────────────┐
│  Step 1: 环境感知                                            │
│  ┌──────────────┐   ┌──────────────────┐                    │
│  │ PostgreSQL   │──>│ Feature          │──> 语义化环境描述   │
│  │ Monitor      │   │ Translator       │                    │
│  └──────────────┘   └──────────────────┘                    │
├─────────────────────────────────────────────────────────────┤
│  Step 2: 智能参数筛选                                        │
│  ┌──────────────┐   ┌──────────────────┐                    │
│  │ Causal       │──>│ MoE Experts      │──> Top-K 关键旋钮  │
│  │ Graph        │   │ (7类子领域专家)    │                    │
│  └──────────────┘   └──────────────────┘                    │
├─────────────────────────────────────────────────────────────┤
│  Step 3: 安全值域剪枝 (Surveyor-Proposer-Corrector)          │
│  ┌──────────┐  ┌───────────┐  ┌───────────┐                │
│  │ Surveyor │─>│ Proposer  │─>│ Corrector │──> 安全值域     │
│  │ 范围勘测  │  │ 值域推荐   │  │ 安全校验   │                │
│  └──────────┘  └───────────┘  └───────────┘                │
├─────────────────────────────────────────────────────────────┤
│  Step 4: 自省式反馈调优循环                                   │
│  ┌────────────────────────────────────────────┐             │
│  │  LLM 生成初始配置                            │             │
│  │       ↓                                     │             │
│  │  应用配置 → 运行 Benchmark → 采集性能指标     │             │
│  │       ↓                                     │             │
│  │  LLM 自省分析 → 生成"文本梯度"               │ ← 核心创新  │
│  │       ↓                                     │             │
│  │  基于梯度生成新配置 → 循环迭代                │             │
│  └────────────────────────────────────────────┘             │
├─────────────────────────────────────────────────────────────┤
│  Step 5: 输出最优配置 + 调优报告                              │
└─────────────────────────────────────────────────────────────┘
```

### 核心模块说明

| 模块 | 文件 | 功能 |
|------|------|------|
| **特征语义翻译器** | `feature_translator.py` | 将数据库原始指标（QPS、缓存命中率等）转化为 LLM 可理解的自然语言描述 |
| **因果知识图谱** | `causal_graph.py` | 构建参数间因果关系图，支持多跳推理（如 shared_buffers → effective_cache_size） |
| **MoE 多智能体专家** | `moe_experts.py` | 7 类子领域专家（内存、I/O、查询执行等）动态评估，加权融合筛选 Top-K 旋钮 |
| **SPC 值域剪枝** | `value_pruner.py` | Surveyor 勘测范围 → Proposer 推荐值域 → Corrector 安全校验 |
| **自省式反馈引擎** | `reflective_engine.py` | 核心引擎：LLM 分析性能变化因果关系，生成文本梯度，驱动配置迭代优化 |
| **LTuner 编排器** | `ltuner_orchestrator.py` | 串联完整工作流：环境感知 → 参数筛选 → 值域剪枝 → 自省反馈循环 → 结果输出 |

## GPTuner System Overview

<img src="/assets/gptuner_overview.png" alt="GPTuner overview" width="800">

**GPTuner** is a manual-reading database tuning system to suggest satisfactory knob configurations with reduced tuning costs. The figure above presents the tuning workflow that involves seven steps:
1. 📌 User provides the DBMS to be tuned (e.g., PostgreSQL or MySQL), the target workload, and the optimization objective (e.g., latency or throughput).
2. 📌 GPTuner collects and refines the heterogeneous knowledge from different sources (e.g., GPT-4, DBMS manuals, and web forums) to construct _Tuning Lake_, a collection of DBMS tuning knowledge.
3. 📌 GPTuner unifies the refined tuning knowledge from _Tuning Lake_ into a structured view accessible to machines (e.g., JSON).
4. 📌 GPTuner reduces the search space dimensionality by selecting important knobs to tune (i.e., fewer knobs to tune means fewer dimensions).
5. 📌 GPTuner optimizes the search space in terms of the value range for each knob based on structured knowledge.
6. 📌 GPTuner explores the optimized space via a novel Coarse-to-Fine Bayesian Optimization framework.
7. 📌 Finally, GPTuner identifies satisfactory knob configurations within resource limits (e.g., the maximum optimization time or iterations specified by users).

## Quick Start
The following instructions have been tested on Ubuntu 20.04 and PostgreSQL v14.9:

### Step 1: Install PostgreSQL
```
sudo apt-get update
sudo apt-get install postgresql-14
```

### Step 2: Install [BenchBase](https://github.com/cmu-db/benchbase) with our script
- Note: the script is tested on `openjdk version "17.0.8.1" 2023-08-24`, (you may need to update openjdk to `version 21` to keep in pace with the newest benchbase), please prepare your JAVA environment first
```
cd ./scripts
sh install_benchbase.sh postgres
```

### Step 3: Install Benchmark with our script

- Note: modify `./benchbase/target/benchbase-postgres/config/postgres/sample_{your_target_benchmark}_config.xml` to customize your tuning setting first
```
sh build_benchmark.sh postgres tpch
```

### Step 4: Install dependencies
```
sudo pip install -r requirements.txt
```

### Step 5: Execute the GPTuner to optimize your DBMS

- Note: modify `configs/postgres.ini` to determine the target DBMS first, the `restart` and `recover` commands depend on the environment and we provide Docker version
- Note: modify `src/run_gptuner.py` to set up your `api_base`, `api_key` and `model` first
- Note: please **update the structured knowledge** based on your hardware resources before running experiments. Just provide your `api_base`, `api_key` and `model`, then GPTuner's knowledge updater will complete this automatically

```
# PYTHONPATH=src python src/run_gptuner.py <dbms> <benchmark> <timeout> <seed>
PYTHONPATH=src python src/run_gptuner.py postgres tpch 180 -seed=100
```
where `<dbms>` specifies the DBMS (e.g., postgres or mysql), `<benchmark>` is the target workload (e.g., tpch or tpcc), `<timeout>` is the maximum time allowed to stress-test the benchmark, `<seed>` is the random seed used by the optimizer.

### Step 6: Execute LTuner (recommended)

```
# 运行 LTuner 自省式反馈调优
PYTHONPATH=src python src/run_ltuner.py
```

### Step 7: Run GPTuner vs LTuner Comparison Experiment

```
# 运行完整对比实验（GPTuner BO vs LTuner Self-Reflective）
cd /root/GPTuner
python src/experiments/run_real_comparison.py
```
实验结果（JSON + 图表）将保存在 `optimization_results/comparison_real/` 目录下。

### Step 8: View the optimization result:
The optimization result is stored in `optimization_results/{dbms}/{stage}/{seed}/runhistory.json`, where `{dbms}` is the target DBMS, `{stage}` is coarse or fine and `{seed}` is the random seed given by user.
- the `data` block contains the following information, we explain the project-related information below. For more details, please refer to [SMAC3 Library](https://github.com/automl/SMAC3).
    - `config_id`: i is the identifier for the knob configuration given by i-th iteration 
    - instance
    - budget
    - seed
    - `cost`: the optimization objective (e.g., throughput or latency)
    - time
    - status
    - starttime
    - endtime
    - additional_info
- the `"configs"` block contains the knob configuration of the i-th iteration, for example:
```
"configs": {
    "1": {
      "effective_io_concurrency": 200,
      "random_page_cost": 1.2 
    },
}
```

## Demo Usage Guide
### Step 1: Complete Steps 1 to 4 in the [Quick Start](#quick-start) section

### Step 2: Execute the Demo
```
PYTHONPATH=src python -m streamlit run src/demo/entrypage.py
```

### Step 3: Follow our [video demonstration](https://youtu.be/Hz5Zck-9TlA) to use the GUI
<img src="/assets/demo_page.png" alt="GPTuner demo page" width="800">

### Step 4: Visualize the Optimization Result
<img src="/assets/demo_result.png" alt="GPTuner demo result" width="800">


## Experimental Result

### Baselines
We compare GPTuner with state-of-the-art methods both using or not using natural language knowledge as input:
- [DB-BERT SIGMOD'22](https://dl.acm.org/doi/10.1145/3514221.3517843): a DBMS tuning tool that uses BERT to read the manuals and use the gained information to guide Reinforcement Learning (RL)
- SMAC: the best Bayesian Optimiztion (BO)-based method evaluated in an [Experimental Evaluation VLDB'22](https://dl.acm.org/doi/10.14778/3538598.3538604)
- GP: the classic Gassian Process-based BO approach used in [iTuned VLDB'09](https://dl.acm.org/doi/10.14778/1687627.1687767) and [OtterTune SIGMOD'17](https://dl.acm.org/doi/10.1145/3035918.3064029)
- DDPG++: a RL-based tuning method proposed in [CDBTune SIGMOD'19](https://dl.acm.org/doi/10.1145/3299869.3300085) and improved in [Inquiry VLDB'21](https://dl.acm.org/doi/10.14778/3450980.3450992)

### Result on PostgreSQL
We compare GPTuner with baselines on different DBMS (PostgreSQL and MySQL), benchmarks (TPC-H and TPC-C) and metrics (throughput and latency). We present the results on PostgreSQL in this repository. For more details, please refer to our [paper](https://web1.arxiv.org/abs/2311.03157) or [technical report](https://github.com/SolidLao/GPTuner/blob/main/gptuner-technical-report.pdf).

<img src="/assets/gptuner_result_postgresql.png" alt="GPTuner result on postgres" width="500">

## LTuner vs GPTuner Comparison

### 实验设置

| 配置项 | 值 |
|--------|-----|
| 数据库 | PostgreSQL 14.20 |
| 基准测试 | TPC-H (OLAP) |
| 优化目标 | 延迟（Latency，越低越好）|
| GPTuner 轮次 | Coarse 30 + Fine 40 = 70 轮 |
| LTuner 轮次 | 20 轮自省反馈 |
| LLM | 通义千问 qwen-plus |
| 目标旋钮数 | 57 个 |

### 实验结果

<img src="/optimization_results/comparison_real/real_tpch_performance.png" alt="Performance Comparison" width="700">

| 指标 | GPTuner (BO) | LTuner (Self-Reflective) | 改善 |
|------|-------------|-------------------------|------|
| 延迟改善 | 2.8% | **12.8%** | LTuner 提升幅度是 GPTuner 的 **4.6 倍** |
| 最佳延迟 | 68,086 μs | **61,065 μs** | 降低 10.3% |
| 所需迭代 | 70 轮 | **20 轮** | 减少 **71.4%** |
| 总耗时 | 14 分钟 | **10 分钟** | 减少 **27.7%** |
| 配置崩溃率 | ~74% | **0%** | LTuner 零崩溃 |

### 收敛曲线对比

<img src="/optimization_results/comparison_real/real_tpch_convergence.png" alt="Convergence Curve" width="700">

- **橙色 (LTuner)**：始终在低延迟区间稳定波动，第 6 轮即达到最佳 61,065 μs
- **蓝色 (GPTuner)**：剧烈震荡，大量迭代因配置崩溃返回极高惩罚值

### 效率对比

<img src="/optimization_results/comparison_real/real_tpch_time.png" alt="Efficiency Comparison" width="700">

### 总览仪表盘

<img src="/optimization_results/comparison_real/real_tpch_dashboard.png" alt="Dashboard" width="700">

### 分析

GPTuner 性能不佳的原因：
1. **SMAC 盲目采样**：Latin Hypercube 初始设计生成极端参数组合（如 shared_buffers 超过物理内存），导致 74%+ 的配置崩溃
2. **高维空间低效**：57 个旋钮的组合空间巨大，70 轮迭代中仅约 18 轮成功执行 benchmark，有效样本严重不足
3. **无语义理解**：贝叶斯优化不理解参数间的语义约束

LTuner 优势：
1. **领域知识驱动**：LLM 理解参数语义，不会生成不安全的配置
2. **文本梯度反馈**：每轮分析性能变化的因果关系，快速收敛到最优区域
3. **零崩溃**：因果知识图谱 + 规则引擎确保配置安全性

## Code Structure
- `configs/`
  - `postgres.ini`: Configuration file to optimize PostgreSQL
  - `mysql.ini`: Configuration file to optimize MySQL
- `optimization_results/`
  - `temp_results/`: Temporary storage for optimization results
  - `comparison_real/`: LTuner vs GPTuner comparison experiment results and charts
  - `postgres/`
    - `coarse/`: Coarse-stage optimization results for PostgreSQL
    - `fine/`: Fine-stage optimization results for PostgreSQL
- `scripts/`
  - `install_benbase.sh`: Script to install the BenchBase benchmark tool
  - `build_benchmark.sh`: Script to build benchmark environments
  - `recover_postgres.sh`: Script to recover the state of PostgreSQL database
  - `recover_mysql.sh`: Script to recover the state of MySQL database
- `knowledge_collection/`
  - `postgres/`
    - `target_knobs.txt`: List of target knobs for PostgreSQL tuning
    - `knob_info/`
      - `system_view.json`: Information from PostgreSQL system views (pg_settings)
      - `official_document.json`: Information from PostgreSQL official documentation
    - `knowledge_sources/`
      - `gpt/`: Knowledge sourced from GPT models
      - `manual/`: Knowledge from DBMS manuals
      - `web/`: Knowledge extracted from web sources
      - `dba/`: Knowledge from database administrators
    - `tuning_lake/`: Data lake for DBMS tuning knowledge
    - `structured_knowledge/`
      - `special/`: Specialized structured knowledge
      - `normal/`: General structured knowledge (enriched with suggested_values, min/max)
- `knowledge_base/`: RAG knowledge base for LTuner
  - `postgres_knowledge_base.pkl`: Pre-built PostgreSQL knowledge base
  - `postgres_faiss.index`: FAISS vector index for semantic retrieval
  - `postgres_embeddings.npy`: Sentence-BERT embeddings
- `example_pool/`: Pool of examples for prompt ensemble algorithm
- `sql`: Provide sql statements if you need query-level knob selection
- `src/`: Source code
  - **`ltuner/`**: **LTuner core engine modules (NEW)**
    - `feature_translator.py`: Feature semantic translator - converts raw DB metrics to natural language
    - `causal_graph.py`: Causal knowledge graph - models parameter dependencies with multi-hop reasoning
    - `moe_experts.py`: MoE multi-agent expert system - 7 sub-domain evaluators with dynamic weighting
    - `value_pruner.py`: Surveyor-Proposer-Corrector value range pruning pipeline
    - `reflective_engine.py`: Self-reflective feedback tuning engine - generates "text gradients" for optimization
    - `ltuner_orchestrator.py`: LTuner main orchestrator - coordinates the entire tuning workflow
  - **`experiments/`**: **Comparison experiment framework (NEW)**
    - `comparison.py`: Comparison runner for GPTuner vs LTuner experiments
    - `run_real_comparison.py`: Real benchmark comparison script (TPC-H)
    - `visualizer.py`: Experiment result visualization (generates PNG charts)
  - **`rag_engine/`**: **RAG retrieval engine (NEW)**
    - `knowledge_builder.py`: Knowledge base builder from structured knowledge
    - `retriever.py`: Semantic retrieval with FAISS + Sentence-BERT
  - **`monitoring/`**: **Real-time monitoring (NEW)**
    - `postgres_monitor.py`: PostgreSQL metrics collector (QPS, cache hit ratio, etc.)
    - `workload_analyzer.py`: Workload pattern analyzer
  - **`rule_engine/`**: **Safety rule engine (NEW)**
    - `safety_engine.py`: Configuration safety validator
    - `parameter_validator.py`: Parameter range and constraint checker
    - `conflict_detector.py`: Parameter conflict detection
  - **`scenario_engine/`**: **Scenario classification (NEW)**
    - `classifier.py`: Workload scenario classifier (OLTP/OLAP/Hybrid)
    - `prompt_templates.py`: Scenario-aware prompt templates
  - `demo/`: Module to execute the GUI (Demonstration Code)
  - `dbms/`
    - `dbms_template.py`: Template for database management systems
    - `postgres.py`: Implementation for PostgreSQL
    - `mysql.py`: Implementation for MySQL
  - `knowledge_handler/`
    - `gpt.py`: Module for interactions with GPT
    - `knowledge_preparation.py`: Module for knowledge preparation (**Sec. 5.1**)
    - `knowledge_transformation.py`: Module for knowledge transformation (**Sec. 5.2**)
  - `space_optimizer/`
    - `knob_selection.py`: Module for knob selection (**Sec. 6.1**)
    - `default_space.py`: Definition of default search space
    - `coarse_space.py`: Definition of coarse search space (**Sec. 6.2**)
    - `fine_space.py`: Definition of fine search space (**Sec. 6.2**)
  - `config_recommender/`
    - `workload_runner.py`: Module to run workloads
    - `coarse_stage.py`: Recommender for coarse stage configuration (**Sec. 7**)
    - `fine_stage.py`: Recommender for fine stage configuration (**Sec. 7**)
  - `run_ltuner.py`: **Main script to run LTuner (NEW)**
  - `run_gptuner.py`: Main script to run GPTuner (original baseline)

## Roadmap
- Paper version (GPTuner)
  - [x] GPTuner uses OpenAI completion API of `gpt-4` or `gpt-3.5-turbo`
  - [x] GPTuner leverages tuning knowledge from `GPT-4`, `DBMS official manuals` and `web contents`
  - [x] GPTuner supports `PostgreSQL` and `MySQL`
  - [x] GPTuner stress-tests workloads through the `BenchBase` tool
- LTuner enhancements (本项目新增)
  - [x] LTuner 自省式反馈调优引擎，替代贝叶斯优化
  - [x] MoE 多智能体专家系统，动态评估参数重要性
  - [x] Surveyor-Proposer-Corrector 三智能体值域剪枝
  - [x] 因果知识图谱，支持参数间多跳推理
  - [x] RAG 语义检索引擎（FAISS + Sentence-BERT）
  - [x] 规则引擎：资源适配、防崩溃、参数冲突检测
  - [x] 场景感知分类（OLTP/OLAP/混合负载）
  - [x] 实时监控模块（PostgreSQL 指标采集与工作负载分析）
  - [x] 支持通义千问 qwen-plus 作为低成本 LLM 替代方案
  - [x] GPTuner vs LTuner 对比实验框架与可视化
- Future implementation (We warmly invite and appreciate your contributions!)
  - [ ] GPTuner employs `locally depolyed large language models` as well
  - [ ] GPTuner collects web contents through `web-gpt` and `web-crawler`
  - [ ] GPTuner uses a `generic` stress-test tool, supporting `any given workload` optimization
  - [ ] GPTuner refines its `knowledge_collection` with a `human-in-the-loop` mechanism
  - [ ] GPTuner supports more `DBMS`
  - [ ] to be continued...

## Citation
If you use this codebase, or otherwise found our work valuable, please cite 📒:
```
@article{10.14778/3659437.3659449,
  author = {Lao, Jiale and Wang, Yibo and Li, Yufei and Wang, Jianping and Zhang, Yunjia and Cheng, Zhiyuan and Chen, Wanghu and Tang, Mingjie and Wang, Jianguo},
  title = {GPTuner: A Manual-Reading Database Tuning System via GPT-Guided Bayesian Optimization},
  year = {2024},
  issue_date = {April 2024},
  publisher = {VLDB Endowment},
  volume = {17},
  number = {8},
  issn = {2150-8097},
  url = {https://doi.org/10.14778/3659437.3659449},
  doi = {10.14778/3659437.3659449},
  abstract = {Modern database management systems (DBMS) expose hundreds of configurable knobs to control system behaviours. Determining the appropriate values for these knobs to improve DBMS performance is a long-standing problem in the database community. As there is an increasing number of knobs to tune and each knob could be in continuous or categorical values, manual tuning becomes impractical. Recently, automatic tuning systems using machine learning methods have shown great potentials. However, existing approaches still incur significant tuning costs or only yield sub-optimal performance. This is because they either ignore the extensive domain knowledge available (e.g., DBMS manuals and forum discussions) and only rely on the runtime feedback of benchmark evaluations to guide the optimization, or they utilize the domain knowledge in a limited way. Hence, we propose GPTuner, a manual-reading database tuning system that leverages domain knowledge extensively and automatically to optimize search space and enhance the runtime feedback-based optimization process. Firstly, we develop a Large Language Model (LLM)-based pipeline to collect and refine heterogeneous knowledge, and propose a prompt ensemble algorithm to unify a structured view of the refined knowledge. Secondly, using the structured knowledge, we (1) design a workload-aware and training-free knob selection strategy, (2) develop a search space optimization technique considering the value range of each knob, and (3) propose a Coarse-to-Fine Bayesian Optimization Framework to explore the optimized space. Finally, we evaluate GPTuner under different benchmarks (TPC-C and TPC-H), metrics (throughput and latency) as well as DBMS (PostgreSQL and MySQL). Compared to the state-of-the-art approaches, GPTuner identifies better configurations in 16x less time on average. Moreover, GPTuner achieves up to 30\% performance improvement (higher throughput or lower latency) over the best-performing alternative.},
  journal = {Proc. VLDB Endow.},
  month = {may},
  pages = {1939–1952},
  numpages = {14}
}

@article{10.1145/3733620.3733641,
author = {Lao, Jiale and Wang, Yibo and Li, Yufei and Wang, Jianping and Zhang, Yunjia and Cheng, Zhiyuan and Chen, Wanghu and Tang, Mingjie and Wang, Jianguo},
title = {GPTuner: An LLM-Based Database Tuning System},
year = {2025},
issue_date = {March 2025},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {54},
number = {1},
issn = {0163-5808},
url = {https://doi.org/10.1145/3733620.3733641},
doi = {10.1145/3733620.3733641},
abstract = {Selecting appropriate values for the configurable knobs of Database Management Systems (DBMS) is essential to improve performance. But because the complexity of this task has surpassed the abilities of even the best human experts, the database community turns to machine learning (ML)- based automatic tuning systems. However, these systems still incur significant tuning costs or only yield suboptimal performance, attributable to their overly high reliance on black-box optimization and the lack of integration with domain knowledge, such as DBMS manuals and forum discussions. Hence, we propose GPTuner, a manual-reading database tuning system that extensively leverages domain knowledge to automatically optimize the search space and enhance the runtime feedback-based optimization process. Firstly, we develop a Large Language Model (LLM)-based pipeline to collect and refine heterogeneous knowledge, and propose a prompt ensemble algorithm to unify a structured view of the refined knowledge. Secondly, using the structured knowledge, we (1) design a workload-aware, trainingfree knob selection strategy, (2) develop a search space optimization technique considering the value range of each knob, (3) propose a Coarse-to-Fine Bayesian Optimization Framework to explore the optimized space. Finally, we evaluate GPTuner under different benchmarks (TPC-C and TPC-H), metrics (throughput and latency) and DBMS (PostgreSQL and MySQL). Compared to state-of-the-art methods, GPTuner identifies better configurations in 16x less time on average. Moreover, GPTuner achieves up to 30\% performance improvement over the best-performing alternative.},
journal = {SIGMOD Rec.},
month = apr,
pages = {101–110},
numpages = {10}
}

@inproceedings{10.1145/3626246.3654739,
    author = {Lao, Jiale and Wang, Yibo and Li, Yufei and Wang, Jianping and Zhang, Yunjia and Cheng, Zhiyuan and Chen, Wanghu and Zhou, Yuanchun and Tang, Mingjie and Wang, Jianguo},
    title = {A Demonstration of GPTuner: A GPT-Based Manual-Reading Database Tuning System},
    year = {2024},
    isbn = {9798400704222},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3626246.3654739},
    doi = {10.1145/3626246.3654739},
    abstract = {Selecting appropriate values for the configurable knobs of Database Management Systems (DBMS) is crucial to improve performance. But because such complexity has surpassed the abilities of even the best human experts, database community turns to machine learning (ML)-based automatic tuning systems. However, these systems still incur significant tuning costs or only yield sub-optimal performance, attributable to their overly high reliance on black-box optimization and an oversight of domain knowledge. This paper demonstrates GPTuner, a manual-reading database tuning system that leverages Large Language Model (LLM) to bridge the gap between black-box optimization and white-box domain knowledge. This demonstration empowers (1) regular users with limited tuning experience to gain qualitative insights on the features of knobs, and optimize their DBMS performance automatically and efficiently, (2) database administrators and experts to further enhance GPTuner by simply contributing their invaluable tuning suggestions in natural language. Finally, we offer visitors the opportunity to explore a range of DBMS and optimization metrics, coupled with the flexibility to tailor their target workloads to their specific needs.},
    booktitle = {Companion of the 2024 International Conference on Management of Data},
    pages = {504–507},
    numpages = {4},
    keywords = {bayesian optimization, database tuning, large language model},
    location = {<conf-loc>, <city>Santiago AA</city>, <country>Chile</country>, </conf-loc>},
    series = {SIGMOD/PODS '24}
}
```
