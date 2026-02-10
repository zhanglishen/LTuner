# LADO 知识库系统使用指南

## 概述

LADO (Lightweight Agentic Database Optimizer) 知识库系统实现了三维增强的数据库参数调优知识管理：

1. **症状维度（Symptom → Knobs）**：从性能问题反查相关参数
2. **约束维度（Constraints）**：硬约束规则和安全边界
3. **依赖维度（Dependencies + Related）**：参数间的功能依赖和资源竞争

## 核心组件

### 1. KnowledgeUnit（知识单元）

表示单个参数的完整知识，包含：

```python
{
    'name': 'shared_buffers',
    'category': 'Resource Usage / Memory',
    'context': 'postmaster',  # user/sighup/postmaster
    'unit': '8kB',
    'description': 'Sets the number of 8kB shared memory buffers...',
    'symptoms': [
        'High disk I/O due to cache misses',
        'OOM error when set too high',
        'Poor query performance'
    ],
    'dependencies': ['shmmax', 'shmall', 'huge_pages'],
    'constraints': {
        'hard_rule': 'value <= RAM * 0.4',
        'min': '16',
        'max': '1073741823'
    },
    'tuning_tips': 'Set to 25–40% of total RAM on dedicated servers...',
    'default_formula': 'RAM * 0.25'
}
```

### 2. KnowledgeBase（知识库管理类）

负责加载、索引、检索结构化知识单元。

```python
from knowledge_handler.knowledge_base import KnowledgeBase

# 初始化知识库
kb = KnowledgeBase(db='postgres')

# 获取统计信息
stats = kb.get_statistics()
print(f"总参数数: {stats['total_units']}")

# 通过症状搜索参数
results = kb.search_by_symptom("high IO wait", top_k=5)
for unit in results:
    print(f"{unit.name}: {unit.description}")

# 获取依赖链
chain = kb.get_dependency_chain('work_mem')
print(f"依赖链: {' → '.join(chain)}")

# 按类别查询
units = kb.get_by_category('Autovacuum')
```

### 3. RAG 检索器增强

升级后的 `RAGRetriever` 支持症状驱动检索和依赖分析。

```python
from rag_engine.retriever import RAGRetriever

# 初始化检索器
retriever = RAGRetriever(db='postgres')

# 症状驱动检索
results = retriever.retrieve_by_symptom("OOM error", top_k=5)
for result in results:
    print(f"{result['knob']}: {result['tuning_tips']}")

# 检索参数及依赖链
info = retriever.retrieve_with_dependencies('shared_buffers')
print(f"依赖链: {info['dependency_chain']}")
print(f"被依赖于: {info['dependent_knobs']}")

# 验证约束
validation = retriever.validate_constraints(
    knob_name='work_mem',
    value='4GB',
    hardware_info={'memory_gb': 16}
)
if not validation['valid']:
    print(f"验证失败: {validation['reason']}")
```

## 使用场景

### 场景 1：监控告警 → 参数推荐

```python
# 1. 从监控系统获取性能症状
symptom = "high IO wait during autovacuum"

# 2. 检索相关参数
kb = KnowledgeBase(db='postgres')
candidates = kb.search_by_symptom(symptom, top_k=3)

# 3. 分析依赖关系
for unit in candidates:
    print(f"\n推荐调整: {unit.name}")
    print(f"  原因: {unit.tuning_tips}")
    if unit.dependencies:
        print(f"  同时检查: {', '.join(unit.dependencies)}")
```

### 场景 2：参数值验证

```python
# 验证推荐值是否符合约束
retriever = RAGRetriever(db='postgres')

validation = retriever.validate_constraints(
    knob_name='shared_buffers',
    value='8GB',
    hardware_info={'memory_gb': 16}
)

if validation['valid']:
    print("✓ 参数值合法")
else:
    print(f"✗ {validation['reason']}")
    print(f"建议: {validation['suggestion']}")
```

### 场景 3：多视角嵌入检索

知识单元支持 4 种视角的嵌入：

1. **Definition View（定义视角）**：用于基于参数名和描述的检索
2. **Symptom View（症状视角）**：用于症状驱动的反向检索
3. **Tuning View（调优视角）**：用于获取调优建议和最佳实践
4. **Dependency View（依赖视角）**：用于分析参数间关系

```python
unit = kb.get_unit('work_mem')
views = unit.get_embedding_texts()

for view_name, text in views.items():
    print(f"{view_name}: {text}")
```

## 知识库构建

### 方法 1：重建完整知识库（包含向量索引）

```bash
cd /root/GPTuner/src/rag_engine
python knowledge_builder.py
```

输出包括：
- `knowledge_base/postgres_documents.pkl`：文档列表
- `knowledge_base/postgres_knowledge_base.pkl`：场景分类
- `knowledge_base/postgres_embeddings.npy`：向量嵌入
- `knowledge_base/postgres_faiss.index`：FAISS 索引

### 方法 2：仅测试知识库（不生成向量）

```bash
cd /root/GPTuner
python scripts/test_knowledge_base.py
```

## 知识库统计（当前状态）

- **总参数数**：153 个
- **平均症状数**：3.86 个/参数
- **有依赖的参数**：152 个
- **有约束的参数**：153 个
- **需要重启的参数**：27 个

### 分类分布（Top 10）

| 类别 | 参数数 |
|------|--------|
| Planner | 37 |
| WAL | 27 |
| Resource Usage | 17 |
| Replication | 13 |
| Autovacuum | 13 |
| Reporting and Logging | 10 |
| Resource Usage / Memory | 8 |
| Statistics | 6 |
| Lock Management | 5 |
| Resource | 3 |

## 测试示例

### 症状检索测试

```python
# 测试 1：高 IO 等待
results = kb.search_by_symptom("high IO wait")
# 返回: autovacuum_vacuum_cost_limit, autovacuum_work_mem, wal_buffers

# 测试 2：内存溢出
results = kb.search_by_symptom("OOM error")
# 返回: logical_decoding_work_mem, work_mem, shared_buffers

# 测试 3：慢查询
results = kb.search_by_symptom("slow query")
# 返回: min_parallel_table_scan_size, enable_nestloop, effective_io_concurrency
```

### 依赖链测试

```python
# work_mem 的完整依赖链
chain = kb.get_dependency_chain('work_mem')
# 返回: ['work_mem', 'max_connections', 'shared_buffers', ...]

# autovacuum_vacuum_cost_delay 的依赖链
chain = kb.get_dependency_chain('autovacuum_vacuum_cost_delay')
# 返回: 包含 26 个参数的完整依赖链
```

## API 参考

### KnowledgeBase 类

| 方法 | 说明 |
|------|------|
| `get_unit(knob_name)` | 获取指定参数的知识单元 |
| `search_by_symptom(query, top_k)` | 症状驱动检索 |
| `get_by_category(category)` | 按类别查询 |
| `get_dependent_knobs(knob_name)` | 获取依赖该参数的所有参数 |
| `get_dependency_chain(knob_name)` | 获取完整依赖链 |
| `get_statistics()` | 获取知识库统计信息 |

### KnowledgeUnit 类

| 方法 | 说明 |
|------|------|
| `get_embedding_texts()` | 获取多视角嵌入文本 |
| `matches_symptom(query)` | 检查是否匹配症状 |
| `get_constraint_rule()` | 获取硬约束规则 |
| `needs_restart()` | 是否需要重启数据库 |

### RAGRetriever 类

| 方法 | 说明 |
|------|------|
| `retrieve_by_symptom(query, top_k)` | 症状驱动检索 |
| `retrieve_with_dependencies(knob_name)` | 检索参数及依赖链 |
| `validate_constraints(knob_name, value)` | 验证参数值约束 |

## 扩展性

### 添加新的症状关键词

编辑 `knowledge_collection/postgres/structured_knowledge/normal/{knob_name}.json`：

```json
{
    "symptoms": [
        "high IO wait",
        "excessive disk usage",
        "slow autovacuum progress"
    ]
}
```

### 添加新的约束规则

```json
{
    "constraints": {
        "hard_rule": "(value <= RAM * 0.4) && (value >= 16MB)",
        "min": "16",
        "max": "1073741823"
    }
}
```

### 添加新的依赖关系

```json
{
    "dependencies": [
        "max_connections",
        "shared_buffers"
    ],
    "related_knobs": [
        "maintenance_work_mem",
        "temp_buffers"
    ]
}
```

## 注意事项

1. **重启需求**：`context='postmaster'` 的参数修改后需要重启数据库
2. **约束验证**：`hard_rule` 支持简单表达式，复杂规则需要规则引擎
3. **依赖传递**：依赖链会递归展开，注意避免循环依赖
4. **症状匹配**：基于关键词分词，英文效果最佳

## 相关文件

- `src/knowledge_handler/knowledge_base.py` - LADO 知识库核心
- `src/rag_engine/knowledge_builder.py` - 知识库构建器
- `src/rag_engine/retriever.py` - RAG 检索器
- `scripts/build_structured_data.py` - LLM 驱动的知识结构化
- `scripts/test_knowledge_base.py` - 知识库测试脚本
- `knowledge_collection/postgres/structured_knowledge/` - 知识库数据

## 许可证

与 GPTuner 主项目保持一致
