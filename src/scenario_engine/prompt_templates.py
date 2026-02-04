"""
场景专属提示模板
为不同场景提供定制化的 Prompt Engineering
"""

class PromptTemplates:
    """场景专属提示模板管理器"""
    
    # 参数选择提示模板
    KNOB_SELECTION_PROMPTS = {
        'OLTP': """
你是一位经验丰富的数据库管理员(DBA)，擅长PostgreSQL性能调优。

## 场景特征
当前场景：**高并发OLTP（在线事务处理）**
- 特点：大量短事务、高QPS、随机IO为主
- 关键指标：响应时间、吞吐量、并发连接数
- 瓶颈：连接管理、缓存命中率、锁竞争

## 任务
从以下候选参数中，评估每个参数对OLTP场景性能的影响，并给出0-1之间的重要性评分。
评分标准：
- 0.9-1.0：对OLTP性能有显著影响，必须调优
- 0.7-0.8：对OLTP性能有较大影响，建议调优
- 0.5-0.6：对OLTP性能有一定影响
- 0.0-0.4：对OLTP性能影响较小

## OLTP场景重点关注参数类别
1. **连接管理**：max_connections, superuser_reserved_connections
2. **缓存配置**：shared_buffers, effective_cache_size
3. **并发控制**：max_parallel_workers_per_gather, max_worker_processes
4. **锁和事务**：deadlock_timeout, lock_timeout
5. **随机IO优化**：random_page_cost, effective_io_concurrency

## 工作负载信息
{workload_info}

## 历史成功案例
{historical_context}

## 候选参数列表
{candidate_knobs}

## 输出格式
请以JSON格式输出评分结果：
{{
    "knob_name": score,  // 每个参数的评分（0-1之间的数字）
    ...
}}
""",
        
        'OLAP': """
你是一位经验丰富的数据库管理员(DBA)，擅长PostgreSQL性能调优。

## 场景特征
当前场景：**批量OLAP（在线分析处理）**
- 特点：复杂查询、大数据扫描、聚合计算
- 关键指标：查询延迟、吞吐量、资源利用率
- 瓶颈：内存不足、CPU瓶颈、磁盘IO

## 任务
从以下候选参数中，评估每个参数对OLAP场景性能的影响，并给出0-1之间的重要性评分。
评分标准：
- 0.9-1.0：对OLAP性能有显著影响，必须调优
- 0.7-0.8：对OLAP性能有较大影响，建议调优
- 0.5-0.6：对OLAP性能有一定影响
- 0.0-0.4：对OLAP性能影响较小

## OLAP场景重点关注参数类别
1. **内存配置**：work_mem, maintenance_work_mem, shared_buffers
2. **并行查询**：max_parallel_workers, max_parallel_workers_per_gather
3. **扫描优化**：seq_page_cost, random_page_cost
4. **排序和聚合**：work_mem, temp_buffers
5. **IO优化**：effective_io_concurrency, wal_buffers

## 工作负载信息
{workload_info}

## 历史成功案例
{historical_context}

## 候选参数列表
{candidate_knobs}

## 输出格式
请以JSON格式输出评分结果：
{{
    "knob_name": score,  // 每个参数的评分（0-1之间的数字）
    ...
}}
""",
        
        'HYBRID': """
你是一位经验丰富的数据库管理员(DBA)，擅长PostgreSQL性能调优。

## 场景特征
当前场景：**混合负载（OLTP + OLAP）**
- 特点：同时存在短事务和复杂查询
- 关键指标：综合性能、资源平衡
- 挑战：需要平衡两类场景的需求

## 任务
从以下候选参数中，评估每个参数对混合场景性能的影响，并给出0-1之间的重要性评分。
重点关注能够同时改善OLTP和OLAP性能的参数，或者可以动态调整的参数。

## 混合场景重点关注参数类别
1. **内存配置**：shared_buffers, effective_cache_size（平衡两类负载）
2. **并发控制**：max_connections, max_worker_processes（支持两类查询）
3. **查询优化器**：random_page_cost, cpu_tuple_cost（平衡不同查询模式）
4. **资源限制**：work_mem（防止单个查询占用过多资源）

## 工作负载信息
{workload_info}

## 历史成功案例
{historical_context}

## 候选参数列表
{candidate_knobs}

## 输出格式
请以JSON格式输出评分结果：
{{
    "knob_name": score,  // 每个参数的评分（0-1之间的数字）
    ...
}}
"""
    }
    
    # 参数值推荐提示模板
    VALUE_RECOMMENDATION_PROMPTS = {
        'OLTP': """
你是一位经验丰富的数据库管理员(DBA)，需要为PostgreSQL的参数推荐合适的值。

## 场景信息
- 场景类型：**高并发OLTP**
- 工作负载特征：{workload_stats}
- 硬件配置：{hardware_info}

## 参数信息
- 参数名：{knob_name}
- 当前值：{current_value}
- 允许范围：{min_value} - {max_value}

## 相关知识库
{rag_context}

## OLTP场景调优原则
1. shared_buffers：建议设置为总内存的25%（避免过大导致双缓存）
2. effective_cache_size：建议设置为总内存的50-75%
3. work_mem：建议较小值（4-16MB），避免大量并发查询时OOM
4. max_connections：根据实际并发数设置，但不宜过大
5. random_page_cost：建议降低到1.1-2.0以鼓励使用索引

## 任务
基于以上信息，为参数 {knob_name} 推荐1-3个合适的值，并说明推荐理由。

## 输出格式（JSON）
{{
    "recommended_values": [值1, 值2, 值3],
    "primary_recommendation": 最推荐的值,
    "reasoning": "推荐理由，包括：1) 为什么选择这个值 2) 与场景的关系 3) 预期效果",
    "risks": "潜在风险或注意事项"
}}
""",
        
        'OLAP': """
你是一位经验丰富的数据库管理员(DBA)，需要为PostgreSQL的参数推荐合适的值。

## 场景信息
- 场景类型：**批量OLAP**
- 工作负载特征：{workload_stats}
- 硬件配置：{hardware_info}

## 参数信息
- 参数名：{knob_name}
- 当前值：{current_value}
- 允许范围：{min_value} - {max_value}

## 相关知识库
{rag_context}

## OLAP场景调优原则
1. work_mem：建议较大值（64-512MB），支持复杂排序和哈希操作
2. maintenance_work_mem：建议大值（512MB-2GB），加速索引构建和VACUUM
3. shared_buffers：建议设置为总内存的25-40%
4. max_parallel_workers_per_gather：根据CPU核心数设置（2-8）
5. random_page_cost：可以降低到1.1以鼓励索引扫描，但seq_page_cost也要相应调整

## 任务
基于以上信息，为参数 {knob_name} 推荐1-3个合适的值，并说明推荐理由。

## 输出格式（JSON）
{{
    "recommended_values": [值1, 值2, 值3],
    "primary_recommendation": 最推荐的值,
    "reasoning": "推荐理由，包括：1) 为什么选择这个值 2) 与场景的关系 3) 预期效果",
    "risks": "潜在风险或注意事项"
}}
""",
        
        'HYBRID': """
你是一位经验丰富的数据库管理员(DBA)，需要为PostgreSQL的参数推荐合适的值。

## 场景信息
- 场景类型：**混合负载（OLTP + OLAP）**
- 工作负载特征：{workload_stats}
- 硬件配置：{hardware_info}

## 参数信息
- 参数名：{knob_name}
- 当前值：{current_value}
- 允许范围：{min_value} - {max_value}

## 相关知识库
{rag_context}

## 混合场景调优原则
1. 平衡原则：参数值应该在OLTP和OLAP的推荐值之间折中
2. shared_buffers：25-30%内存（介于OLTP和OLAP之间）
3. work_mem：16-64MB（防止OLTP的OOM，同时支持一定的OLAP操作）
4. 动态调整：考虑使用连接级参数设置，为不同类型查询设置不同的参数

## 任务
基于以上信息，为参数 {knob_name} 推荐1-3个合适的值，并说明推荐理由。
特别注意平衡OLTP和OLAP两类负载的需求。

## 输出格式（JSON）
{{
    "recommended_values": [值1, 值2, 值3],
    "primary_recommendation": 最推荐的值,
    "reasoning": "推荐理由，包括：1) 为什么选择这个值 2) 如何平衡两类场景 3) 预期效果",
    "risks": "潜在风险或注意事项"
}}
"""
    }
    
    @classmethod
    def get_knob_selection_prompt(cls, scenario, workload_info, candidate_knobs, historical_context=""):
        """
        获取参数选择提示
        
        Args:
            scenario: 场景类型
            workload_info: 工作负载信息
            candidate_knobs: 候选参数列表
            historical_context: 历史上下文
            
        Returns:
            格式化的提示文本
        """
        template = cls.KNOB_SELECTION_PROMPTS.get(scenario, cls.KNOB_SELECTION_PROMPTS['HYBRID'])
        
        # 格式化候选参数
        if isinstance(candidate_knobs, list):
            knobs_str = ", ".join(candidate_knobs)
        else:
            knobs_str = str(candidate_knobs)
            
        # 格式化工作负载信息
        if isinstance(workload_info, dict):
            workload_str = "\n".join([f"- {k}: {v}" for k, v in workload_info.items()])
        else:
            workload_str = str(workload_info)
            
        return template.format(
            workload_info=workload_str,
            historical_context=historical_context or "暂无历史案例",
            candidate_knobs=knobs_str
        )
        
    @classmethod
    def get_value_recommendation_prompt(cls, scenario, knob_name, current_value, 
                                       min_value, max_value, workload_stats,
                                       hardware_info, rag_context):
        """
        获取参数值推荐提示
        
        Args:
            scenario: 场景类型
            knob_name: 参数名
            current_value: 当前值
            min_value: 最小值
            max_value: 最大值
            workload_stats: 工作负载统计
            hardware_info: 硬件信息
            rag_context: RAG检索的上下文
            
        Returns:
            格式化的提示文本
        """
        template = cls.VALUE_RECOMMENDATION_PROMPTS.get(scenario, cls.VALUE_RECOMMENDATION_PROMPTS['HYBRID'])
        
        # 格式化硬件信息
        if isinstance(hardware_info, dict):
            hw_str = "\n".join([f"- {k}: {v}" for k, v in hardware_info.items()])
        else:
            hw_str = str(hardware_info)
            
        # 格式化工作负载统计
        if isinstance(workload_stats, dict):
            wl_str = "\n".join([f"- {k}: {v}" for k, v in workload_stats.items()])
        else:
            wl_str = str(workload_stats)
            
        return template.format(
            knob_name=knob_name,
            current_value=current_value,
            min_value=min_value,
            max_value=max_value,
            workload_stats=wl_str,
            hardware_info=hw_str,
            rag_context=rag_context or "暂无相关知识"
        )


if __name__ == '__main__':
    # 测试提示模板
    print("=== OLTP 参数选择提示示例 ===\n")
    prompt = PromptTemplates.get_knob_selection_prompt(
        scenario='OLTP',
        workload_info={'qps': 500, 'avg_latency': '50ms'},
        candidate_knobs=['shared_buffers', 'work_mem', 'max_connections'],
        historical_context="过去成功案例：shared_buffers=8GB, work_mem=16MB"
    )
    print(prompt[:500] + "...\n")
    
    print("=== OLAP 参数值推荐提示示例 ===\n")
    prompt = PromptTemplates.get_value_recommendation_prompt(
        scenario='OLAP',
        knob_name='work_mem',
        current_value='4MB',
        min_value='64kB',
        max_value='2GB',
        workload_stats={'avg_query_time': '5.2s', 'complex_queries': 'yes'},
        hardware_info={'memory_gb': 64, 'cpu_cores': 16},
        rag_context="work_mem 控制排序和哈希操作的内存..."
    )
    print(prompt[:500] + "...")
