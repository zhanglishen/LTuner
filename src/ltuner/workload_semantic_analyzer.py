#!/usr/bin/env python3
"""
工作负载语义分析器 (Workload Semantic Analyzer)
将原始数据库指标和 benchmark 结果转换为 LLM 可理解的自然语言诊断，
使 LLM 能基于 "参数如何影响查询行为" 的因果证据做出更精准的调优决策。

核心能力：
1. 逐查询延迟变化对比（哪条查询变快/变慢了）
2. EXPLAIN 执行计划模式摘要（Hash Join 溢出、并行度、排序方法等）
3. 系统级诊断（等待事件分布、临时文件统计、表扫描模式）
"""
import os
import sys
import json
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class WorkloadSemanticAnalyzer:
    """
    工作负载语义分析器
    将原始数值数据转换为 LLM 可理解的自然语言诊断文本
    """

    # TPC-H 查询描述映射（帮助 LLM 理解查询用途）
    TPCH_QUERY_DESCRIPTIONS = {
        'Q1': '定价汇总报告(全表扫描+聚合)',
        'Q2': '最低成本供应商(多表JOIN+子查询)',
        'Q3': '运输优先级(三表JOIN+排序)',
        'Q4': '订单优先级检查(EXISTS子查询)',
        'Q5': '本地供应商收入(六表JOIN)',
        'Q6': '预测收入变化(单表扫描+过滤)',
        'Q7': '运输量(多表JOIN+聚合)',
        'Q8': '国家市场份额(八表JOIN)',
        'Q9': '产品利润(六表JOIN+聚合)',
        'Q10': '退货报告(四表JOIN)',
        'Q11': '重要库存标识(聚合+HAVING)',
        'Q12': '运输模式(二表JOIN+CASE)',
        'Q13': '客户分布(LEFT JOIN+聚合)',
        'Q14': '促销效果(二表JOIN)',
        'Q15': '顶级供应商(子查询+聚合)',
        'Q16': '零件供应商关系(三表JOIN+NOT IN)',
        'Q17': '小量订单收入(子查询+聚合)',
        'Q18': '大量客户(子查询+GROUP BY+HAVING)',
        'Q19': '折扣收入(复杂OR条件)',
        'Q20': '潜在零件促销(子查询嵌套)',
        'Q21': '等待供应商(四表JOIN+EXISTS+NOT EXISTS)',
        'Q22': '全球销售机会(子查询+NOT EXISTS)',
    }

    def __init__(self, dbms=None, runner=None, sql_dir: str = './sql/tpch'):
        """
        初始化语义分析器

        Args:
            dbms: PgDBMS 实例（用于执行 EXPLAIN）
            runner: BenchbaseRunner 实例（用于获取逐查询延迟）
            sql_dir: TPC-H SQL 文件目录
        """
        self.dbms = dbms
        self.runner = runner
        self.sql_dir = sql_dir

    # ========== 1. 逐查询延迟变化分析 ==========

    def analyze_query_deltas(self, prev_latencies: Dict[str, float],
                             curr_latencies: Dict[str, float],
                             threshold_pct: float = 5.0) -> str:
        """
        分析两轮 benchmark 的逐查询延迟变化，生成语义化描述。
        只报告变化超过阈值的查询，帮助 LLM 聚焦关键变化。

        Args:
            prev_latencies: 上一轮逐查询延迟 {"Q1": 1380.5, ...}（单位 ms）
            curr_latencies: 当前轮逐查询延迟
            threshold_pct: 只报告变化超过此百分比的查询

        Returns:
            自然语言描述文本
        """
        if not prev_latencies or not curr_latencies:
            return ""

        changes = []
        for q in sorted(curr_latencies.keys(), key=lambda x: int(x[1:])):
            curr = curr_latencies.get(q, 0)
            prev = prev_latencies.get(q, 0)
            if prev > 0 and curr > 0:
                delta_pct = (curr - prev) / prev * 100
                changes.append((q, prev, curr, delta_pct))

        if not changes:
            return ""

        # 按绝对变化幅度排序
        changes.sort(key=lambda x: abs(x[3]), reverse=True)

        # 分类
        improved = [(q, p, c, d) for q, p, c, d in changes if d < -threshold_pct]
        degraded = [(q, p, c, d) for q, p, c, d in changes if d > threshold_pct]
        stable_count = len(changes) - len(improved) - len(degraded)

        lines = ["## 查询级性能变化（本轮 vs 上轮）"]

        if improved:
            lines.append("### 改善的查询:")
            for q, prev, curr, delta in improved[:8]:
                desc = self.TPCH_QUERY_DESCRIPTIONS.get(q, '')
                lines.append(
                    f"  {q}: {prev:.0f}ms -> {curr:.0f}ms "
                    f"(改善 {abs(delta):.1f}%)  [{desc}]"
                )

        if degraded:
            lines.append("### 退化的查询（需关注）:")
            for q, prev, curr, delta in degraded[:8]:
                desc = self.TPCH_QUERY_DESCRIPTIONS.get(q, '')
                lines.append(
                    f"  {q}: {prev:.0f}ms -> {curr:.0f}ms "
                    f"(退化 {delta:.1f}%)  [{desc}]"
                )

        if stable_count > 0:
            lines.append(f"### 其余 {stable_count} 条查询变化 <{threshold_pct}%，省略")

        # 总结
        total_prev = sum(p for _, p, _, _ in changes)
        total_curr = sum(c for _, _, c, _ in changes)
        if total_prev > 0:
            overall_delta = (total_curr - total_prev) / total_prev * 100
            direction = "改善" if overall_delta < 0 else "退化"
            lines.append(f"\n总延迟: {total_prev:.0f}ms -> {total_curr:.0f}ms "
                         f"({direction} {abs(overall_delta):.1f}%)")

        return "\n".join(lines)

    # ========== 2. EXPLAIN 执行计划分析 ==========

    def analyze_explain_plans(self, queries_to_analyze: Optional[List[int]] = None) -> str:
        """
        对 TPC-H 查询执行 EXPLAIN (FORMAT JSON)，提取执行计划关键特征。
        包括：根节点类型、并行度、排序溢出、Hash 溢出等。

        Args:
            queries_to_analyze: 要分析的查询编号列表，默认全部 22 条

        Returns:
            执行计划诊断的自然语言描述
        """
        if not self.dbms:
            return ""

        if queries_to_analyze is None:
            queries_to_analyze = list(range(1, 23))

        plan_summaries = []

        for qi in queries_to_analyze:
            sql_path = os.path.join(self.sql_dir, f'Q{qi}.sql')
            if not os.path.exists(sql_path):
                continue

            try:
                with open(sql_path, 'r') as f:
                    sql = f.read().strip().rstrip(';')

                explain_sql = f"EXPLAIN (FORMAT JSON) {sql}"
                result, _ = self.dbms.get_sql_result(explain_sql)

                if result and result[0] and result[0][0]:
                    plan_json = result[0][0]
                    if isinstance(plan_json, str):
                        plan_json = json.loads(plan_json)

                    if isinstance(plan_json, list) and len(plan_json) > 0:
                        plan_node = plan_json[0].get('Plan', {})
                        summary = self._extract_plan_features(f'Q{qi}', plan_node)
                        if summary:
                            plan_summaries.append(summary)

            except Exception as e:
                plan_summaries.append(f"  - Q{qi}: 分析失败({str(e)[:60]})")

        if not plan_summaries:
            return ""

        lines = ["## 执行计划诊断"]
        lines.extend(plan_summaries)
        return "\n".join(lines)

    def _extract_plan_features(self, query_name: str, plan_node: dict,
                               depth: int = 0) -> str:
        """
        递归提取执行计划的关键特征

        Args:
            query_name: 查询名称（如 Q1）
            plan_node: EXPLAIN JSON 中的 Plan 节点
            depth: 递归深度

        Returns:
            单行摘要文本
        """
        node_type = plan_node.get('Node Type', 'Unknown')
        features = []

        # 1. 根节点类型
        features.append(node_type)

        # 2. 并行度
        workers_planned = plan_node.get('Workers Planned', 0)
        if workers_planned > 0:
            features.append(f"parallel={workers_planned}workers")

        # 3. 排序方法（Sort 节点）
        sort_method = plan_node.get('Sort Method', '')
        if sort_method:
            if 'external' in sort_method.lower():
                features.append(f"排序溢出磁盘({sort_method})!")
            else:
                features.append(f"排序:{sort_method}")

        # 4. Hash 溢出（Hash Join / Hash Aggregate）
        hash_batches = plan_node.get('Hash Batches', 0)
        if hash_batches > 1:
            features.append(f"Hash溢出({hash_batches}batches,work_mem可能不足)")

        # 5. 行数估计偏差
        plan_rows = plan_node.get('Plan Rows', 0)
        actual_rows = plan_node.get('Actual Rows')
        if actual_rows is not None and plan_rows > 0:
            ratio = actual_rows / plan_rows if plan_rows > 0 else 0
            if ratio > 10 or ratio < 0.1:
                features.append(f"行数估计偏差{ratio:.1f}x(统计信息可能过期)")

        # 6. 递归处理子节点，提取溢出/并行等关键特征
        sub_features = []
        for child in plan_node.get('Plans', []):
            child_summary = self._extract_plan_child_features(child)
            if child_summary:
                sub_features.append(child_summary)

        desc = self.TPCH_QUERY_DESCRIPTIONS.get(query_name, '')
        feature_str = ', '.join(features)

        result = f"  - {query_name}: {feature_str}"
        if sub_features:
            result += " | 子节点: " + "; ".join(sub_features[:3])
        if desc:
            result += f"  [{desc}]"

        return result

    def _extract_plan_child_features(self, node: dict) -> str:
        """提取子节点中的关键异常特征（溢出、并行等）"""
        alerts = []
        node_type = node.get('Node Type', '')

        # Hash 溢出
        hash_batches = node.get('Hash Batches', 0)
        if hash_batches > 1:
            alerts.append(f"{node_type}:Hash溢出({hash_batches}batches)")

        # Sort 溢出
        sort_method = node.get('Sort Method', '')
        if 'external' in sort_method.lower():
            alerts.append(f"{node_type}:排序溢出磁盘")

        # 并行
        workers = node.get('Workers Planned', 0)
        if workers > 0:
            alerts.append(f"{node_type}:parallel={workers}")

        # 递归
        for child in node.get('Plans', []):
            child_alert = self._extract_plan_child_features(child)
            if child_alert:
                alerts.append(child_alert)

        return "; ".join(alerts) if alerts else ""

    # ========== 3. 系统级诊断 ==========

    def analyze_system_diagnostics(self, monitor=None) -> str:
        """
        综合等待事件、临时文件、表扫描模式，生成系统级诊断。

        Args:
            monitor: PostgreSQLMonitor 实例

        Returns:
            系统诊断的自然语言描述
        """
        if not monitor:
            return ""

        lines = ["## 系统诊断"]

        # 1. 等待事件分布
        try:
            wait_events = monitor.get_wait_events()
            if wait_events:
                total = sum(e['count'] for e in wait_events)
                lines.append("### 等待事件分布:")
                for ev in wait_events[:5]:
                    pct = ev['count'] / total * 100 if total > 0 else 0
                    interpretation = self._interpret_wait_event(ev['type'], ev['event'])
                    lines.append(
                        f"  - {ev['type']}:{ev['event']} "
                        f"占 {pct:.0f}% ({ev['count']}次) "
                        f"-- {interpretation}"
                    )
            else:
                lines.append("### 等待事件: 当前无活跃等待（采集时数据库可能空闲）")
        except Exception as e:
            lines.append(f"### 等待事件: 采集失败({e})")

        # 2. 临时文件统计
        try:
            temp_stats = monitor.get_temp_file_stats()
            temp_files = temp_stats.get('temp_files', 0)
            temp_mb = temp_stats.get('temp_bytes_mb', 0)
            lines.append("### 临时文件统计:")
            if temp_files > 0:
                severity = "严重" if temp_mb > 1000 else "中等" if temp_mb > 100 else "轻微"
                lines.append(
                    f"  - 临时文件: {temp_files} 个，共 {temp_mb:.1f} MB "
                    f"-> work_mem 不足({severity})，排序/Hash 操作溢出到磁盘"
                )
                lines.append(
                    "  - 建议: 增大 work_mem 可减少磁盘溢出，直接降低排序和 Hash Join 延迟"
                )
            else:
                lines.append("  - 临时文件: 0 个 -> work_mem 充足，无磁盘溢出")
        except Exception as e:
            lines.append(f"### 临时文件: 采集失败({e})")

        # 3. 表扫描模式
        try:
            scan_patterns = monitor.get_table_scan_patterns()
            if scan_patterns:
                lines.append("### 表扫描模式:")
                for sp in scan_patterns[:5]:
                    if sp['seq_pct'] > 80 and sp['seq_scans'] > 100:
                        lines.append(
                            f"  - {sp['table']}: {sp['seq_pct']:.0f}% 顺序扫描 "
                            f"(seq={sp['seq_scans']}, idx={sp['idx_scans']}) "
                            f"-> 大量全表扫描，考虑增大 effective_cache_size "
                            f"或降低 random_page_cost"
                        )
                    else:
                        lines.append(
                            f"  - {sp['table']}: seq={sp['seq_scans']}, "
                            f"idx={sp['idx_scans']} ({sp['seq_pct']:.0f}% seq)"
                        )
        except Exception as e:
            lines.append(f"### 表扫描模式: 采集失败({e})")

        return "\n".join(lines)

    def _interpret_wait_event(self, event_type: str, event_name: str) -> str:
        """解释等待事件的含义及调优建议"""
        interpretations = {
            ('IO', 'DataFileRead'): '磁盘数据读取，瓶颈在I/O → 增大 shared_buffers/effective_cache_size',
            ('IO', 'DataFileWrite'): '磁盘数据写入 → 调整 checkpoint/bgwriter 参数',
            ('IO', 'WALSync'): 'WAL 同步等待 → 考虑 synchronous_commit=off 或增大 wal_buffers',
            ('IO', 'WALWrite'): 'WAL 写入 → 增大 wal_buffers 或 max_wal_size',
            ('IO', 'BufFileRead'): '临时文件读取(排序/Hash溢出) → 增大 work_mem',
            ('IO', 'BufFileWrite'): '临时文件写入(排序/Hash溢出) → 增大 work_mem',
            ('LWLock', 'BufferMapping'): 'shared_buffers 内部竞争 → 可能过大或并发过高',
            ('LWLock', 'WALWriteLock'): 'WAL 写入锁 → 增大 wal_buffers',
            ('LWLock', 'BufferContent'): '缓冲区内容锁 → shared_buffers 竞争',
            ('Lock', 'relation'): '表级锁等待 → 并发事务竞争',
            ('Lock', 'transactionid'): '事务锁等待 → 长事务阻塞',
            ('CPU', 'Computing'): 'CPU 计算中(无等待事件) → 参数调优空间较小',
        }
        return interpretations.get(
            (event_type, event_name),
            f'{event_type} 类型等待'
        )

    # ========== 4. 综合语义上下文生成 ==========

    def generate_semantic_context(self,
                                  prev_query_latencies: Optional[Dict[str, float]] = None,
                                  curr_query_latencies: Optional[Dict[str, float]] = None,
                                  monitor=None,
                                  include_explain: bool = False) -> str:
        """
        汇总所有分析，生成完整的语义上下文供 LLM prompt 使用。

        Args:
            prev_query_latencies: 上轮逐查询延迟
            curr_query_latencies: 当前轮逐查询延迟
            monitor: PostgreSQLMonitor 实例
            include_explain: 是否包含 EXPLAIN 分析（耗时较长，可选）

        Returns:
            完整的语义化上下文文本
        """
        sections = []

        # 1. 逐查询延迟变化
        if prev_query_latencies and curr_query_latencies:
            query_delta = self.analyze_query_deltas(
                prev_query_latencies, curr_query_latencies
            )
            if query_delta:
                sections.append(query_delta)

        # 2. EXPLAIN 执行计划（仅在需要时，如首轮或指标异常时）
        if include_explain:
            explain_summary = self.analyze_explain_plans()
            if explain_summary:
                sections.append(explain_summary)

        # 3. 系统诊断
        if monitor:
            sys_diag = self.analyze_system_diagnostics(monitor)
            if sys_diag:
                sections.append(sys_diag)

        if not sections:
            return ""

        return "\n\n".join(sections)

    def generate_initial_diagnostic(self, monitor=None) -> str:
        """
        生成初始诊断上下文（首轮优化前），包含 EXPLAIN 和系统诊断。
        这帮助 LLM 在第一轮就了解当前的执行计划特征和系统瓶颈。

        Args:
            monitor: PostgreSQLMonitor 实例

        Returns:
            初始诊断文本
        """
        sections = []

        # EXPLAIN 分析（首轮值得花时间做完整分析）
        explain_summary = self.analyze_explain_plans()
        if explain_summary:
            sections.append(explain_summary)

        # 系统诊断
        if monitor:
            sys_diag = self.analyze_system_diagnostics(monitor)
            if sys_diag:
                sections.append(sys_diag)

        if not sections:
            return ""

        return "\n\n".join(sections)
