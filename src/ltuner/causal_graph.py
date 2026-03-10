#!/usr/bin/env python3
"""
因果知识图谱 (Causal Knowledge Graph)
构建 PostgreSQL 参数间的因果依赖关系图，支持多跳推理。
不同于简单的依赖列表，本模块构建带权有向图，
可根据当前性能瓶颈沿因果链路召回具有协同效应的参数集。
"""
import sys
import os
from typing import Dict, List, Set, Optional, Tuple
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class CausalEdge:
    """因果边：描述两个参数间的因果关系"""

    def __init__(self, source: str, target: str, relation: str,
                 weight: float = 1.0, description: str = ""):
        self.source = source
        self.target = target
        self.relation = relation   # 'causes', 'triggers', 'constrains', 'conflicts'
        self.weight = weight       # 因果强度 0-1
        self.description = description

    def __repr__(self):
        return f"{self.source} --[{self.relation} w={self.weight:.2f}]--> {self.target}"


class PerformanceNode:
    """性能指标节点：描述一个可观测的性能指标"""

    def __init__(self, name: str, category: str, description: str = ""):
        self.name = name
        self.category = category   # 'memory', 'io', 'cpu', 'wal', 'lock', 'query'
        self.description = description


class CausalKnowledgeGraph:
    """
    参数因果知识图谱
    构建 参数(Knob) <-> 性能指标(Metric) <-> 参数(Knob) 的三元组因果关系图，
    支持从性能瓶颈反向追溯关键参数，以及参数间的协同效应分析。
    """

    def __init__(self, db: str = 'postgres'):
        self.db = db
        # 参数节点集合
        self.knob_nodes: Dict[str, dict] = {}
        # 性能指标节点集合
        self.metric_nodes: Dict[str, PerformanceNode] = {}
        # 有向边集合：adjacency list
        self.edges: Dict[str, List[CausalEdge]] = defaultdict(list)
        # 反向边索引（用于反向追溯）
        self.reverse_edges: Dict[str, List[CausalEdge]] = defaultdict(list)
        # 参数互斥集
        self.conflict_pairs: List[Tuple[str, str, str]] = []

        # 构建图谱
        self._build_postgresql_causal_graph()

    def _build_postgresql_causal_graph(self):
        """构建 PostgreSQL 核心参数因果图谱"""

        # ========== 注册性能指标节点 ==========
        self._add_metric("buffer_hit_ratio", "memory", "缓冲区命中率")
        self._add_metric("disk_read_rate", "io", "磁盘物理读取速率")
        self._add_metric("checkpoint_frequency", "wal", "检查点触发频率")
        self._add_metric("wal_write_latency", "wal", "WAL 日志写入延迟")
        self._add_metric("query_sort_spill", "query", "查询排序溢出到磁盘")
        self._add_metric("lock_contention", "lock", "锁竞争强度")
        self._add_metric("connection_wait", "cpu", "连接等待时间")
        self._add_metric("parallel_efficiency", "cpu", "并行查询效率")
        self._add_metric("vacuum_efficiency", "io", "自动清理效率")
        self._add_metric("seq_scan_ratio", "query", "顺序扫描比例")
        self._add_metric("temp_file_usage", "io", "临时文件使用量")
        self._add_metric("memory_pressure", "memory", "整体内存压力")
        self._add_metric("transaction_throughput", "cpu", "事务吞吐量")

        # ========== 内存子系统因果链 ==========
        # shared_buffers -> buffer_hit_ratio -> disk_read_rate
        self._add_knob("shared_buffers", "memory")
        self._add_edge("shared_buffers", "buffer_hit_ratio", "causes", 0.95,
                       "增大 shared_buffers 直接提升缓冲区命中率")
        self._add_edge("buffer_hit_ratio", "disk_read_rate", "triggers", 0.90,
                       "命中率降低导致物理磁盘读取增加")
        self._add_edge("shared_buffers", "checkpoint_frequency", "triggers", 0.70,
                       "更大的缓冲池意味着更多脏页，可能触发更频繁的检查点")
        self._add_edge("shared_buffers", "memory_pressure", "causes", 0.80,
                       "增大缓冲池增加整体内存占用")

        # effective_cache_size（影响查询规划器决策）
        self._add_knob("effective_cache_size", "memory")
        self._add_edge("effective_cache_size", "seq_scan_ratio", "causes", 0.75,
                       "较大的 effective_cache_size 使规划器更倾向索引扫描而非顺序扫描")

        # work_mem -> query_sort_spill -> temp_file_usage
        self._add_knob("work_mem", "memory")
        self._add_edge("work_mem", "query_sort_spill", "causes", 0.90,
                       "增大 work_mem 减少排序/哈希操作溢出到磁盘")
        self._add_edge("query_sort_spill", "temp_file_usage", "triggers", 0.85,
                       "排序溢出直接产生临时文件")
        self._add_edge("work_mem", "memory_pressure", "causes", 0.75,
                       "work_mem × 并发连接数 = 总内存开销")

        # maintenance_work_mem -> vacuum_efficiency
        self._add_knob("maintenance_work_mem", "memory")
        self._add_edge("maintenance_work_mem", "vacuum_efficiency", "causes", 0.80,
                       "增大维护内存加速 VACUUM 和 CREATE INDEX")

        # ========== WAL/检查点子系统因果链 ==========
        self._add_knob("max_wal_size", "wal")
        self._add_edge("max_wal_size", "checkpoint_frequency", "constrains", 0.90,
                       "增大 max_wal_size 减少检查点触发频率")
        self._add_edge("checkpoint_frequency", "disk_read_rate", "triggers", 0.60,
                       "频繁检查点增加磁盘写入负担")

        self._add_knob("wal_buffers", "wal")
        self._add_edge("wal_buffers", "wal_write_latency", "causes", 0.75,
                       "较大的 WAL 缓冲区减少 WAL 写入延迟")

        self._add_knob("checkpoint_completion_target", "wal")
        self._add_edge("checkpoint_completion_target", "disk_read_rate", "causes", 0.65,
                       "更高的完成目标使检查点写入更平滑，减少 I/O 峰值")

        self._add_knob("min_wal_size", "wal")
        self._add_edge("min_wal_size", "wal_write_latency", "constrains", 0.50,
                       "min_wal_size 过小可能导致频繁回收 WAL 段")

        # ========== 并发/连接子系统因果链 ==========
        self._add_knob("max_connections", "connection")
        self._add_edge("max_connections", "connection_wait", "causes", 0.85,
                       "增大最大连接数减少连接排队等待")
        self._add_edge("max_connections", "memory_pressure", "causes", 0.80,
                       "每个连接消耗 work_mem + 连接开销，连接数过多可致内存溢出")
        self._add_edge("max_connections", "lock_contention", "triggers", 0.60,
                       "更多并发连接可能加剧锁竞争")

        # ========== 并行查询子系统 ==========
        self._add_knob("max_parallel_workers_per_gather", "parallel")
        self._add_edge("max_parallel_workers_per_gather", "parallel_efficiency", "causes", 0.85,
                       "增加每个 Gather 节点的并行工作者提升并行查询效率")

        self._add_knob("max_parallel_workers", "parallel")
        self._add_edge("max_parallel_workers", "parallel_efficiency", "constrains", 0.80,
                       "全局并行工作者上限约束了实际可用的并行度")

        self._add_knob("max_worker_processes", "parallel")
        self._add_edge("max_worker_processes", "parallel_efficiency", "constrains", 0.70,
                       "后台工作进程上限约束了总并行能力")

        # ========== 查询规划器参数 ==========
        self._add_knob("random_page_cost", "planner")
        self._add_edge("random_page_cost", "seq_scan_ratio", "causes", 0.80,
                       "降低随机页面成本使规划器更倾向使用索引")

        self._add_knob("effective_io_concurrency", "io")
        self._add_edge("effective_io_concurrency", "disk_read_rate", "causes", 0.65,
                       "增大 I/O 并发度可改善 SSD 上的预读性能")

        self._add_knob("default_statistics_target", "planner")
        self._add_edge("default_statistics_target", "seq_scan_ratio", "causes", 0.55,
                       "更细致的统计信息帮助规划器选择更优的执行计划")

        # ========== 自动清理参数 ==========
        self._add_knob("autovacuum_vacuum_cost_delay", "autovacuum")
        self._add_edge("autovacuum_vacuum_cost_delay", "vacuum_efficiency", "causes", 0.70,
                       "降低清理延迟加速 autovacuum 执行")

        # ========== 协同关系（参数间） ==========
        # shared_buffers 增大时需同步调整 max_wal_size
        self._add_edge("shared_buffers", "max_wal_size", "causes", 0.85,
                       "增大 shared_buffers 产生更多脏页，需同步增大 max_wal_size 避免频繁检查点")

        # 并行层级约束
        self._add_edge("max_parallel_workers_per_gather", "max_parallel_workers", "constrains", 0.95,
                       "max_parallel_workers 必须 >= max_parallel_workers_per_gather")
        self._add_edge("max_parallel_workers", "max_worker_processes", "constrains", 0.95,
                       "max_worker_processes 必须 >= max_parallel_workers")

        # ========== 互斥/冲突关系 ==========
        self.conflict_pairs.append(
            ("shared_buffers", "max_connections",
             "两者均消耗内存，shared_buffers 过大时 max_connections 需控制以避免 OOM")
        )

    def _add_metric(self, name: str, category: str, description: str = ""):
        self.metric_nodes[name] = PerformanceNode(name, category, description)

    def _add_knob(self, name: str, category: str):
        self.knob_nodes[name] = {'name': name, 'category': category}

    def _add_edge(self, source: str, target: str, relation: str,
                  weight: float, description: str = ""):
        edge = CausalEdge(source, target, relation, weight, description)
        self.edges[source].append(edge)
        self.reverse_edges[target].append(edge)

    # ========== 图谱查询接口 ==========

    def get_causal_chain(self, knob_name: str, max_depth: int = 3) -> List[List[CausalEdge]]:
        """
        获取参数的正向因果链（参数 -> 影响了什么）

        Args:
            knob_name: 起始参数名
            max_depth: 最大追溯深度

        Returns:
            因果链列表，每条链是边的序列
        """
        chains = []
        self._dfs_forward(knob_name, [], set(), chains, max_depth)
        return chains

    def _dfs_forward(self, node: str, current_chain: List[CausalEdge],
                     visited: Set[str], all_chains: List, max_depth: int):
        if len(current_chain) >= max_depth:
            if current_chain:
                all_chains.append(list(current_chain))
            return

        visited.add(node)
        has_next = False

        for edge in self.edges.get(node, []):
            if edge.target not in visited:
                has_next = True
                current_chain.append(edge)
                self._dfs_forward(edge.target, current_chain, visited, all_chains, max_depth)
                current_chain.pop()

        if not has_next and current_chain:
            all_chains.append(list(current_chain))

        visited.discard(node)

    def trace_bottleneck(self, metric_name: str, max_depth: int = 3) -> List[dict]:
        """
        从性能瓶颈指标反向追溯关键参数（核心功能）

        Args:
            metric_name: 瓶颈指标名称（如 'disk_read_rate', 'lock_contention'）
            max_depth: 最大追溯深度

        Returns:
            相关参数列表，按因果权重排序
        """
        related_knobs = {}
        self._trace_backward(metric_name, 1.0, set(), related_knobs, max_depth)

        # 按累计因果权重排序
        sorted_knobs = sorted(related_knobs.items(), key=lambda x: x[1]['score'], reverse=True)

        results = []
        for knob_name, info in sorted_knobs:
            results.append({
                'knob': knob_name,
                'causal_score': info['score'],
                'path': info['path'],
                'category': self.knob_nodes.get(knob_name, {}).get('category', 'unknown')
            })

        return results

    def _trace_backward(self, node: str, cumulative_weight: float,
                        visited: Set[str], results: dict, max_depth: int, path: str = ""):
        if max_depth <= 0:
            return

        visited.add(node)
        current_path = f"{node}" if not path else f"{node} -> {path}"

        for edge in self.reverse_edges.get(node, []):
            if edge.source not in visited:
                new_weight = cumulative_weight * edge.weight
                new_path = f"{edge.source} --[{edge.relation}]--> {current_path}"

                # 如果是参数节点，记录结果
                if edge.source in self.knob_nodes:
                    if edge.source not in results or results[edge.source]['score'] < new_weight:
                        results[edge.source] = {
                            'score': new_weight,
                            'path': new_path
                        }

                # 继续反向追溯
                self._trace_backward(
                    edge.source, new_weight, visited, results, max_depth - 1, current_path
                )

        visited.discard(node)

    def get_synergy_group(self, knob_name: str) -> List[dict]:
        """
        获取与指定参数具有协同效应的参数组

        Args:
            knob_name: 参数名

        Returns:
            协同参数列表
        """
        synergy = []
        visited = set()
        visited.add(knob_name)

        # 正向查找：该参数直接影响的其他参数
        for edge in self.edges.get(knob_name, []):
            if edge.target in self.knob_nodes and edge.target not in visited:
                synergy.append({
                    'knob': edge.target,
                    'relation': edge.relation,
                    'weight': edge.weight,
                    'description': edge.description,
                    'direction': 'forward'
                })
                visited.add(edge.target)

        # 反向查找：直接影响该参数的其他参数
        for edge in self.reverse_edges.get(knob_name, []):
            if edge.source in self.knob_nodes and edge.source not in visited:
                synergy.append({
                    'knob': edge.source,
                    'relation': edge.relation,
                    'weight': edge.weight,
                    'description': edge.description,
                    'direction': 'backward'
                })
                visited.add(edge.source)

        # 通过共同影响的指标查找间接关联的参数
        affected_metrics = set()
        for edge in self.edges.get(knob_name, []):
            if edge.target in self.metric_nodes:
                affected_metrics.add(edge.target)

        for metric in affected_metrics:
            for edge in self.reverse_edges.get(metric, []):
                if edge.source in self.knob_nodes and edge.source not in visited:
                    synergy.append({
                        'knob': edge.source,
                        'relation': 'co-affects',
                        'weight': edge.weight * 0.7,
                        'description': f"共同影响 {metric}",
                        'direction': 'indirect'
                    })
                    visited.add(edge.source)

        synergy.sort(key=lambda x: x['weight'], reverse=True)
        return synergy

    def get_conflict_warnings(self, knob_set: List[str]) -> List[dict]:
        """
        检查参数集合中的潜在冲突

        Args:
            knob_set: 参数名列表

        Returns:
            冲突警告列表
        """
        warnings = []
        knob_set = set(knob_set)

        for k1, k2, desc in self.conflict_pairs:
            if k1 in knob_set and k2 in knob_set:
                warnings.append({
                    'knob_a': k1,
                    'knob_b': k2,
                    'warning': desc
                })

        return warnings

    def generate_causal_context(self, target_knobs: List[str],
                                 bottleneck_metrics: List[str] = None) -> str:
        """
        为 LLM 生成因果推理上下文文本

        Args:
            target_knobs: 当前关注的参数列表
            bottleneck_metrics: 检测到的瓶颈指标列表

        Returns:
            因果关系的自然语言描述
        """
        lines = ["## 参数因果关系分析", ""]

        # 瓶颈追溯
        if bottleneck_metrics:
            lines.append("### 瓶颈追溯")
            for metric in bottleneck_metrics:
                node = self.metric_nodes.get(metric)
                if node:
                    lines.append(f"\n**瓶颈: {node.description}** ({metric})")
                    traced = self.trace_bottleneck(metric, max_depth=3)
                    for item in traced[:5]:
                        lines.append(
                            f"  - {item['knob']} (因果得分: {item['causal_score']:.2f}) "
                            f"路径: {item['path']}"
                        )
            lines.append("")

        # 参数协同关系
        lines.append("### 参数协同关系")
        for knob in target_knobs[:10]:
            synergy = self.get_synergy_group(knob)
            if synergy:
                partners = [f"{s['knob']}({s['relation']})" for s in synergy[:3]]
                lines.append(f"- {knob} 协同参数: {', '.join(partners)}")

        # 冲突警告
        conflicts = self.get_conflict_warnings(target_knobs)
        if conflicts:
            lines.append("\n### 冲突警告")
            for c in conflicts:
                lines.append(f"- {c['knob_a']} vs {c['knob_b']}: {c['warning']}")

        return "\n".join(lines)

    def get_all_knob_names(self) -> List[str]:
        """获取图谱中所有参数名"""
        return list(self.knob_nodes.keys())

    def get_statistics(self) -> dict:
        """获取图谱统计信息"""
        total_edges = sum(len(edges) for edges in self.edges.values())
        return {
            'knob_nodes': len(self.knob_nodes),
            'metric_nodes': len(self.metric_nodes),
            'total_edges': total_edges,
            'conflict_pairs': len(self.conflict_pairs)
        }
