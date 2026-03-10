#!/usr/bin/env python3
"""
MoE 多智能体专家系统 (Mixture of Experts)
实现 7 类子领域评估器，通过动态权重分配和多专家融合筛选
从数百个候选参数中识别本次调优最关键的参数子集。
"""
import sys
import os
from typing import Dict, List, Optional
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class ExpertEvaluator:
    """专家评估器基类"""

    def __init__(self, name: str, domain: str):
        self.name = name
        self.domain = domain
        # 每个专家关注的参数关键词
        self.relevant_keywords: List[str] = []
        # 专家权重（由 Manager 动态分配）
        self.weight: float = 1.0

    def evaluate(self, knob_name: str, knob_info: dict,
                 workload_context: str, metrics: dict) -> dict:
        """
        评估参数的重要性

        Args:
            knob_name: 参数名
            knob_info: 参数元信息
            workload_context: 负载语义描述
            metrics: 当前性能指标

        Returns:
            {score: 0-100, reason: str}
        """
        raise NotImplementedError


class MemoryExpert(ExpertEvaluator):
    """内存管理专家：评估内存缓冲、缓存相关参数"""

    def __init__(self):
        super().__init__("内存管理专家", "memory")
        self.relevant_keywords = [
            'buffer', 'cache', 'mem', 'memory', 'shared',
            'effective_cache', 'temp_buffers', 'huge_pages'
        ]
        self.core_knobs = {
            'shared_buffers': 95,
            'effective_cache_size': 90,
            'work_mem': 85,
            'maintenance_work_mem': 75,
            'temp_buffers': 60,
            'huge_pages': 50,
        }

    def evaluate(self, knob_name: str, knob_info: dict,
                 workload_context: str, metrics: dict) -> dict:
        # 基础分：根据已知重要性
        base_score = self.core_knobs.get(knob_name, 0)
        if base_score == 0:
            # 关键词匹配
            for kw in self.relevant_keywords:
                if kw in knob_name.lower():
                    base_score = 40
                    break

        # 动态调整：根据缓存命中率
        cache_ratio = metrics.get('cache_hit_ratio', {}).get('buffer', 99)
        if cache_ratio < 85 and knob_name in ('shared_buffers', 'effective_cache_size'):
            base_score = min(100, base_score + 15)
        elif cache_ratio < 70:
            base_score = min(100, base_score + 10)

        reason = ""
        if base_score > 0:
            reason = f"[{self.name}] {knob_name} 属于内存管理类参数"
            if cache_ratio < 85:
                reason += f"，当前缓存命中率 {cache_ratio:.1f}% 偏低，该参数调整优先级提升"

        return {'score': base_score, 'reason': reason}


class IOExpert(ExpertEvaluator):
    """I/O 调度专家：评估磁盘读写、WAL、检查点相关参数"""

    def __init__(self):
        super().__init__("I/O 调度专家", "io")
        self.relevant_keywords = [
            'wal', 'checkpoint', 'io', 'fsync', 'synchronous',
            'bgwriter', 'disk', 'write'
        ]
        self.core_knobs = {
            'max_wal_size': 85,
            'min_wal_size': 60,
            'wal_buffers': 75,
            'checkpoint_completion_target': 80,
            'effective_io_concurrency': 70,
            'random_page_cost': 65,
            'seq_page_cost': 55,
            'bgwriter_delay': 50,
            'bgwriter_lru_maxpages': 50,
        }

    def evaluate(self, knob_name: str, knob_info: dict,
                 workload_context: str, metrics: dict) -> dict:
        base_score = self.core_knobs.get(knob_name, 0)
        if base_score == 0:
            for kw in self.relevant_keywords:
                if kw in knob_name.lower():
                    base_score = 35
                    break

        # 动态调整：物理读取率高时提升 IO 参数优先级
        io_data = metrics.get('io', {})
        blks_read = io_data.get('blks_read_per_sec', 0)
        blks_hit = io_data.get('blks_hit_per_sec', 0)
        if (blks_read + blks_hit) > 0:
            physical_ratio = blks_read / (blks_read + blks_hit) * 100
            if physical_ratio > 20 and base_score > 0:
                base_score = min(100, base_score + 15)

        reason = f"[{self.name}] {knob_name} 属于 I/O 调度类参数" if base_score > 0 else ""
        return {'score': base_score, 'reason': reason}


class QueryExpert(ExpertEvaluator):
    """查询执行专家：评估查询规划器、并行查询相关参数"""

    def __init__(self):
        super().__init__("查询执行专家", "query")
        self.relevant_keywords = [
            'parallel', 'worker', 'cost', 'plan', 'join',
            'statistics', 'geqo', 'jit', 'hash', 'sort'
        ]
        self.core_knobs = {
            'max_parallel_workers_per_gather': 85,
            'max_parallel_workers': 80,
            'max_worker_processes': 75,
            'default_statistics_target': 65,
            'random_page_cost': 70,
            'effective_io_concurrency': 60,
            'from_collapse_limit': 45,
            'join_collapse_limit': 45,
            'jit': 50,
            'geqo': 40,
        }

    def evaluate(self, knob_name: str, knob_info: dict,
                 workload_context: str, metrics: dict) -> dict:
        base_score = self.core_knobs.get(knob_name, 0)
        if base_score == 0:
            for kw in self.relevant_keywords:
                if kw in knob_name.lower():
                    base_score = 35
                    break

        # OLAP 场景下并行查询参数更重要
        if 'OLAP' in workload_context.upper() and 'parallel' in knob_name.lower():
            base_score = min(100, base_score + 15)

        # 有慢查询时提升规划器参数
        slow_count = metrics.get('slow_query_count', 0)
        if slow_count > 0 and base_score > 0:
            base_score = min(100, base_score + 10)

        reason = f"[{self.name}] {knob_name} 属于查询执行类参数" if base_score > 0 else ""
        return {'score': base_score, 'reason': reason}


class ConcurrencyExpert(ExpertEvaluator):
    """并发事务专家：评估连接管理、锁机制相关参数"""

    def __init__(self):
        super().__init__("并发事务专家", "concurrency")
        self.relevant_keywords = [
            'connection', 'lock', 'deadlock', 'transaction',
            'idle', 'timeout', 'statement'
        ]
        self.core_knobs = {
            'max_connections': 90,
            'deadlock_timeout': 60,
            'lock_timeout': 55,
            'statement_timeout': 50,
            'idle_in_transaction_session_timeout': 65,
        }

    def evaluate(self, knob_name: str, knob_info: dict,
                 workload_context: str, metrics: dict) -> dict:
        base_score = self.core_knobs.get(knob_name, 0)
        if base_score == 0:
            for kw in self.relevant_keywords:
                if kw in knob_name.lower():
                    base_score = 35
                    break

        # 高并发时连接类参数更重要
        conn_data = metrics.get('connections', {})
        total_conn = conn_data.get('total', 0)
        if total_conn > 80 and knob_name == 'max_connections':
            base_score = min(100, base_score + 10)

        # 锁竞争时锁参数更重要
        total_locks = metrics.get('total_locks', 0)
        if total_locks > 50 and 'lock' in knob_name.lower():
            base_score = min(100, base_score + 15)

        reason = f"[{self.name}] {knob_name} 属于并发事务类参数" if base_score > 0 else ""
        return {'score': base_score, 'reason': reason}


class BackgroundProcessExpert(ExpertEvaluator):
    """后台进程专家：评估 autovacuum、bgwriter 等后台进程参数"""

    def __init__(self):
        super().__init__("后台进程专家", "background")
        self.relevant_keywords = [
            'autovacuum', 'vacuum', 'bgwriter', 'archiv',
            'log', 'wal_sender', 'wal_receiver'
        ]
        self.core_knobs = {
            'autovacuum_vacuum_cost_delay': 70,
            'autovacuum_max_workers': 65,
            'autovacuum_naptime': 55,
            'bgwriter_delay': 60,
            'bgwriter_lru_maxpages': 55,
            'log_min_duration_statement': 40,
        }

    def evaluate(self, knob_name: str, knob_info: dict,
                 workload_context: str, metrics: dict) -> dict:
        base_score = self.core_knobs.get(knob_name, 0)
        if base_score == 0:
            for kw in self.relevant_keywords:
                if kw in knob_name.lower():
                    base_score = 30
                    break

        reason = f"[{self.name}] {knob_name} 属于后台进程类参数" if base_score > 0 else ""
        return {'score': base_score, 'reason': reason}


class HardwareAdaptExpert(ExpertEvaluator):
    """硬件适配专家：根据硬件配置评估参数重要性"""

    def __init__(self):
        super().__init__("硬件适配专家", "hardware")
        self.relevant_keywords = ['huge_pages', 'io_concurrency', 'page_cost']

    def evaluate(self, knob_name: str, knob_info: dict,
                 workload_context: str, metrics: dict) -> dict:
        import psutil
        total_mem_gb = psutil.virtual_memory().total / (1024 ** 3)
        cpu_count = psutil.cpu_count() or 1

        base_score = 0

        # 大内存环境下内存参数更重要
        if total_mem_gb >= 16 and knob_name in ('shared_buffers', 'effective_cache_size'):
            base_score = 60
        elif total_mem_gb < 4 and knob_name in ('shared_buffers', 'work_mem'):
            base_score = 70  # 小内存更需要精细调优

        # 多核环境下并行参数更重要
        if cpu_count >= 8 and 'parallel' in knob_name.lower():
            base_score = 55

        # SSD 环境下 random_page_cost 需要下调
        if knob_name == 'random_page_cost':
            base_score = 50

        reason = f"[{self.name}] 硬件适配评估: 内存 {total_mem_gb:.0f}GB, CPU {cpu_count} 核" if base_score > 0 else ""
        return {'score': base_score, 'reason': reason}


class SafetyAuditExpert(ExpertEvaluator):
    """安全审计专家：评估参数调整的安全风险"""

    def __init__(self):
        super().__init__("安全审计专家", "safety")
        # 高风险参数（修改需谨慎）
        self.high_risk_knobs = {
            'max_connections': 85,
            'shared_buffers': 80,
            'work_mem': 70,
            'max_wal_size': 60,
        }

    def evaluate(self, knob_name: str, knob_info: dict,
                 workload_context: str, metrics: dict) -> dict:
        base_score = self.high_risk_knobs.get(knob_name, 0)

        # 需要重启的参数风险更高
        context = knob_info.get('context', 'user') if knob_info else 'user'
        if context == 'postmaster' and base_score > 0:
            base_score = min(100, base_score + 10)

        reason = ""
        if base_score > 0:
            reason = f"[{self.name}] {knob_name} 为高影响力参数"
            if context == 'postmaster':
                reason += "（修改需重启数据库）"

        return {'score': base_score, 'reason': reason}


class MoEManager:
    """
    MoE 管理器 - 混合专家协调中心
    负责根据负载画像动态分配专家权重，并融合多专家评分进行参数筛选。
    """

    # 场景 -> 专家权重映射
    SCENARIO_WEIGHTS = {
        'OLTP': {
            '内存管理专家': 0.20,
            'I/O 调度专家': 0.12,
            '查询执行专家': 0.08,
            '并发事务专家': 0.25,
            '后台进程专家': 0.08,
            '硬件适配专家': 0.12,
            '安全审计专家': 0.15,
        },
        'OLAP': {
            '内存管理专家': 0.15,
            'I/O 调度专家': 0.15,
            '查询执行专家': 0.25,
            '并发事务专家': 0.05,
            '后台进程专家': 0.08,
            '硬件适配专家': 0.12,
            '安全审计专家': 0.20,
        },
        'HYBRID': {
            '内存管理专家': 0.18,
            'I/O 调度专家': 0.14,
            '查询执行专家': 0.15,
            '并发事务专家': 0.15,
            '后台进程专家': 0.08,
            '硬件适配专家': 0.12,
            '安全审计专家': 0.18,
        }
    }

    def __init__(self):
        self.experts: List[ExpertEvaluator] = [
            MemoryExpert(),
            IOExpert(),
            QueryExpert(),
            ConcurrencyExpert(),
            BackgroundProcessExpert(),
            HardwareAdaptExpert(),
            SafetyAuditExpert(),
        ]

    def assign_weights(self, scenario: str):
        """
        根据场景动态分配专家权重

        Args:
            scenario: 场景类型 (OLTP/OLAP/HYBRID)
        """
        weights = self.SCENARIO_WEIGHTS.get(scenario, self.SCENARIO_WEIGHTS['HYBRID'])
        for expert in self.experts:
            expert.weight = weights.get(expert.name, 0.1)

        print(f"[MoE Manager] 场景: {scenario}, 专家权重已分配:")
        for expert in self.experts:
            print(f"  - {expert.name}: {expert.weight:.2f}")

    def evaluate_knobs(self, candidate_knobs: List[str],
                       knob_info_dict: dict,
                       workload_context: str,
                       metrics: dict,
                       scenario: str,
                       top_k: int = 30) -> List[dict]:
        """
        多专家融合评估候选参数

        Args:
            candidate_knobs: 候选参数列表
            knob_info_dict: 参数元信息字典
            workload_context: 负载语义描述
            metrics: 当前性能指标
            scenario: 场景类型
            top_k: 返回 Top-K 参数

        Returns:
            排序后的参数评估结果列表
        """
        # 分配权重
        self.assign_weights(scenario)

        # 对每个候选参数进行多专家评估
        knob_scores = []

        for knob in candidate_knobs:
            knob_info = knob_info_dict.get(knob, {})
            expert_evaluations = []
            weighted_score = 0

            for expert in self.experts:
                result = expert.evaluate(knob, knob_info, workload_context, metrics)
                score = result['score']
                weighted_contribution = score * expert.weight
                weighted_score += weighted_contribution

                if score > 0:
                    expert_evaluations.append({
                        'expert': expert.name,
                        'score': score,
                        'weighted': weighted_contribution,
                        'reason': result['reason']
                    })

            if weighted_score > 0:
                # 取贡献最大的专家作为主要理由
                top_expert = max(expert_evaluations, key=lambda x: x['weighted']) if expert_evaluations else None
                knob_scores.append({
                    'knob': knob,
                    'total_score': round(weighted_score, 2),
                    'expert_details': expert_evaluations,
                    'primary_reason': top_expert['reason'] if top_expert else '',
                    'contributing_experts': len(expert_evaluations)
                })

        # 按综合得分排序
        knob_scores.sort(key=lambda x: x['total_score'], reverse=True)

        # 返回 Top-K
        selected = knob_scores[:top_k]

        print(f"\n[MoE Manager] 从 {len(candidate_knobs)} 个候选参数中筛选出 Top-{top_k}:")
        for i, item in enumerate(selected[:10], 1):
            print(f"  {i:2d}. {item['knob']:40s} 综合分: {item['total_score']:6.2f} "
                  f"({item['contributing_experts']} 位专家)")

        return selected

    def generate_selection_report(self, evaluation_results: List[dict]) -> str:
        """
        生成参数筛选报告（供 LLM 参考）

        Args:
            evaluation_results: evaluate_knobs 的返回结果

        Returns:
            自然语言筛选报告
        """
        lines = ["## MoE 多专家参数筛选报告", ""]
        lines.append(f"共筛选出 {len(evaluation_results)} 个关键参数:\n")

        for i, item in enumerate(evaluation_results, 1):
            lines.append(f"### {i}. {item['knob']} (综合评分: {item['total_score']:.1f})")
            for detail in item['expert_details'][:3]:
                lines.append(f"  - {detail['reason']}")
            lines.append("")

        return "\n".join(lines)
