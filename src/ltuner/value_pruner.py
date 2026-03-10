#!/usr/bin/env python3
"""
Surveyor-Proposer-Corrector (SPC) 三智能体值域剪枝模块
在给出具体推荐值前，为每个参数划定安全且合理的值域范围。
- Surveyor（勘测员）: 从因果图谱和知识库检索参数的物理范围和建议范围
- Proposer（提案者）: 结合硬件信息将相对值转为绝对值
- Corrector（校验者）: 执行安全红线检查，过滤冲突配置
"""
import sys
import os
import psutil
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class ValueRange:
    """参数值域描述"""

    def __init__(self, knob_name: str):
        self.knob_name = knob_name
        self.physical_min = None      # 系统物理下限
        self.physical_max = None      # 系统物理上限
        self.suggested_min = None     # 建议下限
        self.suggested_max = None     # 建议上限
        self.recommended_value = None # 推荐值
        self.unit = None
        self.knob_type = 'integer'    # integer/real/enum/bool
        self.enum_values = None
        self.needs_restart = False
        self.safety_notes: List[str] = []

    def to_dict(self) -> dict:
        return {
            'knob': self.knob_name,
            'physical_range': [self.physical_min, self.physical_max],
            'suggested_range': [self.suggested_min, self.suggested_max],
            'recommended': self.recommended_value,
            'unit': self.unit,
            'type': self.knob_type,
            'needs_restart': self.needs_restart,
            'safety_notes': self.safety_notes
        }

    def __repr__(self):
        return (f"ValueRange({self.knob_name}: "
                f"[{self.suggested_min}, {self.suggested_max}] "
                f"rec={self.recommended_value})")


class Surveyor:
    """
    勘测员智能体
    负责从知识库和系统元数据中检索参数的物理范围和建议范围。
    """

    def __init__(self, dbms=None, knowledge_base=None):
        self.dbms = dbms
        self.knowledge_base = knowledge_base

    def survey(self, knob_name: str, knob_info: dict = None) -> ValueRange:
        """
        勘测参数的值域信息

        Args:
            knob_name: 参数名
            knob_info: 参数元信息（来自 pg_settings）

        Returns:
            ValueRange 值域描述
        """
        vr = ValueRange(knob_name)

        # 从系统元信息获取物理范围
        if knob_info:
            vr.physical_min = knob_info.get('min_val')
            vr.physical_max = knob_info.get('max_val')
            vr.unit = knob_info.get('unit')
            vr.knob_type = knob_info.get('vartype', 'integer')
            vr.needs_restart = knob_info.get('context') == 'postmaster'

            if vr.knob_type == 'enum':
                vr.enum_values = knob_info.get('enumvals', [])

        # 从知识库获取建议范围
        if self.knowledge_base:
            unit = self.knowledge_base.get_unit(knob_name) if hasattr(self.knowledge_base, 'get_unit') else None
            if unit:
                constraints = getattr(unit, 'constraints', {})
                if constraints:
                    if 'min' in constraints:
                        vr.suggested_min = constraints['min']
                    if 'max' in constraints:
                        vr.suggested_max = constraints['max']

        # 默认值逻辑
        if vr.suggested_min is None:
            vr.suggested_min = vr.physical_min
        if vr.suggested_max is None:
            vr.suggested_max = vr.physical_max

        return vr


class Proposer:
    """
    提案者智能体
    结合硬件信息将相对值（如 "RAM 的 25%"）转为具体绝对值。
    """

    # PostgreSQL 核心参数的推荐公式
    RECOMMENDATION_FORMULAS = {
        'shared_buffers': {
            'formula': 'ram_mb * ratio',
            'OLTP': {'ratio': 0.25},
            'OLAP': {'ratio': 0.25},
            'HYBRID': {'ratio': 0.25},
            'min_mb': 128,
            'max_ratio': 0.40,
            'unit': 'MB'
        },
        'effective_cache_size': {
            'formula': 'ram_mb * ratio',
            'OLTP': {'ratio': 0.50},
            'OLAP': {'ratio': 0.60},
            'HYBRID': {'ratio': 0.50},
            'min_mb': 512,
            'max_ratio': 0.75,
            'unit': 'MB'
        },
        'work_mem': {
            'formula': 'fixed_mb',
            'OLTP': {'fixed_mb': 16},
            'OLAP': {'fixed_mb': 256},
            'HYBRID': {'fixed_mb': 64},
            'min_mb': 4,
            'max_mb': 2048,
            'unit': 'MB'
        },
        'maintenance_work_mem': {
            'formula': 'fixed_mb',
            'OLTP': {'fixed_mb': 256},
            'OLAP': {'fixed_mb': 1024},
            'HYBRID': {'fixed_mb': 512},
            'min_mb': 64,
            'max_mb': 2048,
            'unit': 'MB'
        },
        'max_connections': {
            'formula': 'fixed',
            'OLTP': {'value': 200},
            'OLAP': {'value': 50},
            'HYBRID': {'value': 100},
            'min': 20,
            'max': 1000
        },
        'max_wal_size': {
            'formula': 'fixed_mb',
            'OLTP': {'fixed_mb': 2048},
            'OLAP': {'fixed_mb': 4096},
            'HYBRID': {'fixed_mb': 2048},
            'min_mb': 256,
            'max_mb': 10240,
            'unit': 'MB'
        },
        'wal_buffers': {
            'formula': 'fixed_mb',
            'OLTP': {'fixed_mb': 16},
            'OLAP': {'fixed_mb': 16},
            'HYBRID': {'fixed_mb': 16},
            'min_mb': 4,
            'max_mb': 64,
            'unit': 'MB'
        },
        'checkpoint_completion_target': {
            'formula': 'fixed',
            'OLTP': {'value': 0.9},
            'OLAP': {'value': 0.9},
            'HYBRID': {'value': 0.9},
            'min': 0.1,
            'max': 0.9
        },
        'random_page_cost': {
            'formula': 'fixed',
            'OLTP': {'value': 1.1},
            'OLAP': {'value': 1.1},
            'HYBRID': {'value': 1.1},
            'min': 1.0,
            'max': 4.0
        },
        'effective_io_concurrency': {
            'formula': 'fixed',
            'OLTP': {'value': 200},
            'OLAP': {'value': 200},
            'HYBRID': {'value': 200},
            'min': 1,
            'max': 1000
        },
        'max_parallel_workers_per_gather': {
            'formula': 'cpu_based',
            'OLTP': {'ratio': 0},
            'OLAP': {'ratio': 0.5},
            'HYBRID': {'ratio': 0.25},
            'min': 0,
            'max': 8
        },
        'max_parallel_workers': {
            'formula': 'cpu_based',
            'OLTP': {'ratio': 0.5},
            'OLAP': {'ratio': 1.0},
            'HYBRID': {'ratio': 0.5},
            'min': 0,
            'max': 16
        },
        'max_worker_processes': {
            'formula': 'cpu_based',
            'OLTP': {'ratio': 1.0},
            'OLAP': {'ratio': 1.0},
            'HYBRID': {'ratio': 1.0},
            'min': 1,
            'max': 32
        },
        'default_statistics_target': {
            'formula': 'fixed',
            'OLTP': {'value': 100},
            'OLAP': {'value': 500},
            'HYBRID': {'value': 200},
            'min': 10,
            'max': 10000
        }
    }

    def __init__(self):
        self.ram_mb = int(psutil.virtual_memory().total / (1024 * 1024))
        self.cpu_count = psutil.cpu_count() or 1

    def propose(self, value_range: ValueRange, scenario: str) -> ValueRange:
        """
        为参数生成具体的推荐值和建议范围

        Args:
            value_range: Surveyor 输出的值域信息
            scenario: 场景类型

        Returns:
            更新后的 ValueRange
        """
        knob = value_range.knob_name
        formula = self.RECOMMENDATION_FORMULAS.get(knob)

        if not formula:
            return value_range

        scenario_config = formula.get(scenario, formula.get('HYBRID', {}))

        if formula['formula'] == 'ram_mb * ratio':
            ratio = scenario_config.get('ratio', 0.25)
            rec_mb = int(self.ram_mb * ratio)
            min_mb = formula.get('min_mb', 128)
            max_mb = int(self.ram_mb * formula.get('max_ratio', 0.5))
            value_range.recommended_value = f"{max(min_mb, rec_mb)}MB"
            value_range.suggested_min = f"{min_mb}MB"
            value_range.suggested_max = f"{max_mb}MB"
            value_range.unit = 'MB'

        elif formula['formula'] == 'fixed_mb':
            rec_mb = scenario_config.get('fixed_mb', 64)
            min_mb = formula.get('min_mb', 4)
            max_mb = formula.get('max_mb', 2048)
            value_range.recommended_value = f"{rec_mb}MB"
            value_range.suggested_min = f"{min_mb}MB"
            value_range.suggested_max = f"{max_mb}MB"
            value_range.unit = 'MB'

        elif formula['formula'] == 'fixed':
            value_range.recommended_value = str(scenario_config.get('value', ''))
            value_range.suggested_min = str(formula.get('min', ''))
            value_range.suggested_max = str(formula.get('max', ''))

        elif formula['formula'] == 'cpu_based':
            ratio = scenario_config.get('ratio', 0.5)
            rec_val = max(int(self.cpu_count * ratio), formula.get('min', 0))
            rec_val = min(rec_val, formula.get('max', 16))
            value_range.recommended_value = str(rec_val)
            value_range.suggested_min = str(formula.get('min', 0))
            value_range.suggested_max = str(min(formula.get('max', 16), self.cpu_count * 2))

        return value_range


class Corrector:
    """
    校验者智能体
    执行安全红线检查，确保推荐值不会导致系统崩溃。
    """

    def __init__(self, dbms=None):
        self.dbms = dbms
        self.ram_mb = int(psutil.virtual_memory().total / (1024 * 1024))

    def correct(self, value_ranges: Dict[str, ValueRange]) -> Dict[str, ValueRange]:
        """
        对所有参数值域进行安全校验和修正

        Args:
            value_ranges: 参数名 -> ValueRange 映射

        Returns:
            校验修正后的值域映射
        """
        # 规则 1: 内存总量检查
        self._check_memory_budget(value_ranges)

        # 规则 2: 并行层级检查
        self._check_parallel_hierarchy(value_ranges)

        # 规则 3: WAL 一致性检查
        self._check_wal_consistency(value_ranges)

        return value_ranges

    def _parse_mb(self, value) -> int:
        """解析内存值为 MB"""
        if value is None:
            return 0
        value = str(value).strip().upper()
        if value.endswith('GB'):
            return int(float(value.replace('GB', '')) * 1024)
        elif value.endswith('MB'):
            return int(float(value.replace('MB', '')))
        elif value.endswith('KB'):
            return int(float(value.replace('KB', '')) / 1024)
        try:
            return int(float(value))
        except (ValueError, TypeError):
            return 0

    def _check_memory_budget(self, value_ranges: Dict[str, ValueRange]):
        """检查内存总预算不超过系统 80%"""
        sb = self._parse_mb(value_ranges.get('shared_buffers', ValueRange('_')).recommended_value)
        wm = self._parse_mb(value_ranges.get('work_mem', ValueRange('_')).recommended_value)
        mm = self._parse_mb(value_ranges.get('maintenance_work_mem', ValueRange('_')).recommended_value)

        max_conn_vr = value_ranges.get('max_connections', ValueRange('_'))
        try:
            max_conn = int(max_conn_vr.recommended_value or 100)
        except (ValueError, TypeError):
            max_conn = 100

        estimated_total = sb + (wm * max_conn) + mm
        budget = int(self.ram_mb * 0.8)

        if estimated_total > budget:
            # 修正：降低 work_mem
            safe_wm = max(4, int((budget - sb - mm) / max(max_conn, 1)))
            if 'work_mem' in value_ranges:
                old_val = value_ranges['work_mem'].recommended_value
                value_ranges['work_mem'].recommended_value = f"{safe_wm}MB"
                value_ranges['work_mem'].safety_notes.append(
                    f"Corrector 修正: work_mem 从 {old_val} 降至 {safe_wm}MB "
                    f"(内存预算 {budget}MB, 预估总使用 {estimated_total}MB)"
                )
                print(f"[Corrector] 内存超限修正: work_mem {old_val} -> {safe_wm}MB")

    def _check_parallel_hierarchy(self, value_ranges: Dict[str, ValueRange]):
        """检查并行参数层级: worker_processes >= parallel_workers >= per_gather"""
        knobs = ['max_parallel_workers_per_gather', 'max_parallel_workers', 'max_worker_processes']
        vals = {}
        for k in knobs:
            vr = value_ranges.get(k)
            if vr and vr.recommended_value:
                try:
                    vals[k] = int(vr.recommended_value)
                except (ValueError, TypeError):
                    pass

        if len(vals) < 2:
            return

        mpwpg = vals.get('max_parallel_workers_per_gather', 0)
        mpw = vals.get('max_parallel_workers', 8)
        mwp = vals.get('max_worker_processes', 8)

        if mpw < mpwpg:
            mpw = mpwpg
            if 'max_parallel_workers' in value_ranges:
                value_ranges['max_parallel_workers'].recommended_value = str(mpw)
                value_ranges['max_parallel_workers'].safety_notes.append(
                    f"Corrector 修正: 调高至 {mpw} 以满足 >= max_parallel_workers_per_gather"
                )

        if mwp < mpw:
            mwp = mpw
            if 'max_worker_processes' in value_ranges:
                value_ranges['max_worker_processes'].recommended_value = str(mwp)
                value_ranges['max_worker_processes'].safety_notes.append(
                    f"Corrector 修正: 调高至 {mwp} 以满足 >= max_parallel_workers"
                )

    def _check_wal_consistency(self, value_ranges: Dict[str, ValueRange]):
        """检查 WAL 配置一致性"""
        max_wal = value_ranges.get('max_wal_size')
        min_wal = value_ranges.get('min_wal_size')

        if max_wal and min_wal:
            max_val = self._parse_mb(max_wal.recommended_value)
            min_val = self._parse_mb(min_wal.recommended_value)
            if min_val >= max_val and max_val > 0:
                new_min = max(80, int(max_val * 0.1))
                min_wal.recommended_value = f"{new_min}MB"
                min_wal.safety_notes.append(
                    f"Corrector 修正: min_wal_size 降至 {new_min}MB (需 < max_wal_size)"
                )


class ValuePruner:
    """
    SPC 值域剪枝器 - 串联三个智能体
    """

    def __init__(self, dbms=None, knowledge_base=None):
        self.surveyor = Surveyor(dbms=dbms, knowledge_base=knowledge_base)
        self.proposer = Proposer()
        self.corrector = Corrector(dbms=dbms)

    def prune(self, target_knobs: List[str],
              knob_info_dict: dict,
              scenario: str) -> Dict[str, ValueRange]:
        """
        执行完整的 SPC 值域剪枝流程

        Args:
            target_knobs: 目标参数列表
            knob_info_dict: 参数元信息
            scenario: 场景类型

        Returns:
            参数名 -> ValueRange 映射
        """
        print(f"\n{'='*60}")
        print(f"SPC 值域剪枝 - 场景: {scenario}")
        print(f"{'='*60}")

        # Phase 1: Surveyor 勘测
        print("\n[Phase 1/3] Surveyor 勘测参数范围...")
        value_ranges = {}
        for knob in target_knobs:
            knob_info = knob_info_dict.get(knob, {})
            vr = self.surveyor.survey(knob, knob_info)
            value_ranges[knob] = vr

        # Phase 2: Proposer 提案
        print("[Phase 2/3] Proposer 生成推荐值...")
        for knob, vr in value_ranges.items():
            value_ranges[knob] = self.proposer.propose(vr, scenario)

        # Phase 3: Corrector 校验
        print("[Phase 3/3] Corrector 安全校验...")
        value_ranges = self.corrector.correct(value_ranges)

        # 输出结果
        print(f"\n值域剪枝完成:")
        for knob, vr in value_ranges.items():
            if vr.recommended_value:
                notes = f" !! {'; '.join(vr.safety_notes)}" if vr.safety_notes else ""
                print(f"  {knob}: [{vr.suggested_min}, {vr.suggested_max}] "
                      f"推荐={vr.recommended_value}{notes}")

        return value_ranges

    def generate_pruning_context(self, value_ranges: Dict[str, ValueRange]) -> str:
        """
        生成值域剪枝上下文（供 LLM 参考）

        Args:
            value_ranges: 剪枝结果

        Returns:
            自然语言描述
        """
        lines = ["## 参数值域安全边界 (SPC 三智能体校验结果)", ""]
        lines.append("| 参数 | 建议范围 | 推荐值 | 安全备注 |")
        lines.append("|------|---------|--------|---------|")

        for knob, vr in value_ranges.items():
            if vr.recommended_value:
                notes = "; ".join(vr.safety_notes) if vr.safety_notes else "通过"
                lines.append(
                    f"| {knob} | [{vr.suggested_min}, {vr.suggested_max}] "
                    f"| {vr.recommended_value} | {notes} |"
                )

        return "\n".join(lines)
