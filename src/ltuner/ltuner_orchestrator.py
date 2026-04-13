#!/usr/bin/env python3
"""
LTuner 主编排器 (LTuner Orchestrator)
串联 LTuner 完整工作流：
1. 环境感知 -> 特征翻译
2. MoE 专家参数筛选
3. 因果图谱检索 + 值域剪枝（SPC 三智能体）
4. 自省反馈迭代优化
5. 结果输出与日志记录
"""
import sys
import os
import json
import time
from typing import Dict, List, Optional
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ltuner.feature_translator import FeatureTranslator
from ltuner.causal_graph import CausalKnowledgeGraph
from ltuner.moe_experts import MoEManager
from ltuner.value_pruner import ValuePruner
from ltuner.reflective_engine import ReflectiveEngine


class LTunerOrchestrator:
    """
    LTuner 主编排器
    串联所有 LTuner 模块，执行完整的自省式反馈调优工作流。
    """

    def __init__(self, dbms, test: str, timeout: int,
                 api_base: str, api_key: str, model: str = "qwen-plus",
                 max_iterations: int = 15,
                 convergence_threshold: float = 0.02,
                 top_k_knobs: int = 20,
                 scenario: str = "HYBRID",
                 output_dir: str = "./optimization_results/ltuner",
                 use_temperature_scheduling: bool = True):
        """
        Args:
            dbms: PgDBMS 实例
            test: benchmark 名称
            timeout: benchmark 超时秒数
            api_base: LLM API 地址
            api_key: LLM API 密钥
            model: LLM 模型名
            max_iterations: 最大迭代次数
            convergence_threshold: 收敛阈值
            top_k_knobs: MoE 筛选 Top-K 参数数
            scenario: 场景类型 (OLTP/OLAP/HYBRID)
            output_dir: 结果输出目录
            use_temperature_scheduling: 是否启用动态温度调度/主动探索/收敛保护
        """
        self.dbms = dbms
        self.test = test
        self.timeout = timeout
        self.api_base = api_base
        self.api_key = api_key
        self.model = model
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.top_k_knobs = top_k_knobs
        self.scenario = scenario
        self.output_dir = output_dir

        # 初始化各模块
        self.translator = FeatureTranslator(dbms)
        self.causal_graph = CausalKnowledgeGraph()
        self.moe_manager = MoEManager()
        self.value_pruner = ValuePruner(dbms=dbms)
        self.reflective_engine = ReflectiveEngine(
            dbms=dbms, test=test, timeout=timeout,
            api_base=api_base, api_key=api_key, model=model,
            max_iterations=max_iterations,
            convergence_threshold=convergence_threshold,
            use_temperature_scheduling=use_temperature_scheduling
        )

        # 场景自动检测
        self._detect_scenario()

        print(f"\n{'='*60}")
        print(f"LTuner 主编排器已初始化")
        print(f"{'='*60}")
        print(f"  数据库: PostgreSQL")
        print(f"  基准测试: {test}")
        print(f"  场景: {self.scenario}")
        print(f"  最大迭代: {max_iterations}")
        print(f"  Top-K 参数: {top_k_knobs}")
        print(f"  LLM: {model}")
        print(f"  输出目录: {output_dir}")
        print(f"{'='*60}\n")

    def _detect_scenario(self):
        """根据 benchmark 类型自动检测场景"""
        oltp_benchmarks = ['tpcc', 'twitter', 'sibench', 'voter', 'tatp', 'smallbank', 'seats']
        olap_benchmarks = ['tpch']

        if self.test in oltp_benchmarks:
            self.scenario = 'OLTP'
        elif self.test in olap_benchmarks:
            self.scenario = 'OLAP'
        else:
            self.scenario = 'HYBRID'

    def run(self) -> Dict:
        """
        执行完整的 LTuner 调优工作流

        Returns:
            完整优化结果报告
        """
        os.makedirs(self.output_dir, exist_ok=True)
        start_time = time.time()

        print(f"\n{'#'*60}")
        print(f"#  LTuner - 基于 LLM 自省式反馈的数据库参数调优系统")
        print(f"#  开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'#'*60}\n")

        workflow_report = {
            'system': 'LTuner',
            'start_time': datetime.now().isoformat(),
            'test': self.test,
            'scenario': self.scenario,
            'steps': {}
        }

        try:
            # ========== Step 1: 环境感知 + 特征翻译 ==========
            print(f"\n{'─'*60}")
            print("Step 1/5: 环境感知与特征翻译")
            print(f"{'─'*60}")

            env_context = self._step_env_translation()
            workflow_report['steps']['step1_env'] = {
                'status': 'completed',
                'env_context_length': len(env_context)
            }

            # ========== Step 2: MoE 多专家参数筛选 ==========
            print(f"\n{'─'*60}")
            print("Step 2/5: MoE 多专家参数筛选")
            print(f"{'─'*60}")

            target_knobs, moe_report = self._step_moe_selection(env_context)
            workflow_report['steps']['step2_moe'] = {
                'status': 'completed',
                'selected_knobs': len(target_knobs),
                'knob_list': target_knobs
            }

            # ========== Step 3: 因果图谱 + SPC 值域剪枝 ==========
            print(f"\n{'─'*60}")
            print("Step 3/5: 因果图谱分析 + SPC 值域剪枝")
            print(f"{'─'*60}")

            causal_context, value_ranges = self._step_causal_and_pruning(target_knobs)
            workflow_report['steps']['step3_pruning'] = {
                'status': 'completed',
                'pruned_knobs': len(value_ranges)
            }

            # ========== Step 4: 自省反馈迭代优化 ==========
            print(f"\n{'─'*60}")
            print("Step 4/5: 自省式反馈迭代优化")
            print(f"{'─'*60}")

            optimize_result = self._step_reflective_optimization(
                target_knobs, value_ranges, causal_context, env_context
            )
            workflow_report['steps']['step4_optimize'] = {
                'status': 'completed',
                'result': optimize_result
            }

            # ========== Step 5: 结果输出与日志 ==========
            print(f"\n{'─'*60}")
            print("Step 5/5: 结果输出与日志记录")
            print(f"{'─'*60}")

            end_time = time.time()
            workflow_report['end_time'] = datetime.now().isoformat()
            workflow_report['total_time_seconds'] = round(end_time - start_time, 2)
            workflow_report['status'] = 'completed'
            workflow_report['final_result'] = {
                'best_config': optimize_result.get('best_config', {}),
                'best_performance': optimize_result.get('best_performance', 0),
                'improvement_percent': optimize_result.get('improvement_percent', 0),
                'total_iterations': optimize_result.get('total_iterations', 0),
                'config_failures': optimize_result.get('config_failures', 0)
            }

            self._save_workflow_report(workflow_report)
            self._print_final_summary(workflow_report)

        except Exception as e:
            workflow_report['status'] = 'failed'
            workflow_report['error'] = str(e)
            workflow_report['end_time'] = datetime.now().isoformat()
            self._save_workflow_report(workflow_report)
            print(f"\n[ERROR] LTuner 工作流执行失败: {e}")
            import traceback
            traceback.print_exc()

        return workflow_report

    def _step_env_translation(self) -> str:
        """Step 1: 环境感知与特征翻译"""
        # 静态环境信息
        static_desc = self.translator.translate_static_features()
        print(static_desc)

        # 尝试获取动态指标
        dynamic_desc = ""
        try:
            from monitoring.postgres_monitor import PostgresMonitor
            monitor = PostgresMonitor(self.dbms)
            metrics = monitor.collect_all_metrics()
            dynamic_desc = self.translator.translate_dynamic_metrics(metrics)
            print(dynamic_desc)
        except Exception as e:
            print(f"[INFO] 动态指标采集不可用: {e}")
            print("[INFO] 将仅使用静态环境信息")

        env_context = static_desc
        if dynamic_desc:
            env_context += "\n\n" + dynamic_desc

        return env_context

    def _step_moe_selection(self, env_context: str):
        """Step 2: MoE 多专家参数筛选"""
        # 获取候选参数列表
        candidate_knobs = list(self.dbms.knob_info.keys())
        print(f"[MoE] 候选参数总数: {len(candidate_knobs)}")

        # 获取当前指标（用于专家评估）
        metrics = self._get_current_metrics()

        # 构建负载描述
        workload_context = f"场景: {self.scenario}, Benchmark: {self.test}"

        # MoE 多专家评估
        evaluation_results = self.moe_manager.evaluate_knobs(
            candidate_knobs=candidate_knobs,
            knob_info_dict=self.dbms.knob_info,
            workload_context=workload_context,
            metrics=metrics,
            scenario=self.scenario,
            top_k=self.top_k_knobs
        )

        # 提取选中的参数名
        target_knobs = [item['knob'] for item in evaluation_results]

        # 过滤掉不可调整的参数（string 类型等）
        target_knobs = [
            k for k in target_knobs
            if k in self.dbms.knob_info
            and self.dbms.knob_info[k].get('vartype', 'string') != 'string'
        ]

        # 生成选择报告
        moe_report = self.moe_manager.generate_selection_report(evaluation_results)
        print(f"\n[MoE] 最终筛选出 {len(target_knobs)} 个可调参数")

        return target_knobs, moe_report

    def _step_causal_and_pruning(self, target_knobs: List[str]):
        """Step 3: 因果图谱分析 + SPC 值域剪枝"""
        # 因果图谱分析
        bottleneck_metrics = self._detect_bottleneck_metrics()
        causal_context = self.causal_graph.generate_causal_context(
            target_knobs=target_knobs,
            bottleneck_metrics=bottleneck_metrics
        )
        print(causal_context[:500])

        # SPC 值域剪枝
        value_ranges = self.value_pruner.prune(
            target_knobs=target_knobs,
            knob_info_dict=self.dbms.knob_info,
            scenario=self.scenario
        )

        # 协同参数扩展：检查是否有因果图谱中关联的参数未被选中
        synergy_additions = set()
        for knob in target_knobs[:10]:
            synergy_group = self.causal_graph.get_synergy_group(knob)
            for sg in synergy_group:
                partner = sg['knob']
                if (partner not in target_knobs
                        and partner in self.dbms.knob_info
                        and self.dbms.knob_info[partner].get('vartype', 'string') != 'string'):
                    synergy_additions.add(partner)

        if synergy_additions:
            print(f"\n[因果图谱] 发现 {len(synergy_additions)} 个协同参数，追加到目标列表")
            for knob in synergy_additions:
                target_knobs.append(knob)
                # 也为追加的参数做值域剪枝
                vr_extra = self.value_pruner.prune(
                    target_knobs=[knob],
                    knob_info_dict=self.dbms.knob_info,
                    scenario=self.scenario
                )
                value_ranges.update(vr_extra)

        return causal_context, value_ranges

    def _step_reflective_optimization(self, target_knobs: List[str],
                                       value_ranges: dict,
                                       causal_context: str,
                                       env_context: str) -> Dict:
        """Step 4: 自省式反馈迭代优化"""
        result = self.reflective_engine.optimize(
            target_knobs=target_knobs,
            value_ranges=value_ranges,
            causal_context=causal_context,
            env_context=env_context,
            output_dir=self.output_dir
        )
        return result

    def _get_current_metrics(self) -> dict:
        """获取当前性能指标，采集失败则返回空字典"""
        try:
            from monitoring.postgres_monitor import PostgresMonitor
            monitor = PostgresMonitor(self.dbms)
            return monitor.collect_all_metrics()
        except Exception:
            return {}

    def _detect_bottleneck_metrics(self) -> List[str]:
        """检测当前性能瓶颈指标"""
        bottlenecks = []
        metrics = self._get_current_metrics()

        if not metrics:
            return bottlenecks

        # 缓存命中率低
        cache_ratio = metrics.get('cache_hit_ratio', {}).get('buffer', 100)
        if cache_ratio < 90:
            bottlenecks.append('buffer_hit_ratio')

        # 物理 IO 高
        io = metrics.get('io', {})
        blks_read = io.get('blks_read_per_sec', 0)
        blks_hit = io.get('blks_hit_per_sec', 0)
        if (blks_read + blks_hit) > 0 and blks_read / (blks_read + blks_hit) > 0.2:
            bottlenecks.append('disk_read_rate')

        # 锁竞争高
        if metrics.get('total_locks', 0) > 50:
            bottlenecks.append('lock_contention')

        # 慢查询多
        if metrics.get('slow_query_count', 0) > 0:
            bottlenecks.append('query_sort_spill')

        return bottlenecks

    def _save_workflow_report(self, report: Dict):
        """保存完整工作流报告"""
        report_path = os.path.join(self.output_dir, 'ltuner_workflow_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        print(f"\n[保存] 工作流报告: {report_path}")

    def _print_final_summary(self, report: Dict):
        """打印最终摘要"""
        result = report.get('final_result', {})
        total_time = report.get('total_time_seconds', 0)

        print(f"\n{'#'*60}")
        print(f"#  LTuner 调优完成")
        print(f"{'#'*60}")
        print(f"  状态: {report.get('status', 'unknown')}")
        print(f"  总耗时: {total_time:.1f} 秒 ({total_time/60:.1f} 分钟)")
        print(f"  场景: {report.get('scenario', '?')}")
        print(f"  调优轮次: {result.get('total_iterations', 0)}")
        print(f"  最佳性能: {result.get('best_performance', 0):.2f}")
        print(f"  性能提升: {result.get('improvement_percent', 0):.1f}%")
        print(f"  配置失败: {result.get('config_failures', 0)} 次")
        print(f"  最佳配置:")
        for k, v in result.get('best_config', {}).items():
            print(f"    {k}: {v}")
        print(f"{'#'*60}\n")
