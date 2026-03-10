#!/usr/bin/env python3
"""
自省式反馈调优引擎 (Self-Reflective Feedback Engine)
核心模块：替代传统贝叶斯优化，采用 LLM 驱动的 "反馈-反思-调整" 迭代循环。
通过文本梯度（Text Gradient）而非数学梯度来确定参数调整方向。
"""
import sys
import os
import json
import time
import threading
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from openai import OpenAI

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config_recommender.workload_runner import BenchbaseRunner
from ltuner.feature_translator import FeatureTranslator
from ltuner.causal_graph import CausalKnowledgeGraph


class IterationRecord:
    """单次迭代记录"""

    def __init__(self, iteration: int):
        self.iteration = iteration
        self.timestamp = datetime.now().isoformat()
        self.config: Dict[str, str] = {}
        self.throughput: float = 0.0
        self.latency: float = 0.0
        self.metrics: dict = {}
        self.reflection_text: str = ""
        self.text_gradient: str = ""
        self.delta_performance: float = 0.0
        self.is_improvement: bool = False
        self.config_failures: int = 0

    def to_dict(self) -> dict:
        return {
            'iteration': self.iteration,
            'timestamp': self.timestamp,
            'config': self.config,
            'throughput': self.throughput,
            'latency': self.latency,
            'delta_performance': self.delta_performance,
            'is_improvement': self.is_improvement,
            'reflection_text': self.reflection_text,
            'text_gradient': self.text_gradient,
            'config_failures': self.config_failures
        }


class ReflectiveEngine:
    """
    自省式反馈调优引擎
    核心逻辑：
    1. LLM 生成初始配置 C_init
    2. 应用配置并运行 benchmark
    3. 计算性能变化 delta_P
    4. LLM 自省分析因果关系，生成文本梯度
    5. LLM 基于文本梯度生成新配置
    6. 迭代直到收敛或达到最大次数
    """

    def __init__(self, dbms, test: str, timeout: int,
                 api_base: str, api_key: str, model: str = "qwen-plus",
                 max_iterations: int = 15,
                 convergence_threshold: float = 0.02):
        """
        Args:
            dbms: PgDBMS 实例
            test: benchmark 名称 (tpcc/tpch 等)
            timeout: benchmark 超时秒数
            api_base: LLM API 地址
            api_key: LLM API 密钥
            model: LLM 模型名
            max_iterations: 最大迭代次数
            convergence_threshold: 收敛阈值（性能增量百分比）
        """
        self.dbms = dbms
        self.test = test
        self.timeout = timeout
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold

        # LLM 客户端
        self.client = OpenAI(api_key=api_key, base_url=api_base)
        self.model = model

        # 组件
        self.translator = FeatureTranslator(dbms)
        self.causal_graph = CausalKnowledgeGraph()
        self.runner = BenchbaseRunner(dbms=dbms, test=test)

        # benchmark 特性
        self.benchmark_copy_db = ['tpcc', 'twitter', 'sibench', 'voter', 'tatp', 'smallbank', 'seats']
        self.benchmark_latency = ['tpch']
        self.is_latency_mode = test in self.benchmark_latency

        # 记录
        self.history: List[IterationRecord] = []
        self.best_config: Dict[str, str] = {}
        self.best_performance: float = float('-inf')  # 初始化为负无穷，确保任何实际性能都能更新
        self.baseline_performance: float = 0.0

    def _call_llm(self, system_prompt: str, user_prompt: str,
                   json_format: bool = False) -> str:
        """调用 LLM"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        kwargs = {
            "messages": messages,
            "model": self.model,
            "temperature": 0.3,
        }
        if json_format:
            kwargs["response_format"] = {"type": "json_object"}

        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content

    def _run_benchmark(self) -> Tuple[float, float]:
        """运行 benchmark 并获取结果"""
        try:
            self.runner.clear_summary_dir()
            t = threading.Thread(target=self.runner.run_benchmark)
            t.start()
            t.join(timeout=self.timeout)

            if t.is_alive():
                print("[ReflectiveEngine] Benchmark 超时，终止")
                self.runner.process.terminate()
                time.sleep(2)
                return 0.0, float('inf')

            throughput = self.runner.get_throughput()
            latency = self.runner.get_latency()
            return throughput, latency
        except Exception as e:
            print(f"[ReflectiveEngine] Benchmark 异常: {e}")
            return 0.0, float('inf')

    # 需要带内存单位后缀的 PostgreSQL 参数
    MEMORY_KNOBS = {
        'shared_buffers', 'work_mem', 'effective_cache_size',
        'maintenance_work_mem', 'wal_buffers', 'temp_buffers',
        'max_wal_size', 'min_wal_size', 'wal_segment_size',
        'autovacuum_work_mem', 'logical_decoding_work_mem',
        'max_stack_depth', 'log_rotation_size',
    }

    def _fix_config_units(self, config: Dict[str, str]) -> Dict[str, str]:
        """修正 LLM 生成配置中的单位问题"""
        import re
        fixed = {}
        for knob, value in config.items():
            val = str(value).strip()
            # 如果是内存参数且值只有数字（无单位后缀），添加 MB
            if knob in self.MEMORY_KNOBS:
                if re.match(r'^\d+(\.\d+)?$', val):
                    num = float(val)
                    if num < 100:
                        # 小数字很可能是 MB 意图（如 work_mem=4 → 4MB）
                        val = f"{int(num)}MB"
                    else:
                        # 较大数字可能已是 MB 意图（如 shared_buffers=2600 → 2600MB）
                        val = f"{int(num)}MB"
                    print(f"[单位修正] {knob}: {value} → {val}")
            fixed[knob] = val
        return fixed

    def _apply_config(self, config: Dict[str, str]) -> bool:
        """应用配置到数据库"""
        # 修正单位问题
        config = self._fix_config_units(config)

        self.dbms.reset_config()
        self.dbms.reconfigure()

        # 如果需要重新加载数据
        if self.test in self.benchmark_copy_db:
            try:
                self.dbms._disconnect()
                self.dbms._connect(f"{self.test}_template")
                self.dbms.copy_db(target_db="benchbase", source_db=f"{self.test}_template")
                time.sleep(12)
                self.dbms._disconnect()
                time.sleep(4)
                self.dbms._connect('benchbase')
                time.sleep(3)
            except Exception as e:
                print(f"[ReflectiveEngine] 数据重载失败: {e}")

        # 设置参数
        for knob, value in config.items():
            self.dbms.set_knob(knob, value)

        success = self.dbms.reconfigure()
        return success

    def _get_performance(self, throughput: float, latency: float) -> float:
        """统一性能度量（越大越好）"""
        if self.is_latency_mode:
            return -latency  # 延迟取负，使其越大越好
        return throughput

    def get_baseline(self) -> float:
        """获取默认配置的基准性能"""
        print("\n[ReflectiveEngine] 测量基准性能...")
        self.dbms.reset_config()
        self.dbms.reconfigure()

        if self.test in self.benchmark_copy_db:
            try:
                self.dbms._disconnect()
                self.dbms._connect(f"{self.test}_template")
                self.dbms.copy_db(target_db="benchbase", source_db=f"{self.test}_template")
                time.sleep(12)
                self.dbms._disconnect()
                time.sleep(4)
                self.dbms._connect('benchbase')
                time.sleep(3)
            except Exception as e:
                print(f"[ReflectiveEngine] 数据重载失败: {e}")

        throughput, latency = self._run_benchmark()
        self.baseline_performance = self._get_performance(throughput, latency)
        print(f"[ReflectiveEngine] 基准: TPS={throughput:.2f}, Latency={latency:.2f}")
        return self.baseline_performance

    def generate_initial_config(self, target_knobs: List[str],
                                  value_ranges: dict,
                                  causal_context: str,
                                  env_context: str) -> Dict[str, str]:
        """
        LLM 生成初始配置方案

        Args:
            target_knobs: 目标参数列表
            value_ranges: SPC 值域剪枝结果
            causal_context: 因果图谱上下文
            env_context: 环境特征上下文

        Returns:
            参数配置字典
        """
        # 构建值域约束描述
        range_desc_lines = []
        for knob in target_knobs:
            vr = value_ranges.get(knob)
            if vr and hasattr(vr, 'recommended_value') and vr.recommended_value:
                range_desc_lines.append(
                    f"- {knob}: 建议范围 [{vr.suggested_min}, {vr.suggested_max}], "
                    f"推荐值 {vr.recommended_value}"
                )
        range_desc = "\n".join(range_desc_lines)

        system_prompt = (
            "你是一位资深的 PostgreSQL DBA，拥有 20 年数据库调优经验。"
            "请根据提供的硬件环境、负载特征和参数因果关系，"
            "为以下参数生成一组高性能初始配置。"
            "必须严格遵守安全值域边界。"
            "以 JSON 格式返回，格式为 {\"parameter_name\": \"value\"}。\n"
            "重要的单位规则：\n"
            "- 内存类参数(shared_buffers, work_mem, effective_cache_size, "
            "maintenance_work_mem, wal_buffers, temp_buffers 等)必须带 MB 或 GB 后缀，"
            "例如 \"512MB\", \"2GB\", \"64MB\"。不要只写数字！\n"
            "- max_wal_size, min_wal_size 必须带 MB 或 GB 后缀\n"
            "- 纯数量参数(max_connections, max_worker_processes 等)只写数字\n"
            "- 比率参数(random_page_cost, seq_page_cost, "
            "checkpoint_completion_target 等)只写数字"
        )

        user_prompt = f"""
{env_context}

{causal_context}

## 参数值域安全边界
{range_desc}

## 任务
请为上述 {len(target_knobs)} 个参数生成一组初始配置。
要求：
1. 所有值必须在安全边界内
2. 考虑参数间的因果协同关系
3. 优先优化对当前瓶颈影响最大的参数
4. 返回纯 JSON，key 为参数名，value 为参数值（字符串格式）
"""
        try:
            response = self._call_llm(system_prompt, user_prompt, json_format=True)
            config = json.loads(response)
            # 过滤只保留目标参数
            config = {k: str(v) for k, v in config.items() if k in target_knobs}
            print(f"[ReflectiveEngine] LLM 生成初始配置: {len(config)} 个参数")
            return config
        except Exception as e:
            print(f"[ReflectiveEngine] LLM 生成配置失败: {e}")
            # 回退到 SPC 推荐值
            config = {}
            for knob in target_knobs:
                vr = value_ranges.get(knob)
                if vr and hasattr(vr, 'recommended_value') and vr.recommended_value:
                    config[knob] = str(vr.recommended_value)
            return config

    def self_reflect(self, current_record: IterationRecord,
                      previous_records: List[IterationRecord],
                      env_context: str,
                      causal_context: str) -> Tuple[str, str]:
        """
        自省分析：LLM 分析性能变化的因果原因，生成文本梯度

        Args:
            current_record: 当前迭代记录
            previous_records: 历史迭代记录
            env_context: 环境上下文
            causal_context: 因果图谱上下文

        Returns:
            (reflection_text, text_gradient)
        """
        # 构建历史概要
        history_lines = []
        for rec in previous_records[-5:]:
            delta_str = f"+{rec.delta_performance:.1f}%" if rec.delta_performance > 0 else f"{rec.delta_performance:.1f}%"
            history_lines.append(
                f"  轮次 {rec.iteration}: TPS={rec.throughput:.1f}, "
                f"Delta={delta_str}, {'改善' if rec.is_improvement else '未改善'}"
            )
        history_desc = "\n".join(history_lines) if history_lines else "  无历史记录"

        # 当前配置
        config_lines = [f"  {k}: {v}" for k, v in current_record.config.items()]
        config_desc = "\n".join(config_lines)

        system_prompt = (
            "你是一位资深的 PostgreSQL DBA 和性能分析专家。"
            "请分析本轮调优的性能变化原因，并给出下一步的调整方向建议。"
            "请用 JSON 格式返回，包含两个字段：\n"
            '{"reflection": "你的分析思考过程", "gradient": "具体的调整方向建议"}'
        )

        delta_str = f"+{current_record.delta_performance:.1f}%" if current_record.delta_performance > 0 else f"{current_record.delta_performance:.1f}%"

        user_prompt = f"""
## 当前状态
- 迭代轮次: {current_record.iteration}
- 当前 TPS: {current_record.throughput:.2f}
- 性能变化: {delta_str}
- 是否改善: {'是' if current_record.is_improvement else '否'}

## 当前配置
{config_desc}

## 历史记录
{history_desc}

{causal_context}

## 任务
1. 分析本轮性能变化的根本原因（基于参数因果关系）
2. 判断当前主要性能瓶颈已转移到哪个子系统
3. 给出下一轮应该调整的参数方向和幅度建议
"""
        try:
            response = self._call_llm(system_prompt, user_prompt, json_format=True)
            result = json.loads(response)
            reflection = result.get('reflection', '')
            gradient = result.get('gradient', '')
            return reflection, gradient
        except Exception as e:
            print(f"[ReflectiveEngine] 自省分析失败: {e}")
            if current_record.is_improvement:
                return "性能有所改善", "继续沿当前方向微调，减小步长"
            else:
                return "性能未改善", "回退变更，尝试调整其他维度的参数"

    def generate_next_config(self, text_gradient: str,
                               current_config: Dict[str, str],
                               target_knobs: List[str],
                               value_ranges: dict) -> Dict[str, str]:
        """
        基于文本梯度生成下一轮配置

        Args:
            text_gradient: 文本梯度（调整方向建议）
            current_config: 当前配置
            target_knobs: 目标参数
            value_ranges: 值域约束

        Returns:
            新配置
        """
        range_lines = []
        for knob in target_knobs:
            vr = value_ranges.get(knob)
            if vr and hasattr(vr, 'recommended_value'):
                range_lines.append(
                    f"- {knob}: 当前值={current_config.get(knob, '?')}, "
                    f"安全范围=[{vr.suggested_min}, {vr.suggested_max}]"
                )
        range_desc = "\n".join(range_lines)

        system_prompt = (
            "你是一位资深 PostgreSQL DBA。根据上一轮的自省分析和调整建议，"
            "生成新一轮的参数配置。必须严格遵守安全范围。"
            "以 JSON 格式返回，格式为 {\"parameter_name\": \"value\"}。\n"
            "单位规则：内存类参数(shared_buffers, work_mem, effective_cache_size, "
            "maintenance_work_mem, wal_buffers, temp_buffers, max_wal_size, "
            "min_wal_size 等)必须带 MB 或 GB 后缀(如 \"512MB\")。"
            "数量参数和比率参数只写数字。"
        )

        user_prompt = f"""
## 上一轮自省建议（文本梯度）
{text_gradient}

## 当前参数值和安全范围
{range_desc}

## 任务
基于自省建议，调整参数值。要求：
1. 严格遵守安全范围
2. 调整幅度合理，避免剧烈变动
3. 返回所有参数的新值（即使未变更也要包含）
"""
        try:
            response = self._call_llm(system_prompt, user_prompt, json_format=True)
            new_config = json.loads(response)
            new_config = {k: str(v) for k, v in new_config.items() if k in target_knobs}

            # 确保所有目标参数都有值
            for knob in target_knobs:
                if knob not in new_config:
                    new_config[knob] = current_config.get(knob, '')

            return new_config
        except Exception as e:
            print(f"[ReflectiveEngine] 生成新配置失败: {e}")
            return current_config

    def optimize(self, target_knobs: List[str],
                  value_ranges: dict,
                  causal_context: str,
                  env_context: str,
                  output_dir: str = "./optimization_results/ltuner") -> Dict:
        """
        执行完整的自省式反馈优化循环

        Args:
            target_knobs: 目标参数
            value_ranges: SPC 值域剪枝结果
            causal_context: 因果图谱上下文
            env_context: 环境特征上下文
            output_dir: 输出目录

        Returns:
            优化结果报告
        """
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"LTuner 自省式反馈优化引擎")
        print(f"目标参数: {len(target_knobs)} 个")
        print(f"最大迭代: {self.max_iterations} 次")
        print(f"收敛阈值: {self.convergence_threshold*100:.1f}%")
        print(f"{'='*60}\n")

        # Step 0: 基准测量
        self.get_baseline()
        prev_performance = self.baseline_performance

        # Step 1: 生成初始配置
        print("\n[Step 1] 生成初始配置...")
        current_config = self.generate_initial_config(
            target_knobs, value_ranges, causal_context, env_context
        )

        no_improvement_count = 0

        for iteration in range(1, self.max_iterations + 1):
            print(f"\n{'─'*60}")
            print(f"迭代轮次 {iteration}/{self.max_iterations}")
            print(f"{'─'*60}")

            record = IterationRecord(iteration)
            record.config = dict(current_config)

            # Step 2: 应用配置并运行 benchmark
            print("[Step 2] 应用配置并运行 benchmark...")
            apply_success = self._apply_config(current_config)
            if not apply_success:
                record.config_failures += 1
                print(f"[WARNING] 配置应用失败，跳过本轮")
                self.history.append(record)
                continue

            throughput, latency = self._run_benchmark()
            current_performance = self._get_performance(throughput, latency)

            record.throughput = throughput
            record.latency = latency

            # Step 3: 计算性能变化
            if prev_performance != 0:
                record.delta_performance = (
                    (current_performance - prev_performance) / abs(prev_performance) * 100
                )
            record.is_improvement = current_performance > prev_performance

            print(f"  TPS: {throughput:.2f}, Latency: {latency:.2f}")
            print(f"  Delta: {record.delta_performance:+.1f}%, "
                  f"{'改善' if record.is_improvement else '未改善'}")

            # 更新最佳
            if current_performance > self.best_performance:
                self.best_performance = current_performance
                self.best_config = dict(current_config)

            # Step 4: 自省分析
            print("[Step 4] LLM 自省分析...")
            reflection, gradient = self.self_reflect(
                record, self.history, env_context, causal_context
            )
            record.reflection_text = reflection
            record.text_gradient = gradient
            print(f"  反思: {reflection[:100]}...")
            print(f"  梯度: {gradient[:100]}...")

            self.history.append(record)

            # Step 5: 收敛检查
            if record.is_improvement:
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            if no_improvement_count >= max(self.max_iterations // 3, 5):
                print(f"\n[收敛] 连续 {no_improvement_count} 轮无改善，提前终止")
                break

            if iteration > self.max_iterations // 2 and abs(record.delta_performance) < self.convergence_threshold * 100:
                print(f"\n[收敛] 性能变化 {record.delta_performance:.2f}% "
                      f"< 阈值 {self.convergence_threshold*100:.1f}%，收敛终止")
                break

            # Step 6: 生成下一轮配置
            if iteration < self.max_iterations:
                print("[Step 6] 基于文本梯度生成下一轮配置...")
                current_config = self.generate_next_config(
                    gradient, current_config, target_knobs, value_ranges
                )
                prev_performance = current_performance

        # 输出结果
        result = self._generate_report(output_dir)
        return result

    def _generate_report(self, output_dir: str) -> Dict:
        """生成优化报告"""
        baseline_abs = abs(self.baseline_performance) if self.baseline_performance != 0 else 1
        improvement = (self.best_performance - self.baseline_performance) / baseline_abs * 100

        report = {
            'system': 'LTuner',
            'method': 'Self-Reflective Feedback',
            'total_iterations': len(self.history),
            'baseline_performance': self.baseline_performance,
            'best_performance': self.best_performance,
            'improvement_percent': improvement,
            'best_config': self.best_config,
            'convergence_iteration': len(self.history),
            'config_failures': sum(r.config_failures for r in self.history),
            'history': [r.to_dict() for r in self.history]
        }

        # 保存报告
        report_path = os.path.join(output_dir, 'ltuner_result.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"\n{'='*60}")
        print(f"LTuner 优化完成")
        print(f"{'='*60}")
        print(f"  总迭代: {len(self.history)} 轮")
        print(f"  基准性能: {self.baseline_performance:.2f}")
        print(f"  最佳性能: {self.best_performance:.2f}")
        print(f"  提升: {improvement:.1f}%")
        print(f"  配置失败: {report['config_failures']} 次")
        print(f"  报告保存: {report_path}")

        return report
