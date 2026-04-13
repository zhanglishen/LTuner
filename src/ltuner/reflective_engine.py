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
from ltuner.workload_semantic_analyzer import WorkloadSemanticAnalyzer


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
                 convergence_threshold: float = 0.02,
                 use_temperature_scheduling: bool = True):
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
            use_temperature_scheduling: 是否启用动态温度调度/主动探索/收敛保护
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

        # 语义分析器
        self.semantic_analyzer = WorkloadSemanticAnalyzer(
            dbms=dbms, runner=self.runner
        )

        # 记录
        self.history: List[IterationRecord] = []
        self.best_config: Dict[str, str] = {}
        self.best_performance: float = float('-inf')  # 初始化为负无穷，确保任何实际性能都能更新
        self.baseline_performance: float = 0.0
        self.prev_query_latencies: Dict[str, float] = {}  # 上轮逐查询延迟
        self.curr_query_latencies: Dict[str, float] = {}  # 当前轮逐查询延迟

        # 温度调度开关
        self.use_temperature_scheduling = use_temperature_scheduling
        # 动态温度调度（Task1）
        self.current_temperature: float = 0.3
        # 收敛保护标志（Task3）
        self._last_chance_explored: bool = False

    def _get_temperature(self, iteration: int) -> float:
        """
        动态 LLM 温度调度：
          前30%  低温(0.2)  → 精准建立基础，快速达到高性能区间
          中30%  高温(0.7)  → 主动探索，寻找更大突破
          中后期 中温(0.4)  → 验证探索成果，稳定提升
          最后20% 极低温(0.1)→ 锁定最优，精准收敛
        若 use_temperature_scheduling=False，则固定返回 0.3
        """
        if not self.use_temperature_scheduling:
            return 0.3
        progress = iteration / max(self.max_iterations, 1)
        if progress < 0.3:
            return 0.2
        elif progress < 0.6:
            return 0.7
        elif progress < 0.8:
            return 0.4
        else:
            return 0.1

    def _call_llm(self, system_prompt: str, user_prompt: str,
                   json_format: bool = False, temperature: float = None) -> str:
        """调用 LLM，支持动态 temperature"""
        if temperature is None:
            temperature = self.current_temperature
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        kwargs = {
            "messages": messages,
            "model": self.model,
            "temperature": temperature,
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
        LLM 生成初始配置方案（Task5：加入随机探索种子，增加不同实验的起点多样性）
        """
        import random
        # 随机选择一个初始调优侧重方向，使不同实验有不同的探索起点
        exploration_hints = [
            "请重点增大 checkpoint 相关参数（checkpoint_completion_target 调高至0.9, max_wal_size 调大）以减少 I/O 抖动",
            "请重点优化 shared_buffers 和 effective_cache_size 的协同比例（shared_buffers=系统内存25%，effective_cache_size=75%）",
            "请重点优化并发相关参数（max_worker_processes, max_parallel_workers, max_parallel_workers_per_gather）以利用多核",
            "请重点优化 bgwriter 和 autovacuum 的后台 I/O 控制参数（bgwriter_lru_maxpages, autovacuum_vacuum_cost_delay）",
            "请重点优化 WAL 和日志参数（wal_buffers, synchronous_commit, wal_compression）以减少写入延迟",
        ]
        exploration_seed = random.choice(exploration_hints)
        print(f"[初始配置] 探索方向种子: {exploration_seed[:50]}...")

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

## 本次调优侧重方向（随机探索种子）
{exploration_seed}

## 任务
请为上述 {len(target_knobs)} 个参数生成一组初始配置。
要求：
1. 所有值必须在安全边界内
2. 考虑参数间的因果协同关系
3. 优先优化对当前瓶颈影响最大的参数，并结合本次侧重方向
4. 返回纯 JSON，key 为参数名，value 为参数值（字符串格式）
"""
        try:
            response = self._call_llm(system_prompt, user_prompt, json_format=True,
                                       temperature=0.5)  # 初始配置用中等温度，增加多样性
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
                      causal_context: str,
                      query_delta_desc: str = "",
                      diagnostic_context: str = "") -> Tuple[str, str]:
        """
        自省分析：LLM 分析性能变化的因果原因，生成文本梯度
        改进：将所有历史轮次的反思与梯度全部传入，供 LLM 全局分析参考。
        语义增强：新增查询级性能变化对比和系统诊断信息，使 LLM 能精确定位参数变化对具体查询的影响。

        Args:
            current_record: 当前迭代记录
            previous_records: 历史迭代记录（全量）
            env_context: 环境上下文
            causal_context: 因果图谱上下文
            query_delta_desc: 逐查询延迟变化的自然语言描述
            diagnostic_context: 系统诊断信息（等待事件/临时文件/表扫描模式）

        Returns:
            (reflection_text, text_gradient)
        """
        baseline = self.baseline_performance

        # 构建全量历史详情（含每轮反思与梯度）
        history_lines = []
        for rec in previous_records:
            delta_str = f"+{rec.delta_performance:.1f}%" if rec.delta_performance > 0 else f"{rec.delta_performance:.1f}%"
            vs_baseline = (rec.throughput - baseline) / abs(baseline) * 100 if baseline else 0
            line = (
                f"  [轮次{rec.iteration:2d}] TPS={rec.throughput:8.1f} "
                f"vs基线{vs_baseline:+.1f}%  逐轮Delta={delta_str}  "
                f"{'✓改善' if rec.is_improvement else '✗未改善'}"
            )
            if rec.reflection_text:
                line += f"\n    └─反思: {rec.reflection_text[:120]}..."
            if rec.text_gradient:
                # text_gradient 可能是 list 或 str
                grad_str = rec.text_gradient
                if isinstance(grad_str, list):
                    grad_str = "; ".join(str(g) for g in grad_str[:3])
                line += f"\n    └─梯度: {str(grad_str)[:120]}..."
            history_lines.append(line)
        history_desc = "\n".join(history_lines) if history_lines else "  无历史记录"

        # 当前配置
        config_lines = [f"  {k}: {v}" for k, v in current_record.config.items()]
        config_desc = "\n".join(config_lines)

        # 历史最优信息
        best_iter = None
        if previous_records:
            best_rec = max(previous_records, key=lambda r: r.throughput)
            best_iter = best_rec.iteration
            best_tps = best_rec.throughput
        else:
            best_tps = 0

        system_prompt = (
            "你是一位资深的 PostgreSQL DBA 和性能分析专家。"
            "请基于完整的历史调优轨迹（含每轮反思与文本梯度），"
            "深度分析本轮性能变化原因，并给出下一步调整建议。\n"
            "特别关注：\n"
            "1. 哪些历史轮次的梯度方向产生了正向效果，哪些导致了性能衰退\n"
            "2. 当前配置是否存在参数冲突或过度调整\n"
            "3. 建议应基于历史最优配置而非当前配置进行微调\n"
            "4. 结合查询级性能变化和系统诊断信息，分析参数变化对具体查询的因果影响\n"
            "请用 JSON 格式返回，包含两个字段：\n"
            '{"reflection": "深度分析思考过程（包含历史梯度回溯和查询级因果分析）", '
            '"gradient": "具体调整方向建议（须指明基于哪个历史轮次配置出发，以及对哪些查询的预期影响）"}'
        )

        delta_str = f"+{current_record.delta_performance:.1f}%" if current_record.delta_performance > 0 else f"{current_record.delta_performance:.1f}%"
        vs_baseline_now = (current_record.throughput - baseline) / abs(baseline) * 100 if baseline else 0

        # 构建查询级变化和诊断信息段落
        query_delta_section = ""
        if query_delta_desc:
            query_delta_section = f"\n{query_delta_desc}\n"
        diagnostic_section = ""
        if diagnostic_context:
            diagnostic_section = f"\n{diagnostic_context}\n"

        user_prompt = f"""
## 当前状态
- 迭代轮次: {current_record.iteration}/{self.max_iterations}
- 基线 TPS: {baseline:.2f}
- 当前 TPS: {current_record.throughput:.2f}（vs基线 {vs_baseline_now:+.1f}%）
- 逐轮 Delta: {delta_str}
- 历史最优: 轮次{best_iter} TPS={best_tps:.1f}（vs基线 {(best_tps-baseline)/abs(baseline)*100:+.1f}%）

## 当前配置
{config_desc}

## 完整历史调优轨迹（含每轮反思与梯度）
{history_desc}

{causal_context}
{query_delta_section}
{diagnostic_section}
## 分析任务
1. 回溯历史梯度：哪些轮次的调整方向带来正向效果？哪些轮次梯度导致了衰退？
2. 诊断当前轮次性能变化的根本原因（参数冲突/过调/正向协同）
3. 结合查询级性能变化，分析参数调整对哪些查询产生了正向/负向影响，及其因果机制
4. 识别当前主要性能瓶颈所在子系统
5. 给出下一轮调整建议：须明确指出应以哪一历史轮次的配置为基础出发点
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
                               value_ranges: dict,
                               exploration_mode: bool = False,
                               diagnostic_context: str = "") -> Dict[str, str]:
        """
        基于文本梯度生成下一轮配置

        Args:
            text_gradient: 文本梯度（调整方向建议）
            current_config: 当前配置（可能是历史最优）
            target_knobs: 目标参数
            value_ranges: 值域约束
            exploration_mode: True=主动探索模式（更大步长+新维度），False=精细微调模式
            diagnostic_context: 系统诊断信息（执行计划+等待事件+内存画像）
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

        # Task4：配置多样性保护——检测近5轮中单调不变的参数
        monotone_knobs = []
        if len(self.history) >= 3:
            for knob in target_knobs:
                recent_vals = [r.config.get(knob) for r in self.history[-5:]
                               if r.config.get(knob)]
                if len(recent_vals) >= 3 and len(set(recent_vals)) == 1:
                    monotone_knobs.append(knob)
        if monotone_knobs:
            diversity_hint = (
                f"\n[多样性约束] 以下参数在近5轮中完全未变化，本轮必须调整其中至少2个："
                f"\n{', '.join(monotone_knobs[:8])}"
            )
        else:
            diversity_hint = ""

        if exploration_mode:
            # 主动探索模式：更大步长，要求调整未充分探索的维度
            mode_instruction = (
                "当前处于【主动探索阶段】。任务是跳出当前局部最优区域，"
                "尝试调整上几轮未变动或变化较少的参数维度"
                "（如 checkpoint_completion_target, max_wal_size, bgwriter_lru_maxpages, "
                "autovacuum_vacuum_cost_delay, random_page_cost 等）。\n"
                "每个参数调整幅度可达当前值的 30-50%，目标是找到新的性能峰值区域。\n"
                "不要拘泥于上一轮的微调方向，大胆探索！"
            )
            step_limit = "（探索模式下，单参数调整幅度可达 30-50%）"
        else:
            mode_instruction = (
                "当前处于【精细微调阶段】。在当前基础配置上按梯度建议进行保守微调，"
                "逐步逼近最优解。"
            )
            step_limit = "（微调模式下，单参数每轮变动幅度不超过 20%）"

        # 构建诊断信息段落
        diagnostic_prompt_section = ""
        if diagnostic_context:
            diagnostic_prompt_section = f"\n{diagnostic_context}\n"

        system_prompt = (
            "你是一位资深 PostgreSQL DBA。根据上一轮的自省分析和调整建议，"
            "生成新一轮的参数配置。必须严格遵守安全范围。"
            "注意：当前基础配置可能来自历史最优轮次（非上一轮），请以此为基础做调整。"
            "以 JSON 格式返回，格式为 {\"parameter_name\": \"value\"}。\n"
            "单位规则：内存类参数(shared_buffers, work_mem, effective_cache_size, "
            "maintenance_work_mem, wal_buffers, temp_buffers, max_wal_size, "
            "min_wal_size 等)必须带 MB 或 GB 后缀(如 \"512MB\")。"
            "数量参数和比率参数只写数字。"
        )

        user_prompt = f"""
## 自省分析建议（文本梯度）
{text_gradient}

## 当前调优模式
{mode_instruction}

## 当前基础配置和安全范围
（注意：此配置可能来自历史最优轮次，请在此基础上按模式要求进行调整）
{range_desc}
{diversity_hint}
{diagnostic_prompt_section}
## 任务
基于自省建议，按当前模式调整配置。要求：
1. 严格遵守安全范围
2. 调整幅度符合当前模式要求 {step_limit}
3. 返回所有参数的新值（即使未变更也要包含）
4. 若梯度建议与历史正向轮次方向一致，则可适度加大步长
5. 结合系统诊断信息，优先调整对瓶颈查询影响最大的参数
"""
        try:
            response = self._call_llm(system_prompt, user_prompt, json_format=True)
            new_config = json.loads(response)
            new_config = {k: str(v) for k, v in new_config.items() if k in target_knobs}

            # 确保所有目标参数都有值
            for knob in target_knobs:
                if knob not in new_config:
                    new_config[knob] = current_config.get(knob, '')

            mode_tag = "[探索]" if exploration_mode else "[微调]"
            print(f"  {mode_tag} 生成新配置：{len(new_config)} 个参数"
                  + (f"，多样性保护参数: {monotone_knobs[:4]}" if monotone_knobs else ""))
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
        # 回滚策略：跌破基线阈值时触发（跌超50%则立即回滚）
        ROLLBACK_THRESHOLD = 0.50  # 跌破基线50%触发回滚
        CRASH_THRESHOLD = 0.10     # TPS低于基线10%视为崩溃，强制终止方向
        rollback_count = 0
        EXPLORATION_INTERVAL = 4   # 每4轮触发一次主动探索（Task2）
        self._last_chance_explored = False  # 收敛保护标志重置（Task3）
        # 无温度调度模式下禁用探索与收敛保护
        enable_exploration = self.use_temperature_scheduling

        for iteration in range(1, self.max_iterations + 1):
            print(f"\n{'─'*60}")
            print(f"迭代轮次 {iteration}/{self.max_iterations}")
            print(f"{'─'*60}")

            # Task1：动态更新 LLM 温度
            self.current_temperature = self._get_temperature(iteration)
            print(f"  [温度调度] LLM temperature={self.current_temperature:.1f}"
                  f"  (进度 {iteration/self.max_iterations*100:.0f}%)")

            record = IterationRecord(iteration)
            record.config = dict(current_config)

            # Step 2: 应用配置并运行 benchmark
            print("[Step 2] 应用配置并运行 benchmark...")
            apply_success = self._apply_config(current_config)
            if not apply_success:
                record.config_failures += 1
                print(f"[WARNING] 配置应用失败，触发回滚到历史最优配置")
                if self.best_config:
                    current_config = dict(self.best_config)
                    rollback_count += 1
                    print(f"[ROLLBACK] 已回滚到历史最优配置（第{rollback_count}次回滚）")
                self.history.append(record)
                continue

            throughput, latency = self._run_benchmark()
            current_performance = self._get_performance(throughput, latency)

            # 采集逐查询延迟（用于语义分析）
            self.prev_query_latencies = dict(self.curr_query_latencies)
            try:
                self.curr_query_latencies = self.runner.get_per_query_latencies()
                if self.curr_query_latencies:
                    print(f"  [语义分析] 采集到 {len(self.curr_query_latencies)} 条查询的逐查询延迟")
            except Exception as e:
                print(f"  [语义分析] 逐查询延迟采集失败: {e}")

            record.throughput = throughput
            record.latency = latency

            # Step 3: 计算性能变化
            if prev_performance != 0:
                record.delta_performance = (
                    (current_performance - prev_performance) / abs(prev_performance) * 100
                )
            record.is_improvement = current_performance > prev_performance

            # 计算 vs 基线的百分比
            vs_baseline_pct = (
                (current_performance - self.baseline_performance) / abs(self.baseline_performance)
                if self.baseline_performance != 0 else 0
            )

            print(f"  TPS: {throughput:.2f}, Latency: {latency:.2f}")
            print(f"  vs基线: {vs_baseline_pct:+.1f}%  逐轮Delta: {record.delta_performance:+.1f}%  "
                  f"{'✓改善' if record.is_improvement else '✗未改善'}")

            # 更新最佳
            if current_performance > self.best_performance:
                self.best_performance = current_performance
                self.best_config = dict(current_config)
                print(f"  [NEW BEST] TPS={throughput:.2f} 已更新历史最优")

            # ===== 回滚策略 =====
            rollback_triggered = False
            if vs_baseline_pct < -ROLLBACK_THRESHOLD * 100:
                # 跌破基线50%：严重崩溃，立即回滚并跳过本轮梯度
                print(f"\n[ROLLBACK] 严重崩溃！TPS vs基线 {vs_baseline_pct:.1f}% < -{ROLLBACK_THRESHOLD*100:.0f}%")
                print(f"[ROLLBACK] 回滚到历史最优配置（TPS={self.best_performance:.1f}）")
                if self.best_config:
                    current_config = dict(self.best_config)
                    rollback_count += 1
                rollback_triggered = True
                record.reflection_text = f"[自动回滚] 性能崩溃至基线{vs_baseline_pct:.1f}%，已回滚至历史最优"
                record.text_gradient = "[回滚] 从历史最优配置出发，缩小调整步长继续探索"
                self.history.append(record)
                no_improvement_count += 1
                # Task3：收敛保护（回滚场景下也触发）
                if no_improvement_count >= max(self.max_iterations // 3, 5):
                    if enable_exploration and not self._last_chance_explored:
                        print(f"[收敛保护] 疑似局部最优，触发最后一次大范围探索")
                        self._last_chance_explored = True
                        no_improvement_count = 0
                    else:
                        print(f"\n[收敛] 连续 {no_improvement_count} 轮无改善，提前终止")
                        break
                prev_performance = self.best_performance
                continue

            # Step 4: 自省分析（传入全量历史，含每轮反思与梯度）
            print("[Step 4] LLM 自省分析（全量历史轨迹 + 语义分析）...")

            # 生成查询级变化描述
            query_delta_desc = ""
            if self.prev_query_latencies and self.curr_query_latencies:
                query_delta_desc = self.semantic_analyzer.analyze_query_deltas(
                    self.prev_query_latencies, self.curr_query_latencies
                )
                if query_delta_desc:
                    print(f"  [语义分析] 已生成查询级性能变化分析")

            # 生成系统诊断信息
            diagnostic_context = ""
            try:
                from monitoring.postgres_monitor import PostgreSQLMonitor
                monitor = PostgreSQLMonitor(self.dbms)
                diagnostic_context = self.semantic_analyzer.analyze_system_diagnostics(monitor)
                if diagnostic_context:
                    print(f"  [语义分析] 已生成系统诊断信息")
            except Exception as e:
                print(f"  [语义分析] 系统诊断采集失败: {e}")

            reflection, gradient = self.self_reflect(
                record, self.history, env_context, causal_context,
                query_delta_desc=query_delta_desc,
                diagnostic_context=diagnostic_context
            )
            record.reflection_text = reflection
            record.text_gradient = gradient
            print(f"  反思: {str(reflection)[:120]}...")
            print(f"  梯度: {str(gradient)[:120]}...")

            self.history.append(record)

            # Step 5: 收敛检查（Task3：加入 _last_chance_explored 保护）
            if record.is_improvement:
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            if no_improvement_count >= max(self.max_iterations // 3, 5):
                if enable_exploration and not self._last_chance_explored:
                    # 还未给过探索机会，不能真正终止
                    print(f"[收敛保护] 连续{no_improvement_count}轮无改善，"
                          f"疑似局部最优，触发最后一次大范围探索后再决定是否终止")
                    self._last_chance_explored = True
                    no_improvement_count = 0
                    # 强制下一轮为探索模式（在 Step 6 会被设为 True）
                    _force_exploration = True
                else:
                    print(f"\n[收敛] 连续无改善 + 探索后仍无突破，真正收敛，终止")
                    break
            else:
                _force_exploration = False

            if iteration > self.max_iterations // 2 and abs(record.delta_performance) < self.convergence_threshold * 100:
                print(f"\n[收敛] 性能变化 {record.delta_performance:.2f}% "
                      f"< 阈值 {self.convergence_threshold*100:.1f}%，收敛终止")
                break

            # Step 6: 生成下一轮配置
            # Task2：判断是否为主动探索轮次（仅温度调度模式下启用）
            if enable_exploration:
                is_exploration_round = _force_exploration or (iteration % EXPLORATION_INTERVAL == 0)
            else:
                is_exploration_round = False

            if iteration < self.max_iterations:
                if vs_baseline_pct < 0 and self.best_config:
                    base_config_for_next = dict(self.best_config)
                    print(f"[Step 6] 性能低于基线，以历史最优配置为起点生成下一轮配置...")
                else:
                    base_config_for_next = current_config
                    if is_exploration_round:
                        print(f"[Step 6] 主动探索轮次（iter%{EXPLORATION_INTERVAL}==0），扩大步长探索新区域...")
                    else:
                        print("[Step 6] 基于文本梯度生成下一轮配置...")
                current_config = self.generate_next_config(
                    gradient, base_config_for_next, target_knobs, value_ranges,
                    exploration_mode=is_exploration_round,
                    diagnostic_context=diagnostic_context
                )
                prev_performance = current_performance

        print(f"\n[优化完成] 共触发回滚 {rollback_count} 次")

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
