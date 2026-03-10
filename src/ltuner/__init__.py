"""
LTuner - 基于大语言模型与自省式反馈的数据库参数自动化调优系统

核心模块：
- feature_translator: 特征语义翻译器，将数据库指标转为自然语言
- causal_graph: 因果知识图谱，参数间因果依赖建模与多跳推理
- moe_experts: MoE 多智能体专家系统，多维度参数重要性评估
- value_pruner: Surveyor-Proposer-Corrector 三智能体值域剪枝
- reflective_engine: 自省式反馈调优引擎，LLM 驱动的迭代优化
- ltuner_orchestrator: 主编排器，串联完整调优工作流
"""

__version__ = "1.0.0"
__name_display__ = "LTuner"
