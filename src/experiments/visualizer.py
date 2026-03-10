#!/usr/bin/env python3
"""
对比实验结果可视化 (Experiment Visualizer)
使用 matplotlib 生成 GPTuner vs LTuner 对比图表：
1. 收敛曲线对比图
2. 最终性能柱状图
3. 调优耗时对比图
4. 安全性对比表
"""
import sys
import os
import json
import numpy as np
from typing import Dict, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# matplotlib 中文支持
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 尝试设置中文字体
def _setup_chinese_font():
    """设置 matplotlib 中文字体"""
    # 常见中文字体路径
    font_candidates = [
        '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
        '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
        '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf',
    ]
    for font_path in font_candidates:
        if os.path.exists(font_path):
            return fm.FontProperties(fname=font_path)
    # 回退到英文
    return None

CHINESE_FONT = _setup_chinese_font()


def _get_label(zh: str, en: str) -> str:
    """根据字体可用性返回中文或英文标签"""
    return zh if CHINESE_FONT else en


class ExperimentVisualizer:
    """实验结果可视化器"""

    # 配色方案
    COLORS = {
        'gptuner': '#2196F3',    # 蓝色
        'ltuner': '#FF5722',     # 橙红色
        'baseline': '#9E9E9E',   # 灰色
        'background': '#FAFAFA',
        'grid': '#E0E0E0',
    }

    def __init__(self, output_dir: str = "./optimization_results/comparison"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def visualize_all(self, comparison_data: Dict,
                      prefix: str = "comparison") -> Dict[str, str]:
        """
        生成所有对比图表

        Args:
            comparison_data: ComparisonRunner 输出的对比结果
            prefix: 文件名前缀

        Returns:
            {图表名: 文件路径}
        """
        saved_files = {}

        # 1. 收敛曲线对比
        path = self.plot_convergence_curve(comparison_data, prefix)
        saved_files['convergence_curve'] = path

        # 2. 最终性能柱状图
        path = self.plot_performance_bar(comparison_data, prefix)
        saved_files['performance_bar'] = path

        # 3. 调优耗时对比
        path = self.plot_time_comparison(comparison_data, prefix)
        saved_files['time_comparison'] = path

        # 4. 安全性对比表
        path = self.plot_safety_table(comparison_data, prefix)
        saved_files['safety_table'] = path

        # 5. 综合仪表盘
        path = self.plot_dashboard(comparison_data, prefix)
        saved_files['dashboard'] = path

        print(f"\n[可视化] 已生成 {len(saved_files)} 张图表:")
        for name, filepath in saved_files.items():
            print(f"  - {name}: {filepath}")

        return saved_files

    def plot_convergence_curve(self, data: Dict, prefix: str) -> str:
        """
        绘制收敛曲线对比图
        X轴=迭代次数, Y轴=TPS/延迟, 两条曲线
        """
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        fig.patch.set_facecolor(self.COLORS['background'])
        ax.set_facecolor(self.COLORS['background'])

        gp = data.get('gptuner', {})
        lt = data.get('ltuner', {})
        test = data.get('test', 'tpcc')
        is_latency = test in ['tpch']

        if is_latency:
            gp_values = gp.get('iteration_latency', [])
            lt_values = lt.get('iteration_latency', [])
            ylabel = _get_label('延迟 (μs)', 'Latency (μs)')
            title = _get_label(
                f'收敛曲线对比 - {test.upper()} (延迟，越低越好)',
                f'Convergence Curve - {test.upper()} (Latency, Lower is Better)'
            )
        else:
            gp_values = gp.get('iteration_tps', [])
            lt_values = lt.get('iteration_tps', [])
            ylabel = _get_label('TPS (事务/秒)', 'TPS (transactions/sec)')
            title = _get_label(
                f'收敛曲线对比 - {test.upper()} (吞吐量，越高越好)',
                f'Convergence Curve - {test.upper()} (Throughput, Higher is Better)'
            )

        # 绘制 GPTuner 曲线
        if gp_values:
            x_gp = list(range(len(gp_values)))
            ax.plot(x_gp, gp_values, color=self.COLORS['gptuner'],
                    linewidth=1.5, alpha=0.7, label='GPTuner (BO)')
            # 绘制最佳点标记
            if is_latency:
                best_idx = int(np.argmin(gp_values))
            else:
                best_idx = int(np.argmax(gp_values))
            ax.scatter([best_idx], [gp_values[best_idx]],
                       color=self.COLORS['gptuner'], s=100, zorder=5,
                       marker='*', edgecolors='black', linewidths=0.5)

        # 绘制 LTuner 曲线
        if lt_values:
            x_lt = list(range(len(lt_values)))
            ax.plot(x_lt, lt_values, color=self.COLORS['ltuner'],
                    linewidth=2.0, label='LTuner (Self-Reflective)')
            if is_latency:
                best_idx = int(np.argmin(lt_values))
            else:
                best_idx = int(np.argmax(lt_values))
            ax.scatter([best_idx], [lt_values[best_idx]],
                       color=self.COLORS['ltuner'], s=100, zorder=5,
                       marker='*', edgecolors='black', linewidths=0.5)

        # 基准线
        baseline = gp.get('baseline_tps', 0) or gp.get('baseline_latency', 0)
        if baseline > 0:
            ax.axhline(y=baseline, color=self.COLORS['baseline'],
                       linestyle='--', linewidth=1, alpha=0.6,
                       label=_get_label('默认配置基准', 'Default Baseline'))

        ax.set_xlabel(_get_label('迭代次数', 'Iteration'), fontsize=12,
                      fontproperties=CHINESE_FONT)
        ax.set_ylabel(ylabel, fontsize=12, fontproperties=CHINESE_FONT)
        ax.set_title(title, fontsize=14, fontweight='bold',
                     fontproperties=CHINESE_FONT)
        ax.legend(fontsize=10, prop=CHINESE_FONT)
        ax.grid(True, alpha=0.3, color=self.COLORS['grid'])

        filepath = os.path.join(self.output_dir, f'{prefix}_convergence.png')
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return filepath

    def plot_performance_bar(self, data: Dict, prefix: str) -> str:
        """绘制最终性能柱状图"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.patch.set_facecolor(self.COLORS['background'])

        gp = data.get('gptuner', {})
        lt = data.get('ltuner', {})
        summary = data.get('comparison_summary', {})

        # 左图：性能提升百分比对比
        ax1 = axes[0]
        ax1.set_facecolor(self.COLORS['background'])
        methods = ['GPTuner\n(BO)', 'LTuner\n(Self-Reflective)']
        improvements = [
            summary.get('gptuner_improvement', 0),
            summary.get('ltuner_improvement', 0)
        ]
        colors = [self.COLORS['gptuner'], self.COLORS['ltuner']]
        bars = ax1.bar(methods, improvements, color=colors, width=0.5, edgecolor='white')
        for bar, val in zip(bars, improvements):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     f'{val:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
        ax1.set_ylabel(_get_label('性能提升 (%)', 'Improvement (%)'),
                       fontsize=11, fontproperties=CHINESE_FONT)
        ax1.set_title(_get_label('最终性能提升对比', 'Final Performance Improvement'),
                      fontsize=13, fontweight='bold', fontproperties=CHINESE_FONT)
        ax1.grid(axis='y', alpha=0.3)

        # 右图：绝对值对比
        ax2 = axes[1]
        ax2.set_facecolor(self.COLORS['background'])
        test = data.get('test', 'tpcc')
        is_latency = test in ['tpch']

        if is_latency:
            gp_val = gp.get('best_latency', 0)
            lt_val = lt.get('best_latency', 0)
            baseline_val = gp.get('baseline_latency', 0)
            ylabel = _get_label('延迟 (μs)', 'Latency (μs)')
            title = _get_label('最佳延迟对比', 'Best Latency Comparison')
        else:
            gp_val = gp.get('best_tps', 0)
            lt_val = lt.get('best_tps', 0)
            baseline_val = gp.get('baseline_tps', 0)
            ylabel = _get_label('TPS', 'TPS')
            title = _get_label('最佳 TPS 对比', 'Best TPS Comparison')

        x_labels = [_get_label('默认', 'Default'), 'GPTuner', 'LTuner']
        values = [baseline_val, gp_val, lt_val]
        bar_colors = [self.COLORS['baseline'], self.COLORS['gptuner'], self.COLORS['ltuner']]
        bars = ax2.bar(x_labels, values, color=bar_colors, width=0.5, edgecolor='white')
        for bar, val in zip(bars, values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     f'{val:.0f}', ha='center', va='bottom', fontsize=11)
        ax2.set_ylabel(ylabel, fontsize=11, fontproperties=CHINESE_FONT)
        ax2.set_title(title, fontsize=13, fontweight='bold', fontproperties=CHINESE_FONT)
        ax2.grid(axis='y', alpha=0.3)

        fig.tight_layout(pad=2.0)
        filepath = os.path.join(self.output_dir, f'{prefix}_performance.png')
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return filepath

    def plot_time_comparison(self, data: Dict, prefix: str) -> str:
        """绘制调优耗时对比图"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.patch.set_facecolor(self.COLORS['background'])

        gp = data.get('gptuner', {})
        lt = data.get('ltuner', {})
        summary = data.get('comparison_summary', {})

        # 左图：总耗时对比
        ax1 = axes[0]
        ax1.set_facecolor(self.COLORS['background'])
        methods = ['GPTuner', 'LTuner']
        total_times = [
            summary.get('gptuner_time_seconds', 0) / 60,  # 转为分钟
            summary.get('ltuner_time_seconds', 0) / 60
        ]
        colors = [self.COLORS['gptuner'], self.COLORS['ltuner']]
        bars = ax1.bar(methods, total_times, color=colors, width=0.4, edgecolor='white')
        for bar, val in zip(bars, total_times):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     f'{val:.0f} min', ha='center', va='bottom', fontsize=11)
        ax1.set_ylabel(_get_label('总耗时 (分钟)', 'Total Time (minutes)'),
                       fontsize=11, fontproperties=CHINESE_FONT)
        ax1.set_title(_get_label('总调优耗时对比', 'Total Tuning Time'),
                      fontsize=13, fontweight='bold', fontproperties=CHINESE_FONT)
        ax1.grid(axis='y', alpha=0.3)

        # 右图：迭代次数对比
        ax2 = axes[1]
        ax2.set_facecolor(self.COLORS['background'])
        iters = [
            summary.get('gptuner_iterations', 0),
            summary.get('ltuner_iterations', 0)
        ]
        bars = ax2.bar(methods, iters, color=colors, width=0.4, edgecolor='white')
        for bar, val in zip(bars, iters):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     f'{val}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        ax2.set_ylabel(_get_label('迭代次数', 'Iterations'),
                       fontsize=11, fontproperties=CHINESE_FONT)
        ax2.set_title(_get_label('调优迭代次数对比', 'Tuning Iterations'),
                      fontsize=13, fontweight='bold', fontproperties=CHINESE_FONT)
        ax2.grid(axis='y', alpha=0.3)

        # 添加时间减少标注
        reduction = summary.get('time_reduction_percent', 0)
        fig.text(0.5, 0.02,
                 _get_label(
                     f'LTuner 总耗时减少 {reduction:.1f}%，迭代次数减少 {summary.get("iteration_reduction", 0):.1f}%',
                     f'LTuner reduces time by {reduction:.1f}%, iterations by {summary.get("iteration_reduction", 0):.1f}%'
                 ),
                 ha='center', fontsize=11, style='italic',
                 fontproperties=CHINESE_FONT)

        fig.tight_layout(pad=2.0, rect=[0, 0.06, 1, 1])
        filepath = os.path.join(self.output_dir, f'{prefix}_time.png')
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return filepath

    def plot_safety_table(self, data: Dict, prefix: str) -> str:
        """绘制安全性对比表"""
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        fig.patch.set_facecolor(self.COLORS['background'])
        ax.set_facecolor(self.COLORS['background'])
        ax.axis('off')

        summary = data.get('comparison_summary', {})
        gp = data.get('gptuner', {})
        lt = data.get('ltuner', {})

        # 表格数据
        col_labels = ['', 'GPTuner (BO)', 'LTuner (Self-Reflective)']
        table_data = [
            [
                _get_label('配置失败次数', 'Config Failures'),
                str(summary.get('gptuner_failures', 0)),
                str(summary.get('ltuner_failures', 0))
            ],
            [
                _get_label('迭代次数', 'Iterations'),
                str(summary.get('gptuner_iterations', 0)),
                str(summary.get('ltuner_iterations', 0))
            ],
            [
                _get_label('性能提升 (%)', 'Improvement (%)'),
                f"{summary.get('gptuner_improvement', 0):.1f}%",
                f"{summary.get('ltuner_improvement', 0):.1f}%"
            ],
            [
                _get_label('总耗时', 'Total Time'),
                f"{summary.get('gptuner_time_seconds', 0)/60:.0f} min",
                f"{summary.get('ltuner_time_seconds', 0)/60:.0f} min"
            ],
            [
                _get_label('每轮均耗时', 'Avg Time/Iter'),
                f"{summary.get('gptuner_time_seconds', 0) / max(summary.get('gptuner_iterations', 1), 1):.0f} s",
                f"{summary.get('ltuner_time_seconds', 0) / max(summary.get('ltuner_iterations', 1), 1):.0f} s"
            ],
        ]

        table = ax.table(
            cellText=table_data,
            colLabels=col_labels,
            cellLoc='center',
            loc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 1.8)

        # 表头样式
        for j in range(len(col_labels)):
            table[0, j].set_facecolor('#37474F')
            table[0, j].set_text_props(color='white', fontweight='bold')

        # 数据行交替颜色
        for i in range(1, len(table_data) + 1):
            for j in range(len(col_labels)):
                if i % 2 == 0:
                    table[i, j].set_facecolor('#ECEFF1')

        ax.set_title(
            _get_label('GPTuner vs LTuner 综合对比', 'GPTuner vs LTuner Comparison'),
            fontsize=14, fontweight='bold', pad=20,
            fontproperties=CHINESE_FONT
        )

        filepath = os.path.join(self.output_dir, f'{prefix}_safety.png')
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return filepath

    def plot_dashboard(self, data: Dict, prefix: str) -> str:
        """绘制综合仪表盘（4合1）"""
        fig = plt.figure(figsize=(16, 12))
        fig.patch.set_facecolor(self.COLORS['background'])

        gp = data.get('gptuner', {})
        lt = data.get('ltuner', {})
        summary = data.get('comparison_summary', {})
        test = data.get('test', 'tpcc')
        is_latency = test in ['tpch']

        # ===== 左上：收敛曲线 =====
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.set_facecolor(self.COLORS['background'])

        if is_latency:
            gp_vals = gp.get('iteration_latency', [])
            lt_vals = lt.get('iteration_latency', [])
            ylabel = _get_label('延迟 (μs)', 'Latency')
        else:
            gp_vals = gp.get('iteration_tps', [])
            lt_vals = lt.get('iteration_tps', [])
            ylabel = 'TPS'

        if gp_vals:
            ax1.plot(range(len(gp_vals)), gp_vals,
                     color=self.COLORS['gptuner'], alpha=0.7,
                     linewidth=1.2, label='GPTuner')
        if lt_vals:
            ax1.plot(range(len(lt_vals)), lt_vals,
                     color=self.COLORS['ltuner'],
                     linewidth=2.0, label='LTuner')
        ax1.set_xlabel(_get_label('迭代', 'Iteration'), fontproperties=CHINESE_FONT)
        ax1.set_ylabel(ylabel, fontproperties=CHINESE_FONT)
        ax1.set_title(_get_label('收敛曲线对比', 'Convergence Curve'),
                      fontweight='bold', fontproperties=CHINESE_FONT)
        ax1.legend(fontsize=9, prop=CHINESE_FONT)
        ax1.grid(alpha=0.3)

        # ===== 右上：性能提升柱状图 =====
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.set_facecolor(self.COLORS['background'])
        methods = ['GPTuner', 'LTuner']
        improvements = [
            summary.get('gptuner_improvement', 0),
            summary.get('ltuner_improvement', 0)
        ]
        colors = [self.COLORS['gptuner'], self.COLORS['ltuner']]
        bars = ax2.bar(methods, improvements, color=colors, width=0.4)
        for bar, val in zip(bars, improvements):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                     f'{val:.1f}%', ha='center', fontweight='bold')
        ax2.set_ylabel(_get_label('提升 (%)', 'Improvement (%)'),
                       fontproperties=CHINESE_FONT)
        ax2.set_title(_get_label('性能提升对比', 'Performance Improvement'),
                      fontweight='bold', fontproperties=CHINESE_FONT)
        ax2.grid(axis='y', alpha=0.3)

        # ===== 左下：耗时对比 =====
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.set_facecolor(self.COLORS['background'])
        total_times = [
            summary.get('gptuner_time_seconds', 0) / 60,
            summary.get('ltuner_time_seconds', 0) / 60
        ]
        bars = ax3.bar(methods, total_times, color=colors, width=0.4)
        for bar, val in zip(bars, total_times):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                     f'{val:.0f}m', ha='center', fontweight='bold')
        ax3.set_ylabel(_get_label('耗时 (分钟)', 'Time (min)'),
                       fontproperties=CHINESE_FONT)
        ax3.set_title(_get_label('调优耗时对比', 'Tuning Time'),
                      fontweight='bold', fontproperties=CHINESE_FONT)
        ax3.grid(axis='y', alpha=0.3)

        # ===== 右下：安全性（失败次数）=====
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.set_facecolor(self.COLORS['background'])
        failures = [
            summary.get('gptuner_failures', 0),
            summary.get('ltuner_failures', 0)
        ]
        bars = ax4.bar(methods, failures, color=colors, width=0.4)
        for bar, val in zip(bars, failures):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                     str(val), ha='center', fontweight='bold')
        ax4.set_ylabel(_get_label('失败次数', 'Failures'),
                       fontproperties=CHINESE_FONT)
        ax4.set_title(_get_label('配置失败次数对比', 'Config Failures'),
                      fontweight='bold', fontproperties=CHINESE_FONT)
        ax4.grid(axis='y', alpha=0.3)

        winner = summary.get('winner', 'LTuner')
        fig.suptitle(
            _get_label(
                f'LTuner vs GPTuner 综合对比仪表盘 - {test.upper()}',
                f'LTuner vs GPTuner Dashboard - {test.upper()}'
            ),
            fontsize=16, fontweight='bold', y=0.98,
            fontproperties=CHINESE_FONT
        )

        fig.tight_layout(pad=2.0, rect=[0, 0, 1, 0.95])
        filepath = os.path.join(self.output_dir, f'{prefix}_dashboard.png')
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return filepath


def generate_comparison_charts(comparison_json_path: str,
                                output_dir: str = None) -> Dict[str, str]:
    """
    便捷函数：从 JSON 文件加载对比结果并生成所有图表

    Args:
        comparison_json_path: comparison.py 输出的 JSON 文件路径
        output_dir: 图表输出目录

    Returns:
        {图表名: 文件路径}
    """
    with open(comparison_json_path, 'r') as f:
        data = json.load(f)

    if output_dir is None:
        output_dir = os.path.dirname(comparison_json_path)

    test = data.get('test', 'unknown')
    visualizer = ExperimentVisualizer(output_dir=output_dir)
    return visualizer.visualize_all(data, prefix=f"comparison_{test}")


# 测试入口
if __name__ == '__main__':
    from experiments.comparison import ComparisonRunner

    # 生成模拟数据
    output_dir = "./optimization_results/comparison"
    runner = ComparisonRunner(output_dir=output_dir)

    # TPCC 对比
    result_tpcc = runner.run_comparison(test='tpcc', mode='simulate', seed=42)
    viz = ExperimentVisualizer(output_dir=output_dir)
    charts_tpcc = viz.visualize_all(result_tpcc, prefix='comparison_tpcc')

    # TPCH 对比
    result_tpch = runner.run_comparison(test='tpch', mode='simulate', seed=42)
    charts_tpch = viz.visualize_all(result_tpch, prefix='comparison_tpch')

    print("\n所有图表生成完成!")
    print(f"输出目录: {output_dir}")
