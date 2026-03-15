#!/usr/bin/env python3
"""
对比实验结果可视化 (Experiment Visualizer)
支持 N 种方法动态对比 (GPTuner / SMAC-only / LTuner)
支持 TPC-H (延迟) 和 TPC-C (吞吐量) 指标
支持多 session 误差带 (中位数 + 四分位范围)
"""
import sys
import os
import json
import numpy as np
from typing import Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# matplotlib 中文支持
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


def _setup_chinese_font():
    """设置 matplotlib 中文字体"""
    font_candidates = [
        '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
        '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
        '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf',
    ]
    for font_path in font_candidates:
        if os.path.exists(font_path):
            return fm.FontProperties(fname=font_path)
    return None

CHINESE_FONT = _setup_chinese_font()


def _get_label(zh: str, en: str) -> str:
    return zh if CHINESE_FONT else en


# ============================================================
# 方法配置 (颜色 / 标签)
# ============================================================
METHOD_CONFIG = {
    'gptuner': {'color': '#2196F3', 'label': 'GPTuner (BO)', 'marker': 's'},
    'ltuner':  {'color': '#FF5722', 'label': 'LTuner (Self-Reflective)', 'marker': 'o'},
    'smac':    {'color': '#4CAF50', 'label': 'SMAC-only (Pure BO)', 'marker': '^'},
}
BASELINE_COLOR = '#9E9E9E'
BG_COLOR = '#FAFAFA'
GRID_COLOR = '#E0E0E0'


def _method_color(name):
    return METHOD_CONFIG.get(name, {}).get('color', '#795548')

def _method_label(name):
    return METHOD_CONFIG.get(name, {}).get('label', name)

def _method_marker(name):
    return METHOD_CONFIG.get(name, {}).get('marker', 'o')


class ExperimentVisualizer:
    """实验结果可视化器 - 支持动态方法数量"""

    def __init__(self, output_dir: str = "./optimization_results/comparison"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    # ----------------------------------------------------------
    # 公共入口
    # ----------------------------------------------------------
    def visualize_all(self, comparison_data: Dict,
                      prefix: str = "comparison") -> Dict[str, str]:
        saved = {}
        saved['convergence_curve'] = self.plot_convergence_curve(comparison_data, prefix)
        saved['performance_bar']   = self.plot_performance_bar(comparison_data, prefix)
        saved['time_comparison']   = self.plot_time_comparison(comparison_data, prefix)
        saved['safety_table']      = self.plot_safety_table(comparison_data, prefix)
        saved['dashboard']         = self.plot_dashboard(comparison_data, prefix)

        # 多 session 误差带 (如果有多个 session)
        if comparison_data.get('sessions_count', 1) > 1:
            saved['multi_session'] = self.plot_multi_session_convergence(comparison_data, prefix)

        print(f"\n[可视化] 已生成 {len(saved)} 张图表:")
        for name, fp in saved.items():
            print(f"  - {name}: {fp}")
        return saved

    # ----------------------------------------------------------
    # 工具方法
    # ----------------------------------------------------------
    @staticmethod
    def _detect_methods(data: Dict) -> List[str]:
        """从数据中检测参与对比的方法列表"""
        if 'methods' in data:
            return data['methods']
        methods = []
        for k in ['gptuner', 'smac', 'ltuner']:
            if k in data and isinstance(data[k], dict):
                methods.append(k)
        return methods or ['gptuner', 'ltuner']

    @staticmethod
    def _is_latency(data: Dict) -> bool:
        return data.get('test', 'tpch') in ['tpch']

    @staticmethod
    def _get_values(method_data: Dict, is_latency: bool) -> List[float]:
        if is_latency:
            return method_data.get('iteration_latency', [])
        return method_data.get('iteration_tps', [])

    @staticmethod
    def _get_best(method_data: Dict, is_latency: bool) -> float:
        if is_latency:
            return method_data.get('best_latency', 0)
        return method_data.get('best_tps', 0)

    @staticmethod
    def _get_baseline(data: Dict, is_latency: bool) -> float:
        if is_latency:
            return data.get('baseline_latency', 0)
        return data.get('baseline_tps', 0)

    @staticmethod
    def _count_crashes(values, baseline, is_latency):
        if not values or baseline <= 0:
            return 0
        if is_latency:
            return sum(1 for v in values if v > baseline * 1.5)
        else:
            return sum(1 for v in values if v < baseline * 0.5)

    @staticmethod
    def _cumulative_best(values, is_latency):
        if not values:
            return []
        result, best = [], values[0]
        for v in values:
            best = min(best, v) if is_latency else max(best, v)
            result.append(best)
        return result

    # ----------------------------------------------------------
    # 1. 收敛曲线 (双面板)
    # ----------------------------------------------------------
    def plot_convergence_curve(self, data: Dict, prefix: str) -> str:
        methods = self._detect_methods(data)
        is_lat = self._is_latency(data)
        baseline = self._get_baseline(data, is_lat)
        ylabel = _get_label('延迟 (μs)', 'Latency (μs)') if is_lat \
                 else _get_label('TPS (事务/秒)', 'TPS (tx/s)')

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.patch.set_facecolor(BG_COLOR)
        ax1.set_facecolor(BG_COLOR); ax2.set_facecolor(BG_COLOR)

        # === 左图: Best Found So Far ===
        for m in methods:
            md = data.get(m, {})
            vals = self._get_values(md, is_lat)
            cum = self._cumulative_best(vals, is_lat)
            if cum:
                ax1.plot(range(len(cum)), cum, color=_method_color(m),
                         linewidth=2.0 if m != 'ltuner' else 2.5,
                         alpha=0.85, label=_method_label(m))
                ax1.annotate(f'{cum[-1]:.0f}', xy=(len(cum)-1, cum[-1]),
                             fontsize=8, color=_method_color(m),
                             textcoords="offset points", xytext=(5, 5))
        if baseline > 0:
            ax1.axhline(y=baseline, color=BASELINE_COLOR, linestyle='--',
                        linewidth=1.5, alpha=0.7,
                        label=f'Baseline ({baseline:.0f})')
        ax1.set_xlabel(_get_label('迭代次数', 'Iteration'), fontsize=12,
                       fontproperties=CHINESE_FONT)
        ax1.set_ylabel(ylabel, fontsize=12, fontproperties=CHINESE_FONT)
        ax1.set_title(_get_label('累计最佳值 (Best Found So Far)',
                                 'Best Found So Far'),
                      fontsize=13, fontweight='bold', fontproperties=CHINESE_FONT)
        ax1.legend(fontsize=9, prop=CHINESE_FONT,
                   loc='upper right' if is_lat else 'lower right')
        ax1.grid(True, alpha=0.3, color=GRID_COLOR)

        # === 右图: Per-Iteration (clipped crashes) ===
        crash_threshold = baseline * 1.5 if is_lat else baseline * 0.5
        all_valid = []
        for m in methods:
            md = data.get(m, {})
            vals = self._get_values(md, is_lat)
            if is_lat:
                all_valid.extend([v for v in vals if v <= crash_threshold])
            else:
                all_valid.extend([v for v in vals if v >= crash_threshold])
        if all_valid:
            y_min = min(all_valid) * 0.9
            y_max = max(max(all_valid), baseline) * 1.15
        else:
            y_min, y_max = baseline * 0.8, baseline * 1.3

        for m in methods:
            md = data.get(m, {})
            vals = self._get_values(md, is_lat)
            if not vals:
                continue
            valid_x, valid_y, crash_x, crash_y = [], [], [], []
            for i, v in enumerate(vals):
                is_crash = (is_lat and v > crash_threshold) or \
                           (not is_lat and v < crash_threshold)
                if is_crash:
                    crash_x.append(i); crash_y.append(y_max * 0.98)
                else:
                    valid_x.append(i); valid_y.append(v)
            ax2.scatter(valid_x, valid_y, color=_method_color(m),
                        s=20, alpha=0.7, zorder=4, marker=_method_marker(m),
                        label=f'{_method_label(m)}')
            if crash_x:
                ax2.scatter(crash_x, crash_y, color='red', s=35, marker='x',
                            alpha=0.5, zorder=4,
                            label=f'{m} crash ({len(crash_x)})')

        if baseline > 0:
            ax2.axhline(y=baseline, color=BASELINE_COLOR, linestyle='--',
                        linewidth=1.5, alpha=0.7)
        ax2.set_ylim(y_min, y_max)
        ax2.set_xlabel(_get_label('迭代次数', 'Iteration'), fontsize=12,
                       fontproperties=CHINESE_FONT)
        ax2.set_ylabel(ylabel, fontsize=12, fontproperties=CHINESE_FONT)
        ax2.set_title(_get_label('每轮实际值 (裁剪崩溃)', 'Per-Iteration (Clipped)'),
                      fontsize=13, fontweight='bold', fontproperties=CHINESE_FONT)
        ax2.legend(fontsize=8, prop=CHINESE_FONT)
        ax2.grid(True, alpha=0.3, color=GRID_COLOR)

        test_name = data.get('test', 'tpch').upper()
        hint = '延迟越低越好' if is_lat else '吞吐量越高越好'
        fig.suptitle(_get_label(f'收敛曲线对比 - {test_name} ({hint})',
                                f'Convergence - {test_name}'),
                     fontsize=15, fontweight='bold', fontproperties=CHINESE_FONT, y=1.02)
        fig.tight_layout(pad=2.0)
        fp = os.path.join(self.output_dir, f'{prefix}_convergence.png')
        fig.savefig(fp, dpi=150, bbox_inches='tight'); plt.close(fig)
        return fp

    # ----------------------------------------------------------
    # 2. 最终性能柱状图
    # ----------------------------------------------------------
    def plot_performance_bar(self, data: Dict, prefix: str) -> str:
        methods = self._detect_methods(data)
        is_lat = self._is_latency(data)
        baseline = self._get_baseline(data, is_lat)

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.patch.set_facecolor(BG_COLOR)

        # 左: 性能提升百分比
        ax1 = axes[0]; ax1.set_facecolor(BG_COLOR)
        labels = [_method_label(m).split('(')[0].strip() for m in methods]
        improvements = [data.get(m, {}).get('improvement_percent', 0) for m in methods]
        colors = [_method_color(m) for m in methods]
        bars = ax1.bar(labels, improvements, color=colors, width=0.5, edgecolor='white')
        for bar, val in zip(bars, improvements):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                     f'{val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
        ax1.set_ylabel(_get_label('性能提升 (%)', 'Improvement (%)'),
                       fontsize=11, fontproperties=CHINESE_FONT)
        ax1.set_title(_get_label('性能提升对比', 'Performance Improvement'),
                      fontsize=13, fontweight='bold', fontproperties=CHINESE_FONT)
        ax1.grid(axis='y', alpha=0.3)

        # 右: 绝对值 (Baseline + 各方法)
        ax2 = axes[1]; ax2.set_facecolor(BG_COLOR)
        x_labels = [_get_label('默认', 'Default')] + labels
        values = [baseline]
        bar_colors = [BASELINE_COLOR]
        for m in methods:
            md = data.get(m, {})
            val = md.get('best_latency', 0) if is_lat else md.get('best_tps', 0)
            values.append(val)
            bar_colors.append(_method_color(m))
        bars = ax2.bar(x_labels, values, color=bar_colors, width=0.5, edgecolor='white')
        for bar, val in zip(bars, values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                     f'{val:.0f}', ha='center', va='bottom', fontsize=10)
        metric_label = _get_label('延迟 (μs)', 'Latency') if is_lat else 'TPS'
        ax2.set_ylabel(metric_label, fontsize=11, fontproperties=CHINESE_FONT)
        ax2.set_title(_get_label('最佳值对比', 'Best Value Comparison'),
                      fontsize=13, fontweight='bold', fontproperties=CHINESE_FONT)
        ax2.grid(axis='y', alpha=0.3)

        fig.tight_layout(pad=2.0)
        fp = os.path.join(self.output_dir, f'{prefix}_performance.png')
        fig.savefig(fp, dpi=150, bbox_inches='tight'); plt.close(fig)
        return fp

    # ----------------------------------------------------------
    # 3. 调优耗时对比
    # ----------------------------------------------------------
    def plot_time_comparison(self, data: Dict, prefix: str) -> str:
        methods = self._detect_methods(data)
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.patch.set_facecolor(BG_COLOR)

        labels = [_method_label(m).split('(')[0].strip() for m in methods]
        colors = [_method_color(m) for m in methods]

        # 左: 总耗时
        ax1 = axes[0]; ax1.set_facecolor(BG_COLOR)
        times = [data.get(m, {}).get('total_time_seconds', 0) / 60 for m in methods]
        bars = ax1.bar(labels, times, color=colors, width=0.4, edgecolor='white')
        for bar, val in zip(bars, times):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                     f'{val:.0f}m', ha='center', fontsize=11)
        ax1.set_ylabel(_get_label('总耗时 (分钟)', 'Time (min)'),
                       fontsize=11, fontproperties=CHINESE_FONT)
        ax1.set_title(_get_label('总调优耗时', 'Total Tuning Time'),
                      fontsize=13, fontweight='bold', fontproperties=CHINESE_FONT)
        ax1.grid(axis='y', alpha=0.3)

        # 右: 迭代次数
        ax2 = axes[1]; ax2.set_facecolor(BG_COLOR)
        iters = [data.get(m, {}).get('total_iterations', 0) for m in methods]
        bars = ax2.bar(labels, iters, color=colors, width=0.4, edgecolor='white')
        for bar, val in zip(bars, iters):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                     str(val), ha='center', fontsize=12, fontweight='bold')
        ax2.set_ylabel(_get_label('迭代次数', 'Iterations'),
                       fontsize=11, fontproperties=CHINESE_FONT)
        ax2.set_title(_get_label('调优迭代次数', 'Tuning Iterations'),
                      fontsize=13, fontweight='bold', fontproperties=CHINESE_FONT)
        ax2.grid(axis='y', alpha=0.3)

        fig.tight_layout(pad=2.0)
        fp = os.path.join(self.output_dir, f'{prefix}_time.png')
        fig.savefig(fp, dpi=150, bbox_inches='tight'); plt.close(fig)
        return fp

    # ----------------------------------------------------------
    # 4. 安全性对比表
    # ----------------------------------------------------------
    def plot_safety_table(self, data: Dict, prefix: str) -> str:
        methods = self._detect_methods(data)
        is_lat = self._is_latency(data)
        baseline = self._get_baseline(data, is_lat)

        fig, ax = plt.subplots(1, 1, figsize=(4 + len(methods) * 3, 4.5))
        fig.patch.set_facecolor(BG_COLOR); ax.set_facecolor(BG_COLOR)
        ax.axis('off')

        col_labels = [''] + [_method_label(m) for m in methods]
        rows = []

        # Row 1: 崩溃率
        crash_row = [_get_label('崩溃率', 'Crash Rate')]
        for m in methods:
            md = data.get(m, {})
            vals = self._get_values(md, is_lat)
            crashes = self._count_crashes(vals, baseline, is_lat)
            total = len(vals)
            rate = crashes / max(total, 1) * 100
            crash_row.append(f'{crashes}/{total} ({rate:.0f}%)')
        rows.append(crash_row)

        # Row 2: 迭代次数
        iter_row = [_get_label('迭代次数', 'Iterations')]
        for m in methods:
            iter_row.append(str(data.get(m, {}).get('total_iterations', 0)))
        rows.append(iter_row)

        # Row 3: 性能提升
        impr_row = [_get_label('性能提升', 'Improvement')]
        for m in methods:
            impr_row.append(f"{data.get(m, {}).get('improvement_percent', 0):.1f}%")
        rows.append(impr_row)

        # Row 4: 总耗时
        time_row = [_get_label('总耗时', 'Total Time')]
        for m in methods:
            t = data.get(m, {}).get('total_time_seconds', 0)
            time_row.append(f'{t/60:.0f} min')
        rows.append(time_row)

        # Row 5: 每轮均耗时
        avg_row = [_get_label('每轮均耗时', 'Avg/Iter')]
        for m in methods:
            md = data.get(m, {})
            t = md.get('total_time_seconds', 0)
            n = max(md.get('total_iterations', 1), 1)
            avg_row.append(f'{t/n:.0f} s')
        rows.append(avg_row)

        table = ax.table(cellText=rows, colLabels=col_labels,
                         cellLoc='center', loc='center')
        table.auto_set_font_size(False); table.set_fontsize(10)
        table.scale(1.2, 1.8)

        for j in range(len(col_labels)):
            table[0, j].set_facecolor('#37474F')
            table[0, j].set_text_props(color='white', fontweight='bold')
        for i in range(1, len(rows) + 1):
            for j in range(len(col_labels)):
                if i % 2 == 0:
                    table[i, j].set_facecolor('#ECEFF1')

        test_name = data.get('test', 'tpch').upper()
        ax.set_title(_get_label(f'综合对比 - {test_name}',
                                f'Comparison - {test_name}'),
                     fontsize=14, fontweight='bold', pad=20, fontproperties=CHINESE_FONT)

        fp = os.path.join(self.output_dir, f'{prefix}_safety.png')
        fig.savefig(fp, dpi=150, bbox_inches='tight'); plt.close(fig)
        return fp

    # ----------------------------------------------------------
    # 5. 综合仪表盘 (4合1)
    # ----------------------------------------------------------
    def plot_dashboard(self, data: Dict, prefix: str) -> str:
        methods = self._detect_methods(data)
        is_lat = self._is_latency(data)
        baseline = self._get_baseline(data, is_lat)
        ylabel = _get_label('延迟 (μs)', 'Latency') if is_lat else 'TPS'

        fig = plt.figure(figsize=(16, 12))
        fig.patch.set_facecolor(BG_COLOR)

        # == 左上: 收敛曲线 ==
        ax1 = fig.add_subplot(2, 2, 1); ax1.set_facecolor(BG_COLOR)
        for m in methods:
            md = data.get(m, {})
            vals = self._get_values(md, is_lat)
            cum = self._cumulative_best(vals, is_lat)
            if cum:
                ax1.plot(range(len(cum)), cum, color=_method_color(m),
                         linewidth=2.0, label=_method_label(m).split('(')[0].strip())
        if baseline > 0:
            ax1.axhline(y=baseline, color=BASELINE_COLOR, linestyle='--',
                        linewidth=1.2, alpha=0.6, label=f'Baseline ({baseline:.0f})')
        ax1.set_xlabel(_get_label('迭代', 'Iter'), fontproperties=CHINESE_FONT)
        ax1.set_ylabel(ylabel, fontproperties=CHINESE_FONT)
        ax1.set_title(_get_label('累计最佳收敛', 'Best So Far'),
                      fontweight='bold', fontproperties=CHINESE_FONT)
        ax1.legend(fontsize=8, prop=CHINESE_FONT); ax1.grid(alpha=0.3)

        # == 右上: 性能提升 ==
        ax2 = fig.add_subplot(2, 2, 2); ax2.set_facecolor(BG_COLOR)
        labels = [_method_label(m).split('(')[0].strip() for m in methods]
        imprs = [data.get(m, {}).get('improvement_percent', 0) for m in methods]
        colors = [_method_color(m) for m in methods]
        bars = ax2.bar(labels, imprs, color=colors, width=0.4)
        for bar, val in zip(bars, imprs):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                     f'{val:.1f}%', ha='center', fontweight='bold', fontsize=10)
        ax2.set_ylabel(_get_label('提升 (%)', 'Improvement (%)'),
                       fontproperties=CHINESE_FONT)
        ax2.set_title(_get_label('性能提升', 'Improvement'),
                      fontweight='bold', fontproperties=CHINESE_FONT)
        ax2.grid(axis='y', alpha=0.3)

        # == 左下: 耗时 ==
        ax3 = fig.add_subplot(2, 2, 3); ax3.set_facecolor(BG_COLOR)
        times = [data.get(m, {}).get('total_time_seconds', 0) / 60 for m in methods]
        bars = ax3.bar(labels, times, color=colors, width=0.4)
        for bar, val in zip(bars, times):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                     f'{val:.0f}m', ha='center', fontweight='bold')
        ax3.set_ylabel(_get_label('耗时 (分)', 'Time (min)'), fontproperties=CHINESE_FONT)
        ax3.set_title(_get_label('调优耗时', 'Tuning Time'),
                      fontweight='bold', fontproperties=CHINESE_FONT)
        ax3.grid(axis='y', alpha=0.3)

        # == 右下: 崩溃次数 ==
        ax4 = fig.add_subplot(2, 2, 4); ax4.set_facecolor(BG_COLOR)
        crashes = []
        for m in methods:
            md = data.get(m, {})
            vals = self._get_values(md, is_lat)
            crashes.append(self._count_crashes(vals, baseline, is_lat))
        bars = ax4.bar(labels, crashes, color=colors, width=0.4)
        y_top = max(max(crashes) * 1.3 + 1, 3) if crashes else 3
        ax4.set_ylim(bottom=0, top=y_top)
        for bar, val in zip(bars, crashes):
            ax4.text(bar.get_x() + bar.get_width()/2,
                     max(bar.get_height(), 0) + y_top * 0.03,
                     str(val), ha='center', fontweight='bold', fontsize=12)
        ax4.set_ylabel(_get_label('崩溃次数', 'Crashes'), fontproperties=CHINESE_FONT)
        ax4.set_title(_get_label('配置崩溃次数', 'Config Crashes'),
                      fontweight='bold', fontproperties=CHINESE_FONT)
        ax4.grid(axis='y', alpha=0.3)

        test_name = data.get('test', 'tpch').upper()
        fig.suptitle(_get_label(f'综合对比仪表盘 - {test_name}',
                                f'Dashboard - {test_name}'),
                     fontsize=16, fontweight='bold', y=0.98, fontproperties=CHINESE_FONT)
        fig.tight_layout(pad=2.0, rect=[0, 0, 1, 0.95])
        fp = os.path.join(self.output_dir, f'{prefix}_dashboard.png')
        fig.savefig(fp, dpi=150, bbox_inches='tight'); plt.close(fig)
        return fp

    # ----------------------------------------------------------
    # 6. 多 Session 误差带收敛图 (中位数 + 四分位)
    # ----------------------------------------------------------
    def plot_multi_session_convergence(self, data: Dict, prefix: str) -> str:
        """
        当 sessions_count > 1 时, 绘制误差带:
        - 中心线: 各 session 累计最佳的中位数
        - 阴影: Q1-Q3 四分位范围
        """
        methods = self._detect_methods(data)
        is_lat = self._is_latency(data)
        baseline = self._get_baseline(data, is_lat)
        all_sessions = data.get('all_sessions', [])
        if not all_sessions:
            return ''

        ylabel = _get_label('延迟 (μs)', 'Latency') if is_lat \
                 else _get_label('TPS', 'TPS')

        fig, ax = plt.subplots(figsize=(12, 6))
        fig.patch.set_facecolor(BG_COLOR); ax.set_facecolor(BG_COLOR)

        for m in methods:
            # Collect cumulative-best curves from all sessions
            session_curves = []
            for sess in all_sessions:
                md = sess.get(m, {})
                vals = self._get_values(md, is_lat)
                cum = self._cumulative_best(vals, is_lat)
                if cum:
                    session_curves.append(cum)
            if not session_curves:
                continue

            # Align lengths (pad with last value)
            max_len = max(len(c) for c in session_curves)
            aligned = []
            for c in session_curves:
                padded = c + [c[-1]] * (max_len - len(c))
                aligned.append(padded)
            arr = np.array(aligned)
            median = np.median(arr, axis=0)
            q1 = np.percentile(arr, 25, axis=0)
            q3 = np.percentile(arr, 75, axis=0)
            x = list(range(max_len))

            ax.plot(x, median, color=_method_color(m), linewidth=2.0,
                    label=f'{_method_label(m)} (median)')
            ax.fill_between(x, q1, q3, color=_method_color(m), alpha=0.2,
                            label=f'{_method_label(m)} (Q1-Q3)')

        if baseline > 0:
            ax.axhline(y=baseline, color=BASELINE_COLOR, linestyle='--',
                       linewidth=1.5, alpha=0.7, label=f'Baseline ({baseline:.0f})')

        n_sessions = len(all_sessions)
        test_name = data.get('test', 'tpch').upper()
        ax.set_xlabel(_get_label('迭代次数', 'Iteration'), fontsize=12,
                      fontproperties=CHINESE_FONT)
        ax.set_ylabel(ylabel, fontsize=12, fontproperties=CHINESE_FONT)
        ax.set_title(_get_label(f'多次重复收敛曲线 ({n_sessions} sessions) - {test_name}',
                                f'Multi-Session Convergence ({n_sessions} runs) - {test_name}'),
                     fontsize=14, fontweight='bold', fontproperties=CHINESE_FONT)
        ax.legend(fontsize=9, prop=CHINESE_FONT)
        ax.grid(True, alpha=0.3, color=GRID_COLOR)

        fig.tight_layout()
        fp = os.path.join(self.output_dir, f'{prefix}_multi_session.png')
        fig.savefig(fp, dpi=150, bbox_inches='tight'); plt.close(fig)
        return fp


# ============================================================
# 便捷函数
# ============================================================
def generate_comparison_charts(comparison_json_path: str,
                               output_dir: str = None) -> Dict[str, str]:
    with open(comparison_json_path, 'r') as f:
        data = json.load(f)
    if output_dir is None:
        output_dir = os.path.dirname(comparison_json_path)
    test = data.get('test', 'unknown')
    viz = ExperimentVisualizer(output_dir=output_dir)
    return viz.visualize_all(data, prefix=f"comparison_{test}")


if __name__ == '__main__':
    from experiments.comparison import ComparisonRunner
    output_dir = "./optimization_results/comparison"
    runner = ComparisonRunner(output_dir=output_dir)
    result_tpch = runner.run_comparison(test='tpch', mode='simulate', seed=42)
    viz = ExperimentVisualizer(output_dir=output_dir)
    viz.visualize_all(result_tpch, prefix='comparison_tpch')
    print("\n图表生成完成!")
