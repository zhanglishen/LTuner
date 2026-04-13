#!/usr/bin/env python3
"""
TPC-C 三方对比图：GPTuner vs LTuner(无温度调节) vs LTuner(有温度调节)
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os

# ---------- 字体设置 ----------
# 优先使用系统中文字体
import matplotlib.font_manager as fm
zh_fonts = [f.name for f in fm.fontManager.ttflist
            if any(k in f.name.lower() for k in ['simhei','noto sans cjk','wqy','droid sans fallback','source han'])]
if zh_fonts:
    rcParams['font.sans-serif'] = [zh_fonts[0]] + ['DejaVu Sans']
else:
    rcParams['font.sans-serif'] = ['DejaVu Sans']
rcParams['axes.unicode_minus'] = False

# ---------- 数据加载 ----------
# GPTuner (第二次实验, 70轮 BO)
with open('/root/GPTuner/optimization_results/comparison_real/tpcc_gptuner_ltuner_20260405_101641/comparison_real_tpcc.json') as f:
    run2 = json.load(f)

# LTuner v2 (无温度调节, 第二次实验, 12轮)
with open('/root/GPTuner/optimization_results/comparison_real/tpcc_gptuner_ltuner_20260405_101641/ltuner/ltuner_result.json') as f:
    ltuner_v2 = json.load(f)

# LTuner v3 (有温度调节, 第三次实验, 18轮)
with open('/root/GPTuner/optimization_results/comparison_real/tpcc_ltuner_20260405_162059/ltuner/ltuner_result.json') as f:
    ltuner_v3 = json.load(f)

# 提取数据
baseline = run2['baseline_tps']  # ~541.5

gptuner_tps = np.array(run2['gptuner']['iteration_tps'])
gptuner_tps_clean = np.where(gptuner_tps <= 0, np.nan, gptuner_tps)  # 崩溃轮次设为 NaN
gptuner_best = run2['gptuner']['best_tps']
gptuner_iters = len(gptuner_tps)
gptuner_improve = run2['gptuner']['improvement_percent']

lv2_tps = np.array([r['throughput'] for r in ltuner_v2['history']])
lv2_best = ltuner_v2['best_performance']
lv2_iters = len(lv2_tps)
lv2_improve = ltuner_v2['improvement_percent']
lv2_baseline = ltuner_v2['baseline_performance']

lv3_tps = np.array([r['throughput'] for r in ltuner_v3['history']])
lv3_best = ltuner_v3['best_performance']
lv3_iters = len(lv3_tps)
lv3_improve = ltuner_v3['improvement_percent']
lv3_baseline = ltuner_v3['baseline_performance']

# 累积最优
def cummax(arr):
    result = np.copy(arr).astype(float)
    for i in range(1, len(result)):
        if np.isnan(result[i]) or result[i] < result[i-1]:
            result[i] = result[i-1]
    return result

gptuner_cummax = cummax(gptuner_tps_clean)
lv2_cummax = cummax(lv2_tps)
lv3_cummax = cummax(lv3_tps)

# 颜色方案
C_GP = '#3498DB'   # GPTuner - 蓝色
C_V2 = '#E74C3C'   # LTuner 无温度 - 红色
C_V3 = '#27AE60'   # LTuner 有温度 - 绿色
C_BL = '#7F8C8D'   # 基线 - 灰色

output_dir = '/root/GPTuner/optimization_results/comparison_real/tpcc_ltuner_20260405_162059'

# ============================================================
# 图1: 综合对比大图 (2x2)
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('TPC-C Benchmark: GPTuner vs LTuner vs Enhanced LTuner',
             fontsize=18, fontweight='bold', y=0.98)

# --- (a) 收敛曲线 ---
ax = axes[0, 0]
ax.plot(range(1, gptuner_iters+1), gptuner_tps_clean, '.', color=C_GP, alpha=0.3, markersize=3)
ax.plot(range(1, gptuner_iters+1), gptuner_cummax, '-', color=C_GP, linewidth=2.5,
        label=f'GPTuner (BO, {gptuner_iters} iters, +{gptuner_improve:.0f}%)')
ax.plot(range(1, lv2_iters+1), lv2_tps, 'o-', color=C_V2, linewidth=2, markersize=5,
        label=f'LTuner (no temp, {lv2_iters} iters, +{lv2_improve:.0f}%)')
ax.plot(range(1, lv3_iters+1), lv3_tps, 's-', color=C_V3, linewidth=2.5, markersize=5,
        label=f'Enhanced LTuner ({lv3_iters} iters, +{lv3_improve:.0f}%)')
ax.axhline(y=baseline, color=C_BL, linestyle='--', linewidth=1.5, alpha=0.6,
           label=f'Baseline: {baseline:.0f} TPS')
ax.set_xlabel('Iteration', fontsize=12, fontweight='bold')
ax.set_ylabel('Throughput (TPS)', fontsize=12, fontweight='bold')
ax.set_title('(a) Convergence Curve', fontsize=14, fontweight='bold')
ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xlim(0, max(gptuner_iters, lv3_iters) + 2)

# --- (b) 性能柱状图 ---
ax = axes[0, 1]
methods = ['GPTuner\n(BO)', 'LTuner\n(No Temp)', 'Enhanced\nLTuner']
best_values = [gptuner_best, lv2_best, lv3_best]
improve_values = [gptuner_improve, lv2_improve, lv3_improve]
colors = [C_GP, C_V2, C_V3]

bars = ax.bar(methods, best_values, color=colors, alpha=0.85, edgecolor='black', linewidth=0.8, width=0.6)
ax.axhline(y=baseline, color=C_BL, linestyle='--', linewidth=2, alpha=0.7)
ax.text(2.45, baseline + 20, f'Baseline: {baseline:.0f}', fontsize=9, color=C_BL, fontweight='bold')

for bar, val, imp in zip(bars, best_values, improve_values):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 30,
            f'{val:.0f} TPS\n(+{imp:.0f}%)',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_ylabel('Best Throughput (TPS)', fontsize=12, fontweight='bold')
ax.set_title('(b) Best Performance Comparison', fontsize=14, fontweight='bold')
ax.set_ylim(0, max(best_values) * 1.18)
ax.grid(True, alpha=0.3, linestyle='--', axis='y')

# --- (c) 效率对比（每轮收益 & 总耗时）---
ax = axes[1, 0]

# 左轴：每轮平均 TPS 提升
avg_gain_per_iter = [(gptuner_best - baseline) / gptuner_iters,
                     (lv2_best - lv2_baseline) / lv2_iters,
                     (lv3_best - lv3_baseline) / lv3_iters]

bars1 = ax.bar(np.arange(3) - 0.2, avg_gain_per_iter, width=0.35, color=colors, alpha=0.85,
               edgecolor='black', linewidth=0.8, label='Avg TPS Gain / Iter')
for bar, val in zip(bars1, avg_gain_per_iter):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
            f'{val:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_ylabel('Avg TPS Gain per Iteration', fontsize=11, fontweight='bold')
ax.set_title('(c) Optimization Efficiency', fontsize=14, fontweight='bold')

# 右轴：总迭代数
ax2 = ax.twinx()
iter_counts = [gptuner_iters, lv2_iters, lv3_iters]
bars2 = ax2.bar(np.arange(3) + 0.2, iter_counts, width=0.35, color=colors, alpha=0.4,
                edgecolor='black', linewidth=0.8, hatch='///', label='Total Iterations')
for bar, val in zip(bars2, iter_counts):
    ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
             f'{val}', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax2.set_ylabel('Total Iterations', fontsize=11, fontweight='bold')

ax.set_xticks(np.arange(3))
ax.set_xticklabels(methods, fontsize=10)
ax.grid(True, alpha=0.3, linestyle='--', axis='y')

lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)

# --- (d) 综合指标表格 ---
ax = axes[1, 1]
ax.axis('off')

# 计算方差和振幅
def calc_delta_stats(tps_arr):
    deltas = []
    for i in range(1, len(tps_arr)):
        if not np.isnan(tps_arr[i]) and not np.isnan(tps_arr[i-1]) and tps_arr[i-1] > 0:
            deltas.append((tps_arr[i] - tps_arr[i-1]) / tps_arr[i-1] * 100)
    if deltas:
        return np.var(deltas), max(deltas) - min(deltas)
    return 0, 0

gp_var, gp_amp = calc_delta_stats(gptuner_tps_clean)
v2_var, v2_amp = calc_delta_stats(lv2_tps)
v3_var, v3_amp = calc_delta_stats(lv3_tps)

# 崩溃次数
gp_crashes = int(np.sum(gptuner_tps <= 0))

table_data = [
    ['Metric', 'GPTuner (BO)', 'LTuner\n(No Temp)', 'Enhanced\nLTuner'],
    ['Baseline TPS', f'{baseline:.0f}', f'{lv2_baseline:.0f}', f'{lv3_baseline:.0f}'],
    ['Best TPS', f'{gptuner_best:.0f}', f'{lv2_best:.0f}', f'{lv3_best:.0f}'],
    ['Improvement', f'+{gptuner_improve:.0f}%', f'+{lv2_improve:.0f}%', f'+{lv3_improve:.0f}%'],
    ['Iterations', str(gptuner_iters), str(lv2_iters), str(lv3_iters)],
    ['Crashes', str(gp_crashes), '0', '0'],
    ['Efficiency\n(TPS/iter)', f'{avg_gain_per_iter[0]:.1f}', f'{avg_gain_per_iter[1]:.1f}', f'{avg_gain_per_iter[2]:.1f}'],
    ['Variance', f'{gp_var:.0f}', f'{v2_var:.1f}', f'{v3_var:.0f}'],
    ['Amplitude', f'{gp_amp:.0f}%', f'{v2_amp:.1f}%', f'{v3_amp:.0f}%'],
    ['Winner', '', '', '***'],
]

table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                 colWidths=[0.22, 0.22, 0.22, 0.22])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2.2)

for i in range(len(table_data)):
    for j in range(len(table_data[0])):
        cell = table[(i, j)]
        if i == 0:
            cell.set_facecolor('#2C3E50')
            cell.set_text_props(weight='bold', color='white', fontsize=9)
        elif j == 1:
            cell.set_facecolor('#D6EAF8')  # GPTuner 蓝底
        elif j == 2:
            cell.set_facecolor('#FADBD8')  # LTuner v2 红底
        elif j == 3:
            cell.set_facecolor('#D5F5E3')  # LTuner v3 绿底
        else:
            cell.set_facecolor('#F8F9FA')
        cell.set_edgecolor('#BDC3C7')
        cell.set_linewidth(0.5)

    # 高亮 Winner 行
    if i == len(table_data) - 1:
        for j in range(len(table_data[0])):
            cell = table[(i, j)]
            if j == 3:
                cell.set_text_props(weight='bold', color='#27AE60', fontsize=11)

ax.set_title('(d) Comprehensive Metrics', fontsize=14, fontweight='bold', pad=20)

plt.tight_layout(rect=[0, 0, 1, 0.96])
fp1 = f'{output_dir}/tpcc_three_way_comparison.png'
plt.savefig(fp1, dpi=150, bbox_inches='tight')
print(f'[1/3] {fp1}')
plt.close()

# ============================================================
# 图2: 归一化收敛曲线（论文核心图）
# ============================================================
fig, ax = plt.subplots(figsize=(12, 6))

# 归一化为相对基线提升百分比
gp_norm = (gptuner_cummax - baseline) / baseline * 100
v2_norm = (lv2_cummax - lv2_baseline) / lv2_baseline * 100
v3_norm = (lv3_cummax - lv3_baseline) / lv3_baseline * 100

ax.plot(range(1, gptuner_iters+1), gp_norm, '-', color=C_GP, linewidth=2.5, alpha=0.9,
        label=f'GPTuner (BO): +{gptuner_improve:.0f}%')
ax.plot(range(1, lv2_iters+1), v2_norm, 'o-', color=C_V2, linewidth=2, markersize=6, alpha=0.9,
        label=f'LTuner (no temp sched.): +{lv2_improve:.0f}%')
ax.plot(range(1, lv3_iters+1), v3_norm, 's-', color=C_V3, linewidth=2.5, markersize=6, alpha=0.9,
        label=f'Enhanced LTuner (w/ temp sched.): +{lv3_improve:.0f}%')

# 标注关键点
ax.annotate(f'Best: +{gptuner_improve:.0f}%\n(iter {np.nanargmax(gptuner_tps_clean)+1})',
            xy=(np.nanargmax(gptuner_tps_clean)+1, gptuner_improve),
            xytext=(55, gptuner_improve+40),
            fontsize=9, fontweight='bold', color=C_GP,
            arrowprops=dict(arrowstyle='->', color=C_GP, lw=1.5))

ax.annotate(f'Best: +{lv2_improve:.0f}%\n(iter {np.argmax(lv2_tps)+1})',
            xy=(np.argmax(lv2_tps)+1, lv2_improve),
            xytext=(20, lv2_improve+40),
            fontsize=9, fontweight='bold', color=C_V2,
            arrowprops=dict(arrowstyle='->', color=C_V2, lw=1.5))

ax.annotate(f'Best: +{lv3_improve:.0f}%\n(iter {np.argmax(lv3_tps)+1})',
            xy=(np.argmax(lv3_tps)+1, lv3_improve),
            xytext=(12, lv3_improve+20),
            fontsize=9, fontweight='bold', color=C_V3,
            arrowprops=dict(arrowstyle='->', color=C_V3, lw=1.5))

# 里程碑线
for pct in [100, 200, 300, 400]:
    ax.axhline(y=pct, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
    ax.text(71, pct+5, f'+{pct}%', fontsize=8, color='gray', alpha=0.7)

ax.set_xlabel('Iteration', fontsize=13, fontweight='bold')
ax.set_ylabel('Improvement over Baseline (%)', fontsize=13, fontweight='bold')
ax.set_title('TPC-C: Normalized Convergence (Cumulative Best)', fontsize=15, fontweight='bold')
ax.legend(loc='upper left', fontsize=11, framealpha=0.9)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xlim(0, max(gptuner_iters, lv3_iters) + 5)
ax.set_ylim(-10, max(lv3_improve, gptuner_improve) * 1.15)

plt.tight_layout()
fp2 = f'{output_dir}/tpcc_normalized_convergence.png'
plt.savefig(fp2, dpi=150, bbox_inches='tight')
print(f'[2/3] {fp2}')
plt.close()

# ============================================================
# 图3: 论文用简洁柱状图
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('TPC-C Performance Comparison: Three Tuning Methods',
             fontsize=15, fontweight='bold', y=1.02)

# (a) 性能提升百分比
ax = axes[0]
bars = ax.bar(methods, improve_values, color=colors, alpha=0.85, edgecolor='black', linewidth=0.8)
for bar, val in zip(bars, improve_values):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 5,
            f'+{val:.0f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
ax.set_ylabel('Performance Improvement (%)', fontsize=11, fontweight='bold')
ax.set_title('(a) Overall Improvement', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, linestyle='--', axis='y')

# (b) 迭代效率
ax = axes[1]
bars = ax.bar(methods, avg_gain_per_iter, color=colors, alpha=0.85, edgecolor='black', linewidth=0.8)
for bar, val in zip(bars, avg_gain_per_iter):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
            f'{val:.1f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
ax.set_ylabel('Avg TPS Gain per Iteration', fontsize=11, fontweight='bold')
ax.set_title('(b) Iteration Efficiency', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, linestyle='--', axis='y')

# (c) 迭代次数与崩溃次数
ax = axes[2]
x = np.arange(3)
w = 0.35
bars1 = ax.bar(x - w/2, iter_counts, w, color=colors, alpha=0.85, edgecolor='black', linewidth=0.8, label='Iterations')
crash_counts = [gp_crashes, 0, 0]
bars2 = ax.bar(x + w/2, crash_counts, w, color=['#E74C3C', '#95A5A6', '#95A5A6'], alpha=0.6,
               edgecolor='black', linewidth=0.8, hatch='xxx', label='Crashes')
for bar, val in zip(bars1, iter_counts):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
            str(val), ha='center', va='bottom', fontsize=11, fontweight='bold')
for bar, val in zip(bars2, crash_counts):
    if val > 0:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                str(val), ha='center', va='bottom', fontsize=11, fontweight='bold', color='red')
ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=10)
ax.set_ylabel('Count', fontsize=11, fontweight='bold')
ax.set_title('(c) Iterations & Crashes', fontsize=13, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, linestyle='--', axis='y')

plt.tight_layout()
fp3 = f'{output_dir}/tpcc_paper_bar_chart.png'
plt.savefig(fp3, dpi=150, bbox_inches='tight')
print(f'[3/3] {fp3}')
plt.close()

print(f'\nAll charts saved to: {output_dir}/')
