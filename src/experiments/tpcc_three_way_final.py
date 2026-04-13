#!/usr/bin/env python3
"""
TPC-C 三方对比图：GPTuner vs LTuner(No Temp) vs Enhanced LTuner
基于实验 tpcc_gptuner_ltuner_no_temp_ltuner_20260405_204250
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os

# ---------- 字体 ----------
import matplotlib.font_manager as fm
zh_fonts = [f.name for f in fm.fontManager.ttflist
            if any(k in f.name.lower() for k in ['simhei','noto sans cjk','wqy','droid sans fallback'])]
if zh_fonts:
    rcParams['font.sans-serif'] = [zh_fonts[0]] + ['DejaVu Sans']
else:
    rcParams['font.sans-serif'] = ['DejaVu Sans']
rcParams['axes.unicode_minus'] = False

# ---------- 数据加载 ----------
base = '/root/GPTuner/optimization_results/comparison_real/tpcc_gptuner_ltuner_no_temp_ltuner_20260405_204250'

# GPTuner
with open(f'{base}/comparison_real_tpcc.json') as f:
    main = json.load(f)
baseline = main['baseline_tps']
gp = main['gptuner']
gp_tps_raw = np.array(gp['iteration_tps'], dtype=float)
gp_tps_clean = np.where(gp_tps_raw <= 0, np.nan, gp_tps_raw)
gp_best = gp['best_tps']
gp_improve = gp['improvement_percent']
gp_iters = len(gp_tps_raw)
gp_crashes = int(np.sum(gp_tps_raw <= 0))
gp_time = gp['total_time_seconds']

# LTuner No Temp
with open(f'{base}/ltuner_no_temp/ltuner_result.json') as f:
    lv2_raw = json.load(f)
lv2_tps = np.array([r['throughput'] for r in lv2_raw['history']])
lv2_best = lv2_raw['best_performance']
lv2_improve = lv2_raw['improvement_percent']
lv2_iters = lv2_raw['total_iterations']
lv2_baseline = lv2_raw['baseline_performance']
lv2_time = sum(r.get('elapsed_seconds', 300) for r in lv2_raw['history'])

# Enhanced LTuner
with open(f'{base}/ltuner/ltuner_result.json') as f:
    lv3_raw = json.load(f)
lv3_tps = np.array([r['throughput'] for r in lv3_raw['history']])
lv3_best = lv3_raw['best_performance']
lv3_improve = lv3_raw['improvement_percent']
lv3_iters = lv3_raw['total_iterations']
lv3_baseline = lv3_raw['baseline_performance']
lv3_time = sum(r.get('elapsed_seconds', 300) for r in lv3_raw['history'])

# 累积最优
def cummax(arr):
    r = np.copy(arr).astype(float)
    for i in range(1, len(r)):
        if np.isnan(r[i]) or r[i] < r[i-1]:
            r[i] = r[i-1]
    return r

gp_cm = cummax(gp_tps_clean)
lv2_cm = cummax(lv2_tps)
lv3_cm = cummax(lv3_tps)

# 归一化提升%
gp_norm = (gp_cm - baseline) / baseline * 100
lv2_norm = (lv2_cm - lv2_baseline) / lv2_baseline * 100
lv3_norm = (lv3_cm - lv3_baseline) / lv3_baseline * 100

# 颜色
C_GP = '#2980B9'   # GPTuner
C_V2 = '#C0392B'   # LTuner No Temp
C_V3 = '#27AE60'   # Enhanced LTuner
C_BL = '#7F8C8D'   # 基线

# 平均效率
avg_gain = [
    (gp_best - baseline) / gp_iters,
    (lv2_best - lv2_baseline) / lv2_iters,
    (lv3_best - lv3_baseline) / lv3_iters,
]

output_dir = base
os.makedirs(output_dir, exist_ok=True)

# ============================================================
# 图1: 综合大图 2x2
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('TPC-C Benchmark: Three-Way Comparison\n'
             'GPTuner (BO) vs LTuner (No Temp) vs Enhanced LTuner (w/ Temp)',
             fontsize=16, fontweight='bold', y=0.99)

# (a) 收敛曲线
ax = axes[0, 0]
ax.plot(range(1, gp_iters+1), gp_tps_clean, '.', color=C_GP, alpha=0.25, markersize=3, label='_nolegend_')
ax.plot(range(1, gp_iters+1), gp_cm, '-', color=C_GP, linewidth=2.5,
        label=f'GPTuner (BO): {gp_iters} iters, +{gp_improve:.0f}%')
ax.plot(range(1, lv2_iters+1), lv2_tps, '.', color=C_V2, alpha=0.3, markersize=4, label='_nolegend_')
ax.plot(range(1, lv2_iters+1), lv2_cm, 'o-', color=C_V2, linewidth=2, markersize=5,
        label=f'LTuner (No Temp): {lv2_iters} iters, +{lv2_improve:.0f}%')
ax.plot(range(1, lv3_iters+1), lv3_tps, '.', color=C_V3, alpha=0.3, markersize=4, label='_nolegend_')
ax.plot(range(1, lv3_iters+1), lv3_cm, 's-', color=C_V3, linewidth=2.5, markersize=6,
        label=f'Enhanced LTuner: {lv3_iters} iters, +{lv3_improve:.0f}%')
ax.axhline(y=baseline, color=C_BL, linestyle='--', linewidth=1.5, alpha=0.6)
ax.text(gp_iters*0.72, baseline+30, f'Baseline: {baseline:.0f}', color=C_BL, fontsize=9)
ax.set_xlabel('Iteration', fontsize=12, fontweight='bold')
ax.set_ylabel('Throughput (TPS)', fontsize=12, fontweight='bold')
ax.set_title('(a) Convergence Curve (Cumulative Best)', fontsize=13, fontweight='bold')
ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xlim(0, gp_iters + 3)

# (b) 性能柱状图
ax = axes[0, 1]
methods = ['GPTuner\n(BO)', 'LTuner\n(No Temp)', 'Enhanced\nLTuner']
best_vals = [gp_best, lv2_best, lv3_best]
improve_vals = [gp_improve, lv2_improve, lv3_improve]
colors = [C_GP, C_V2, C_V3]
bars = ax.bar(methods, best_vals, color=colors, alpha=0.85, edgecolor='black', linewidth=0.8, width=0.55)
ax.axhline(y=baseline, color=C_BL, linestyle='--', linewidth=2, alpha=0.7)
ax.text(2.45, baseline+40, f'Baseline\n{baseline:.0f}', color=C_BL, fontsize=8, fontweight='bold')
for bar, val, imp in zip(bars, best_vals, improve_vals):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 30,
            f'{val:.0f}\n(+{imp:.0f}%)', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.set_ylabel('Best Throughput (TPS)', fontsize=12, fontweight='bold')
ax.set_title('(b) Best Performance Comparison', fontsize=13, fontweight='bold')
ax.set_ylim(0, max(best_vals) * 1.18)
ax.grid(True, alpha=0.3, linestyle='--', axis='y')

# (c) 效率双轴
ax = axes[1, 0]
x = np.arange(3)
w = 0.35
bars1 = ax.bar(x - w/2, avg_gain, w, color=colors, alpha=0.85, edgecolor='black', linewidth=0.8)
for bar, val in zip(bars1, avg_gain):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
            f'{val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.set_ylabel('Avg TPS Gain / Iteration', fontsize=11, fontweight='bold')
ax.set_title('(c) Optimization Efficiency', fontsize=13, fontweight='bold')
ax2 = ax.twinx()
iter_counts = [gp_iters, lv2_iters, lv3_iters]
crash_counts = [gp_crashes, 0, 0]
bars2a = ax2.bar(x + w/2 - 0.08, iter_counts, w*0.6, color=colors, alpha=0.35,
                 edgecolor='black', linewidth=0.8, hatch='///', label='Total Iterations')
bars2b = ax2.bar(x + w/2 + 0.18, crash_counts, w*0.6, color=[C_GP,'#95A5A6','#95A5A6'],
                 alpha=0.5, edgecolor='black', linewidth=0.8, hatch='xxx', label='Crashes')
for bar, val in zip(bars2a, iter_counts):
    ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height()+0.5,
             str(val), ha='center', va='bottom', fontsize=10, fontweight='bold')
if gp_crashes > 0:
    for bar in bars2b:
        if bar.get_height() > 0:
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height()+0.3,
                     str(gp_crashes), ha='center', va='bottom', fontsize=9, fontweight='bold', color='red')
ax2.set_ylabel('Count', fontsize=11, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=10)
lines1, labs1 = ax.get_legend_handles_labels()
lines2, labs2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labs1 + labs2, loc='upper right', fontsize=8)
ax.grid(True, alpha=0.3, linestyle='--', axis='y')

# (d) 指标表格
ax = axes[1, 1]
ax.axis('off')

# 方差/振幅
def pct_var(arr):
    d = np.diff(arr) / arr[:-1] * 100
    return np.std(d), np.max(d) - np.min(d)
gpv, gpa = pct_var(gp_tps_clean[~np.isnan(gp_tps_clean)])
lv2v, lv2a = pct_var(lv2_tps)
lv3v, lv3a = pct_var(lv3_tps)

table_data = [
    ['Metric', 'GPTuner (BO)', 'LTuner\n(No Temp)', 'Enhanced\nLTuner'],
    ['Baseline TPS', f'{baseline:.0f}', f'{lv2_baseline:.0f}', f'{lv3_baseline:.0f}'],
    ['Best TPS', f'{gp_best:.0f}', f'{lv2_best:.0f}', f'{lv3_best:.0f}'],
    ['Improvement', f'+{gp_improve:.0f}%', f'+{lv2_improve:.0f}%', f'+{lv3_improve:.0f}%'],
    ['Iterations', str(gp_iters), str(lv2_iters), str(lv3_iters)],
    ['Crashes', str(gp_crashes), '0', '0'],
    ['Efficiency\n(TPS/iter)', f'{avg_gain[0]:.1f}', f'{avg_gain[1]:.1f}', f'{avg_gain[2]:.1f}'],
    ['Time (min)', f'{gp_time/60:.0f}', f'{lv2_time/60:.0f}', f'{lv3_time/60:.0f}'],
    ['Converged', 'No', 'No', 'Yes'],
]

table = ax.table(cellText=table_data, cellLoc='center', loc='center', colWidths=[0.24, 0.22, 0.22, 0.22])
table.auto_set_font_size(False)
table.set_fontsize(9.5)
table.scale(1, 2.1)

for i in range(len(table_data)):
    for j in range(len(table_data[0])):
        cell = table[(i, j)]
        if i == 0:
            cell.set_facecolor('#2C3E50')
            cell.set_text_props(weight='bold', color='white', fontsize=9)
        elif j == 1: cell.set_facecolor('#D6EAF8')
        elif j == 2: cell.set_facecolor('#FADBD8')
        elif j == 3: cell.set_facecolor('#D5F5E3')
        else: cell.set_facecolor('#F8F9FA')
        cell.set_edgecolor('#BDC3C7')
        cell.set_linewidth(0.5)

ax.set_title('(d) Comprehensive Metrics', fontsize=13, fontweight='bold', pad=15)

plt.tight_layout(rect=[0, 0, 1, 0.96])
fp1 = f'{output_dir}/tpcc_three_way_comparison.png'
plt.savefig(fp1, dpi=150, bbox_inches='tight')
print(f'[1/3] {fp1}')
plt.close()

# ============================================================
# 图2: 归一化收敛曲线（论文核心图）
# ============================================================
fig, ax = plt.subplots(figsize=(13, 6.5))

ax.plot(range(1, gp_iters+1), gp_norm, '-', color=C_GP, linewidth=2.5,
        label=f'GPTuner (BO): +{gp_improve:.0f}% in {gp_iters} iters, {gp_crashes} crashes')
ax.plot(range(1, lv2_iters+1), lv2_norm, 'o-', color=C_V2, linewidth=2.5, markersize=7,
        label=f'LTuner (No Temp): +{lv2_improve:.0f}% in {lv2_iters} iters, no crashes')
ax.plot(range(1, lv3_iters+1), lv3_norm, 's-', color=C_V3, linewidth=2.5, markersize=7,
        label=f'Enhanced LTuner (w/ Temp Sched.): +{lv3_improve:.0f}% in {lv3_iters} iters, early convergence')

# 标注峰值
ax.annotate(f'+{gp_improve:.0f}%\n(iter {np.nanargmax(gp_tps_clean)+1})',
            xy=(np.nanargmax(gp_tps_clean)+1, gp_improve),
            xytext=(50, gp_improve+30), fontsize=9, fontweight='bold', color=C_GP,
            arrowprops=dict(arrowstyle='->', color=C_GP, lw=1.5))
ax.annotate(f'+{lv2_improve:.0f}%\n(iter {np.argmax(lv2_tps)+1})',
            xy=(np.argmax(lv2_tps)+1, lv2_improve),
            xytext=(40, lv2_improve-80), fontsize=9, fontweight='bold', color=C_V2,
            arrowprops=dict(arrowstyle='->', color=C_V2, lw=1.5))
ax.annotate(f'+{lv3_improve:.0f}%\n(iter {np.argmax(lv3_tps)+1})',
            xy=(np.argmax(lv3_tps)+1, lv3_improve),
            xytext=(25, lv3_improve-50), fontsize=9, fontweight='bold', color=C_V3,
            arrowprops=dict(arrowstyle='->', color=C_V3, lw=1.5))

for pct in [100, 200, 300, 400, 500]:
    ax.axhline(y=pct, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
    ax.text(gp_iters+1, pct+5, f'+{pct}%', fontsize=8, color='gray', alpha=0.7)

ax.axhline(y=0, color=C_BL, linestyle='--', linewidth=1, alpha=0.4)
ax.set_xlabel('Iteration', fontsize=13, fontweight='bold')
ax.set_ylabel('Improvement over Baseline (%)', fontsize=13, fontweight='bold')
ax.set_title('TPC-C: Normalized Convergence Comparison\n'
             '(Cumulative Best Throughput relative to Baseline)',
             fontsize=14, fontweight='bold')
ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xlim(0, gp_iters + 5)
ax.set_ylim(-10, max(gp_improve, lv2_improve, lv3_improve) * 1.15)

plt.tight_layout()
fp2 = f'{output_dir}/tpcc_normalized_convergence.png'
plt.savefig(fp2, dpi=150, bbox_inches='tight')
print(f'[2/3] {fp2}')
plt.close()

# ============================================================
# 图3: 论文柱状三联图
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
fig.suptitle('TPC-C: Performance vs Efficiency — Three Tuning Methods',
             fontsize=14, fontweight='bold', y=1.02)

# (a) 性能提升
ax = axes[0]
bars = ax.bar(methods, improve_vals, color=colors, alpha=0.85, edgecolor='black', linewidth=0.8)
for bar, val in zip(bars, improve_vals):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 8,
            f'+{val:.0f}%', ha='center', va='bottom', fontsize=13, fontweight='bold')
ax.set_ylabel('Performance Improvement (%)', fontsize=11, fontweight='bold')
ax.set_title('(a) Overall Improvement', fontsize=12, fontweight='bold')
ax.set_ylim(0, max(improve_vals) * 1.15)
ax.grid(True, alpha=0.3, linestyle='--', axis='y')

# (b) 迭代效率
ax = axes[1]
bars = ax.bar(methods, avg_gain, color=colors, alpha=0.85, edgecolor='black', linewidth=0.8)
for bar, val in zip(bars, avg_gain):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
            f'{val:.1f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
ax.set_ylabel('Avg TPS Gain per Iteration', fontsize=11, fontweight='bold')
ax.set_title('(b) Iteration Efficiency', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, linestyle='--', axis='y')

# (c) 迭代数/崩溃
ax = axes[2]
x = np.arange(3)
w = 0.3
b1 = ax.bar(x - w/2, iter_counts, w, color=colors, alpha=0.85, edgecolor='black', linewidth=0.8, label='Iterations')
b2 = ax.bar(x + w/2, crash_counts, w, color=[C_GP,'#95A5A6','#95A5A6'], alpha=0.45,
            edgecolor='black', linewidth=0.8, hatch='xxx', label='Crashes')
for bar, val in zip(b1, iter_counts):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height()+0.5,
            str(val), ha='center', va='bottom', fontsize=11, fontweight='bold')
for bar, val in zip(b2, crash_counts):
    if val > 0:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height()+0.3,
                str(val), ha='center', va='bottom', fontsize=10, fontweight='bold', color='red')
ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=10)
ax.set_ylabel('Count', fontsize=11, fontweight='bold')
ax.set_title('(c) Iterations & Crashes', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, linestyle='--', axis='y')

plt.tight_layout()
fp3 = f'{output_dir}/tpcc_paper_bar_chart.png'
plt.savefig(fp3, dpi=150, bbox_inches='tight')
print(f'[3/3] {fp3}')
plt.close()

print(f'\nAll charts saved to: {output_dir}/')
