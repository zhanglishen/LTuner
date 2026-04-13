#!/usr/bin/env python3
"""TPC-H 两轮三方对比实验 — 图表生成与数据汇总"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ─────────── 数据加载 ───────────
R1_DIR = 'optimization_results/comparison_real/tpch_gptuner_ltuner_no_temp_ltuner_20260406_172420'
R2_DIR = 'optimization_results/comparison_real/tpch_gptuner_ltuner_no_temp_ltuner_20260406_212400'
OUT_DIR = 'optimization_results/comparison_real/tpch_combined_analysis_v2'
os.makedirs(OUT_DIR, exist_ok=True)

def load_round(base):
    with open(f'{base}/comparison_real_tpch.json') as f:
        main = json.load(f)
    baseline_lat = main['baseline_latency']

    # GPTuner
    gp = main['gptuner']
    gp_lat = gp['iteration_latency']

    # LTuner No Temp
    with open(f'{base}/ltuner_no_temp/ltuner_result.json') as f:
        lv2_raw = json.load(f)
    lv2_lat = [r['latency'] for r in lv2_raw['history']]
    lv2_configs = [r.get('config', {}) for r in lv2_raw['history']]

    # Enhanced LTuner
    with open(f'{base}/ltuner/ltuner_result.json') as f:
        lv3_raw = json.load(f)
    lv3_lat = [r['latency'] for r in lv3_raw['history']]
    lv3_configs = [r.get('config', {}) for r in lv3_raw['history']]

    return {
        'baseline': baseline_lat,
        'gp_lat': gp_lat, 'gp_time': gp['total_time_seconds'],
        'lv2_lat': lv2_lat, 'lv2_configs': lv2_configs,
        'lv2_time': lv2_raw.get('total_time_seconds', 0),
        'lv3_lat': lv3_lat, 'lv3_configs': lv3_configs,
        'lv3_time': lv3_raw.get('total_time_seconds', 0),
    }

r1 = load_round(R1_DIR)
r2 = load_round(R2_DIR)

# ─────────── 颜色 ───────────
C_GP, C_V2, C_V3 = '#e74c3c', '#3498db', '#2ecc71'

# ─────────── 辅助 ───────────
def cummin(arr):
    out, m = [], float('inf')
    for v in arr:
        m = min(m, v)
        out.append(m)
    return out

def ms(us):
    return us / 1000.0

# ───────── 图1: 两轮各自的收敛曲线 (1×2) ─────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for idx, (rd, title) in enumerate([(r1, 'Round 1'), (r2, 'Round 2')]):
    ax = axes[idx]
    bl = rd['baseline']

    gp_n = len(rd['gp_lat'])
    gp_cm = cummin(rd['gp_lat'])
    ax.plot(range(1, gp_n+1), [ms(x) for x in rd['gp_lat']], '.', color=C_GP, alpha=0.25, ms=3)
    ax.plot(range(1, gp_n+1), [ms(x) for x in gp_cm], '-', color=C_GP, lw=2, label=f'GPTuner (best={ms(min(rd["gp_lat"])):.0f}ms)')

    lv2_n = len(rd['lv2_lat'])
    lv2_cm = cummin(rd['lv2_lat'])
    ax.plot(range(1, lv2_n+1), [ms(x) for x in rd['lv2_lat']], '.', color=C_V2, alpha=0.25, ms=4)
    ax.plot(range(1, lv2_n+1), [ms(x) for x in lv2_cm], 'o-', color=C_V2, lw=2, ms=4, label=f'LTuner NoTemp (best={ms(min(rd["lv2_lat"])):.0f}ms)')

    lv3_n = len(rd['lv3_lat'])
    lv3_cm = cummin(rd['lv3_lat'])
    ax.plot(range(1, lv3_n+1), [ms(x) for x in rd['lv3_lat']], '.', color=C_V3, alpha=0.25, ms=4)
    ax.plot(range(1, lv3_n+1), [ms(x) for x in lv3_cm], 's-', color=C_V3, lw=2, ms=4, label=f'Enhanced LTuner (best={ms(min(rd["lv3_lat"])):.0f}ms)')

    ax.axhline(y=ms(bl), color='gray', ls='--', lw=1, alpha=0.7, label=f'Baseline ({ms(bl):.0f}ms)')
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Avg Latency (ms)', fontsize=12)
    ax.set_title(f'{title} — Convergence (lower is better)', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
fp1 = os.path.join(OUT_DIR, 'tpch_convergence_two_rounds.png')
fig.savefig(fp1, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {fp1}")

# ───────── 图2: 两轮汇总柱状图 (延迟改善 + 效率) ─────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# (a) 最优延迟
methods = ['GPTuner\n(BO)', 'LTuner\n(No Temp)', 'Enhanced\nLTuner']
colors = [C_GP, C_V2, C_V3]

r1_best = [ms(min(r1['gp_lat'])), ms(min(r1['lv2_lat'])), ms(min(r1['lv3_lat']))]
r2_best = [ms(min(r2['gp_lat'])), ms(min(r2['lv2_lat'])), ms(min(r2['lv3_lat']))]

ax = axes[0]
x = np.arange(3)
w = 0.35
bars1 = ax.bar(x - w/2, r1_best, w, color=colors, alpha=0.7, edgecolor='white', label='Round 1')
bars2 = ax.bar(x + w/2, r2_best, w, color=colors, alpha=1.0, edgecolor='white', label='Round 2')
for b, v in zip(bars1, r1_best):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+10, f'{v:.0f}', ha='center', va='bottom', fontsize=8)
for b, v in zip(bars2, r2_best):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+10, f'{v:.0f}', ha='center', va='bottom', fontsize=8)
ax.axhline(y=ms(r1['baseline']), color='gray', ls='--', alpha=0.5, label='Baseline R1')
ax.axhline(y=ms(r2['baseline']), color='gray', ls=':', alpha=0.5, label='Baseline R2')
ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=10)
ax.set_ylabel('Best Latency (ms)', fontsize=11)
ax.set_title('(a) Best Latency (lower is better)', fontsize=12, fontweight='bold')
ax.legend(fontsize=8, loc='upper left')
ax.grid(axis='y', alpha=0.3)

# (b) 延迟改善百分比
r1_impr = [(r1['baseline']-min(r1['gp_lat']))/r1['baseline']*100,
           (r1['baseline']-min(r1['lv2_lat']))/r1['baseline']*100,
           (r1['baseline']-min(r1['lv3_lat']))/r1['baseline']*100]
r2_impr = [(r2['baseline']-min(r2['gp_lat']))/r2['baseline']*100,
           (r2['baseline']-min(r2['lv2_lat']))/r2['baseline']*100,
           (r2['baseline']-min(r2['lv3_lat']))/r2['baseline']*100]

ax = axes[1]
bars1 = ax.bar(x - w/2, r1_impr, w, color=colors, alpha=0.7, edgecolor='white')
bars2 = ax.bar(x + w/2, r2_impr, w, color=colors, alpha=1.0, edgecolor='white')
for b, v in zip(bars1, r1_impr):
    ax.text(b.get_x()+b.get_width()/2, max(v, 0)+1, f'{v:.1f}%', ha='center', va='bottom', fontsize=8)
for b, v in zip(bars2, r2_impr):
    ax.text(b.get_x()+b.get_width()/2, max(v, 0)+1, f'{v:.1f}%', ha='center', va='bottom', fontsize=8)
ax.axhline(y=0, color='black', lw=0.5)
ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=10)
ax.set_ylabel('Latency Reduction (%)', fontsize=11)
ax.set_title('(b) Latency Improvement (higher is better)', fontsize=12, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# (c) 时间效率
r1_time = [r1['gp_time']/60, r1['lv2_time']/60, r1['lv3_time']/60]
r2_time = [r2['gp_time']/60, r2['lv2_time']/60, r2['lv3_time']/60]

ax = axes[2]
bars1 = ax.bar(x - w/2, r1_time, w, color=colors, alpha=0.7, edgecolor='white')
bars2 = ax.bar(x + w/2, r2_time, w, color=colors, alpha=1.0, edgecolor='white')
for b, v in zip(bars1, r1_time):
    ax.text(b.get_x()+b.get_width()/2, v+0.5, f'{v:.0f}m', ha='center', va='bottom', fontsize=8)
for b, v in zip(bars2, r2_time):
    ax.text(b.get_x()+b.get_width()/2, v+0.5, f'{v:.0f}m', ha='center', va='bottom', fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=10)
ax.set_ylabel('Total Time (minutes)', fontsize=11)
ax.set_title('(c) Total Optimization Time', fontsize=12, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
fp2 = os.path.join(OUT_DIR, 'tpch_bar_comparison.png')
fig.savefig(fp2, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {fp2}")

# ───────── 图3: 两轮平均的综合仪表板 (2×2) ─────────
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# (a) 两轮平均改善
avg_impr = [(r1_impr[i]+r2_impr[i])/2 for i in range(3)]
ax = axes[0, 0]
bars = ax.bar(methods, avg_impr, color=colors, edgecolor='white', width=0.5)
for b, v in zip(bars, avg_impr):
    ax.text(b.get_x()+b.get_width()/2, max(v, 0)+1, f'{v:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.axhline(y=0, color='black', lw=0.5)
ax.set_ylabel('Avg Latency Reduction (%)', fontsize=11)
ax.set_title('(a) Average Improvement (2 rounds)', fontsize=13, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# (b) 迭代次数
r1_iters = [len(r1['gp_lat']), len(r1['lv2_lat']), len(r1['lv3_lat'])]
r2_iters = [len(r2['gp_lat']), len(r2['lv2_lat']), len(r2['lv3_lat'])]
avg_iters = [(r1_iters[i]+r2_iters[i])/2 for i in range(3)]

ax = axes[0, 1]
bars = ax.bar(methods, avg_iters, color=colors, edgecolor='white', width=0.5)
for b, v in zip(bars, avg_iters):
    ax.text(b.get_x()+b.get_width()/2, v+0.5, f'{v:.0f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.set_ylabel('Avg Iterations', fontsize=11)
ax.set_title('(b) Average Iterations Used', fontsize=13, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# (c) 每轮延迟降低量（效率）
r1_eff = [(r1['baseline']-min(r1['gp_lat']))/len(r1['gp_lat']),
          (r1['baseline']-min(r1['lv2_lat']))/len(r1['lv2_lat']),
          (r1['baseline']-min(r1['lv3_lat']))/len(r1['lv3_lat'])]
r2_eff = [(r2['baseline']-min(r2['gp_lat']))/len(r2['gp_lat']),
          (r2['baseline']-min(r2['lv2_lat']))/len(r2['lv2_lat']),
          (r2['baseline']-min(r2['lv3_lat']))/len(r2['lv3_lat'])]
avg_eff = [(r1_eff[i]+r2_eff[i])/2 for i in range(3)]

ax = axes[1, 0]
bars = ax.bar(methods, [ms(e) for e in avg_eff], color=colors, edgecolor='white', width=0.5)
for b, v in zip(bars, avg_eff):
    label = f'{ms(v):.1f}ms/iter'
    ax.text(b.get_x()+b.get_width()/2, max(ms(v), 0)+0.2, label, ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.axhline(y=0, color='black', lw=0.5)
ax.set_ylabel('Avg Latency Reduction per Iter (ms)', fontsize=11)
ax.set_title('(c) Optimization Efficiency', fontsize=13, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# (d) 结果表
ax = axes[1, 1]
ax.axis('off')
r1_bl_ms, r2_bl_ms = ms(r1['baseline']), ms(r2['baseline'])
table_data = [
    ['', 'GPTuner (BO)', 'LTuner (NoTemp)', 'Enhanced LTuner'],
    ['R1 Baseline', f'{r1_bl_ms:.0f}ms', f'{r1_bl_ms:.0f}ms', f'{r1_bl_ms:.0f}ms'],
    ['R1 Best Lat', f'{r1_best[0]:.0f}ms', f'{r1_best[1]:.0f}ms', f'{r1_best[2]:.0f}ms'],
    ['R1 Improve', f'{r1_impr[0]:.1f}%', f'{r1_impr[1]:.1f}%', f'{r1_impr[2]:.1f}%'],
    ['R1 Iters', str(r1_iters[0]), str(r1_iters[1]), str(r1_iters[2])],
    ['R2 Baseline', f'{r2_bl_ms:.0f}ms', f'{r2_bl_ms:.0f}ms', f'{r2_bl_ms:.0f}ms'],
    ['R2 Best Lat', f'{r2_best[0]:.0f}ms', f'{r2_best[1]:.0f}ms', f'{r2_best[2]:.0f}ms'],
    ['R2 Improve', f'{r2_impr[0]:.1f}%', f'{r2_impr[1]:.1f}%', f'{r2_impr[2]:.1f}%'],
    ['R2 Iters', str(r2_iters[0]), str(r2_iters[1]), str(r2_iters[2])],
    ['Avg Improve', f'{avg_impr[0]:.1f}%', f'{avg_impr[1]:.1f}%', f'{avg_impr[2]:.1f}%'],
]
table = ax.table(cellText=table_data, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.0, 1.6)
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_facecolor('#f0f0f0')
        cell.set_text_props(fontweight='bold')
    if col == 0:
        cell.set_text_props(fontweight='bold')
ax.set_title('(d) Summary Table', fontsize=13, fontweight='bold', pad=20)

plt.tight_layout()
fp3 = os.path.join(OUT_DIR, 'tpch_dashboard.png')
fig.savefig(fp3, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {fp3}")

print("\nAll charts generated successfully!")
print(f"Output directory: {OUT_DIR}")
